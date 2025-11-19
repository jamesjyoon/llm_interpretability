from __future__ import annotations

__doc__ = """Utility for running zero-shot and LoRA-fine-tuned LLaMA style models on binary classification datasets.

This module is designed so it can be executed end-to-end on Google Colab. It
loads a dataset, evaluates a zero-shot baseline, optionally fine-tunes a LoRA
adapter, and generates LIME explanations for both models.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, NoReturn, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import matthews_corrcoef

try:
    from lime.lime_text import LimeTextExplainer  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("The `lime` package is required for interpretability. Install it via `pip install lime`.") from exc

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    default_data_collator,
    set_seed,
    Trainer,
    TrainingArguments,
)

try:  # pragma: no cover - defensive import for nicer gated model errors
    from huggingface_hub import login as hf_login
    from huggingface_hub.errors import GatedRepoError, RepoAccessError
except Exception:  # pragma: no cover - fallback when huggingface_hub is unavailable
    hf_login = None
    GatedRepoError = RepoAccessError = type("_DummyHFError", (Exception,), {})


HF_ACCESS_ERRORS = (OSError, GatedRepoError, RepoAccessError)


def _raise_hf_access_error(target: str, model_name: str, exc: Exception) -> NoReturn:
    """Surface actionable guidance when gated Hugging Face assets are requested."""

    base_message = (
        f"Failed to load the {target} for `{model_name}`.\n"
        f"If the repository is gated, visit https://huggingface.co/{model_name} to request access "
        "and authenticate before rerunning the script, for example by executing\n"
        "`from huggingface_hub import login; login(\"YOUR_TOKEN\")` in your Colab runtime.\n"
        "Alternatively, rerun with `--model-name` set to a public checkpoint."
    )
    raise SystemExit(f"{base_message}\nOriginal error: {exc}") from exc


LabelTokenMap = Dict[int, int]


class RestrictedLabelLogitsProcessor(LogitsProcessor):
    """Force generation to stay within a fixed label vocabulary.

    Some base checkpoints occasionally emit stray punctuation or whitespace tokens when
    asked to generate a single-digit label. This logits processor masks all
    vocabulary entries except the provided label tokens so greedy decoding cannot
    wander outside the expected label space.
    """

    def __init__(self, allowed_token_ids: Sequence[int]):
        if not allowed_token_ids:
            raise ValueError("At least one label token id must be supplied for restriction.")
        self.allowed_token_ids = torch.tensor(sorted(set(int(t) for t in allowed_token_ids)))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:  # type: ignore[override]
        if self.allowed_token_ids.max().item() >= scores.size(-1):
            raise ValueError("Allowed token id exceeds the vocabulary size.")

        original = scores
        restricted = torch.full_like(original, torch.finfo(original.dtype).min)
        allowed = self.allowed_token_ids.to(original.device)
        expanded = allowed.unsqueeze(0).expand(original.size(0), -1)
        restricted.scatter_(1, expanded, original.gather(1, expanded))
        return restricted


def _maybe_login_to_hf(token: Optional[str]) -> None:
    """Authenticate with Hugging Face when a token is supplied."""

    if not token:
        return
    if hf_login is None:  # pragma: no cover - import guard for minimal installs
        print(
            "Hugging Face token provided but `huggingface_hub` is unavailable; install it "
            "to enable automatic authentication."
        )
        return
    hf_login(token, add_to_git_credential=False)


def _configure_cuda_allocator() -> None:
    """Set a fragmentation-friendly CUDA allocator configuration when available."""

    if not torch.cuda.is_available():
        return

    env_keys = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    recommended = "expandable_segments:True"

    for env_key in env_keys:
        current = os.environ.get(env_key)
        if current is None:
            os.environ[env_key] = recommended
            print(
                f"Set {env_key}=expandable_segments:True to reduce CUDA memory fragmentation."
            )
        elif recommended not in current:
            os.environ[env_key] = f"{current},{recommended}"
            print(
                f"Appended expandable_segments:True to {env_key} to reduce CUDA memory fragmentation."
            )


def _load_label_token_map(tokenizer, label_space: Sequence[int]) -> LabelTokenMap:
    """Return a mapping from dataset labels to their token ids.

    LLaMA-style tokenizers encode numbers with a leading space as a single
    token (e.g., ``" 0"`` becomes ``"▁0"``).  Using the space-prefixed
    representation ensures the labels align with how prompts are constructed
    elsewhere in this module and avoids situations where ``"0"`` would be
    split across multiple tokens.
    """

    if not label_space:
        raise ValueError("At least one label must be provided to build the token map.")

    label_token_map: LabelTokenMap = {}
    for label in label_space:
        label_text = f" {label}"
        token_ids = tokenizer(
            label_text, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        if not token_ids:
            raise ValueError(f"Tokenizer could not encode label {label}.")
        label_token_map[int(label)] = token_ids[-1]
    return label_token_map


@dataclass
class ExperimentConfig:
    """Configuration for the classification experiment."""
    model_name: str = "meta-llama/Llama-3.2-1B"
    dataset_name: str = "mteb/tweet_sentiment_extraction"
    dataset_config: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "test"
    text_field: str = "text"
    label_field: str = "label"
    train_subset: Optional[int] = 800
    eval_subset: Optional[int] = 200
    random_seed: int = 42
    learning_rate: float = 2e-4
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    max_seq_length: int = 516
    max_target_length: int = 4
    output_dir: str = "outputs/imdb"
    interpretability_example_count: int = 5
    interpretability_batch_size: int = 8
    lime_num_features: int = 10
    lime_num_samples: int = 512
    run_lime: bool = True
    load_in_4bit: bool = True
    label_space: Optional[Sequence[int]] = (0, 1)


class PromptFormatter:
    """Converts sentences into classification prompts."""

    def __init__(self, label_space: Sequence[int]) -> None:
        self.label_space = list(label_space)
        label_list = ", ".join(str(label) for label in self.label_space)
        if set(self.label_space) == {0, 1}:
            instruction = "Respond with only one of the digits 0 (for negative) or 1 (for positive)."
        else:
            instruction = (
                "Respond with only one of the digits "
                + label_list
                + " to indicate the sentiment class."
            )
        self.template = "{instruction}\nTweet: {sentence}\nLabel:"
        self.instruction = instruction

    def build_prompt(self, sentence: str) -> str:
        return self.template.format(instruction=self.instruction, sentence=sentence)


def _prepare_dataset(
    dataset: DatasetDict, config: ExperimentConfig, tokenizer, formatter: PromptFormatter
) -> DatasetDict:
    required_splits = {config.train_split, config.eval_split}
    if not required_splits.issubset(dataset):
        raise ValueError(
            (
                f"Dataset `{config.dataset_name}`"
                + (f" with config `{config.dataset_config}`" if config.dataset_config else "")
                + f" must contain the splits {sorted(required_splits)}."
            )
        )

    mask_notice = {"emitted": False}

    def _format_examples(examples):
        prompts = [formatter.build_prompt(sentence) for sentence in examples[config.text_field]]
        full_sequences = [
            f"{prompt} {label}" for prompt, label in zip(prompts, examples[config.label_field])
        ]
        model_inputs = tokenizer(
            full_sequences,
            max_length=config.max_seq_length,
            truncation=True,
            padding="max_length",
        )

        # The causal-LM objective should only supervise the final label token. Mask the
        # prompt portion of each sequence with ``-100`` so the cross-entropy loss
        # focuses on the classification target instead of the copied instruction text.
        attention = model_inputs["attention_mask"]
        labels = []
        for input_ids, mask in zip(model_inputs["input_ids"], attention):
            masked = [-100] * len(input_ids)
            # ``attention_mask`` marks non-padding tokens with 1s. The final
            # non-padding position corresponds to the label token we appended.
            seq_length = int(sum(mask))
            if seq_length > 0:
                label_index = seq_length - 1
                masked[label_index] = input_ids[label_index]
            labels.append(masked)
            if not mask_notice["emitted"]:
                print(
                    "Masking prompt tokens with -100 so the fine-tuning loss only supervises the appended label token."
                )
                mask_notice["emitted"] = True
        model_inputs["labels"] = labels
        return model_inputs

    remove_columns = sorted({
        column
        for split in required_splits
        for column in dataset[split].column_names
    })
    processed = dataset.map(
        _format_examples,
        batched=True,
        remove_columns=remove_columns,
    )

    if config.train_subset:
        train_dataset = processed[config.train_split].shuffle(seed=config.random_seed)
        train_count = min(config.train_subset, train_dataset.num_rows)
        processed[config.train_split] = train_dataset.select(range(train_count))
    if config.eval_subset:
        validation_dataset = processed[config.eval_split].shuffle(seed=config.random_seed)
        validation_count = min(config.eval_subset, validation_dataset.num_rows)
        processed[config.eval_split] = validation_dataset.select(range(validation_count))
    return processed


def _autoscale_for_device_capacity(config: ExperimentConfig) -> None:
    """Lower max sequence length and subsets on small GPUs to mitigate OOM."""

    if not torch.cuda.is_available():
        return

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # Prefer keeping the user-requested configuration when ample memory exists.
    if total_gb >= 22:
        return

    # Tighten sequence length and dataset sizes when running on ~16 GB cards.
    recommended_max_seq_length = min(config.max_seq_length, 1024)
    recommended_train_subset = None if config.train_subset is None else min(config.train_subset, 3500)
    recommended_eval_subset = None if config.eval_subset is None else min(config.eval_subset, 1400)

    if recommended_max_seq_length < config.max_seq_length:
        print(
            "Detected <22 GB GPU; lowering max_seq_length from "
            f"{config.max_seq_length} to {recommended_max_seq_length} to reduce activation memory."
        )
        config.max_seq_length = recommended_max_seq_length

    if recommended_train_subset is not None and recommended_train_subset < config.train_subset:
        print(
            f"Reducing train_subset from {config.train_subset} to {recommended_train_subset} "
            "to shorten fine-tuning batches on constrained GPUs."
        )
        config.train_subset = recommended_train_subset

    if recommended_eval_subset is not None and recommended_eval_subset < config.eval_subset:
        print(
            f"Reducing eval_subset from {config.eval_subset} to {recommended_eval_subset} "
            "to keep evaluation memory usage lower."
        )
        config.eval_subset = recommended_eval_subset


def _truncate_tokenized_dataset(dataset: DatasetDict, max_length: int) -> DatasetDict:
    """Trim tokenized columns to a smaller maximum length to reduce GPU memory usage."""

    def _truncate(example):
        for key in ("input_ids", "attention_mask", "labels"):
            if key in example:
                example[key] = example[key][:max_length]
        return example

    print(f"Truncating tokenized dataset to max_seq_length={max_length} to mitigate OOM.")
    return dataset.map(_truncate, load_from_cache_file=False)


def _prepare_zero_shot_texts(
    config: ExperimentConfig, original_dataset: DatasetDict
) -> Tuple[List[str], List[int]]:
    validation_split = original_dataset[config.eval_split]
    if config.eval_subset:
        eval_count = min(config.eval_subset, len(validation_split))
        validation_split = validation_split.shuffle(seed=config.random_seed)
        validation_split = validation_split.select(range(eval_count))
    texts = list(validation_split[config.text_field])
    labels = list(validation_split[config.label_field])
    return texts, labels


def _resolve_label_space(config: ExperimentConfig, dataset: DatasetDict) -> List[int]:
    """Infer the set of labels present in the dataset if not provided explicitly."""

    if config.label_space is not None:
        return sorted({int(label) for label in config.label_space})

    label_values: set[int] = set()
    for split in {config.train_split, config.eval_split} & set(dataset.keys()):
        column = dataset[split][config.label_field]
        label_values.update(int(label) for label in column)

    if not label_values:
        raise ValueError("Unable to infer label space from the dataset; please specify `label_space`.")

    return sorted(label_values)


def _filter_dataset_by_labels(
    dataset: DatasetDict, config: ExperimentConfig, label_space: Sequence[int]
) -> DatasetDict:
    """Restrict the dataset to the requested label ids."""

    allowed_labels = {int(label) for label in label_space}

    def predicate(example):
        return int(example[config.label_field]) in allowed_labels

    return dataset.filter(predicate)


def _classification_probabilities(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: Sequence[str],
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
    *,
    max_batch_size: Optional[int] = None,
) -> np.ndarray:
    # Lime can request thousands of perturbed samples at once. Microbatch the
    # probability computation to avoid a single oversized forward pass on the GPU.
    batch_size = max_batch_size or len(prompts)
    probabilities: List[np.ndarray] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        sequence_lengths = inputs["attention_mask"].sum(dim=-1) - 1
        final_logits = logits[torch.arange(logits.size(0), device=device), sequence_lengths]
        ordered_labels = list(sorted(label_token_map))
        label_token_ids = torch.tensor(
            [label_token_map[label] for label in ordered_labels], device=device
        )
        label_logits = final_logits[:, label_token_ids]
        probs = torch.softmax(label_logits, dim=-1)
        probabilities.append(probs.detach().cpu().numpy())

    return np.concatenate(probabilities, axis=0)


def _build_probability_fn(
    model: AutoModelForCausalLM,
    tokenizer,
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
    *,
    max_batch_size: Optional[int] = None,
    formatter: Optional[PromptFormatter] = None,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Return a callable that exposes class probabilities for raw texts."""

    def _predict(batch_texts: Sequence[str]) -> np.ndarray:
        normalized: List[str]
        if isinstance(batch_texts, np.ndarray):
            normalized = [str(item) for item in batch_texts.tolist()]
        else:
            normalized = [str(item) for item in batch_texts]
        if formatter is not None:
            normalized = [formatter.build_prompt(text) for text in normalized]
        return _classification_probabilities(
            model,
            tokenizer,
            normalized,
            label_token_map,
            device,
            max_length,
            max_batch_size=max_batch_size,
        )

    return _predict


def _generation_eos_ids(tokenizer, label_token_map: LabelTokenMap) -> List[int]:
    eos_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    for label_id in label_token_map.values():
        if label_id not in eos_ids:
            eos_ids.append(label_id)
    return eos_ids


def _generate_class_predictions(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: Sequence[str],
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
) -> Tuple[List[int], np.ndarray]:
    inputs = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    eos_token_id = _generation_eos_ids(tokenizer, label_token_map)
    logits_processor = LogitsProcessorList(
        [RestrictedLabelLogitsProcessor(label_token_map.values())]
    )
    model.eval()
    with torch.no_grad():
        generation = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=logits_processor,
            return_dict_in_generate=True,
            output_scores=True,
        )

    scores = generation.scores[0]
    ordered_labels = list(sorted(label_token_map))
    label_token_ids = torch.tensor(
        [label_token_map[label] for label in ordered_labels], device=scores.device
    )
    label_logits = scores[:, label_token_ids]
    probs = torch.softmax(label_logits, dim=-1)
    pred_indices = probs.argmax(dim=-1).detach().cpu().tolist()
    predictions = [ordered_labels[index] for index in pred_indices]

    generated_tokens = generation.sequences[:, -1]
    mismatched_indices: List[int] = []
    for idx, token_id in enumerate(generated_tokens.tolist()):
        if token_id not in label_token_map.values():
            mismatched_indices.append(idx)
    if mismatched_indices:
        print(
            "Warning: the model generated tokens outside the expected {0,1} label space "
            f"for batch indices {mismatched_indices}. Predictions fall back to argmax probabilities."
        )

    return predictions, probs.detach().cpu().numpy()


def _batched(iterator: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(iterator), batch_size):
        yield iterator[i : i + batch_size]


def _build_confusion_matrix(
    labels: Sequence[int], predictions: Sequence[int], ordered_labels: Sequence[int]
) -> torch.Tensor:
    """Construct a confusion matrix aligned to ``ordered_labels``."""

    label_to_index = {int(label): idx for idx, label in enumerate(ordered_labels)}
    matrix = torch.zeros((len(ordered_labels), len(ordered_labels)), dtype=torch.long)
    for true_label, pred_label in zip(labels, predictions):
        if int(true_label) not in label_to_index or int(pred_label) not in label_to_index:
            continue
        row = label_to_index[int(true_label)]
        col = label_to_index[int(pred_label)]
        matrix[row, col] += 1
    return matrix


def _per_class_metrics(
    confusion: torch.Tensor, ordered_labels: Sequence[int]
) -> Dict[str, Dict[str, float]]:
    """Compute precision/recall/F1/support for every class."""

    per_class: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(ordered_labels):
        tp = confusion[idx, idx].item()
        predicted = confusion[:, idx].sum().item()
        actual = confusion[idx, :].sum().item()
        precision = tp / predicted if predicted > 0 else 0.0
        recall = tp / actual if actual > 0 else 0.0
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
        per_class[str(int(label))] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(actual),
        }
    return per_class


def _weighted_average(metric: torch.Tensor, weights: torch.Tensor) -> float:
    total = float(weights.sum().item())
    if total <= 0:
        return 0.0
    return float((metric * weights).sum().item() / total)


def _aggregate_metrics(
    confusion: torch.Tensor,
    per_class: Dict[str, Dict[str, float]],
    ordered_labels: Sequence[int],
) -> Tuple[float, float, float, float]:
    total = float(confusion.sum().item())
    accuracy = float(confusion.diag().sum().item() / total) if total > 0 else 0.0

    supports = torch.tensor(
        [per_class[str(int(label))]["support"] for label in ordered_labels], dtype=torch.float32
    )
    precisions = torch.tensor(
        [per_class[str(int(label))]["precision"] for label in ordered_labels], dtype=torch.float32
    )
    recalls = torch.tensor(
        [per_class[str(int(label))]["recall"] for label in ordered_labels], dtype=torch.float32
    )
    f1_scores = torch.tensor(
        [per_class[str(int(label))]["f1"] for label in ordered_labels], dtype=torch.float32
    )

    if len(ordered_labels) == 2:
        if 1 in ordered_labels:
            positive_idx = ordered_labels.index(1)
        else:
            positive_idx = len(ordered_labels) - 1
        precision = float(precisions[positive_idx].item())
        recall = float(recalls[positive_idx].item())
        f1 = float(f1_scores[positive_idx].item())
    else:
        precision = _weighted_average(precisions, supports)
        recall = _weighted_average(recalls, supports)
        f1 = _weighted_average(f1_scores, supports)

    return accuracy, precision, recall, f1


def evaluate_zero_shot(
    model: AutoModelForCausalLM,
    tokenizer,
    texts: Sequence[str],
    labels: Sequence[int],
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
    formatter: PromptFormatter,
    batch_size: int = 8,
) -> Dict[str, float]:
    predictions: List[int] = []
    probability_rows: List[List[float]] = []
    ordered_labels = list(sorted(label_token_map))
    formatted_prompts = [formatter.build_prompt(text) for text in texts]
    for batch_prompts in _batched(formatted_prompts, batch_size):
        preds, probs = _generate_class_predictions(
            model,
            tokenizer,
            batch_prompts,
            label_token_map,
            device,
            max_length,
        )
        predictions.extend(preds)
        probability_rows.extend(probs.astype(float).tolist())

    confusion = _build_confusion_matrix(labels, predictions, ordered_labels)
    per_class = _per_class_metrics(confusion, ordered_labels)
    accuracy, precision, recall, f1 = _aggregate_metrics(confusion, per_class, ordered_labels)
    mcc = 0.0
    if predictions:
        try:
            mcc = float(matthews_corrcoef(list(labels), list(predictions)))
        except ValueError:
            mcc = 0.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }

    if probability_rows:
        probability_array = np.array(probability_rows, dtype=float)
        metrics["mean_max_probability"] = float(np.max(probability_array, axis=1).mean())
        if len(ordered_labels) == 2 and 1 in ordered_labels:
            positive_index = ordered_labels.index(1)
            metrics["positive_probability_mean"] = float(
                probability_array[:, positive_index].mean()
            )
    return metrics


def train_lora_classifier(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    processed_dataset: DatasetDict,
) -> Tuple[PeftModel, bool]:
    model_gradient_dtype = getattr(model, "dtype", torch.float32)
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    peft_model.gradient_checkpointing_enable()

    def _build_trainer(batch_size: int, dataset: DatasetDict) -> Trainer:
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            bf16=model_gradient_dtype == torch.bfloat16,
            fp16=model_gradient_dtype == torch.float16,
            gradient_checkpointing=True,
        )

        return Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=dataset[config.train_split],
            eval_dataset=dataset[config.eval_split],
            data_collator=default_data_collator,
        )

    current_batch_size = max(1, config.per_device_train_batch_size)
    current_dataset = processed_dataset
    current_max_seq_length = config.max_seq_length
    trainer: Optional[Trainer] = None
    sequence_length_reduced = False
    while True:
        trainer = _build_trainer(current_batch_size, current_dataset)
        try:
            trainer.train()
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch_size == 1:
                if current_max_seq_length > 256:
                    next_max_seq_length = max(256, current_max_seq_length // 2)
                    print(
                        "CUDA OOM at batch size 1; lowering max_seq_length from "
                        f"{current_max_seq_length} to {next_max_seq_length} and retrying."
                    )
                    current_max_seq_length = next_max_seq_length
                    config.max_seq_length = next_max_seq_length
                    current_dataset = _truncate_tokenized_dataset(
                        current_dataset, next_max_seq_length
                    )
                    sequence_length_reduced = True
                    del trainer
                    continue
                print(
                    "CUDA OOM occurred even at batch size 1 and minimum sequence length; "
                    "consider lowering `max_seq_length` or using a smaller model checkpoint."
                )
                raise
            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"CUDA OOM at batch size {current_batch_size}; retrying with batch size {next_batch_size}."
            )
            current_batch_size = next_batch_size
            del trainer
            continue

    peft_model.eval()
    return peft_model, sequence_length_reduced


def run_lime_text_explanations(
    texts: Sequence[str],
    predict_fn: Callable[[Sequence[str]], np.ndarray],
    class_names: Sequence[str],
    num_features: int,
    *,
    num_samples: int,
) -> List[Dict[str, object]]:
    """Generate LIME explanations for a handful of prompts."""

    if not texts:
        return []

    explainer = LimeTextExplainer(class_names=list(class_names))
    results: List[Dict[str, object]] = []
    for text in texts:
        explanation = explainer.explain_instance(
            text, predict_fn, num_features=num_features, num_samples=num_samples
        )
        probabilities = predict_fn([text])[0]
        results.append(
            {
                "text": text,
                "predicted_label": class_names[int(np.argmax(probabilities))],
                "prediction_confidence": float(np.max(probabilities)),
                "token_weights": [(token, float(weight)) for token, weight in explanation.as_list()],
            }
        )
    return results


def collect_text_interpretability_outputs(
    *,
    model: AutoModelForCausalLM,
    tokenizer,
    label_token_map: LabelTokenMap,
    device: torch.device,
    config: ExperimentConfig,
    texts: Sequence[str],
    formatter: PromptFormatter,
    output_prefix: str,
) -> Dict[str, object]:
    """Run interpretability suites for a given model."""

    class_names = [str(label) for label in sorted(label_token_map)]
    predict_fn = _build_probability_fn(
        model,
        tokenizer,
        label_token_map,
        device,
        config.max_seq_length,
        max_batch_size=config.interpretability_batch_size,
        formatter=formatter,
    )
    summary: Dict[str, object] = {}

    lime_samples = texts[: config.interpretability_example_count]
    lime_outputs = run_lime_text_explanations(
        lime_samples,
        predict_fn,
        class_names,
        config.lime_num_features,
        num_samples=config.lime_num_samples,
    )
    if lime_outputs:
        lime_path = os.path.join(config.output_dir, f"{output_prefix}_lime.json")
        with open(lime_path, "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(lime_outputs), handle, indent=2)
        lime_plot_path = _plot_lime_explanations(lime_outputs, config.output_dir, output_prefix)
        summary["lime"] = {
            "output_path": lime_path,
            "plot_path": lime_plot_path,
            "examples": lime_outputs,
        }
        print(
            f"Saved {output_prefix} LIME explanations for {len(lime_outputs)} "
            f"example{'s' if len(lime_outputs) != 1 else ''}."
        )

    return summary


def _ensure_json_serializable(value):
    """Recursively convert numpy/tensor objects into JSON-friendly Python types."""

    if isinstance(value, torch.Tensor):
        return _ensure_json_serializable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _ensure_json_serializable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_ensure_json_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _ensure_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (np.generic,)):
        return value.item()
    if hasattr(value, "tolist"):
        return _ensure_json_serializable(value.tolist())
    return value


def _save_interpretability_outputs(summary: Dict[str, object], output_dir: str) -> Optional[str]:
    """Persist the generated interpretability artifacts to disk.

    The interpretability summary can contain comparison statistics and
    intermediate metadata.  This helper extracts the per-model outputs for LIME
    and writes them to a single JSON file so they can be consumed after the
    experiment run.
    """

    outputs: Dict[str, Dict[str, object]] = {}
    for model_key in ("zero_shot", "fine_tuned"):
        model_summary = summary.get(model_key)
        if not isinstance(model_summary, dict):
            continue

        method_outputs: Dict[str, object] = {}
        lime_summary = model_summary.get("lime")
        if lime_summary:
            method_outputs["lime"] = lime_summary

        if method_outputs:
            outputs[model_key] = method_outputs

    if not outputs:
        return None

    output_path = os.path.join(output_dir, "interpretability_outputs.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_ensure_json_serializable(outputs), handle, indent=2)

    return output_path


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _plot_metric_bars(
    zero_shot_metrics: Dict[str, float],
    fine_tuned_metrics: Optional[Dict[str, float]],
    output_dir: str,
    *,
    require_fine_tuned: bool = False,
) -> None:
    """Visualize classification metrics and save the figure.

    The bar chart focuses on core metrics that are available for both the
    zero-shot baseline and the optional fine-tuned model.  Matplotlib is
    imported lazily so that the experiment can still run in lightweight
    environments that do not preinstall plotting libraries.
    """

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency in Colab
        print(
            "matplotlib is not installed; skipping metric visualization. Install it with `pip install matplotlib` to view the bar chart."
        )
        return

    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    zero_values = [zero_shot_metrics.get(metric, float("nan")) for metric in metrics]
    tuned_values: Optional[List[float]] = None
    if fine_tuned_metrics is not None:
        tuned_values = [fine_tuned_metrics.get(metric, float("nan")) for metric in metrics]
    elif require_fine_tuned:
        raise RuntimeError(
            "Fine-tuned metrics were expected for comparison but were unavailable; rerun with --finetune enabled."
        )
    else:
        print("Fine-tuned metrics were not provided; plotting zero-shot scores only.")

    x = np.arange(len(metrics))
    width = 0.35 if tuned_values is not None else 0.6

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2 if tuned_values is not None else x, zero_values, width, label="Zero-shot")
    if tuned_values is not None:
        ax.bar(x + width / 2, tuned_values, width, label="Fine-tuned")

    ax.set_xticks(x)
    ax.set_xticklabels([metric.upper() for metric in metrics])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    output_path = os.path.join(output_dir, "metrics_comparison.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    inline_displayed = False
    try:  # Attempt inline display for notebook and Colab workflows.
        from IPython import get_ipython  # type: ignore
        from IPython.display import display  # type: ignore
    except ImportError:
        pass
    else:
        ipython = get_ipython()
        if ipython is not None and getattr(ipython, "kernel", None) is not None:
            display(fig)
            inline_displayed = True

    plt.close(fig)

    absolute_path = os.path.abspath(output_path)
    message = f"Saved metric visualization to {absolute_path}."
    if not inline_displayed:
        message += " Inline display is unavailable in this environment; open the PNG to view the chart."
    print(message)


def _plot_confusion_matrix(
    confusion: np.ndarray, labels: Sequence[int], output_dir: str, prefix: str
) -> str:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency in Colab
        print("matplotlib is not installed; skipping confusion matrix visualization.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    tick_labels = [str(label) for label in labels]
    ax.set(
        xticks=np.arange(confusion.shape[1]),
        yticks=np.arange(confusion.shape[0]),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"{prefix.replace('_', ' ').title()} Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = confusion.max() / 2.0 if confusion.size else 0.0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(
                j,
                i,
                format(int(confusion[i, j])),
                ha="center",
                va="center",
                color="white" if confusion[i, j] > thresh else "black",
            )

    fig.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    absolute_path = os.path.abspath(output_path)
    print(f"Saved confusion matrix visualization to {absolute_path}.")
    return absolute_path


def _plot_lime_explanations(
    lime_outputs: Sequence[Dict[str, object]], output_dir: str, prefix: str
) -> Optional[str]:
    """Visualize LIME token weights for each analyzed example."""

    if not lime_outputs:
        return None

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        print(
            "matplotlib is not installed; skipping LIME visualization. Install it with `pip install matplotlib` to view the plots."
        )
        return None

    n_rows = len(lime_outputs)
    fig_height = max(3, 3 * n_rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_height))
    if n_rows == 1:
        axes = [axes]

    for ax, example in zip(axes, lime_outputs):
        token_weights = example.get("token_weights", [])  # type: ignore[assignment]
        if not token_weights:
            ax.axis("off")
            continue
        tokens, weights = zip(*token_weights)  # type: ignore[misc]
        indices = np.arange(len(tokens))
        colors = ["#2c7bb6" if weight >= 0 else "#d7191c" for weight in weights]
        ax.barh(indices, weights, color=colors)
        ax.set_yticks(indices)
        ax.set_yticklabels(tokens)
        ax.invert_yaxis()
        ax.set_xlabel("LIME weight")
        ax.set_title(f"{prefix.replace('_', ' ').title()} example LIME attribution")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_lime_plot.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    absolute_path = os.path.abspath(output_path)
    print(
        "Saved LIME visualization to "
        f"{absolute_path}. Inline display may be unavailable; open the PNG to view the chart."
    )
    return output_path


def run_experiment(args: argparse.Namespace) -> None:
    provided_token = args.huggingface_token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_TOKEN"
    )
    _maybe_login_to_hf(provided_token)
    _configure_cuda_allocator()

    config = ExperimentConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        text_field=args.text_field,
        label_field=args.label_field,
        train_subset=args.train_subset,
        eval_subset=args.eval_subset,
        run_lime=args.run_lime,
        interpretability_example_count=args.interpretability_example_count,
        interpretability_batch_size=args.interpretability_batch_size,
        lime_num_features=args.lime_num_features,
        lime_num_samples=args.lime_num_samples,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        output_dir=args.output_dir,
        label_space=args.label_space,
    )

    set_seed(config.random_seed)
    _autoscale_for_device_capacity(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.load_in_4bit and device.type != "cuda":
        print("4-bit quantization requested but CUDA is unavailable; falling back to full precision.")
        config.load_in_4bit = False

    _ensure_output_dir(config.output_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    except HF_ACCESS_ERRORS as exc:
        _raise_hf_access_error("tokenizer", config.model_name, exc)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model_kwargs = {}
    if config.load_in_4bit:
        base_model_kwargs.update({"load_in_4bit": True, "device_map": "auto"})

    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **base_model_kwargs)
    except HF_ACCESS_ERRORS as exc:
        _raise_hf_access_error("model", config.model_name, exc)
    if not config.load_in_4bit:
        model.to(device)

    if config.dataset_config:
        raw_dataset = load_dataset(config.dataset_name, config.dataset_config)
    else:
        raw_dataset = load_dataset(config.dataset_name)

    label_space = _resolve_label_space(config, raw_dataset)
    if config.label_space is not None:
        raw_dataset = _filter_dataset_by_labels(raw_dataset, config, label_space)
    label_token_map = _load_label_token_map(tokenizer, label_space)
    formatter = PromptFormatter(label_space)
    processed_dataset = _prepare_dataset(raw_dataset, config, tokenizer, formatter)

    tuned_model: Optional[PeftModel] = None
    fine_tuned_metrics: Optional[Dict[str, float]] = None
    sequence_length_reduced = False
    zero_shot_texts, zero_shot_labels = _prepare_zero_shot_texts(config, raw_dataset)
    if args.finetune:
        tuned_model, sequence_length_reduced = train_lora_classifier(
            config, model, processed_dataset
        )
        tuned_model.save_pretrained(os.path.join(config.output_dir, "lora_adapter"))

        fine_tuned_metrics = evaluate_zero_shot(
            tuned_model,
            tokenizer,
            zero_shot_texts,
            zero_shot_labels,
            label_token_map,
            device,
            config.max_seq_length,
            formatter,
        )
        print("Fine-tuned evaluation metrics:")
        print(json.dumps(_ensure_json_serializable(fine_tuned_metrics), indent=2))
        with open(os.path.join(config.output_dir, "fine_tuned_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(fine_tuned_metrics), handle, indent=2)

    zero_shot_metrics = evaluate_zero_shot(
        model,
        tokenizer,
        zero_shot_texts,
        zero_shot_labels,
        label_token_map,
        device,
        config.max_seq_length,
        formatter,
    )

    if sequence_length_reduced:
        print(
            "Evaluated zero-shot baseline after fine-tuning adjusted max_seq_length to keep comparisons aligned."
        )

    print("Zero-shot evaluation metrics:")
    print(json.dumps(_ensure_json_serializable(zero_shot_metrics), indent=2))

    with open(os.path.join(config.output_dir, "zero_shot_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(_ensure_json_serializable(zero_shot_metrics), handle, indent=2)

    _plot_metric_bars(
        zero_shot_metrics,
        fine_tuned_metrics,
        config.output_dir,
        require_fine_tuned=args.finetune,
    )
    label_order = list(sorted(label_token_map))
    if zero_shot_metrics.get("confusion_matrix"):
        _plot_confusion_matrix(
            np.array(zero_shot_metrics["confusion_matrix"]), label_order, config.output_dir, "zero_shot"
        )
    else:
        print("Zero-shot confusion matrix unavailable; skipping visualization.")
    if args.finetune and not fine_tuned_metrics:
        raise RuntimeError("Fine-tuned metrics are required to plot comparisons but were missing.")
    if fine_tuned_metrics and fine_tuned_metrics.get("confusion_matrix"):
        fine_tuned_confusion_path = _plot_confusion_matrix(
            np.array(fine_tuned_metrics["confusion_matrix"]), label_order, config.output_dir, "fine_tuned"
        )
        print(f"Saved fine-tuned confusion matrix visualization to {fine_tuned_confusion_path}.")
    elif args.finetune:
        raise RuntimeError(
            "Fine-tuned confusion matrix was not generated; ensure evaluation produced predictions."
        )

    interpretability_summary: Optional[Dict[str, object]] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            "Cleared CUDA cache before interpretability; the attribution runs operate on small batches "
            "and are unlikely to cause OOM unless the GPU is already at capacity. "
            "Lower --lime-num-samples or disable --run-lime if attribution memory errors persist."
        )

    if config.run_lime:
        lime_samples = zero_shot_texts[: config.interpretability_example_count]
        if lime_samples:
            zero_summary = collect_text_interpretability_outputs(
                model=model,
                tokenizer=tokenizer,
                label_token_map=label_token_map,
                device=device,
                config=config,
                texts=lime_samples,
                formatter=formatter,
                output_prefix="zero_shot",
            )
            if zero_summary:
                interpretability_summary = {"zero_shot": zero_summary}

            if tuned_model is not None:
                tuned_summary = collect_text_interpretability_outputs(
                    model=tuned_model,
                    tokenizer=tokenizer,
                    label_token_map=label_token_map,
                    device=device,
                    config=config,
                    texts=lime_samples,
                    formatter=formatter,
                    output_prefix="fine_tuned",
                )
                if tuned_summary:
                    tuned_lime = tuned_summary.get("lime", {})
                    tuned_plot = tuned_lime.get("plot_path") if isinstance(tuned_lime, dict) else None
                    if tuned_plot:
                        print(f"Saved fine-tuned LIME visualization to {tuned_plot}.")
                    elif args.finetune:
                        raise RuntimeError(
                            "Fine-tuned LIME plot was expected but missing; ensure matplotlib is installed "
                            "and that --run-lime remains enabled."
                        )
                    if interpretability_summary is None:
                        interpretability_summary = {}
                    interpretability_summary["fine_tuned"] = tuned_summary
                elif args.finetune:
                    raise RuntimeError("Fine-tuned LIME explanations were expected but missing.")
        else:
            print("No samples available for interpretability analysis; skipping LIME generation.")

    if interpretability_summary is not None:
        summary_path = os.path.join(config.output_dir, "interpretability_metrics.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(interpretability_summary), handle, indent=2)
        print(f"Saved interpretability comparison metrics to {summary_path}.")

        outputs_path = _save_interpretability_outputs(interpretability_summary, config.output_dir)
        if outputs_path:
            print(f"Saved interpretability outputs to {outputs_path}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a classification interpretability experiment.")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset-name", default="mteb/tweet_sentiment_extraction")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--label-field", default="label")
    parser.add_argument(
        "--label-space",
        nargs="*",
        type=int,
        default=[0, 1],
        help="Explicit list of label ids to model (defaults to binary sentiment {0,1} unless overridden)",
    )
    parser.add_argument("--train-subset", type=int, default=5000)
    parser.add_argument("--eval-subset", type=int, default=2000)
    parser.add_argument("--run-lime", action="store_true")
    parser.add_argument("--no-run-lime", dest="run_lime", action="store_false")
    parser.set_defaults(run_lime=True)
    parser.add_argument("--interpretability-example-count", type=int, default=5)
    parser.add_argument(
        "--interpretability-batch-size",
        type=int,
        default=8,
        help="Maximum batch size for interpretability prediction calls to avoid OOM.",
    )
    parser.add_argument("--lime-num-features", type=int, default=10)
    parser.add_argument(
        "--lime-num-samples",
        type=int,
        default=512,
        help="Number of perturbed samples per LIME explanation; lower to reduce GPU load.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization; reduce to lower GPU memory usage.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--output-dir", default="outputs/mteb/tweet_sentiment_extraction")
    parser.add_argument(
        "--huggingface-token",
        default=None,
        help="Personal access token for gated Hugging Face repositories."
        " If omitted, the script will fall back to the HF_TOKEN or HUGGINGFACE_TOKEN"
        " environment variables when present.",
    )
    return parser
    
if __name__ == "__main__":
    parser = build_parser()
    run_experiment(parser.parse_args())
