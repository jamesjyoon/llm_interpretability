from __future__ import annotations

__doc__ = """Utility for running zero-shot and LoRA-fine-tuned LLaMA style models on binary classification datasets.

This module is designed so it can be executed end-to-end on Google Colab. It
loads a dataset, evaluates a zero-shot baseline, optionally fine-tunes a LoRA
adapter, and computes SHAP token attributions for both models.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, NoReturn, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import matthews_corrcoef

try:
    import shap  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "The `shap` package is required for attribution analysis. Install it via `pip install shap`."
    ) from exc

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
    train_subset: Optional[int] = 500
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
    output_dir: str = "outputs/mteb/tweet_sentiment_extraction"
    run_shap: bool = True
    shap_max_evals: int = 200
    shap_example_count: int = 50
    interpretability_example_count: int = 5
    interpretability_batch_size: int = 8
    lime_num_features: int = 10
    lime_num_samples: int = 512
    shap_surrogate_samples: int = 200
    shap_top_features: int = 12
    shap_vectorizer_max_features: int = 5000
    load_in_4bit: bool = True
    label_space: Optional[Sequence[int]] = None


class PromptFormatter:
    """Converts sentences into classification prompts."""

    def __init__(self, label_space: Sequence[int]) -> None:
        self.label_space = list(label_space)
        label_list = ", ".join(str(label) for label in self.label_space)
        if set(self.label_space) == {0, 1}:
            instruction = (
                "Respond with only the digit `1` for positive sentiment and `0` for negative sentiment."
            )
        else:
            instruction = (
                "Respond with only one of the digits "
                + label_list
                + " to indicate the sentiment class."
            )
        self.template = (
            "You are a sentiment classifier.\n"
            f"{instruction}\n"
            "Tweet: {sentence}\n"
            "Label:"
        )

    def build_prompt(self, sentence: str) -> str:
        return self.template.format(sentence=sentence)


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
    config: ExperimentConfig, original_dataset: DatasetDict, formatter: PromptFormatter
) -> Tuple[List[str], List[int]]:
    validation_split = original_dataset[config.eval_split]
    if config.eval_subset:
        eval_count = min(config.eval_subset, len(validation_split))
        validation_split = validation_split.shuffle(seed=config.random_seed)
        validation_split = validation_split.select(range(eval_count))
    texts = [formatter.build_prompt(sentence) for sentence in validation_split[config.text_field]]
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
) -> Callable[[Sequence[str]], np.ndarray]:
    """Return a callable that exposes class probabilities for raw prompts."""

    def _predict(batch_texts: Sequence[str]) -> np.ndarray:
        normalized: List[str]
        if isinstance(batch_texts, np.ndarray):
            normalized = [str(item) for item in batch_texts.tolist()]
        else:
            normalized = [str(item) for item in batch_texts]
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
    prompts: Sequence[str],
    labels: Sequence[int],
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
    batch_size: int = 8,
) -> Dict[str, float]:
    predictions: List[int] = []
    probability_rows: List[List[float]] = []
    ordered_labels = list(sorted(label_token_map))
    for batch_prompts in _batched(list(prompts), batch_size):
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


def _shap_masker(tokenizer):
    return shap.maskers.Text(tokenizer)


def compute_shap_values(
    model: AutoModelForCausalLM,
    tokenizer,
    texts: Sequence[str],
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
    max_evals: int,
    algorithm: Optional[str] = None,
) -> shap.Explanation:
    masker = _shap_masker(tokenizer)

    def predict_fn(batch_texts: List[str]) -> np.ndarray:
        return _classification_probabilities(
            model,
            tokenizer,
            batch_texts,
            label_token_map,
            device,
            max_length,
        )

    output_names = [str(label) for label in sorted(label_token_map)]
    explainer_kwargs = {"output_names": output_names}
    if algorithm is not None:
        explainer_kwargs["algorithm"] = algorithm
    explainer = shap.Explainer(predict_fn, masker, **explainer_kwargs)
    return explainer(texts, max_evals=max_evals)


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


def run_shap_surrogate(
    texts: Sequence[str],
    predict_fn: Callable[[Sequence[str]], np.ndarray],
    class_names: Sequence[str],
    *,
    top_features: int,
    vectorizer_max_features: int,
    random_seed: int,
) -> Optional[Dict[str, object]]:
    """Train a tree surrogate model and explain it with SHAP."""

    if not texts:
        return None

    sample_texts = list(texts)
    probabilities = predict_fn(sample_texts)
    surrogate_labels = probabilities.argmax(axis=1)

    vectorizer = TfidfVectorizer(max_features=vectorizer_max_features, ngram_range=(1, 2))
    features = vectorizer.fit_transform(sample_texts)
    feature_names = vectorizer.get_feature_names_out()
    if features.shape[1] == 0:
        return None

    model = GradientBoostingClassifier(random_state=random_seed)
    model.fit(features.toarray(), surrogate_labels)
    explainer = shap.TreeExplainer(model)
    dense_features = features.toarray()
    shap_values = explainer.shap_values(dense_features)

    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(values) for values in shap_values], axis=0)
        per_example = stacked.mean(axis=0)
    else:
        per_example = np.abs(shap_values)
        if per_example.ndim == 3:
            per_example = per_example.mean(axis=0)

    feature_importance = per_example.mean(axis=0)
    top_count = min(top_features, len(feature_names))
    top_indices = np.argsort(-feature_importance)[:top_count]
    top_tokens = [
        {"feature": str(feature_names[idx]), "importance": float(feature_importance[idx])}
        for idx in top_indices
        if 0 <= idx < len(feature_names)
    ]

    per_example_summaries: List[Dict[str, object]] = []
    preview_examples = min(5, len(sample_texts))
    for example_idx in range(preview_examples):
        contributions = [
            {
                "feature": str(feature_names[idx]),
                "importance": float(per_example[example_idx, idx]),
            }
            for idx in top_indices
            if 0 <= idx < len(feature_names)
        ]
        per_example_summaries.append(
            {
                "text": sample_texts[example_idx],
                "predicted_label": class_names[int(surrogate_labels[example_idx])],
                "top_feature_contributions": contributions,
            }
        )

    return {
        "surrogate_training_examples": len(sample_texts),
        "top_features": top_tokens,
        "per_example_top_features": per_example_summaries,
    }


def collect_text_interpretability_outputs(
    *,
    model: AutoModelForCausalLM,
    tokenizer,
    label_token_map: LabelTokenMap,
    device: torch.device,
    config: ExperimentConfig,
    shap_texts: Sequence[str],
    tree_texts: Sequence[str],
    output_prefix: str,
) -> Tuple[Optional[shap.Explanation], Dict[str, object]]:
    """Run interpretability suites for a given model."""

    class_names = [str(label) for label in sorted(label_token_map)]
    predict_fn = _build_probability_fn(
        model,
        tokenizer,
        label_token_map,
        device,
        config.max_seq_length,
        max_batch_size=config.interpretability_batch_size,
    )
    summary: Dict[str, object] = {}

    lime_samples = shap_texts[: config.interpretability_example_count]
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
        summary["lime"] = {"output_path": lime_path, "examples": lime_outputs}
        print(
            f"Saved {output_prefix} LIME explanations for {len(lime_outputs)} "
            f"example{'s' if len(lime_outputs) != 1 else ''}."
        )

    kernel_explanation: Optional[shap.Explanation] = None
    if shap_texts:
        kernel_explanation = compute_shap_values(
            model,
            tokenizer,
            shap_texts,
            label_token_map,
            device,
            config.max_seq_length,
            config.shap_max_evals,
            algorithm="kernel",
        )
        kernel_path = os.path.join(config.output_dir, f"{output_prefix}_kernel_shap.json")
        save_shap_values(kernel_explanation, kernel_path)
        kernel_summary = summarize_shap_importance(kernel_explanation)
        kernel_summary["per_example_stats"] = _collect_shap_statistics(kernel_explanation)
        kernel_plot_path = _plot_shap_summary(kernel_explanation, config.output_dir, f"{output_prefix}_kernel")
        if kernel_plot_path:
            kernel_summary["visualization_path"] = kernel_plot_path
        kernel_summary["output_path"] = kernel_path
        summary["kernel_shap"] = kernel_summary
        print(
            f"Saved {output_prefix} KernelSHAP explanations for {len(shap_texts)} "
            f"example{'s' if len(shap_texts) != 1 else ''}."
        )

    ig_summaries: List[Dict[str, object]] = []
    for text in shap_texts[: config.interpretability_example_count]:
        ig_summary = _integrated_gradients_attributions(
            model=model,
            tokenizer=tokenizer,
            text=text,
            label_token_map=label_token_map,
            device=device,
            max_length=config.max_seq_length,
        )
        if ig_summary:
            ig_summaries.append(ig_summary)
    if ig_summaries:
        ig_path = os.path.join(config.output_dir, f"{output_prefix}_integrated_gradients.json")
        with open(ig_path, "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(ig_summaries), handle, indent=2)
        summary["integrated_gradients"] = {"output_path": ig_path, "examples": ig_summaries}
        print(
            f"Saved {output_prefix} Integrated Gradients attributions for {len(ig_summaries)} "
            f"example{'s' if len(ig_summaries) != 1 else ''}."
        )

    lrp_summaries: List[Dict[str, object]] = []
    for text in shap_texts[: config.interpretability_example_count]:
        lrp_summary = _lrp_token_attributions(
            model=model,
            tokenizer=tokenizer,
            text=text,
            label_token_map=label_token_map,
            device=device,
            max_length=config.max_seq_length,
        )
        if lrp_summary:
            lrp_summaries.append(lrp_summary)
    if lrp_summaries:
        lrp_path = os.path.join(config.output_dir, f"{output_prefix}_lrp.json")
        with open(lrp_path, "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(lrp_summaries), handle, indent=2)
        summary["lrp"] = {"output_path": lrp_path, "examples": lrp_summaries}
        print(
            f"Saved {output_prefix} Layer-wise Relevance Propagation scores for {len(lrp_summaries)} "
            f"example{'s' if len(lrp_summaries) != 1 else ''}."
        )

    attention_rollouts: List[Dict[str, object]] = []
    attention_flows: List[Dict[str, object]] = []
    self_explanations: List[Dict[str, str]] = []
    for text in shap_texts[: config.interpretability_example_count]:
        rollout = _attention_rollout_attributions(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
            max_length=config.max_seq_length,
        )
        if rollout:
            attention_rollouts.append(rollout)
        flow = _attention_flow_attributions(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
            max_length=config.max_seq_length,
        )
        if flow:
            attention_flows.append(flow)
        explanation = _generate_self_explanation(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
            max_length=config.max_seq_length,
        )
        self_explanations.append(explanation)

    if attention_rollouts:
        rollout_path = os.path.join(config.output_dir, f"{output_prefix}_attention_rollout.json")
        with open(rollout_path, "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(attention_rollouts), handle, indent=2)
        rollout_plots = _plot_attention_importances(
            examples=attention_rollouts,
            output_dir=config.output_dir,
            prefix=output_prefix,
            method="attention_rollout",
        )
        summary["attention_rollout"] = {
            "output_path": rollout_path,
            "examples": attention_rollouts,
            "plots": rollout_plots,
        }
        print(
            f"Saved {output_prefix} attention rollout explanations for {len(attention_rollouts)} "
            f"example{'s' if len(attention_rollouts) != 1 else ''}."
        )
    if attention_flows:
        flow_path = os.path.join(config.output_dir, f"{output_prefix}_attention_flow.json")
        with open(flow_path, "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(attention_flows), handle, indent=2)
        flow_plots = _plot_attention_importances(
            examples=attention_flows,
            output_dir=config.output_dir,
            prefix=output_prefix,
            method="attention_flow",
        )
        summary["attention_flow"] = {
            "output_path": flow_path,
            "examples": attention_flows,
            "plots": flow_plots,
        }
        print(
            f"Saved {output_prefix} attention flow explanations for {len(attention_flows)} "
            f"example{'s' if len(attention_flows) != 1 else ''}."
        )
    if self_explanations:
        self_exp_path = os.path.join(config.output_dir, f"{output_prefix}_self_explanations.json")
        with open(self_exp_path, "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(self_explanations), handle, indent=2)
        self_exp_text = _save_self_explanations_text(
            examples=self_explanations, output_dir=config.output_dir, prefix=output_prefix
        )
        summary["self_explanations"] = {
            "output_path": self_exp_path,
            "text_dump": self_exp_text,
            "examples": self_explanations,
        }
        print(
            f"Saved {output_prefix} self explanations for {len(self_explanations)} "
            f"example{'s' if len(self_explanations) != 1 else ''}."
        )

    shap_subset = tree_texts[: config.shap_surrogate_samples]
    shap_summary = run_shap_surrogate(
        shap_subset,
        predict_fn,
        class_names,
        top_features=config.shap_top_features,
        vectorizer_max_features=config.shap_vectorizer_max_features,
        random_seed=config.random_seed,
    )
    if shap_summary:
        tree_path = os.path.join(config.output_dir, f"{output_prefix}_shap.json")
        with open(tree_path, "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(shap_summary), handle, indent=2)
        shap_summary["output_path"] = tree_path
        summary["shap"] = shap_summary
        print(
            f"Saved {output_prefix} SHAP surrogate explanations using {len(shap_subset)} "
            f"example{'s' if len(shap_subset) != 1 else ''}."
        )

    representer = compute_representer_points(
        target_texts=shap_texts[: config.interpretability_example_count],
        reference_texts=shap_subset,
    )
    if representer:
        rep_path = os.path.join(config.output_dir, f"{output_prefix}_representer_points.json")
        with open(rep_path, "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(representer), handle, indent=2)
        summary["representer_points"] = {"output_path": rep_path, "examples": representer}
        print(
            f"Saved {output_prefix} representer points for {len(representer)} "
            f"example{'s' if len(representer) != 1 else ''}."
        )

    return kernel_explanation, summary


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


def _token_attribution_template(
    model: AutoModelForCausalLM,
    tokenizer,
    text: str,
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
) -> Tuple[Optional[List[str]], Optional[torch.Tensor], Optional[int]]:
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    label_token_ids = torch.tensor([label_token_map[label] for label in sorted(label_token_map)], device=device)

    embedding_layer = model.get_input_embeddings()
    input_embeddings = embedding_layer(inputs["input_ids"])
    attention_mask = inputs["attention_mask"]
    model.zero_grad()
    input_embeddings.requires_grad_(True)

    outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
    seq_length = int(attention_mask.sum().item())
    logits = outputs.logits[0, seq_length - 1, label_token_ids]
    target_index = int(torch.argmax(logits).item())
    logits[target_index].backward()

    grad = input_embeddings.grad
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0)) if input_embeddings.grad is not None else None
    return tokens, grad, target_index


def _integrated_gradients_attributions(
    *,
    model: AutoModelForCausalLM,
    tokenizer,
    text: str,
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
    steps: int = 20,
) -> Optional[Dict[str, object]]:
    tokens, _, target_index = _token_attribution_template(
        model, tokenizer, text, label_token_map, device, max_length
    )
    if tokens is None:
        return None

    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    attention_mask = inputs["attention_mask"]
    label_token_ids = torch.tensor([label_token_map[label] for label in sorted(label_token_map)], device=device)
    embedding_layer = model.get_input_embeddings()
    input_embeddings = embedding_layer(inputs["input_ids"])
    baseline = torch.zeros_like(input_embeddings)
    delta = input_embeddings - baseline
    total_grads = torch.zeros_like(input_embeddings)

    for alpha in torch.linspace(0, 1, steps, device=device):
        scaled = baseline + alpha * delta
        scaled.requires_grad_(True)
        model.zero_grad()
        outputs = model(inputs_embeds=scaled, attention_mask=attention_mask)
        seq_length = int(attention_mask.sum().item())
        logits = outputs.logits[0, seq_length - 1, label_token_ids]
        logits[target_index].backward()
        if scaled.grad is not None:
            total_grads += scaled.grad.detach()

    attributions = (delta * total_grads / float(steps)).squeeze(0)
    token_scores = attributions.norm(dim=-1).detach().cpu().numpy().tolist()
    return {
        "text": text,
        "target_label": int(sorted(label_token_map)[target_index]),
        "tokens": tokens,
        "attribution_scores": token_scores,
    }


def _lrp_token_attributions(
    *,
    model: AutoModelForCausalLM,
    tokenizer,
    text: str,
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
) -> Optional[Dict[str, object]]:
    tokens, grad, target_index = _token_attribution_template(
        model, tokenizer, text, label_token_map, device, max_length
    )
    if tokens is None or grad is None:
        return None

    embedding_layer = model.get_input_embeddings()
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    input_embeddings = embedding_layer(inputs["input_ids"].to(device))
    relevance = (input_embeddings * grad).squeeze(0).sum(dim=-1)
    scores = relevance.detach().cpu().numpy().tolist()

    return {
        "text": text,
        "target_label": int(sorted(label_token_map)[target_index]),
        "tokens": tokens,
        "relevance_scores": scores,
    }


def compute_representer_points(
    *, target_texts: Sequence[str], reference_texts: Sequence[str], top_k: int = 3
) -> List[Dict[str, object]]:
    if not target_texts or not reference_texts:
        return []

    vectorizer = TfidfVectorizer(max_features=5000)
    reference_matrix = vectorizer.fit_transform(reference_texts)
    target_matrix = vectorizer.transform(target_texts)
    similarity = target_matrix @ reference_matrix.T

    summaries: List[Dict[str, object]] = []
    for row_index, text in enumerate(target_texts):
        row = similarity.getrow(row_index)
        if row.nnz == 0:
            continue
        top_indices = np.argsort(-row.toarray().ravel())[:top_k]
        neighbors = []
        for idx in top_indices:
            neighbors.append(
                {
                    "reference_text": reference_texts[idx],
                    "similarity": float(row[0, idx]),
                }
            )
        summaries.append({"text": text, "representer_points": neighbors})

    return summaries


def _save_interpretability_outputs(summary: Dict[str, object], output_dir: str) -> Optional[str]:
    """Persist the generated interpretability artifacts to disk.

    The interpretability summary can contain comparison statistics and
    intermediate metadata.  This helper extracts the per-model outputs for LIME,
    KernelSHAP, SHAP, Integrated Gradients, LRP, and representer point methods
    and writes them to a single JSON file so they can be consumed after the
    experiment run.
    """

    outputs: Dict[str, Dict[str, object]] = {}
    for model_key in ("zero_shot", "fine_tuned"):
        model_summary = summary.get(model_key)
        if not isinstance(model_summary, dict):
            continue

        method_outputs: Dict[str, object] = {}
        for method in (
            "lime",
            "kernel_shap",
            "shap",
            "integrated_gradients",
            "lrp",
            "representer_points",
            "attention_rollout",
            "attention_flow",
            "self_explanations",
        ):
            method_summary = model_summary.get(method)
            if method_summary:
                method_outputs[method] = method_summary

        if method_outputs:
            outputs[model_key] = method_outputs

    if not outputs:
        return None

    output_path = os.path.join(output_dir, "interpretability_outputs.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_ensure_json_serializable(outputs), handle, indent=2)

    return output_path


def _decode_tokens(tokenizer, input_ids: torch.Tensor, seq_length: int) -> List[str]:
    return tokenizer.convert_ids_to_tokens(input_ids[:seq_length].tolist())


def _attention_rollout_attributions(
    *,
    model: AutoModelForCausalLM,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int,
) -> Optional[Dict[str, object]]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        return None

    seq_length = int(inputs["attention_mask"].sum().item())
    layer_attn = torch.stack([layer[0] for layer in attentions], dim=0)
    mean_attn = layer_attn.mean(dim=1)
    eye = torch.eye(mean_attn.size(-1), device=mean_attn.device)
    augmented = mean_attn + eye
    normalized = augmented / augmented.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    rollout = normalized[0]
    for layer in normalized[1:]:
        rollout = rollout @ layer

    target_index = seq_length - 1
    token_scores = rollout[target_index, :seq_length].detach().cpu().tolist()
    tokens = _decode_tokens(tokenizer, inputs["input_ids"][0], seq_length)

    return {
        "text": text,
        "tokens": tokens,
        "target_index": int(target_index),
        "importance": token_scores,
        "method": "attention_rollout",
    }


def _attention_flow_attributions(
    *,
    model: AutoModelForCausalLM,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int,
) -> Optional[Dict[str, object]]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        return None

    seq_length = int(inputs["attention_mask"].sum().item())
    layer_attn = torch.stack([layer[0] for layer in attentions], dim=0)
    mean_attn = layer_attn.mean(dim=1)
    eye = torch.eye(mean_attn.size(-1), device=mean_attn.device)
    normalized = (mean_attn + eye) / (mean_attn + eye).sum(dim=-1, keepdim=True).clamp_min(1e-6)

    flow = torch.zeros(normalized.size(-1), device=device)
    flow[seq_length - 1] = 1.0
    for layer in reversed(normalized):
        flow = layer.transpose(0, 1) @ flow
    token_scores = flow[:seq_length].detach().cpu().tolist()
    tokens = _decode_tokens(tokenizer, inputs["input_ids"][0], seq_length)

    return {
        "text": text,
        "tokens": tokens,
        "target_index": int(seq_length - 1),
        "importance": token_scores,
        "method": "attention_flow",
    }


def _generate_self_explanation(
    *,
    model: AutoModelForCausalLM,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int,
) -> Dict[str, str]:
    prompt = f"{text}\n\nExplain why this review is positive. Highlight key phrases."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
    explanation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {"text": text, "prompt": prompt, "explanation": explanation}


def _plot_attention_importances(
    *,
    examples: Sequence[Dict[str, object]],
    output_dir: str,
    prefix: str,
    method: str,
    top_k: int = 15,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency in Colab
        print(
            "matplotlib is not installed; skipping attention attribution plots. Install it with `pip install matplotlib` to generate the figures."
        )
        return []

    saved_paths: List[str] = []
    for index, example in enumerate(examples):
        tokens = example.get("tokens")
        scores = example.get("importance")
        if not isinstance(tokens, list) or not isinstance(scores, list) or not tokens:
            continue
        # Truncate/normalize to match lengths in case of padding.
        token_scores = np.array(scores, dtype=float)[: len(tokens)]
        ordering = np.argsort(np.abs(token_scores))[::-1]
        ordering = ordering[: min(top_k, len(ordering))]
        ordered_tokens = [tokens[pos] for pos in ordering][::-1]
        ordered_scores = [float(token_scores[pos]) for pos in ordering][::-1]

        height = max(3.0, 0.35 * len(ordered_tokens) + 1.0)
        fig, ax = plt.subplots(figsize=(9, height))
        bars = ax.barh(np.arange(len(ordered_tokens)), ordered_scores, color="#8c6bb1")
        ax.set_yticks(np.arange(len(ordered_tokens)))
        ax.set_yticklabels(ordered_tokens)
        ax.set_xlabel("Token importance")
        ax.set_title(f"Top attention weights ({method.replace('_', ' ').title()} #{index})")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        for bar, score in zip(bars, ordered_scores):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {score:.3f}", va="center")

        fig.tight_layout()
        output_path = os.path.join(
            output_dir, f"{prefix}_{method}_example_{index}.png"
        )
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        saved_paths.append(os.path.abspath(output_path))

    if saved_paths:
        print(
            f"Saved {len(saved_paths)} attention attribution plots for {method.replace('_', ' ')} to {output_dir}."
        )
    return saved_paths


def _save_self_explanations_text(
    examples: Sequence[Dict[str, str]], output_dir: str, prefix: str
) -> Optional[str]:
    if not examples:
        return None

    path = os.path.join(output_dir, f"{prefix}_self_explanations.txt")
    with open(path, "w", encoding="utf-8") as handle:
        for idx, example in enumerate(examples):
            handle.write(f"Example {idx}\n")
            handle.write("Prompt:\n")
            handle.write(example.get("prompt", "") + "\n")
            handle.write("Explanation:\n")
            handle.write(example.get("explanation", "") + "\n")
            handle.write("\n" + "-" * 40 + "\n\n")

    print(f"Saved self-explanation texts to {os.path.abspath(path)}.")
    return path


def _serialize_shap(explanation: shap.Explanation) -> Dict[str, object]:
    return {
        "values": _ensure_json_serializable(explanation.values),
        "base_values": _ensure_json_serializable(explanation.base_values),
        "data": _ensure_json_serializable(explanation.data),
        "feature_names": _ensure_json_serializable(explanation.feature_names),
        "output_names": _ensure_json_serializable(explanation.output_names),
    }


def save_shap_values(explanation: shap.Explanation, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_serialize_shap(explanation), handle, indent=2)


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _plot_metric_bars(
    zero_shot_metrics: Dict[str, float],
    fine_tuned_metrics: Optional[Dict[str, float]],
    output_dir: str,
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
) -> None:
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
    print(f"Saved confusion matrix visualization to {os.path.abspath(output_path)}.")


def _plot_shap_summary(
    explanation: shap.Explanation,
    output_dir: str,
    prefix: str,
    top_k: int = 20,
) -> Optional[str]:
    """Aggregate SHAP scores and visualize the highest-impact tokens."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency in Colab
        print("matplotlib is not installed; skipping SHAP visualization.")
        return None

    token_scores = defaultdict(float)
    for tokens, values in _iter_shap_examples(explanation):
        if not tokens:
            continue
        scores = np.abs(_normalize_token_scores(values, len(tokens)))
        if scores.size == 0:
            continue
        for token, score in zip(tokens, scores):
            token_scores[token] += float(score)

    if not token_scores:
        print(f"No SHAP scores available to visualize for {prefix}; skipping plot.")
        return None

    top_items = sorted(token_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    tokens = [token for token, _ in top_items]
    scores = [score for _, score in top_items]

    # Reverse for a top-to-bottom horizontal bar chart.
    tokens = tokens[::-1]
    scores = scores[::-1]

    height = max(4.0, 0.35 * len(tokens) + 1.0)
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(np.arange(len(tokens)), scores, color="#4c72b0")
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Total |SHAP| score")
    ax.set_title(f"Top token contributions ({prefix.replace('_', ' ').title()})")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_shap_summary.png")
    fig.savefig(output_path, dpi=200)

    inline_displayed = False
    try:
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
    message = f"Saved SHAP visualization to {absolute_path}."
    if not inline_displayed:
        message += " Inline display is unavailable in this environment; open the PNG to review the summary."
    print(message)
    return absolute_path


def _iter_object_container(container) -> Iterator:
    if isinstance(container, np.ndarray) and container.dtype == object:
        for item in container:
            yield item
    elif isinstance(container, (list, tuple)):
        for item in container:
            yield item
    else:
        yield container


def _to_float_array(value) -> np.ndarray:
    try:
        array = np.array(value)
        if array.dtype == object:
            array = np.array(array.tolist(), dtype=float)
        return array.astype(float)
    except Exception:
        return np.array([], dtype=float)


def _select_label_dimension(values: np.ndarray) -> np.ndarray:
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim == 1:
        return values
    if values.shape[-1] == 1:
        return values[..., 0]
    return values.mean(axis=-1)


def _normalize_token_scores(values: np.ndarray, token_count: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(token_count, dtype=float)
    label_values = np.asarray(_select_label_dimension(values)).reshape(-1)
    if label_values.size > token_count:
        label_values = label_values[:token_count]
    if label_values.size < token_count:
        label_values = np.pad(label_values, (0, token_count - label_values.size))
    return label_values.astype(float)


def _normalize_distribution(scores: np.ndarray) -> np.ndarray:
    total = float(scores.sum())
    if scores.size == 0 or total <= 0:
        return np.zeros(scores.size, dtype=float)
    return (scores / total).astype(float)


def _gini_coefficient(distribution: np.ndarray) -> float:
    if distribution.size == 0:
        return 0.0
    sorted_values = np.sort(distribution)
    if sorted_values[-1] == 0:
        return 0.0
    cumulative = np.cumsum(sorted_values)
    total = cumulative[-1]
    if total <= 0:
        return 0.0
    n = distribution.size
    gini = (n + 1 - 2 * np.sum(cumulative / total)) / n
    return float(max(gini, 0.0))


def _entropy(distribution: np.ndarray) -> float:
    if distribution.size == 0:
        return 0.0
    positive = distribution[distribution > 0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def _iter_shap_examples(explanation: shap.Explanation) -> Iterator[Tuple[List[str], np.ndarray]]:
    for tokens, values in zip(
        _iter_object_container(explanation.data),
        _iter_object_container(explanation.values),
    ):
        token_list = list(tokens)
        value_array = _to_float_array(values)
        yield token_list, value_array


def summarize_shap_importance(explanation: shap.Explanation) -> Dict[str, object]:
    token_scores = defaultdict(float)
    example_means: List[float] = []
    example_stds: List[float] = []
    sparsity_values: List[float] = []
    entropy_values: List[float] = []

    for tokens, values in _iter_shap_examples(explanation):
        if not tokens:
            continue
        scores = np.abs(_normalize_token_scores(values, len(tokens)))
        if scores.size == 0:
            continue
        example_means.append(float(scores.mean()))
        example_stds.append(float(scores.std()))
        distribution = _normalize_distribution(scores)
        sparsity_values.append(_gini_coefficient(distribution))
        entropy_values.append(_entropy(distribution))
        for token, score in zip(tokens, scores):
            token_scores[token] += float(score)

    summary: Dict[str, object] = {}
    if example_means:
        summary["mean_absolute_token_importance"] = float(np.mean(example_means))
        summary["std_absolute_token_importance"] = float(np.std(example_means))
        summary["median_absolute_token_importance"] = float(np.median(example_means))
    if example_stds:
        summary["mean_token_importance_std"] = float(np.mean(example_stds))
    if sparsity_values:
        summary["mean_token_gini"] = float(np.mean(sparsity_values))
    if entropy_values:
        summary["mean_token_entropy"] = float(np.mean(entropy_values))

    if token_scores:
        top_tokens = sorted(token_scores.items(), key=lambda item: item[1], reverse=True)[:5]
        summary["top_tokens"] = [token for token, _ in top_tokens]
        summary["top_token_scores"] = {token: score for token, score in top_tokens}

    return summary


def _collect_shap_statistics(explanation: shap.Explanation) -> List[Dict[str, object]]:
    statistics: List[Dict[str, object]] = []
    for index, (tokens, values) in enumerate(_iter_shap_examples(explanation)):
        if not tokens:
            continue
        scores = np.abs(_normalize_token_scores(values, len(tokens)))
        if scores.size == 0:
            continue
        ordering = np.argsort(scores)[::-1]
        top_tokens = [
            {"token": tokens[pos], "score": float(scores[pos])}
            for pos in ordering[:5]
        ]
        statistics.append(
            {
                "example_index": index,
                "mean_abs_importance": float(scores.mean()),
                "std_abs_importance": float(scores.std()),
                "token_count": len(tokens),
                "top_tokens": top_tokens,
            }
        )
    return statistics


def _pooled_stddev(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a < 2 and n_b < 2:
        return 0.0
    var_a = np.var(sample_a, ddof=1) if n_a > 1 else 0.0
    var_b = np.var(sample_b, ddof=1) if n_b > 1 else 0.0
    denominator = max(n_a + n_b - 2, 1)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / denominator if denominator else 0.0
    return float(np.sqrt(max(pooled, 0.0)))


def compare_shap_explanations(
    zero_shot: shap.Explanation, fine_tuned: shap.Explanation
) -> Dict[str, float]:
    cosine_similarities: List[float] = []
    jaccard_scores: List[float] = []
    zero_means: List[float] = []
    tuned_means: List[float] = []
    per_example_correlations: List[float] = []
    spearman_scores: List[float] = []
    zero_sparsity: List[float] = []
    tuned_sparsity: List[float] = []
    zero_entropy: List[float] = []
    tuned_entropy: List[float] = []
    top5_overlap: List[float] = []
    top10_overlap: List[float] = []

    for (zero_tokens, zero_values), (tuned_tokens, tuned_values) in zip(
        _iter_shap_examples(zero_shot),
        _iter_shap_examples(fine_tuned),
    ):
        if not zero_tokens or not tuned_tokens or len(zero_tokens) != len(tuned_tokens):
            continue

        zero_scores = np.abs(_normalize_token_scores(zero_values, len(zero_tokens)))
        tuned_scores = np.abs(_normalize_token_scores(tuned_values, len(tuned_tokens)))
        if zero_scores.size == 0 or tuned_scores.size == 0:
            continue

        denom = float(np.linalg.norm(zero_scores) * np.linalg.norm(tuned_scores))
        if denom > 0:
            cosine_similarities.append(float(np.dot(zero_scores, tuned_scores) / denom))

        zero_means.append(float(zero_scores.mean()))
        tuned_means.append(float(tuned_scores.mean()))

        correlation_matrix = np.corrcoef(zero_scores, tuned_scores)
        corr_value = correlation_matrix[0, 1]
        if np.isfinite(corr_value):
            per_example_correlations.append(float(corr_value))

        if zero_scores.size > 1 and tuned_scores.size > 1:
            rho, _ = spearmanr(zero_scores, tuned_scores)
            if np.isfinite(rho):
                spearman_scores.append(float(rho))

        zero_dist = _normalize_distribution(zero_scores)
        tuned_dist = _normalize_distribution(tuned_scores)
        zero_sparsity.append(_gini_coefficient(zero_dist))
        tuned_sparsity.append(_gini_coefficient(tuned_dist))
        zero_entropy.append(_entropy(zero_dist))
        tuned_entropy.append(_entropy(tuned_dist))

        top_k = min(5, len(zero_scores), len(tuned_scores))
        if top_k == 0:
            continue
        zero_top = set(np.argsort(zero_scores)[-top_k:])
        tuned_top = set(np.argsort(tuned_scores)[-top_k:])
        union = zero_top | tuned_top
        if union:
            jaccard_scores.append(len(zero_top & tuned_top) / len(union))
            top5_overlap.append(len(zero_top & tuned_top) / top_k)

        top_k10 = min(10, len(zero_scores), len(tuned_scores))
        if top_k10:
            zero_top10 = set(np.argsort(zero_scores)[-top_k10:])
            tuned_top10 = set(np.argsort(tuned_scores)[-top_k10:])
            top10_overlap.append(len(zero_top10 & tuned_top10) / top_k10)

    comparison: Dict[str, float] = {}
    if cosine_similarities:
        comparison["mean_token_cosine_similarity"] = float(np.mean(cosine_similarities))
    if jaccard_scores:
        comparison["mean_top_token_jaccard"] = float(np.mean(jaccard_scores))
    if top5_overlap:
        comparison["mean_top5_overlap"] = float(np.mean(top5_overlap))
    if top10_overlap:
        comparison["mean_top10_overlap"] = float(np.mean(top10_overlap))
    if zero_means and tuned_means:
        mean_diff = np.array(tuned_means) - np.array(zero_means)
        comparison["mean_abs_importance_difference"] = float(np.mean(mean_diff))
        pooled_std = _pooled_stddev(zero_means, tuned_means)
        if pooled_std > 0:
            comparison["cohens_d_abs_importance"] = float(
                (np.mean(tuned_means) - np.mean(zero_means)) / pooled_std
            )
    if per_example_correlations:
        comparison["mean_token_importance_correlation"] = float(
            np.mean(per_example_correlations)
        )
    if spearman_scores:
        comparison["mean_spearman_correlation"] = float(np.mean(spearman_scores))
    if zero_sparsity:
        comparison["mean_zero_shot_gini"] = float(np.mean(zero_sparsity))
    if tuned_sparsity:
        comparison["mean_fine_tuned_gini"] = float(np.mean(tuned_sparsity))
    if zero_entropy:
        comparison["mean_zero_shot_entropy"] = float(np.mean(zero_entropy))
    if tuned_entropy:
        comparison["mean_fine_tuned_entropy"] = float(np.mean(tuned_entropy))
    return comparison


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
        run_shap=args.run_shap,
        shap_example_count=args.shap_example_count,
        shap_max_evals=args.shap_max_evals,
        interpretability_example_count=args.interpretability_example_count,
        interpretability_batch_size=args.interpretability_batch_size,
        lime_num_features=args.lime_num_features,
        lime_num_samples=args.lime_num_samples,
        shap_surrogate_samples=args.shap_surrogate_samples,
        shap_top_features=args.shap_top_features,
        shap_vectorizer_max_features=args.shap_vectorizer_max_features,
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
    zero_shot_texts, zero_shot_labels = _prepare_zero_shot_texts(config, raw_dataset, formatter)
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
    )

    if sequence_length_reduced:
        print(
            "Evaluated zero-shot baseline after fine-tuning adjusted max_seq_length to keep comparisons aligned."
        )

    print("Zero-shot evaluation metrics:")
    print(json.dumps(_ensure_json_serializable(zero_shot_metrics), indent=2))

    with open(os.path.join(config.output_dir, "zero_shot_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(_ensure_json_serializable(zero_shot_metrics), handle, indent=2)

    _plot_metric_bars(zero_shot_metrics, fine_tuned_metrics, config.output_dir)
    label_order = list(sorted(label_token_map))
    if zero_shot_metrics.get("confusion_matrix"):
        _plot_confusion_matrix(
            np.array(zero_shot_metrics["confusion_matrix"]), label_order, config.output_dir, "zero_shot"
        )
    if fine_tuned_metrics and fine_tuned_metrics.get("confusion_matrix"):
        _plot_confusion_matrix(
            np.array(fine_tuned_metrics["confusion_matrix"]), label_order, config.output_dir, "fine_tuned"
        )

    interpretability_summary: Optional[Dict[str, object]] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            "Cleared CUDA cache before interpretability; the attribution runs operate on small batches "
            "and are unlikely to cause OOM unless the GPU is already at capacity. "
            "Lower --shap-example-count or disable --run-shap if attribution memory errors persist."
        )

    if config.run_shap:
        shap_samples = zero_shot_texts[: config.shap_example_count]
        shap_surrogate_samples = zero_shot_texts[: config.shap_surrogate_samples]
        if shap_samples:
            zero_kernel, zero_summary = collect_text_interpretability_outputs(
                model=model,
                tokenizer=tokenizer,
                label_token_map=label_token_map,
                device=device,
                config=config,
                shap_texts=shap_samples,
                tree_texts=shap_surrogate_samples,
                output_prefix="zero_shot",
            )
            if zero_summary:
                interpretability_summary = {"zero_shot": zero_summary}

            tuned_kernel: Optional[shap.Explanation] = None
            if tuned_model is not None:
                tuned_kernel, tuned_summary = collect_text_interpretability_outputs(
                    model=tuned_model,
                    tokenizer=tokenizer,
                    label_token_map=label_token_map,
                    device=device,
                    config=config,
                    shap_texts=shap_samples,
                    tree_texts=shap_surrogate_samples,
                    output_prefix="fine_tuned",
                )
                if tuned_summary:
                    if interpretability_summary is None:
                        interpretability_summary = {}
                    interpretability_summary["fine_tuned"] = tuned_summary

            if (
                interpretability_summary is not None
                and zero_kernel is not None
                and tuned_kernel is not None
            ):
                comparison = compare_shap_explanations(zero_kernel, tuned_kernel)
                interpretability_summary.setdefault("comparison", {})["kernel_shap"] = comparison
        else:
            print(
                "No samples available for interpretability analysis; skipping LIME/SHAP generation."
            )

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
        help="Explicit list of label ids to model (defaults to inferring from the dataset)",
    )
    parser.add_argument("--train-subset", type=int, default=5000)
    parser.add_argument("--eval-subset", type=int, default=2000)
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--no-run-shap", dest="run_shap", action="store_false")
    parser.set_defaults(run_shap=True)
    parser.add_argument("--shap-example-count", type=int, default=50)
    parser.add_argument("--shap-max-evals", type=int, default=200)
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
    parser.add_argument("--shap-surrogate-samples", type=int, default=200)
    parser.add_argument("--shap-top-features", type=int, default=12)
    parser.add_argument("--shap-vectorizer-max-features", type=int, default=5000)
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
