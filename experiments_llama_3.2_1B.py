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
    """Force generation to stay within a fixed label vocabulary."""

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
    """Return a mapping from dataset labels to their token ids."""

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

        attention = model_inputs["attention_mask"]
        labels = []
        for input_ids, mask in zip(model_inputs["input_ids"], attention):
            masked = [-100] * len(input_ids)
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
    if total_gb >= 22:
        return

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
