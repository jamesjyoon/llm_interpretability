from __future__ import annotations

__doc__ = """Utility for running zero-shot and LoRA-fine-tuned LLaMA style models on binary classification datasets.

This module is designed so it can be executed end-to-end on Google Colab. It
loads a dataset, evaluates a zero-shot baseline, optionally fine-tunes a LoRA
adapter, and computes SHAP token attributions for both models. In addition to
precision, recall, F1, and Matthews Correlation Coefficient (MCC) comparisons, the script now evaluates interpretability
characteristics across multiple explanation techniques including standard SHAP,
Kernel SHAP, TreeSHAP surrogates, and LIME.
"""

import argparse
import copy
import json
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, NoReturn, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from scipy.stats import spearmanr

try:  # pragma: no cover - PyTorch 2.0+ exposes this helper
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass

if torch.cuda.is_available():  # pragma: no cover - depends on runtime
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

try:
    import shap  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "The `shap` package is required for attribution analysis. Install it via `pip install shap`."
    ) from exc

try:  # pragma: no cover - optional dependency
    from lime.lime_text import LimeTextExplainer
except Exception:  # pragma: no cover - environments without lime fall back gracefully later
    LimeTextExplainer = None

try:  # pragma: no cover - optional dependency for TreeSHAP surrogate training
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.tree import DecisionTreeRegressor
except Exception:  # pragma: no cover - environments without sklearn fall back gracefully later
    TfidfVectorizer = DecisionTreeRegressor = None

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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
TokenAttribution = Tuple[List[str], np.ndarray]


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


def _balanced_subset(
    dataset_split: Dataset,
    target_count: Optional[int],
    label_field: str,
    seed: int,
) -> Dataset:
    """Return a stratified subset that preserves label balance when downsampling."""

    if target_count is None:
        return dataset_split
    available = dataset_split.num_rows
    if target_count >= available:
        return dataset_split

    labels = [int(label) for label in dataset_split[label_field]]
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)
    rng = np.random.default_rng(seed)
    for indices in label_to_indices.values():
        rng.shuffle(indices)

    selected: List[int] = []
    ordered_labels = sorted(label_to_indices)
    while len(selected) < target_count and ordered_labels:
        for label in list(ordered_labels):
            indices = label_to_indices[label]
            if not indices:
                ordered_labels.remove(label)
                continue
            selected.append(indices.pop())
            if len(selected) >= target_count:
                break
            if not indices:
                ordered_labels.remove(label)
    selected.sort()
    return dataset_split.select(selected)


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
    train_subset: Optional[int] = None
    eval_subset: Optional[int] = None
    random_seed: int = 42
    learning_rate: float = 3e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    max_seq_length: int = 512
    max_target_length: int = 4
    max_new_tokens: int = 3
    eval_batch_size: int = 16
    output_dir: str = "outputs/tweet_sentiment_extraction"
    run_shap: bool = True
    shap_max_evals: int = 200
    shap_example_count: int = 10
    interpretability_methods: Sequence[str] = ("kernel_shap", "tree_shap", "lime")
    lime_num_features: int = 10
    lime_num_samples: int = 500
    tree_shap_max_features: int = 200
    tree_shap_max_depth: int = 6
    load_in_4bit: bool = True
    label_space: Optional[Sequence[int]] = None
    fast_mode: bool = False
    dataloader_num_workers: Optional[int] = None
    auto_adjust_max_seq_length: bool = True
    length_sample_size: int = 2000
    max_length_percentile: float = 99.5


def _apply_fast_mode_overrides(config: ExperimentConfig) -> ExperimentConfig:
    """Return a copy of the config with faster defaults when fast mode is enabled."""

    if not config.fast_mode:
        return config

    fast_config = copy.deepcopy(config)
    fast_config.train_subset = fast_config.train_subset or 8000
    fast_config.eval_subset = fast_config.eval_subset or 2000
    fast_config.num_train_epochs = min(fast_config.num_train_epochs, 2.0)
    fast_config.shap_max_evals = min(fast_config.shap_max_evals, 120)
    fast_config.shap_example_count = min(fast_config.shap_example_count, 6)
    fast_config.lime_num_samples = min(fast_config.lime_num_samples, 300)
    fast_config.tree_shap_max_features = min(fast_config.tree_shap_max_features, 150)
    fast_config.eval_batch_size = max(fast_config.eval_batch_size, 32)
    fast_config.length_sample_size = min(fast_config.length_sample_size, 1000)
    if fast_config.dataloader_num_workers is None:
        fast_config.dataloader_num_workers = max(1, os.cpu_count() or 1)
    return fast_config


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


def _maybe_auto_adjust_sequence_length(
    config: ExperimentConfig,
    dataset: DatasetDict,
    tokenizer,
    formatter: PromptFormatter,
) -> ExperimentConfig:
    """Shrink ``max_seq_length`` when the dataset's prompts are already short."""

    if not config.auto_adjust_max_seq_length:
        return config
    if config.train_split not in dataset:
        return config

    split = dataset[config.train_split]
    sample_size = min(config.length_sample_size, split.num_rows)
    if sample_size <= 0:
        return config

    rng = np.random.default_rng(config.random_seed)
    if sample_size >= split.num_rows:
        indices = list(range(split.num_rows))
    else:
        indices = rng.choice(split.num_rows, size=sample_size, replace=False)

    prompt_lengths: List[int] = []
    for raw_index in indices:
        record = split[int(raw_index)]
        sentence = record[config.text_field]
        prompt = formatter.build_prompt(sentence)
        tokenized = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        prompt_lengths.append(len(tokenized["input_ids"]) + config.max_target_length)

    if not prompt_lengths:
        return config

    target_percentile = max(90.0, min(100.0, float(config.max_length_percentile)))
    percentile_length = float(np.percentile(prompt_lengths, target_percentile))
    safety_margin = 4
    trimmed_length = int(math.ceil(percentile_length + safety_margin))
    min_reasonable = max(config.max_target_length + 8, 32)
    adjusted = min(config.max_seq_length, max(trimmed_length, min_reasonable))
    if adjusted < config.max_seq_length:
        print(
            "Auto-adjusted max_seq_length from "
            f"{config.max_seq_length} to {adjusted} based on the {target_percentile}th percentile "
            "of prompt lengths."
        )
        config.max_seq_length = adjusted
    return config


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

    # ``DatasetDict`` inherits from ``dict`` but ``dict.copy`` returns a plain
    # ``dict`` without dataset helper methods such as ``map``.  Re-wrap the
    # object explicitly to keep the DatasetDict behavior while avoiding
    # in-place mutation of the caller's dataset.
    dataset = DatasetDict(dataset)
    if config.train_subset:
        dataset[config.train_split] = _balanced_subset(
            dataset[config.train_split],
            config.train_subset,
            config.label_field,
            config.random_seed,
        )
    if config.eval_subset:
        dataset[config.eval_split] = _balanced_subset(
            dataset[config.eval_split],
            config.eval_subset,
            config.label_field,
            config.random_seed,
        )

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

    return processed


def _prepare_zero_shot_texts(
    config: ExperimentConfig, original_dataset: DatasetDict, formatter: PromptFormatter
) -> Tuple[List[str], List[int]]:
    validation_split = original_dataset[config.eval_split]
    validation_split = _balanced_subset(
        validation_split,
        config.eval_subset,
        config.label_field,
        config.random_seed,
    )
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


def _autocast_context(model: AutoModelForCausalLM, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    model_dtype = getattr(model, "dtype", torch.float32)
    if model_dtype in {torch.float16, torch.bfloat16}:
        return torch.cuda.amp.autocast(dtype=model_dtype)
    return torch.cuda.amp.autocast()


def _classification_probabilities(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: Sequence[str],
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    inputs = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.inference_mode():
        with _autocast_context(model, device):
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
    return probs.detach().cpu().numpy()


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
    max_new_tokens: int,
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
    model.eval()
    with torch.inference_mode():
        with _autocast_context(model, device):
            generation = model.generate(
                **inputs,
                max_new_tokens=max(1, int(max_new_tokens)),
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
    with torch.no_grad():
        generation = model.generate(
            **inputs,
            max_new_tokens=max(1, int(max_new_tokens)),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
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


def _matthews_correlation(confusion: torch.Tensor) -> float:
    """Compute the Matthews Correlation Coefficient for any confusion matrix."""

    if confusion.numel() == 0:
        return 0.0

    confusion = confusion.to(torch.double)
    total = float(confusion.sum().item())
    if total <= 0:
        return 0.0

    actual = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    diag_sum = float(confusion.diag().sum().item())
    numerator = diag_sum * total - float((actual * predicted).sum().item())

    denom_actual = total ** 2 - float(actual.pow(2).sum().item())
    denom_predicted = total ** 2 - float(predicted.pow(2).sum().item())
    if denom_actual <= 0.0 or denom_predicted <= 0.0:
        return 0.0

    denominator = math.sqrt(denom_actual * denom_predicted)
    if denominator <= 0.0:
        return 0.0

    return float(numerator / denominator)


def _aggregate_metrics(
    confusion: torch.Tensor,
    per_class: Dict[str, Dict[str, float]],
    ordered_labels: Sequence[int],
) -> Tuple[float, float, float, float, float]:
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

    mcc = _matthews_correlation(confusion)

    return accuracy, precision, recall, f1, mcc


def evaluate_zero_shot(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: Sequence[str],
    labels: Sequence[int],
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
    max_new_tokens: int = 1,
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
            max_new_tokens,
        )
        predictions.extend(preds)
        probability_rows.extend(probs.astype(float).tolist())

    confusion = _build_confusion_matrix(labels, predictions, ordered_labels)
    per_class = _per_class_metrics(confusion, ordered_labels)
    accuracy, precision, recall, f1, mcc = _aggregate_metrics(
        confusion, per_class, ordered_labels
    )

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
) -> PeftModel:
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

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=model_gradient_dtype == torch.bfloat16,
        fp16=model_gradient_dtype == torch.float16,
        dataloader_num_workers=config.dataloader_num_workers or 0,
        tf32=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=processed_dataset[config.train_split],
        eval_dataset=processed_dataset[config.eval_split],
        data_collator=default_data_collator,
    )

    trainer.train()
    peft_model.eval()
    return peft_model


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
    algorithm: str = "auto",
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
    explainer = shap.Explainer(
        predict_fn, masker, output_names=output_names, algorithm=algorithm
    )
    return explainer(texts, max_evals=max_evals)


def compute_kernel_shap_attributions(
    model: AutoModelForCausalLM,
    tokenizer,
    texts: Sequence[str],
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
    max_evals: int,
) -> List[TokenAttribution]:
    """Return token-level Kernel SHAP approximations.

    ``shap.Explainer`` does not accept ``algorithm="kernel"`` when used with
    ``maskers.Text`` (the combination needed for Hugging Face tokenizers).  The
    permutation-based explainer shares the same Kernel SHAP sampling strategy,
    so we explicitly request that algorithm here to emulate Kernel SHAP while
    still receiving properly tokenized outputs.
    """

    explanation = compute_shap_values(
        model,
        tokenizer,
        texts,
        label_token_map,
        device,
        max_length,
        max_evals,
        algorithm="permutation",
    )
    return list(_iter_shap_examples(explanation))


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
        print("matplotlib is not installed; skipping metric visualization.")
        return

    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    zero_values = [zero_shot_metrics.get(metric, float("nan")) for metric in metrics]
    tuned_values: Optional[List[float]] = None
    if fine_tuned_metrics is not None:
        tuned_values = [fine_tuned_metrics.get(metric, float("nan")) for metric in metrics]

    x = np.arange(len(metrics))
    width = 0.35 if tuned_values is not None else 0.6

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2 if tuned_values is not None else x, zero_values, width, label="Zero-shot")
    if tuned_values is not None:
        ax.bar(x + width / 2, tuned_values, width, label="Fine-tuned")

    ax.set_xticks(x)
    ax.set_xticklabels([metric.upper() for metric in metrics])
    ax.set_ylim(-1.05, 1.05)
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


def _aggregate_top_token_scores(
    examples: Sequence[TokenAttribution],
    top_k: int = 20,
) -> Tuple[List[str], List[float]]:
    token_scores = defaultdict(float)
    for tokens, values in examples:
        if not tokens:
            continue
        scores = np.abs(_normalize_token_scores(values, len(tokens)))
        if scores.size == 0:
            continue
        for token, score in zip(tokens, scores):
            token_scores[token] += float(score)

    if not token_scores:
        return [], []

    top_items = sorted(token_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    tokens = [token for token, _ in top_items][::-1]
    scores = [score for _, score in top_items][::-1]
    return tokens, scores


def _plot_generic_token_summary(
    examples: Sequence[TokenAttribution],
    output_dir: str,
    prefix: str,
    title: str,
    top_k: int = 20,
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency in Colab
        print(f"matplotlib is not installed; skipping {title.lower()} visualization.")
        return None

    tokens, scores = _aggregate_top_token_scores(examples, top_k=top_k)
    if not tokens:
        print(f"No token scores available to visualize for {prefix}; skipping plot.")
        return None

    height = max(4.0, 0.35 * len(tokens) + 1.0)
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(np.arange(len(tokens)), scores, color="#55a868")
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Total |importance| score")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_token_summary.png")
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
    message = f"Saved interpretability visualization to {absolute_path}."
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


def summarize_token_attributions(examples: Sequence[TokenAttribution]) -> Dict[str, object]:
    token_scores = defaultdict(float)
    example_means: List[float] = []
    example_stds: List[float] = []
    sparsity_values: List[float] = []
    entropy_values: List[float] = []

    for tokens, values in examples:
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


def collect_token_statistics(examples: Sequence[TokenAttribution]) -> List[Dict[str, object]]:
    statistics: List[Dict[str, object]] = []
    for index, (tokens, values) in enumerate(examples):
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


def _tokens_to_dict(tokens: Sequence[str], scores: np.ndarray) -> Dict[str, float]:
    mapping: Dict[str, float] = defaultdict(float)
    for token, score in zip(tokens, scores):
        mapping[token] += float(score)
    return mapping


def compare_token_attributions(
    zero_examples: Sequence[TokenAttribution],
    tuned_examples: Sequence[TokenAttribution],
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

    for (zero_tokens, zero_values), (tuned_tokens, tuned_values) in zip(zero_examples, tuned_examples):
        if not zero_tokens and not tuned_tokens:
            continue

        zero_scores = np.abs(_normalize_token_scores(zero_values, len(zero_tokens)))
        tuned_scores = np.abs(_normalize_token_scores(tuned_values, len(tuned_tokens)))
        zero_dict = _tokens_to_dict(zero_tokens, zero_scores)
        tuned_dict = _tokens_to_dict(tuned_tokens, tuned_scores)
        union = sorted(set(zero_dict) | set(tuned_dict))
        if not union:
            continue
        zero_vector = np.array([zero_dict.get(token, 0.0) for token in union], dtype=float)
        tuned_vector = np.array([tuned_dict.get(token, 0.0) for token in union], dtype=float)
        if zero_vector.size == 0 or tuned_vector.size == 0:
            continue

        denom = float(np.linalg.norm(zero_vector) * np.linalg.norm(tuned_vector))
        if denom > 0:
            cosine_similarities.append(float(np.dot(zero_vector, tuned_vector) / denom))

        zero_means.append(float(zero_vector.mean()))
        tuned_means.append(float(tuned_vector.mean()))

        correlation_matrix = np.corrcoef(zero_vector, tuned_vector)
        corr_value = correlation_matrix[0, 1]
        if np.isfinite(corr_value):
            per_example_correlations.append(float(corr_value))

        if zero_vector.size > 1 and tuned_vector.size > 1:
            rho, _ = spearmanr(zero_vector, tuned_vector)
            if np.isfinite(rho):
                spearman_scores.append(float(rho))

        zero_dist = _normalize_distribution(zero_vector)
        tuned_dist = _normalize_distribution(tuned_vector)
        zero_sparsity.append(_gini_coefficient(zero_dist))
        tuned_sparsity.append(_gini_coefficient(tuned_dist))
        zero_entropy.append(_entropy(zero_dist))
        tuned_entropy.append(_entropy(tuned_dist))

        top_k = min(5, len(zero_dict), len(tuned_dict))
        if top_k == 0:
            continue
        zero_top_tokens = set(sorted(zero_dict, key=zero_dict.get, reverse=True)[:top_k])
        tuned_top_tokens = set(sorted(tuned_dict, key=tuned_dict.get, reverse=True)[:top_k])
        union_top = zero_top_tokens | tuned_top_tokens
        if union_top:
            jaccard_scores.append(len(zero_top_tokens & tuned_top_tokens) / len(union_top))
            top5_overlap.append(len(zero_top_tokens & tuned_top_tokens) / top_k)

        top_k10 = min(10, len(zero_dict), len(tuned_dict))
        if top_k10:
            zero_top10 = set(sorted(zero_dict, key=zero_dict.get, reverse=True)[:top_k10])
            tuned_top10 = set(sorted(tuned_dict, key=tuned_dict.get, reverse=True)[:top_k10])
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


def summarize_shap_importance(explanation: shap.Explanation) -> Dict[str, object]:
    examples = list(_iter_shap_examples(explanation))
    return summarize_token_attributions(examples)


def _collect_shap_statistics(explanation: shap.Explanation) -> List[Dict[str, object]]:
    examples = list(_iter_shap_examples(explanation))
    return collect_token_statistics(examples)


def compare_shap_explanations(
    zero_shot: shap.Explanation, fine_tuned: shap.Explanation
) -> Dict[str, float]:
    zero_examples = list(_iter_shap_examples(zero_shot))
    tuned_examples = list(_iter_shap_examples(fine_tuned))
    return compare_token_attributions(zero_examples, tuned_examples)


def _preferred_label_index(label_token_map: LabelTokenMap) -> int:
    ordered_labels = sorted(label_token_map)
    if not ordered_labels:
        return 0
    positive_label = ordered_labels[-1]
    return ordered_labels.index(positive_label)


def compute_lime_attributions(
    model: AutoModelForCausalLM,
    tokenizer,
    texts: Sequence[str],
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
    num_features: int,
    num_samples: int,
) -> List[TokenAttribution]:
    if LimeTextExplainer is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "LIME is not installed. Install it via `pip install lime` to compute LIME metrics."
        )

    ordered_labels = [str(label) for label in sorted(label_token_map)]
    explainer = LimeTextExplainer(class_names=ordered_labels)

    def predict_fn(batch_texts: List[str]) -> np.ndarray:
        return _classification_probabilities(
            model,
            tokenizer,
            batch_texts,
            label_token_map,
            device,
            max_length,
        )

    target_index = _preferred_label_index(label_token_map)
    examples: List[TokenAttribution] = []
    for text in texts:
        explanation = explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            labels=[target_index],
            num_samples=num_samples,
        )
        feature_list = explanation.as_list(label=target_index)
        tokens = [feature for feature, _ in feature_list]
        values = np.array([score for _, score in feature_list], dtype=float)
        examples.append((tokens, values))
    return examples


def compute_tree_shap_attributions(
    model: AutoModelForCausalLM,
    tokenizer,
    texts: Sequence[str],
    label_token_map: LabelTokenMap,
    device: torch.device,
    max_length: int,
    max_features: int,
    max_depth: int,
) -> List[TokenAttribution]:
    if TfidfVectorizer is None or DecisionTreeRegressor is None:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for TreeSHAP surrogate explanations. Install it via `pip install scikit-learn`."
        )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    feature_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()
    dense_matrix = feature_matrix.toarray()

    def predict_fn(batch_texts: Sequence[str]) -> np.ndarray:
        probs = _classification_probabilities(
            model,
            tokenizer,
            batch_texts,
            label_token_map,
            device,
            max_length,
        )
        return probs[:, _preferred_label_index(label_token_map)]

    target_probs = predict_fn(texts)
    regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    regressor.fit(dense_matrix, target_probs)
    tree_explainer = shap.TreeExplainer(regressor, feature_perturbation="interventional")
    values = tree_explainer.shap_values(dense_matrix)
    if isinstance(values, list):  # pragma: no cover - TreeExplainer multi-output fallback
        values = values[0]

    examples: List[TokenAttribution] = []
    for row in values:
        examples.append((feature_names, np.array(row, dtype=float)))
    return examples


def _compute_method_examples(
    method: str,
    model: AutoModelForCausalLM,
    tokenizer,
    texts: Sequence[str],
    label_token_map: LabelTokenMap,
    device: torch.device,
    config: ExperimentConfig,
) -> List[TokenAttribution]:
    normalized = method.lower()
    if normalized == "kernel_shap":
        return compute_kernel_shap_attributions(
            model,
            tokenizer,
            texts,
            label_token_map,
            device,
            config.max_seq_length,
            config.shap_max_evals,
        )
    if normalized == "tree_shap":
        return compute_tree_shap_attributions(
            model,
            tokenizer,
            texts,
            label_token_map,
            device,
            config.max_seq_length,
            config.tree_shap_max_features,
            config.tree_shap_max_depth,
        )
    if normalized == "lime":
        return compute_lime_attributions(
            model,
            tokenizer,
            texts,
            label_token_map,
            device,
            config.max_seq_length,
            config.lime_num_features,
            config.lime_num_samples,
        )
    raise ValueError(f"Unsupported interpretability method: {method}")


def _summarize_examples(
    examples: List[TokenAttribution],
    output_dir: Optional[str] = None,
    method: Optional[str] = None,
    variant: Optional[str] = None,
) -> Dict[str, object]:
    summary = summarize_token_attributions(examples)
    summary["per_example_stats"] = collect_token_statistics(examples)
    if output_dir and method and variant:
        visualization_path = _plot_generic_token_summary(
            examples,
            output_dir,
            f"{variant}_{method.lower()}",
            title=f"{method.replace('_', ' ').title()} token contributions ({variant.replace('_', ' ')})",
        )
        if visualization_path:
            summary["visualization_path"] = visualization_path
    return summary


def evaluate_interpretability_method(
    method: str,
    model: AutoModelForCausalLM,
    tokenizer,
    texts: Sequence[str],
    label_token_map: LabelTokenMap,
    device: torch.device,
    config: ExperimentConfig,
    tuned_model: Optional[PeftModel],
) -> Dict[str, object]:
    zero_examples = _compute_method_examples(
        method, model, tokenizer, texts, label_token_map, device, config
    )
    result: Dict[str, object] = {
        "zero_shot": _summarize_examples(
            zero_examples,
            output_dir=config.output_dir,
            method=method,
            variant="zero_shot",
        )
    }

    tuned_examples: Optional[List[TokenAttribution]] = None
    if tuned_model is not None:
        tuned_examples = _compute_method_examples(
            method,
            tuned_model,
            tokenizer,
            texts,
            label_token_map,
            device,
            config,
        )
        result["fine_tuned"] = _summarize_examples(
            tuned_examples,
            output_dir=config.output_dir,
            method=method,
            variant="fine_tuned",
        )
    if tuned_examples is not None:
        result["comparison"] = compare_token_attributions(zero_examples, tuned_examples)
    return result


def run_experiment(args: argparse.Namespace) -> None:
    provided_token = args.huggingface_token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_TOKEN"
    )
    _maybe_login_to_hf(provided_token)

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
        random_seed=args.random_seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_shap=args.run_shap,
        shap_example_count=args.shap_example_count,
        shap_max_evals=args.shap_max_evals,
        interpretability_methods=args.interpretability_methods,
        lime_num_features=args.lime_num_features,
        lime_num_samples=args.lime_num_samples,
        tree_shap_max_features=args.tree_shap_max_features,
        tree_shap_max_depth=args.tree_shap_max_depth,
        load_in_4bit=args.load_in_4bit,
        max_seq_length=args.max_seq_length,
        max_target_length=args.max_target_length,
        max_new_tokens=args.max_new_tokens,
        eval_batch_size=args.eval_batch_size,
        output_dir=args.output_dir,
        label_space=args.label_space,
        fast_mode=args.fast_mode,
        dataloader_num_workers=args.dataloader_num_workers,
        auto_adjust_max_seq_length=args.auto_adjust_max_seq_length,
        length_sample_size=args.length_sample_size,
        max_length_percentile=args.max_length_percentile,
    )

    config = _apply_fast_mode_overrides(config)
    if config.fast_mode:
        print(
            "Fast mode enabled: using up to "
            f"{config.train_subset or 'all'} training samples, "
            f"{config.eval_subset or 'all'} eval samples, "
            f"{config.num_train_epochs} training epochs, and "
            f"{config.shap_example_count} SHAP examples."
        )

    set_seed(config.random_seed)
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
    config = _maybe_auto_adjust_sequence_length(config, raw_dataset, tokenizer, formatter)
    processed_dataset = _prepare_dataset(raw_dataset, config, tokenizer, formatter)

    zero_shot_texts, zero_shot_labels = _prepare_zero_shot_texts(config, raw_dataset, formatter)
    zero_shot_metrics = evaluate_zero_shot(
        model,
        tokenizer,
        zero_shot_texts,
        zero_shot_labels,
        label_token_map,
        device,
        config.max_seq_length,
        config.max_new_tokens,
        batch_size=config.eval_batch_size,
    )
    print("Zero-shot evaluation metrics:")
    print(json.dumps(_ensure_json_serializable(zero_shot_metrics), indent=2))

    with open(os.path.join(config.output_dir, "zero_shot_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(_ensure_json_serializable(zero_shot_metrics), handle, indent=2)

    zero_shot_model = model
    tuned_model: Optional[PeftModel] = None
    fine_tuned_metrics: Optional[Dict[str, float]] = None
    if args.finetune:
        preserve_zero_shot_model = config.run_shap or bool(config.interpretability_methods)
        if preserve_zero_shot_model:
            zero_shot_model = copy.deepcopy(model)
        tuned_model = train_lora_classifier(config, model, processed_dataset)
        tuned_model.save_pretrained(os.path.join(config.output_dir, "lora_adapter"))
        fine_tuned_metrics = evaluate_zero_shot(
            tuned_model,
            tokenizer,
            zero_shot_texts,
            zero_shot_labels,
            label_token_map,
            device,
            config.max_seq_length,
            config.max_new_tokens,
            batch_size=config.eval_batch_size,
        )
        print("Fine-tuned evaluation metrics:")
        print(json.dumps(_ensure_json_serializable(fine_tuned_metrics), indent=2))
        with open(os.path.join(config.output_dir, "fine_tuned_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(fine_tuned_metrics), handle, indent=2)

    _plot_metric_bars(zero_shot_metrics, fine_tuned_metrics, config.output_dir)

    interpretability_summary: Optional[Dict[str, object]] = None
    additional_method_results: Dict[str, object] = {}
    if config.run_shap:
        shap_samples = zero_shot_texts[: config.shap_example_count]
        if shap_samples:
            zero_shot_shap = compute_shap_values(
                zero_shot_model,
                tokenizer,
                shap_samples,
                label_token_map,
                device,
                config.max_seq_length,
                config.shap_max_evals,
            )
            save_shap_values(zero_shot_shap, os.path.join(config.output_dir, "zero_shot_shap.json"))
            print(f"Saved zero-shot SHAP explanations for {len(shap_samples)} examples.")
            zero_summary = summarize_shap_importance(zero_shot_shap)
            zero_summary["per_example_stats"] = _collect_shap_statistics(zero_shot_shap)
            zero_plot_path = _plot_shap_summary(
                zero_shot_shap, config.output_dir, "zero_shot"
            )
            if zero_plot_path:
                zero_summary["visualization_path"] = zero_plot_path
            interpretability_summary = {"zero_shot": zero_summary}

            tuned_shap: Optional[shap.Explanation] = None
            if tuned_model is not None:
                tuned_shap = compute_shap_values(
                    tuned_model,
                    tokenizer,
                    shap_samples,
                    label_token_map,
                    device,
                    config.max_seq_length,
                    config.shap_max_evals,
                )
                save_shap_values(
                    tuned_shap, os.path.join(config.output_dir, "fine_tuned_shap.json")
                )
                print("Saved fine-tuned SHAP explanations.")
                tuned_summary = summarize_shap_importance(tuned_shap)
                tuned_summary["per_example_stats"] = _collect_shap_statistics(tuned_shap)
                tuned_plot_path = _plot_shap_summary(
                    tuned_shap, config.output_dir, "fine_tuned"
                )
                if tuned_plot_path:
                    tuned_summary["visualization_path"] = tuned_plot_path
                interpretability_summary["fine_tuned"] = tuned_summary
                comparison = compare_shap_explanations(zero_shot_shap, tuned_shap)
                zero_stats = interpretability_summary["zero_shot"].get("per_example_stats", [])
                tuned_stats = tuned_summary.get("per_example_stats", [])
                if len(zero_stats) == len(tuned_stats) and zero_stats:
                    comparison["per_example_mean_abs_difference"] = [
                        float(
                            tuned_stats[idx]["mean_abs_importance"]
                            - zero_stats[idx]["mean_abs_importance"]
                        )
                        for idx in range(len(zero_stats))
                    ]
                zero_top = set(interpretability_summary["zero_shot"].get("top_tokens", []))
                tuned_top = set(tuned_summary.get("top_tokens", []))
                union = zero_top | tuned_top
                if union:
                    comparison["top_token_jaccard_overall"] = len(zero_top & tuned_top) / len(union)
                interpretability_summary["comparison"] = comparison

            if config.interpretability_methods:
                for method in config.interpretability_methods:
                    try:
                        method_result = evaluate_interpretability_method(
                            method,
                            zero_shot_model,
                            tokenizer,
                            shap_samples,
                            label_token_map,
                            device,
                            config,
                            tuned_model,
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic aid
                        method_result = {"error": str(exc)}
                    additional_method_results[method] = method_result
        else:
            print("No samples available for SHAP analysis; skipping attribution generation.")

    if additional_method_results:
        if interpretability_summary is None:
            interpretability_summary = {}
        interpretability_summary["additional_methods"] = additional_method_results

    if interpretability_summary is not None:
        summary_path = os.path.join(config.output_dir, "interpretability_metrics.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(interpretability_summary), handle, indent=2)
        print(f"Saved interpretability comparison metrics to {summary_path}.")


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
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--eval-subset", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument(
        "--auto-adjust-max-seq-length",
        dest="auto_adjust_max_seq_length",
        action="store_true",
        help="Estimate prompt lengths and shrink max_seq_length automatically when safe.",
    )
    parser.add_argument(
        "--no-auto-adjust-max-seq-length",
        dest="auto_adjust_max_seq_length",
        action="store_false",
    )
    parser.set_defaults(auto_adjust_max_seq_length=True)
    parser.add_argument(
        "--length-sample-size",
        type=int,
        default=2000,
        help="Number of training prompts to sample when auto-adjusting the sequence length.",
    )
    parser.add_argument(
        "--max-length-percentile",
        type=float,
        default=99.5,
        help="Percentile of sampled prompt lengths to cover before trimming max_seq_length.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Apply throughput-oriented defaults (balanced subsampling, fewer epochs, and"
        " lighter interpretability settings) to get quicker feedback.",
    )
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--no-run-shap", dest="run_shap", action="store_false")
    parser.set_defaults(run_shap=True)
    parser.add_argument("--shap-example-count", type=int, default=10)
    parser.add_argument("--shap-max-evals", type=int, default=200)
    parser.add_argument(
        "--interpretability-methods",
        nargs="*",
        default=["kernel_shap", "tree_shap", "lime"],
        help="Additional explanation techniques to compare (e.g., kernel_shap tree_shap lime).",
    )
    parser.add_argument("--lime-num-features", type=int, default=10)
    parser.add_argument("--lime-num-samples", type=int, default=500)
    parser.add_argument("--tree-shap-max-features", type=int, default=200)
    parser.add_argument("--tree-shap-max-depth", type=int, default=6)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--output-dir", default="outputs/tweet_sentiment_extraction")
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
