from __future__ import annotations

__doc__ = """Utility for running zero-shot and LoRA-fine-tuned LLaMA style models on binary classification datasets.

This module is designed so it can be executed end-to-end on Google Colab. It
loads a dataset, evaluates a zero-shot baseline, optionally fine-tunes a LoRA
adapter, and computes SHAP token attributions for both models.
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, NoReturn, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    import shap  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "The `shap` package is required for attribution analysis. Install it via `pip install shap`."
    ) from exc

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


@dataclass
class ExperimentConfig:
    """Configuration for the baseline experiment."""

    model_name: str = "meta-llama/Llama-3.2-1B"
    dataset_name: str = "mteb/tweet_sentiment_extraction"
    dataset_config: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "test"
    text_field: str = "text"
    label_field: str = "label"
    train_subset: Optional[int] = 2000
    eval_subset: Optional[int] = 1000
    random_seed: int = 42
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    max_seq_length: int = 512
    max_target_length: int = 4
    output_dir: str = "outputs/tweet_sentiment_extraction"
    run_shap: bool = True
    shap_max_evals: int = 200
    shap_example_count: int = 10
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
                "Respond with only one of the digits {" + label_list + "} to indicate the sentiment class."
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

    def _format_examples(examples):
        prompts = [formatter.build_prompt(sentence) for sentence in examples[config.text_field]]
        full_sequences = [f"{prompt} {label}" for prompt, label in zip(prompts, examples[config.label_field])]
        model_inputs = tokenizer(
            full_sequences,
            max_length=config.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
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
    with torch.no_grad():
        generation = model.generate(
            **inputs,
            max_new_tokens=1,
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

    label_set = sorted(set(labels) | set(predictions))
    average = "binary" if len(label_set) == 2 else "weighted"

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, average=average, zero_division=0)
        ),
        "f1": float(f1_score(labels, predictions, average=average, zero_division=0)),
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
    explainer = shap.Explainer(predict_fn, masker, output_names=output_names)
    return explainer(texts, max_evals=max_evals)


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

    metrics = ["accuracy", "precision", "recall", "f1"]
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

    for tokens, values in _iter_shap_examples(explanation):
        if not tokens:
            continue
        scores = np.abs(_normalize_token_scores(values, len(tokens)))
        if scores.size == 0:
            continue
        example_means.append(float(scores.mean()))
        for token, score in zip(tokens, scores):
            token_scores[token] += float(score)

    summary: Dict[str, object] = {}
    if example_means:
        summary["mean_absolute_token_importance"] = float(np.mean(example_means))
        summary["std_absolute_token_importance"] = float(np.std(example_means))

    if token_scores:
        top_tokens = sorted(token_scores.items(), key=lambda item: item[1], reverse=True)[:5]
        summary["top_tokens"] = [token for token, _ in top_tokens]
        summary["top_token_scores"] = {token: score for token, score in top_tokens}

    return summary


def compare_shap_explanations(
    zero_shot: shap.Explanation, fine_tuned: shap.Explanation
) -> Dict[str, float]:
    cosine_similarities: List[float] = []
    jaccard_scores: List[float] = []

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

        top_k = min(5, len(zero_scores), len(tuned_scores))
        if top_k == 0:
            continue
        zero_top = set(np.argsort(zero_scores)[-top_k:])
        tuned_top = set(np.argsort(tuned_scores)[-top_k:])
        union = zero_top | tuned_top
        if union:
            jaccard_scores.append(len(zero_top & tuned_top) / len(union))

    comparison: Dict[str, float] = {}
    if cosine_similarities:
        comparison["mean_token_cosine_similarity"] = float(np.mean(cosine_similarities))
    if jaccard_scores:
        comparison["mean_top_token_jaccard"] = float(np.mean(jaccard_scores))
    return comparison


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
        run_shap=args.run_shap,
        shap_example_count=args.shap_example_count,
        shap_max_evals=args.shap_max_evals,
        load_in_4bit=args.load_in_4bit,
        output_dir=args.output_dir,
        label_space=args.label_space,
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
    )
    print("Zero-shot evaluation metrics:")
    print(json.dumps(_ensure_json_serializable(zero_shot_metrics), indent=2))

    with open(os.path.join(config.output_dir, "zero_shot_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(_ensure_json_serializable(zero_shot_metrics), handle, indent=2)

    tuned_model: Optional[PeftModel] = None
    fine_tuned_metrics: Optional[Dict[str, float]] = None
    if args.finetune:
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
        )
        print("Fine-tuned evaluation metrics:")
        print(json.dumps(_ensure_json_serializable(fine_tuned_metrics), indent=2))
        with open(os.path.join(config.output_dir, "fine_tuned_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(_ensure_json_serializable(fine_tuned_metrics), handle, indent=2)

    _plot_metric_bars(zero_shot_metrics, fine_tuned_metrics, config.output_dir)

    interpretability_summary: Optional[Dict[str, object]] = None
    if config.run_shap:
        shap_samples = zero_shot_texts[: config.shap_example_count]
        if shap_samples:
            zero_shot_shap = compute_shap_values(
                model,
                tokenizer,
                shap_samples,
                label_token_map,
                device,
                config.max_seq_length,
                config.shap_max_evals,
            )
            save_shap_values(zero_shot_shap, os.path.join(config.output_dir, "zero_shot_shap.json"))
            print(f"Saved zero-shot SHAP explanations for {len(shap_samples)} examples.")
            interpretability_summary = {"zero_shot": summarize_shap_importance(zero_shot_shap)}

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
                save_shap_values(tuned_shap, os.path.join(config.output_dir, "fine_tuned_shap.json"))
                print("Saved fine-tuned SHAP explanations.")
                tuned_summary = summarize_shap_importance(tuned_shap)
                if interpretability_summary is None:
                    interpretability_summary = {}
                interpretability_summary["fine_tuned"] = tuned_summary
                if "zero_shot" in interpretability_summary:
                    comparison = compare_shap_explanations(zero_shot_shap, tuned_shap)
                    zero_top = set(interpretability_summary["zero_shot"].get("top_tokens", []))
                    tuned_top = set(tuned_summary.get("top_tokens", []))
                    union = zero_top | tuned_top
                    if union:
                        comparison["top_token_jaccard_overall"] = len(zero_top & tuned_top) / len(union)
                    interpretability_summary["comparison"] = comparison
        else:
            print("No samples available for SHAP analysis; skipping attribution generation.")

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
    parser.add_argument("--train-subset", type=int, default=2000)
    parser.add_argument("--eval-subset", type=int, default=1000)
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--no-run-shap", dest="run_shap", action="store_false")
    parser.set_defaults(run_shap=True)
    parser.add_argument("--shap-example-count", type=int, default=10)
    parser.add_argument("--shap-max-evals", type=int, default=200)
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
