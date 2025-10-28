"""Utility for running zero-shot and LoRA-fine-tuned LLaMA style models on binary classification datasets.

This module is designed so it can be executed end-to-end on Google Colab. It
loads a dataset, evaluates a zero-shot baseline, optionally fine-tunes a LoRA
adapter, and computes SHAP token attributions for both models.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


LabelTokenMap = Dict[int, int]


def _load_label_token_map(tokenizer) -> LabelTokenMap:
    """Return a mapping from label integers to their token ids.

    LLaMA-style tokenizers encode numbers with a leading space as a single
    token (e.g., ``" 0"`` becomes ``"▁0"``).  Using the space-prefixed
    representation ensures the labels align with how prompts are constructed
    elsewhere in this module and avoids situations where ``"0"`` would be
    split across multiple tokens.
    """
    label_token_map: LabelTokenMap = {}
    for label in (0, 1):
        label_text = f" {label}"
        token_ids = tokenizer(
            label_text, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        if not token_ids:
            raise ValueError(f"Tokenizer could not encode label {label}.")
        label_token_map[label] = token_ids[-1]
    return label_token_map


@dataclass
class ExperimentConfig:
    """Configuration for the baseline experiment."""

    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_name: str = "glue"
    dataset_config: str = "sst2"
    text_field: str = "sentence"
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
    output_dir: str = "outputs/collab_experiment"
    run_shap: bool = True
    shap_max_evals: int = 200
    shap_example_count: int = 10
    load_in_4bit: bool = True


class PromptFormatter:
    """Converts sentences into classification prompts."""

    def __init__(self) -> None:
        self.template = (
            "You are a sentiment classifier.\n"
            "Return `1` for positive sentiment and `0` for negative sentiment.\n"
            "Sentence: {sentence}\n"
            "Label:"
        )

    def build_prompt(self, sentence: str) -> str:
        return self.template.format(sentence=sentence)


def _prepare_dataset(
    dataset: DatasetDict, config: ExperimentConfig, tokenizer, formatter: PromptFormatter
) -> DatasetDict:
    required_splits = {"train", "validation"}
    if not required_splits.issubset(dataset):
        raise ValueError(
            f"Dataset `{config.dataset_name}` with config `{config.dataset_config}` must contain train and validation splits."
        )

    def _format_examples(examples):
        prompts = [formatter.build_prompt(sentence) for sentence in examples[config.text_field]]
        full_sequences = [f"{prompt} {label}" for prompt, label in zip(prompts, examples[config.label_field])]
        model_inputs = tokenizer(
            full_sequences,
            max_length=config.max_seq_length,
            truncation=True,
            padding="longest",
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    processed = dataset.map(_format_examples, batched=True, remove_columns=dataset["train"].column_names)

    if config.train_subset:
        train_dataset = processed["train"].shuffle(seed=config.random_seed)
        train_count = min(config.train_subset, train_dataset.num_rows)
        processed["train"] = train_dataset.select(range(train_count))
    if config.eval_subset:
        validation_dataset = processed["validation"].shuffle(seed=config.random_seed)
        validation_count = min(config.eval_subset, validation_dataset.num_rows)
        processed["validation"] = validation_dataset.select(range(validation_count))
    return processed


def _prepare_zero_shot_texts(
    config: ExperimentConfig, original_dataset: DatasetDict, formatter: PromptFormatter
) -> Tuple[List[str], List[int]]:
    validation_split = original_dataset["validation"]
    if config.eval_subset:
        eval_count = min(config.eval_subset, len(validation_split))
        validation_split = validation_split.shuffle(seed=config.random_seed)
        validation_split = validation_split.select(range(eval_count))
    texts = [formatter.build_prompt(sentence) for sentence in validation_split[config.text_field]]
    labels = list(validation_split[config.label_field])
    return texts, labels


def _zero_shot_probabilities(
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
    label_token_ids = torch.tensor([label_token_map[0], label_token_map[1]], device=device)
    label_logits = final_logits[:, label_token_ids]
    probs = torch.softmax(label_logits, dim=-1)
    return probs.detach().cpu().numpy()


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
    probabilities: List[float] = []
    for batch_prompts in _batched(list(prompts), batch_size):
        probs = _zero_shot_probabilities(model, tokenizer, batch_prompts, label_token_map, device, max_length)
        preds = probs.argmax(axis=-1)
        predictions.extend(preds.tolist())
        probabilities.extend(probs[:, 1].tolist())

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    metrics["positive_probability_mean"] = float(np.mean(probabilities))
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
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
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
        return _zero_shot_probabilities(model, tokenizer, batch_texts, label_token_map, device, max_length)

    explainer = shap.Explainer(predict_fn, masker, output_names=["0", "1"])
    return explainer(texts, max_evals=max_evals)


def _serialize_shap(explanation: shap.Explanation) -> Dict[str, object]:
    def _ensure_serializable(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    return {
        "values": _ensure_serializable(explanation.values),
        "base_values": _ensure_serializable(explanation.base_values),
        "data": _ensure_serializable(explanation.data),
        "feature_names": _ensure_serializable(explanation.feature_names),
        "output_names": _ensure_serializable(explanation.output_names),
    }


def save_shap_values(explanation: shap.Explanation, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_serialize_shap(explanation), handle, indent=2)


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_experiment(args: argparse.Namespace) -> None:
    config = ExperimentConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_subset=args.train_subset,
        eval_subset=args.eval_subset,
        run_shap=args.run_shap,
        shap_example_count=args.shap_example_count,
        shap_max_evals=args.shap_max_evals,
        load_in_4bit=args.load_in_4bit,
        output_dir=args.output_dir,
    )

    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.load_in_4bit and device.type != "cuda":
        print("4-bit quantization requested but CUDA is unavailable; falling back to full precision.")
        config.load_in_4bit = False

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model_kwargs = {}
    if config.load_in_4bit:
        base_model_kwargs.update({"load_in_4bit": True, "device_map": "auto"})

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **base_model_kwargs)
    if not config.load_in_4bit:
        model.to(device)

    label_token_map = _load_label_token_map(tokenizer)
    formatter = PromptFormatter()

    raw_dataset = load_dataset(config.dataset_name, config.dataset_config)
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
    print(json.dumps(zero_shot_metrics, indent=2))

    _ensure_output_dir(config.output_dir)
    with open(os.path.join(config.output_dir, "zero_shot_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(zero_shot_metrics, handle, indent=2)

    tuned_model: Optional[PeftModel] = None
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
        print(json.dumps(fine_tuned_metrics, indent=2))
        with open(os.path.join(config.output_dir, "fine_tuned_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(fine_tuned_metrics, handle, indent=2)

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
        else:
            print("No samples available for SHAP analysis; skipping attribution generation.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run binary classification interpretability experiment.")
    parser.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset-name", default="glue")
    parser.add_argument("--dataset-config", default="sst2")
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
    parser.add_argument("--output-dir", default="outputs/collab_experiment")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    run_experiment(parser.parse_args())
