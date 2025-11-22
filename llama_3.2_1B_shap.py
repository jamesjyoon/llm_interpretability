# Move this try-except block to the top, before the main transformers import
from __future__ import annotations

__doc__ = """Utility for running zero-shot and LoRA-fine-tuned LLaMA style models on binary classification datasets.
Fixed: Custom SHAP plotting to avoid 'partition tree' errors, and enabled finetuning by default.
"""

import argparse
import json
import os
import time
from typing import Callable, Dict, Iterable, Iterator, List, NoReturn, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader, WeightedRandomSampler

# --- Import SHAP instead of LIME ---
try:
    import shap
except ImportError as exc:
    raise SystemExit("The `shap` package is required. Install it via `pip install shap`.") from exc

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    from huggingface_hub import login as hf_login
    from huggingface_hub.errors import GatedRepoError, RepoAccessError
except Exception:
    hf_login = None
    GatedRepoError = RepoAccessError = type("_DummyHFError", (Exception,), {})


HF_ACCESS_ERRORS = (OSError, GatedRepoError, RepoAccessError)


# --- Helper Classes & Functions ---

LabelTokenMap = Dict[int, int]


class SentimentClassificationModel(nn.Module):
    """Wraps a causal LM with a dedicated sentiment head and calibration hooks."""

    def __init__(self, base_model, num_labels: int, label_token_map: LabelTokenMap | None = None):
        super().__init__()
        self.base_model = base_model
        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Base model must expose a `hidden_size` in its config to attach a classification head.")
        self.classification_head = nn.Linear(hidden_size, num_labels)
        self.temperature: Optional[float] = None
        self.platt_coef: Optional[float] = None
        self.platt_intercept: Optional[float] = None

        if label_token_map:
            try:
                token_ids = [label_token_map[k] for k in sorted(label_token_map.keys())]
                with torch.no_grad():
                    self.classification_head.weight.copy_(self.base_model.lm_head.weight[token_ids])
                    nn.init.zeros_(self.classification_head.bias)
            except Exception:
                # If anything fails we fall back to default init
                pass

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        hidden_states = outputs.hidden_states[-1]
        seq_lens = attention_mask.sum(dim=-1) - 1
        pooled = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), seq_lens]
        logits = self.classification_head(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BalancedTrainer(Trainer):
    """Trainer that applies class-balanced sampling to fight label collapse."""

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset")

        labels = self.train_dataset["labels"]
        unique, counts = np.unique(labels, return_counts=True)
        freq = {int(u): c for u, c in zip(unique, counts)}
        weights = [1.0 / freq[int(lbl)] for lbl in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

class RestrictedLabelLogitsProcessor(LogitsProcessor):
    """Force generation to stay within a fixed label vocabulary."""
    def __init__(self, allowed_token_ids: Sequence[int]):
        if not allowed_token_ids:
            raise ValueError("At least one label token id must be supplied for restriction.")
        self.allowed_token_ids = torch.tensor(sorted(set(int(t) for t in allowed_token_ids)))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.allowed_token_ids.max().item() >= scores.size(-1):
            raise ValueError("Allowed token id exceeds the vocabulary size.")
        original = scores
        restricted = torch.full_like(original, torch.finfo(original.dtype).min)
        allowed = self.allowed_token_ids.to(original.device)
        expanded = allowed.unsqueeze(0).expand(original.size(0), -1)
        restricted.scatter_(1, expanded, original.gather(1, expanded))
        return restricted

def _maybe_login_to_hf(token: Optional[str]) -> None:
    if token and hf_login:
        hf_login(token, add_to_git_credential=False)

def _configure_cuda_allocator() -> None:
    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

def _load_label_token_map(tokenizer, label_space: Sequence[int]) -> LabelTokenMap:
    label_token_map: LabelTokenMap = {}
    for label in label_space:
        label_text = f" {label}"
        token_ids = tokenizer(label_text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        label_token_map[int(label)] = token_ids[-1]
    return label_token_map

@dataclass
class ExperimentConfig:
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
    learning_rate: float = 5e-5
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    max_seq_length: int = 516
    output_dir: str = "outputs/experiment_shap"
    
    # Interpretability
    interpretability_example_count: int = 3 
    shap_max_evals: int = 100
    run_shap: bool = True
    
    load_in_4bit: bool = True
    finetune: bool = True # Changed default to True
    label_space: Optional[Sequence[int]] = (0, 1)

class PromptFormatter:
    def __init__(self, label_space: Sequence[int]) -> None:
        self.label_space = list(label_space)
        label_list = ", ".join(str(label) for label in self.label_space)
        instruction = f"Respond with only one of the digits {label_list} to indicate the sentiment class."
        self.template = "{instruction}\nTweet: {sentence}\nLabel:"
        self.instruction = instruction

    def build_prompt(self, sentence: str) -> str:
        return self.template.format(instruction=self.instruction, sentence=sentence)

def _prepare_dataset(dataset: DatasetDict, config: ExperimentConfig, tokenizer, formatter: PromptFormatter) -> DatasetDict:
    def _format_examples(examples):
        prompts = [formatter.build_prompt(s) for s in examples[config.text_field]]
        full_seqs = [f"{p} {l}" for p, l in zip(prompts, examples[config.label_field])]
        model_inputs = tokenizer(full_seqs, max_length=config.max_seq_length, truncation=True, padding="max_length")

        attention = model_inputs["attention_mask"]
        model_inputs["labels"] = [int(l) for l in examples[config.label_field]]
        return model_inputs

    required_cols = sorted(set(dataset[config.train_split].column_names))
    processed = dataset.map(_format_examples, batched=True, remove_columns=required_cols)
    
    if config.train_subset:
        processed[config.train_split] = processed[config.train_split].shuffle(seed=config.random_seed).select(range(min(config.train_subset, len(processed[config.train_split]))))
    if config.eval_subset:
        processed[config.eval_split] = processed[config.eval_split].shuffle(seed=config.random_seed).select(range(min(config.eval_subset, len(processed[config.eval_split]))))
    return processed

def _truncate_tokenized_dataset(dataset: DatasetDict, max_length: int) -> DatasetDict:
    def _truncate(example):
        for key in ("input_ids", "attention_mask", "labels"):
            if key in example: example[key] = example[key][:max_length]
        return example
    return dataset.map(_truncate, load_from_cache_file=False)

def _prepare_texts_labels(config: ExperimentConfig, dataset: DatasetDict) -> Tuple[List[str], List[int]]:
    split = dataset[config.eval_split]
    if config.eval_subset:
        split = split.shuffle(seed=config.random_seed).select(range(min(config.eval_subset, len(split))))
    return list(split[config.text_field]), list(split[config.label_field])

def _resolve_label_space(config: ExperimentConfig, dataset: DatasetDict) -> List[int]:
    if config.label_space: return sorted(set(config.label_space))
    return sorted({int(x) for x in dataset[config.train_split][config.label_field]})

def _filter_dataset(dataset: DatasetDict, config: ExperimentConfig, label_space: Sequence[int]) -> DatasetDict:
    allowed = set(label_space)
    return dataset.filter(lambda x: int(x[config.label_field]) in allowed)


def _fit_temperature(logits: np.ndarray, labels: List[int]) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    temperature = torch.nn.Parameter(torch.ones(1, device=logits_tensor.device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = torch.nn.CrossEntropyLoss()

    def _closure():
        optimizer.zero_grad()
        loss = criterion(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(temperature.detach().cpu())


def _fit_platt_scaler(logits: np.ndarray, labels: List[int]) -> Tuple[Optional[float], Optional[float]]:
    if logits.shape[1] != 2:
        return None, None
    scores = logits[:, 1] - logits[:, 0]
    clf = LogisticRegression(max_iter=100)
    clf.fit(scores.reshape(-1, 1), labels)
    return float(clf.coef_[0][0]), float(clf.intercept_[0])


def _temperature_calibrator(temperature: Optional[float]) -> Callable[[np.ndarray], np.ndarray]:
    def _apply(logits: np.ndarray) -> np.ndarray:
        logits_tensor = torch.tensor(logits, dtype=torch.float)
        if temperature:
            logits_tensor = logits_tensor / temperature
        return torch.softmax(logits_tensor, dim=-1).cpu().numpy()

    return _apply


def _platt_calibrator(coef: Optional[float], intercept: Optional[float]) -> Callable[[np.ndarray], np.ndarray]:
    def _apply(logits: np.ndarray) -> np.ndarray:
        if coef is None or intercept is None or logits.shape[1] != 2:
            return torch.softmax(torch.tensor(logits, dtype=torch.float), dim=-1).cpu().numpy()
        scores = logits[:, 1] - logits[:, 0]
        probs_pos = 1 / (1 + np.exp(-(scores * coef + intercept)))
        return np.stack([1 - probs_pos, probs_pos], axis=1)

    return _apply

# --- Core Evaluation & SHAP Functions ---

def _build_logits_fn(model, tokenizer, device, max_length, formatter):
    def _predict(texts: Sequence[str]) -> np.ndarray:
        prompts = [formatter.build_prompt(str(t)) for t in texts]
        logits_list = []
        batch_size = 8
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
            logits_list.append(outputs.logits.cpu().numpy())
        return np.concatenate(logits_list, axis=0)

    return _predict


def _compute_metrics(labels: List[int], probs: np.ndarray, label_space: Sequence[int]):
    sorted_labels = list(sorted(label_space))
    pred_indices = np.argmax(probs, axis=1)
    preds = [sorted_labels[i] for i in pred_indices]
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

    cm = confusion_matrix(labels, preds, labels=sorted_labels)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    try:
        mcc = matthews_corrcoef(labels, preds)
    except Exception:
        mcc = 0.0
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "mcc": mcc,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_model(model, tokenizer, texts, labels, device, max_length, formatter, label_space, calibrate: bool = True):
    logits_fn = _build_logits_fn(model, tokenizer, device, max_length, formatter)
    logits = logits_fn(texts)
    raw_probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    metrics = {"raw": _compute_metrics(labels, raw_probs, label_space)}
    calibration = {}

    if calibrate:
        temp = _fit_temperature(logits, labels)
        temp_probs = _temperature_calibrator(temp)(logits)
        metrics["temperature"] = _compute_metrics(labels, temp_probs, label_space)
        calibration["temperature"] = temp

        coef, intercept = _fit_platt_scaler(logits, labels)
        platt_probs = _platt_calibrator(coef, intercept)(logits)
        metrics["platt"] = _compute_metrics(labels, platt_probs, label_space)
        calibration["platt"] = {"coef": coef, "intercept": intercept}

    return metrics, calibration

def run_shap(model, tokenizer, texts, device, config, formatter, prefix, calibrator: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    """Runs SHAP and saves plots."""
    print(f"Running SHAP for {prefix}...")
    class_names = [str(l) for l in sorted(formatter.label_space)]

    logits_fn = _build_logits_fn(model, tokenizer, device, config.max_seq_length, formatter)

    def predict_fn(text_batch: Sequence[str]) -> np.ndarray:
        logits = logits_fn(text_batch)
        if calibrator:
            return calibrator(logits)
        return torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_fn, masker, output_names=class_names)
    
    samples = texts[:config.interpretability_example_count]
    
    # Calculate SHAP values
    shap_values = explainer(samples, max_evals=config.shap_max_evals)
    
    json_output = []
    for i, text in enumerate(samples):
        probs = predict_fn([text])[0]
        pred_idx = np.argmax(probs)
        
        # Extract weights for the predicted class
        vals = shap_values[i].values
        if len(vals.shape) > 1:
            vals = vals[:, pred_idx]
            
        tokens = shap_values[i].data
        # Some maskers return bytes or arrays, ensure strings
        if isinstance(tokens, np.ndarray): tokens = tokens.tolist()
        tokens = [str(t) for t in tokens]
        
        json_output.append({
            "text": text,
            "predicted_label": class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "token_weights": list(zip(tokens, vals.tolist()))
        })

    with open(os.path.join(config.output_dir, f"{prefix}_shap.json"), "w") as f:
        json.dump(json_output, f, indent=2)

    # Generate Plots manually using the extracted JSON data
    _plot_shap_manual(json_output, config.output_dir, prefix)
    
    return shap_values

# --- Plotting Functions ---

def _plot_shap_manual(shap_data, output_dir, prefix):
    """
    Manually plots SHAP values using matplotlib to avoid shap.plots.bar errors.
    shap_data: List of dicts containing 'token_weights' (list of [token, weight])
    """
    if not plt: return

    for i, example in enumerate(shap_data):
        weights = example['token_weights']
        if not weights: continue
        
        # Unzip
        tokens, vals = zip(*weights)
        
        # Convert to numpy for easier handling
        vals = np.array(vals)
        tokens = np.array(tokens)
        
        # Sort by absolute value to show most important features, limit to top 15
        indices = np.argsort(np.abs(vals))
        if len(indices) > 15:
            indices = indices[-15:]
            
        sorted_vals = vals[indices]
        sorted_tokens = tokens[indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        y_pos = np.arange(len(sorted_tokens))
        colors = ['#ff0051' if x > 0 else '#008bfb' for x in sorted_vals] # SHAP standard colors (Red=Positive, Blue=Negative)
        
        ax.barh(y_pos, sorted_vals, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_tokens)
        ax.set_xlabel("SHAP Value (Impact on model output)")
        ax.set_title(f"{prefix} Example {i+1}: Pred {example['predicted_label']}")
        
        plt.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_shap_plot_ex{i}.png")
        plt.savefig(path)
        plt.close()
        
    print(f"Saved manual SHAP plots to {output_dir}")

def _plot_confusion_matrix(cm, labels, output_dir, prefix):
    if not plt: return
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(f'{prefix.replace("_", " ").title()} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"))
    plt.close()

def _plot_comparison(zs_metrics, ft_metrics, output_dir):
    if not plt or not ft_metrics: return
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    z_vals = [zs_metrics[m] for m in metrics]
    f_vals = [ft_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, z_vals, width, label='Zero Shot')
    ax.bar(x + width/2, f_vals, width, label='Fine Tuned')
    
    ax.set_ylabel('Score')
    ax.set_title('Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")

# --- Main Execution Flow ---

def run_experiment(args: argparse.Namespace) -> None:
    _maybe_login_to_hf(args.huggingface_token or os.environ.get("HF_TOKEN"))
    _configure_cuda_allocator()
    
    config = ExperimentConfig(
        model_name=args.model_name, output_dir=args.output_dir,
        load_in_4bit=args.load_in_4bit, run_shap=args.run_shap,
        finetune=args.finetune,
        train_subset=args.train_subset,
        eval_subset=args.eval_subset,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Tokenizer & Data
    print("Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    dataset = load_dataset(config.dataset_name, config.dataset_config) if config.dataset_config else load_dataset(config.dataset_name)
    label_space = _resolve_label_space(config, dataset)
    dataset = _filter_dataset(dataset, config, label_space)
    
    label_token_map = _load_label_token_map(tokenizer, label_space)
    formatter = PromptFormatter(label_space)

    # 2. Load BASE Model
    print("Loading Base Model...")
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map="auto" if config.load_in_4bit else None,
    )
    base_model.config.output_hidden_states = True
    if not config.load_in_4bit: base_model.to(device)
    model = SentimentClassificationModel(base_model, num_labels=len(label_space), label_token_map=label_token_map)
    model.to(device)

    # 3. Prepare Evaluation Data
    eval_texts, eval_labels = _prepare_texts_labels(config, dataset)

    # ==========================================
    # PHASE 1: ZERO-SHOT
    # ==========================================
    print("--- Phase 1: Running Zero-Shot Evaluation ---")
    zs_metrics, zs_calibration = evaluate_model(model, tokenizer, eval_texts, eval_labels, device, config.max_seq_length, formatter, label_space)
    print("Zero-Shot Metrics:", zs_metrics)

    with open(os.path.join(config.output_dir, "zero_shot_metrics.json"), "w") as f: json.dump(zs_metrics, f, indent=2)
    _plot_confusion_matrix(zs_metrics['raw']['confusion_matrix'], label_space, config.output_dir, "zero_shot")

    zero_shot_calibrator = _temperature_calibrator(zs_calibration.get("temperature")) if zs_calibration else None
    if config.run_shap:
        run_shap(model, tokenizer, eval_texts, device, config, formatter, "zero_shot", zero_shot_calibrator)

    # ==========================================
    # PHASE 2: FINE-TUNING
    # ==========================================
    ft_metrics = None
    if args.finetune:
        print("--- Phase 2: Starting Fine-Tuning ---")
        
        processed_dataset = _prepare_dataset(dataset, config, tokenizer, formatter)
        
        if config.load_in_4bit: base_model = prepare_model_for_kbit_training(base_model)
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        base_model.config.use_cache = False
        peft_model = get_peft_model(base_model, peft_config)
        peft_model.print_trainable_parameters()

        classification_model = SentimentClassificationModel(peft_model, num_labels=len(label_space), label_token_map=label_token_map)
        classification_model.to(device)

        trainer = BalancedTrainer(
            model=classification_model,
            train_dataset=processed_dataset[config.train_split],
            eval_dataset=processed_dataset[config.eval_split],
            args=TrainingArguments(
                output_dir=config.output_dir,
                num_train_epochs=config.num_train_epochs,
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                logging_steps=10,
                save_strategy="no",
                fp16=True
            ),
            data_collator=default_data_collator
        )
        
        try:
            trainer.train()
        except Exception as e:
            print(f"Training failed: {e}")
        
        classification_model.eval()

        print("Evaluating Fine-Tuned Model...")
        ft_metrics, ft_calibration = evaluate_model(classification_model, tokenizer, eval_texts, eval_labels, device, config.max_seq_length, formatter, label_space)
        print("Fine-Tuned Metrics:", ft_metrics)

        with open(os.path.join(config.output_dir, "fine_tuned_metrics.json"), "w") as f: json.dump(ft_metrics, f, indent=2)
        _plot_confusion_matrix(ft_metrics['raw']['confusion_matrix'], label_space, config.output_dir, "fine_tuned")

        fine_tuned_calibrator = _temperature_calibrator(ft_calibration.get("temperature")) if ft_calibration else None

        if config.run_shap:
            run_shap(classification_model, tokenizer, eval_texts, device, config, formatter, "fine_tuned", fine_tuned_calibrator)

    # ==========================================
    # PHASE 3: COMPARISON
    # ==========================================
    if ft_metrics:
        _plot_comparison(zs_metrics["raw"], ft_metrics.get("raw"), config.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset-name", default="mteb/tweet_sentiment_extraction")
    parser.add_argument("--dataset-config", default=None)
    
    # Updated args
    parser.add_argument("--finetune", action="store_true", default=True) # DEFAULT TRUE
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--no-run-shap", dest="run_shap", action="store_false")
    parser.set_defaults(run_shap=True)
    parser.add_argument("--shap-max-evals", type=int, default=100)
    
    parser.add_argument("--train-subset", type=int, default=4000)
    parser.add_argument("--eval-subset", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output-dir", default="outputs/experiment_shap")
    parser.add_argument("--huggingface-token", default=None)
    return parser

if __name__ == "__main__":
    parser = build_parser()
    run_experiment(parser.parse_args())
