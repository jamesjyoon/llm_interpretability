from __future__ import annotations

"""Fixed version with proper fine-tuning that improves metrics."""

import argparse
import dataclasses
import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments, default_data_collator, set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

LabelTokenMap = Dict[int, int]


class SentimentClassificationModel(nn.Module):
    """Classification model that properly handles training."""
    
    def __init__(self, base_model, num_labels: int, label_token_map: LabelTokenMap | None = None, 
                 class_weights: Optional[torch.Tensor] = None, freeze_head: bool = False):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        
        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Base model must expose `hidden_size`.")
        
        self.classification_head = nn.Linear(hidden_size, num_labels)
        self.class_weights = class_weights
        
        # Initialize from label token embeddings if available
        if label_token_map:
            try:
                token_ids = [label_token_map[k] for k in sorted(label_token_map.keys())]
                with torch.no_grad():
                    lm_head = base_model.lm_head if hasattr(base_model, 'lm_head') else base_model.base_model.lm_head
                    self.classification_head.weight.copy_(lm_head.weight[token_ids])
                    nn.init.zeros_(self.classification_head.bias)
            except Exception as e:
                print(f"Could not initialize from lm_head: {e}")
        
        if freeze_head:
            for p in self.classification_head.parameters():
                p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Remove 'labels' from kwargs to avoid passing to base model
        kwargs.pop('labels', None)
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        
        hidden_states = outputs.hidden_states[-1]
        
        # Pool from last non-padded token
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=-1) - 1
        else:
            seq_lens = torch.full((hidden_states.size(0),), hidden_states.size(1) - 1, 
                                  device=hidden_states.device)
        
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, seq_lens]
        
        logits = self.classification_head(pooled)
        
        loss = None
        if labels is not None:
            weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = F.cross_entropy(logits, labels, weight=weight)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=getattr(outputs, 'attentions', None),
        )
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Forward gradient checkpointing to base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(**kwargs)


class BalancedTrainer(Trainer):
    """Trainer with balanced sampling for imbalanced datasets."""
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Training requires train_dataset")
        
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


@dataclasses.dataclass
class ExperimentConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    dataset_name: str = "mteb/tweet_sentiment_extraction"
    dataset_config: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "test"
    text_field: str = "text"
    label_field: str = "label"
    train_subset: Optional[int] = 8000
    eval_subset: Optional[int] = 2000
    random_seed: int = 42
    learning_rate: float = 2e-4  # INCREASED for classification head
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_r: int = 16  # INCREASED for better adaptation
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 256  # REDUCED - tweets are short
    output_dir: str = "outputs/experiment"
    interpretability_example_count: int = 5
    occlusion_batch_size: int = 16
    run_occlusion: bool = True
    load_in_4bit: bool = True
    finetune: bool = True
    label_space: Optional[Sequence[int]] = (0, 1)
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01


class PromptFormatter:
    def __init__(self, label_space: Sequence[int]) -> None:
        self.label_space = list(label_space)
        label_list = ", ".join(str(l) for l in self.label_space)
        self.instruction = f"Classify sentiment as {label_list}."
        self.template = "{instruction}\nText: {sentence}\nSentiment:"
    
    def build_prompt(self, sentence: str) -> str:
        return self.template.format(instruction=self.instruction, sentence=sentence)


def _load_label_token_map(tokenizer, label_space: Sequence[int]) -> LabelTokenMap:
    label_token_map = {}
    for label in label_space:
        token_ids = tokenizer(f" {label}", add_special_tokens=False)["input_ids"]
        label_token_map[int(label)] = token_ids[-1]
    return label_token_map


def _compute_class_weights(labels: Sequence[int], label_space: Sequence[int]) -> torch.Tensor:
    counts = {int(l): 0 for l in label_space}
    for lbl in labels:
        counts[int(lbl)] = counts.get(int(lbl), 0) + 1
    
    total = sum(counts.values())
    n_classes = len(label_space)
    weights = []
    for lbl in sorted(counts.keys()):
        count = counts[lbl]
        # Balanced class weights formula
        weights.append(total / (n_classes * count) if count > 0 else 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def _prepare_dataset(dataset: DatasetDict, config: ExperimentConfig, tokenizer, formatter: PromptFormatter) -> DatasetDict:
    """Prepare dataset with proper tokenization for classification."""
    
    def _format_examples(examples):
        prompts = [formatter.build_prompt(s) for s in examples[config.text_field]]
        
        # Tokenize prompts only (not with labels appended)
        model_inputs = tokenizer(
            prompts, 
            max_length=config.max_seq_length, 
            truncation=True, 
            padding="max_length",
            return_tensors=None  # Return lists
        )
        
        # Classification labels (integers)
        model_inputs["labels"] = [int(l) for l in examples[config.label_field]]
        return model_inputs
    
    cols_to_remove = dataset[config.train_split].column_names
    processed = dataset.map(_format_examples, batched=True, remove_columns=cols_to_remove)
    
    if config.train_subset:
        n = min(config.train_subset, len(processed[config.train_split]))
        processed[config.train_split] = processed[config.train_split].shuffle(seed=config.random_seed).select(range(n))
    if config.eval_subset:
        n = min(config.eval_subset, len(processed[config.eval_split]))
        processed[config.eval_split] = processed[config.eval_split].shuffle(seed=config.random_seed).select(range(n))
    
    return processed


def _build_logits_fn(model, tokenizer, device, max_length, formatter):
    """Build prediction function for evaluation."""
    
    def _predict(texts: Sequence[str]) -> np.ndarray:
        prompts = [formatter.build_prompt(str(t)) for t in texts]
        logits_list = []
        batch_size = 8
        
        model.eval()
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, 
                             max_length=max_length, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            logits_list.append(outputs.logits.cpu().numpy())
        
        return np.concatenate(logits_list, axis=0)
    
    return _predict


def _compute_metrics(labels: List[int], probs: np.ndarray, label_space: Sequence[int]) -> Dict:
    sorted_labels = sorted(label_space)
    pred_indices = np.argmax(probs, axis=1)
    preds = [sorted_labels[i] for i in pred_indices]
    
    cm = confusion_matrix(labels, preds, labels=sorted_labels)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    
    try:
        mcc = matthews_corrcoef(labels, preds)
    except Exception:
        mcc = 0.0
    
    return {
        "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        "mcc": mcc, "confusion_matrix": cm.tolist(),
    }


def _fit_temperature(logits: np.ndarray, labels: List[int]) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float)
    labels_t = torch.tensor(labels, dtype=torch.long)
    temp = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([temp], lr=0.01, max_iter=50)
    
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / temp.clamp(min=0.1), labels_t)
        loss.backward()
        return loss
    
    opt.step(closure)
    return float(temp.detach().clamp(min=0.1))


def _temperature_calibrator(temperature: Optional[float]) -> Callable[[np.ndarray], np.ndarray]:
    def _apply(logits: np.ndarray) -> np.ndarray:
        t = torch.tensor(logits, dtype=torch.float)
        if temperature:
            t = t / temperature
        return torch.softmax(t, dim=-1).numpy()
    return _apply


def evaluate_model(model, tokenizer, texts, labels, device, max_length, formatter, label_space, calibrate=True):
    """Evaluate model and optionally calibrate."""
    logits_fn = _build_logits_fn(model, tokenizer, device, max_length, formatter)
    logits = logits_fn(texts)
    
    raw_probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    metrics = {"raw": _compute_metrics(labels, raw_probs, label_space)}
    calibration = {}
    
    if calibrate:
        temp = _fit_temperature(logits, labels)
        temp_probs = _temperature_calibrator(temp)(logits)
        metrics["temperature"] = _compute_metrics(labels, temp_probs, label_space)
        calibration["temperature"] = temp
    
    return metrics, calibration


def _select_best_variant(metrics: Dict) -> Tuple[str, Dict]:
    best_name, best_metrics = "raw", metrics.get("raw", {})
    best_f1 = best_metrics.get("f1", -1.0)
    
    for name, values in metrics.items():
        if values.get("f1", -1.0) > best_f1:
            best_f1 = values["f1"]
            best_name, best_metrics = name, values
    
    return best_name, best_metrics


def run_shap(shap_values, shap_outputs, output_dir, prefix):
    pass


def run_occlusion(model, tokenizer, texts, device, config, formatter, prefix, calibrator=None):
    """Run occlusion-based interpretability and save results."""
    print(f"Running Occlusion for {prefix}...")
    
    class_names = [str(l) for l in sorted(formatter.label_space)]
    samples = texts[:config.interpretability_example_count]
    
    def predict_fn(text_list):
        """Prediction function returning probabilities."""
        prompts = [formatter.build_prompt(str(t)) for t in text_list]
        logits_list = []
        
        model.eval()
        for i in range(0, len(prompts), config.occlusion_batch_size):
            batch = prompts[i:i + config.occlusion_batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                             max_length=config.max_seq_length, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            logits_list.append(outputs.logits.cpu().numpy())
        
        logits = np.concatenate(logits_list, axis=0)
        
        if calibrator:
            return calibrator(logits)
        return torch.softmax(torch.tensor(logits), dim=-1).numpy()
    
    def tokenize_text(text: str) -> List[str]:
        """Simple whitespace tokenization."""
        import re
        return re.findall(r'\S+', text)
    
    def compute_occlusion_scores(text: str, mask_token: str = "[MASK]") -> List[Tuple[str, float]]:
        """Compute importance scores by occluding each token."""
        tokens = tokenize_text(text)
        if not tokens:
            return []
        
        # Get baseline prediction
        baseline_probs = predict_fn([text])[0]
        pred_idx = np.argmax(baseline_probs)
        baseline_score = baseline_probs[pred_idx]
        
        # Occlude each token and measure change
        occluded_texts = []
        for i in range(len(tokens)):
            occluded = tokens[:i] + [mask_token] + tokens[i+1:]
            occluded_texts.append(" ".join(occluded))
        
        # Batch predict
        occluded_probs = predict_fn(occluded_texts)
        
        # Calculate importance as drop in predicted class probability
        token_scores = []
        for i, token in enumerate(tokens):
            occluded_score = occluded_probs[i, pred_idx]
            importance = baseline_score - occluded_score  # Positive = token helps prediction
            token_scores.append((token, float(importance)))
        
        return token_scores
    
    # Process each sample
    occlusion_outputs = []
    for text in samples:
        probs = predict_fn([text])[0]
        pred_idx = np.argmax(probs)
        pred_label = class_names[pred_idx]
        
        token_weights = compute_occlusion_scores(text)
        # Sort by absolute importance
        token_weights_sorted = sorted(token_weights, key=lambda x: abs(x[1]), reverse=True)
        
        occlusion_outputs.append({
            "text": text,
            "predicted_label": pred_label,
            "predicted_probs": {cn: float(p) for cn, p in zip(class_names, probs)},
            "token_weights": token_weights,  # Original order
            "token_weights_ranked": token_weights_sorted[:20],  # Top 20 by importance
        })
    
    # Save JSON
    with open(os.path.join(config.output_dir, f"{prefix}_occlusion.json"), "w") as f:
        json.dump(occlusion_outputs, f, indent=2)
    
    # Plot
    _plot_occlusion(occlusion_outputs, config.output_dir, prefix)
    
    return occlusion_outputs


def _plot_occlusion(occlusion_outputs, output_dir, prefix):
    """Plot occlusion-based feature importance."""
    if not plt:
        return
    
    n = len(occlusion_outputs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]
    
    for ax, item in zip(axes, occlusion_outputs):
        weights = item['token_weights_ranked'][:15]  # Top 15 features
        if not weights:
            continue
        
        tokens, vals = zip(*weights)
        y_pos = np.arange(len(tokens))
        # Positive = helps prediction (green), Negative = hurts prediction (red)
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in vals]
        
        ax.barh(y_pos, vals, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Importance (Δ probability when occluded)')
        ax.set_title(f"Predicted: {item['predicted_label']} | Text: {item['text'][:60]}...")
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_occlusion_plot.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")
    
    # Create heatmap visualization showing tokens in sequence
    _plot_occlusion_heatmap(occlusion_outputs, output_dir, prefix)


def _plot_occlusion_heatmap(occlusion_outputs, output_dir, prefix):
    """Plot text with color-coded importance."""
    if not plt:
        return
    
    n = len(occlusion_outputs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2 * n))
    if n == 1:
        axes = [axes]
    
    for ax, item in zip(axes, occlusion_outputs):
        weights = item['token_weights']
        if not weights:
            continue
        
        tokens, vals = zip(*weights)
        vals = np.array(vals)
        
        # Normalize values for coloring
        max_abs = max(abs(vals.min()), abs(vals.max())) if len(vals) > 0 else 1
        if max_abs == 0:
            max_abs = 1
        norm_vals = vals / max_abs
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Plot tokens with background colors
        x_pos = 0.02
        for token, nv in zip(tokens, norm_vals):
            # Green for positive, red for negative
            if nv > 0:
                bg_color = (0.18, 0.8, 0.44, min(abs(nv), 1) * 0.7)
            else:
                bg_color = (0.91, 0.3, 0.24, min(abs(nv), 1) * 0.7)
            
            txt = ax.text(x_pos, 0.5, token + " ", fontsize=10, 
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, edgecolor='none'),
                         verticalalignment='center')
            
            # Get text width for next position
            fig.canvas.draw()
            bbox = txt.get_window_extent()
            x_pos += (bbox.width / fig.dpi / fig.get_figwidth()) + 0.005
            
            if x_pos > 0.95:  # Wrap text
                break
        
        ax.set_title(f"Pred: {item['predicted_label']} | Green=helps, Red=hurts", fontsize=10, loc='left')
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_occlusion_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def _plot_confusion_matrix(cm, labels, output_dir, prefix):
    if not plt: return
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    ax.set_title(f'{prefix.replace("_", " ").title()} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"))
    plt.close()


def _plot_comparison(zs_metrics, ft_metrics, output_dir, zs_var, ft_var):
    if not plt or not ft_metrics: return
    
    keys = ['accuracy', 'precision', 'recall', 'f1', 'mcc']
    z_vals = [zs_metrics.get(m, 0) for m in keys]
    f_vals = [ft_metrics.get(m, 0) for m in keys]
    
    x = np.arange(len(keys))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, z_vals, width, label=f'Zero-Shot ({zs_var})', color='steelblue')
    bars2 = ax.bar(x + width/2, f_vals, width, label=f'Fine-Tuned ({ft_var})', color='darkorange')
    
    # Add value labels
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    ax.set_ylabel('Score')
    ax.set_title('Zero-Shot vs Fine-Tuned Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in keys])
    ax.legend()
    ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
    plt.close()
    print(f"Saved comparison plot")


def run_experiment(args: argparse.Namespace) -> None:
    """Main experiment runner."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = ExperimentConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        load_in_4bit=args.load_in_4bit,
        run_occlusion=args.run_occlusion,
        finetune=args.finetune,
    )
    
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(config.dataset_name, config.dataset_config) if config.dataset_config else load_dataset(config.dataset_name)
    label_space = sorted(set(config.label_space)) if config.label_space else sorted({int(x) for x in dataset[config.train_split][config.label_field]})
    
    # Filter to binary if needed
    allowed = set(label_space)
    dataset = dataset.filter(lambda x: int(x[config.label_field]) in allowed)
    
    label_token_map = _load_label_token_map(tokenizer, label_space)
    formatter = PromptFormatter(label_space)
    
    # Load base model
    print("Loading base model...")
    quant_config = None
    if config.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        device_map="auto" if config.load_in_4bit else None,
        torch_dtype=torch.bfloat16 if not config.load_in_4bit else None,
    )
    base_model.config.output_hidden_states = True
    
    if not config.load_in_4bit:
        base_model.to(device)
    
    # Create classification model for zero-shot
    zs_model = SentimentClassificationModel(base_model, num_labels=len(label_space), label_token_map=label_token_map)
    zs_model.to(device)
    
    # Prepare eval data
    eval_split = dataset[config.eval_split]
    if config.eval_subset:
        eval_split = eval_split.shuffle(seed=config.random_seed).select(range(min(config.eval_subset, len(eval_split))))
    eval_texts = list(eval_split[config.text_field])
    eval_labels = list(eval_split[config.label_field])
    
    # === ZERO-SHOT EVALUATION ===
    print("\n" + "="*50)
    print("PHASE 1: Zero-Shot Evaluation")
    print("="*50)
    
    zs_metrics, zs_calib = evaluate_model(zs_model, tokenizer, eval_texts, eval_labels, device, config.max_seq_length, formatter, label_space)
    zs_variant, zs_best = _select_best_variant(zs_metrics)
    
    print(f"\nZero-Shot Results ({zs_variant}):")
    print(f"  Accuracy:  {zs_best['accuracy']:.4f}")
    print(f"  Precision: {zs_best['precision']:.4f}")
    print(f"  Recall:    {zs_best['recall']:.4f}")
    print(f"  F1:        {zs_best['f1']:.4f}")
    print(f"  MCC:       {zs_best['mcc']:.4f}")
    
    with open(os.path.join(config.output_dir, "zero_shot_metrics.json"), "w") as f:
        json.dump(zs_metrics, f, indent=2)
    _plot_confusion_matrix(zs_best['confusion_matrix'], label_space, config.output_dir, "zero_shot")
    
    # Run Occlusion for zero-shot
    if config.run_occlusion:
        zs_calibrator = _temperature_calibrator(zs_calib.get("temperature")) if zs_calib else None
        run_occlusion(zs_model, tokenizer, eval_texts, device, config, formatter, "zero_shot", zs_calibrator)
    
    # === FINE-TUNING ===
    ft_metrics = None
    ft_best = None
    ft_variant = None
    
    if config.finetune:
        print("\n" + "="*50)
        print("PHASE 2: Fine-Tuning")
        print("="*50)
        
        # Prepare training data
        processed_dataset = _prepare_dataset(dataset, config, tokenizer, formatter)
        class_weights = _compute_class_weights(processed_dataset[config.train_split]["labels"], label_space)
        
        print(f"Training samples: {len(processed_dataset[config.train_split])}")
        print(f"Class weights: {class_weights.tolist()}")
        
        # Setup LoRA
        if config.load_in_4bit:
            base_model = prepare_model_for_kbit_training(base_model)
        
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        base_model.config.use_cache = False
        peft_model = get_peft_model(base_model, peft_config)
        peft_model.print_trainable_parameters()
        
        # Create classification model with PEFT base
        ft_model = SentimentClassificationModel(
            peft_model,
            num_labels=len(label_space),
            label_token_map=label_token_map,
            class_weights=class_weights,
        )
        ft_model.to(device)
        
        # Ensure classification head is trainable
        for param in ft_model.classification_head.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in ft_model.parameters())
        print(f"Total trainable params (including head): {trainable:,} / {total:,}")
        
        # Training args
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            report_to="none",
            dataloader_pin_memory=False,
            remove_unused_columns=False,  # IMPORTANT: keep our labels
        )
        
        trainer = BalancedTrainer(
            model=ft_model,
            args=training_args,
            train_dataset=processed_dataset[config.train_split],
            eval_dataset=processed_dataset[config.eval_split],
            data_collator=default_data_collator,
        )
        
        print("\nStarting training...")
        trainer.train()
        
        # Evaluate fine-tuned model
        print("\nEvaluating fine-tuned model...")
        ft_model.eval()
        
        ft_metrics, ft_calib = evaluate_model(ft_model, tokenizer, eval_texts, eval_labels, device, config.max_seq_length, formatter, label_space)
        ft_variant, ft_best = _select_best_variant(ft_metrics)
        
        print(f"\nFine-Tuned Results ({ft_variant}):")
        print(f"  Accuracy:  {ft_best['accuracy']:.4f}")
        print(f"  Precision: {ft_best['precision']:.4f}")
        print(f"  Recall:    {ft_best['recall']:.4f}")
        print(f"  F1:        {ft_best['f1']:.4f}")
        print(f"  MCC:       {ft_best['mcc']:.4f}")
        
        with open(os.path.join(config.output_dir, "fine_tuned_metrics.json"), "w") as f:
            json.dump(ft_metrics, f, indent=2)
        _plot_confusion_matrix(ft_best['confusion_matrix'], label_space, config.output_dir, "fine_tuned")
        
        # Run Occlusion for fine-tuned
        if config.run_occlusion:
            ft_calibrator = _temperature_calibrator(ft_calib.get("temperature")) if ft_calib else None
            run_occlusion(ft_model, tokenizer, eval_texts, device, config, formatter, "fine_tuned", ft_calibrator)
    
    # === COMPARISON ===
    if ft_best:
        print("\n" + "="*50)
        print("COMPARISON")
        print("="*50)
        
        print(f"\n{'Metric':<12} {'Zero-Shot':>12} {'Fine-Tuned':>12} {'Δ':>12}")
        print("-" * 50)
        for m in ['accuracy', 'precision', 'recall', 'f1', 'mcc']:
            zs_val = zs_best.get(m, 0)
            ft_val = ft_best.get(m, 0)
            delta = ft_val - zs_val
            sign = "+" if delta > 0 else ""
            print(f"{m:<12} {zs_val:>12.4f} {ft_val:>12.4f} {sign}{delta:>11.4f}")
        
        _plot_comparison(zs_best, ft_best, config.output_dir, zs_variant, ft_variant)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--finetune", action="store_true", default=True)
    parser.add_argument("--no-finetune", dest="finetune", action="store_false")
    parser.add_argument("--run-occlusion", action="store_true", default=False)
    parser.add_argument("--no-occlusion", dest="run_occlusion", action="store_false")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--output-dir", default="outputs/experiment")
    return parser


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
