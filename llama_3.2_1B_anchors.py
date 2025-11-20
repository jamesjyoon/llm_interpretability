from __future__ import annotations

__doc__ = """Utility for running zero-shot and LoRA-fine-tuned LLaMA style models.
Fixed: Added JSON serialization helper to handle numpy arrays returned by Alibi Anchors.
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

# --- Import ALIBI and SPACY ---
try:
    from alibi.explainers import AnchorText
    import spacy 
except ImportError as exc:
    raise SystemExit("The `alibi` and `spacy` packages are required. Install via `pip install alibi spacy`.") from exc

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

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

try:
    from huggingface_hub import login as hf_login
    from huggingface_hub.errors import GatedRepoError, RepoAccessError
except Exception:
    hf_login = None
    GatedRepoError = RepoAccessError = type("_DummyHFError", (Exception,), {})


HF_ACCESS_ERRORS = (OSError, GatedRepoError, RepoAccessError)


# --- Helper Classes & Functions ---

LabelTokenMap = Dict[int, int]

def _ensure_json_serializable(value):
    """Recursively convert numpy/tensor types to standard Python types for JSON."""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, list):
        return [_ensure_json_serializable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_ensure_json_serializable(v) for v in value)
    if isinstance(value, dict):
        return {k: _ensure_json_serializable(v) for k, v in value.items()}
    return value

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
    train_subset: Optional[int] = 4000
    eval_subset: Optional[int] = 2000
    random_seed: int = 42
    learning_rate: float = 5e-5
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    max_seq_length: int = 516
    output_dir: str = "outputs/experiment_anchors"
    
    # Interpretability
    interpretability_example_count: int = 3 
    run_anchors: bool = True
    anchor_threshold: float = 0.95
    
    load_in_4bit: bool = True
    finetune: bool = True
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
        labels = []
        for input_ids, mask in zip(model_inputs["input_ids"], attention):
            masked = [-100] * len(input_ids)
            seq_len = int(sum(mask))
            if seq_len > 0:
                masked[seq_len - 1] = input_ids[seq_len - 1]
            labels.append(masked)
        model_inputs["labels"] = labels
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

# --- Core Evaluation & Anchor Functions ---

def _build_probability_fn(model, tokenizer, label_token_map, device, max_length, formatter):
    """Returns a function that takes raw texts and returns probabilities."""
    def _predict(texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        prompts = [formatter.build_prompt(t) for t in texts]
        
        probs_list = []
        batch_size = 8
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            seq_lens = inputs.attention_mask.sum(dim=-1) - 1
            final_logits = logits[torch.arange(logits.size(0), device=device), seq_lens]
            
            sorted_labels = sorted(label_token_map.keys())
            target_ids = torch.tensor([label_token_map[l] for l in sorted_labels], device=device)
            label_logits = final_logits[:, target_ids]
            probs = torch.softmax(label_logits, dim=-1)
            probs_list.append(probs.cpu().numpy())
            
        return np.concatenate(probs_list, axis=0)
    return _predict

def evaluate_model(model, tokenizer, texts, labels, label_token_map, device, max_length, formatter):
    preds = []
    sorted_labels = sorted(label_token_map.keys())
    
    predict_fn = _build_probability_fn(model, tokenizer, label_token_map, device, max_length, formatter)
    probs_all = predict_fn(texts)
    pred_indices = np.argmax(probs_all, axis=1)
    preds = [sorted_labels[i] for i in pred_indices]
    
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
    cm = confusion_matrix(labels, preds, labels=sorted_labels)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    try:
        mcc = matthews_corrcoef(labels, preds)
    except:
        mcc = 0.0

    return {
        "accuracy": acc, "precision": p, "recall": r, "f1": f1, "mcc": mcc,
        "confusion_matrix": cm.tolist()
    }

def run_anchors(model, tokenizer, texts, label_token_map, device, config, formatter, prefix):
    """Runs Anchors with a blank SpaCy model to avoid download issues."""
    print(f"Running Anchors for {prefix}...")
    
    prob_fn = _build_probability_fn(model, tokenizer, label_token_map, device, config.max_seq_length, formatter)
    predict_fn = lambda x: np.argmax(prob_fn(x), axis=1)
    
    class_names = [str(l) for l in sorted(label_token_map.keys())]
    
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy 'en_core_web_sm' not found. Using blank 'en' model (no download required).")
        nlp = spacy.blank("en")

    explainer = AnchorText(predictor=predict_fn, sampling_strategy='unknown', nlp=nlp)
    
    samples = texts[:config.interpretability_example_count]
    results = []
    
    for i, text in enumerate(samples):
        print(f"Explaining example {i+1}/{len(samples)}...")
        try:
            explanation = explainer.explain(text, threshold=config.anchor_threshold)
            
            pred_idx = predict_fn([text])[0]
            pred_label = class_names[pred_idx]
            
            results.append({
                "text": text,
                "predicted_label": pred_label,
                "anchor_words": explanation.anchor,
                "precision": explanation.precision,
                "coverage": explanation.coverage
            })
        except Exception as e:
            print(f"Error explaining example {i}: {e}")

    # Save JSON with cleaner
    with open(os.path.join(config.output_dir, f"{prefix}_anchors.json"), "w") as f:
        json.dump(_ensure_json_serializable(results), f, indent=2)

    # Generate Plots
    _plot_anchors_text(results, config.output_dir, prefix)
    
    return results

# --- Plotting Functions ---

def _plot_anchors_text(anchor_results, output_dir, prefix):
    if not plt: return

    for i, res in enumerate(anchor_results):
        text = res['text']
        anchors = res['anchor_words']
        pred_label = res['predicted_label']
        precision = res['precision']
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        
        ax.text(0.5, 0.9, f"{prefix} Example {i+1} (Pred: {pred_label})", 
                ha='center', fontsize=14, weight='bold')
        
        wrapped_text = "\n".join([text[j:j+80] for j in range(0, len(text), 80)])
        ax.text(0.5, 0.6, f"ORIGINAL TEXT:\n{wrapped_text}", 
                ha='center', va='center', fontsize=10, style='italic')
        
        # Anchors might be numpy strings, convert for display
        anchors_list = [str(w) for w in anchors]
        anchor_str = ", ".join([f"'{w}'" for w in anchors_list])
        
        ax.text(0.5, 0.3, f"ANCHOR RULE (Precision: {precision:.2f}):\nIF words [{anchor_str}] present\nTHEN Pred = {pred_label}", 
                ha='center', va='center', fontsize=12, color='darkred', weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", fc="mistyrose", ec="red", lw=2))

        plt.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_anchor_plot_ex{i}.png")
        plt.savefig(path)
        plt.close()

    print(f"Saved Anchor visual rules to {output_dir}")

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
        load_in_4bit=args.load_in_4bit, run_anchors=args.run_anchors,
        finetune=args.finetune,
        train_subset=args.train_subset,
        eval_subset=args.eval_subset,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset(config.dataset_name, config.dataset_config) if config.dataset_config else load_dataset(config.dataset_name)
    label_space = _resolve_label_space(config, dataset)
    dataset = _filter_dataset(dataset, config, label_space)
    
    label_token_map = _load_label_token_map(tokenizer, label_space)
    formatter = PromptFormatter(label_space)
    
    print("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        load_in_4bit=config.load_in_4bit, 
        device_map="auto" if config.load_in_4bit else None
    )
    if not config.load_in_4bit: model.to(device)

    eval_texts, eval_labels = _prepare_texts_labels(config, dataset)

    # ==========================================
    # PHASE 1: ZERO-SHOT
    # ==========================================
    print("--- Phase 1: Running Zero-Shot Evaluation ---")
    zs_metrics = evaluate_model(model, tokenizer, eval_texts, eval_labels, label_token_map, device, config.max_seq_length, formatter)
    print("Zero-Shot Metrics:", zs_metrics)
    
    with open(os.path.join(config.output_dir, "zero_shot_metrics.json"), "w") as f: json.dump(_ensure_json_serializable(zs_metrics), f, indent=2)
    _plot_confusion_matrix(zs_metrics['confusion_matrix'], label_space, config.output_dir, "zero_shot")
    
    if config.run_anchors:
        run_anchors(model, tokenizer, eval_texts, label_token_map, device, config, formatter, "zero_shot")

    # ==========================================
    # PHASE 2: FINE-TUNING
    # ==========================================
    ft_metrics = None
    if args.finetune:
        print("--- Phase 2: Starting Fine-Tuning ---")
        
        processed_dataset = _prepare_dataset(dataset, config, tokenizer, formatter)
        
        if config.load_in_4bit: model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
        model.config.use_cache = False
        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()
        
        trainer = Trainer(
            model=peft_model,
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
        
        peft_model.eval()
        
        print("Evaluating Fine-Tuned Model...")
        ft_metrics = evaluate_model(peft_model, tokenizer, eval_texts, eval_labels, label_token_map, device, config.max_seq_length, formatter)
        print("Fine-Tuned Metrics:", ft_metrics)
        
        with open(os.path.join(config.output_dir, "fine_tuned_metrics.json"), "w") as f: json.dump(_ensure_json_serializable(ft_metrics), f, indent=2)
        _plot_confusion_matrix(ft_metrics['confusion_matrix'], label_space, config.output_dir, "fine_tuned")
        
        if config.run_anchors:
            run_anchors(peft_model, tokenizer, eval_texts, label_token_map, device, config, formatter, "fine_tuned")

    # ==========================================
    # PHASE 3: COMPARISON
    # ==========================================
    if ft_metrics:
        _plot_comparison(zs_metrics, ft_metrics, config.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset-name", default="mteb/tweet_sentiment_extraction")
    parser.add_argument("--dataset-config", default=None)
    
    # Updated args
    parser.add_argument("--finetune", action="store_true", default=True) 
    
    parser.add_argument("--run-anchors", action="store_true")
    parser.add_argument("--no-run-anchors", dest="run_anchors", action="store_false")
    parser.set_defaults(run_anchors=True)
    
    parser.add_argument("--train-subset", type=int, default=4000)
    parser.add_argument("--eval-subset", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output-dir", default="outputs/experiment_anchors")
    parser.add_argument("--huggingface-token", default=None)
    return parser

if __name__ == "__main__":
    parser = build_parser()
    run_experiment(parser.parse_args())
