# llama_3.2_1B_sst2_final_with_bar_chart.py

from __future__ import annotations

import argparse
import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm


class PromptFormatter:
    def __init__(self):
        self.template = "Classify the sentiment as 0 (negative) or 1 (positive).\nText: {text}\nSentiment:"

    def format(self, text: str) -> str:
        return self.template.format(text=text)


def make_predict_fn(model, tokenizer, formatter, device):
    token_0 = tokenizer("0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]

    def predict_fn(texts):
        if isinstance(texts, str):
            texts = [texts]
        probs = []
        for i in range(0, len(texts), 4):
            batch = texts[i:i+4]
            prompts = [formatter.format(t) for t in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                             truncation=True, max_length=256).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits[:, -1, :].float()
            prob_1 = torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1]
            probs.extend(prob_1.cpu().numpy())
            del inputs, logits
            torch.cuda.empty_cache()
        return np.stack([1 - np.array(probs), np.array(probs)], axis=1)
    return predict_fn


def evaluate_model_safe(model, tokenizer, dataset, formatter, device, batch_size=8):
    model.eval()
    token_0 = tokenizer("0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]

    all_preds, all_labels = [], []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i+batch_size]
        texts = batch["sentence"]
        labels = batch["label"]

        prompts = [formatter.format(t) for t in texts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :].float()
        prob_1 = torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1]
        preds = (prob_1 > 0.5).int().cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels)
        del inputs, logits, prob_1
        torch.cuda.empty_cache()

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "mcc": round(float(mcc), 4),
        "confusion_matrix": cm.tolist()
    }, np.array(all_preds)


def plot_confusion(cm, title, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title, fontsize=16, pad=20)
    plt.colorbar()
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center",
                     color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=18)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_all_explanations(model, tokenizer, formatter, dataset, device, title_prefix, output_dir):
    explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    predict_fn = make_predict_fn(model, tokenizer, formatter, device)

    _, preds = evaluate_model_safe(model, tokenizer, dataset, formatter, device, batch_size=16)
    correct_neg = [i for i, (l, p) in enumerate(zip(dataset["label"], preds)) if l == 0 and p == 0]
    correct_pos = [i for i, (l, p) in enumerate(zip(dataset["label"], preds)) if l == 1 and p == 1]

    selected = (random.sample(correct_neg, min(5, len(correct_neg))) +
                random.sample(correct_pos, min(5, len(correct_pos))))
    random.shuffle(selected)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    for plot_idx in range(len(selected)):
        idx = selected[plot_idx]
        text = dataset[idx]["sentence"]
        true_label = "Pos" if dataset[idx]["label"] == 1 else "Neg"

        exp = explainer.explain_instance(text, predict_fn, num_features=10, num_samples=500)
        temp_path = f"/tmp/lime_{plot_idx}.png"
        exp.as_pyplot_figure()
        plt.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        img = plt.imread(temp_path)
        axes[plot_idx].imshow(img)
        axes[plot_idx].set_title(f"{true_label}: {text[:60]}...", fontsize=10)
        axes[plot_idx].axis("off")
        os.remove(temp_path)

    plt.suptitle(f"{title_prefix} - LIME Explanations", fontsize=28, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/lime_{title_prefix.lower().replace(' ', '_')}.png", dpi=200)
    plt.close('all')
    print(f"LIME grid saved: {title_prefix}")


def plot_metrics_comparison(zs_metrics, ft_metrics, output_dir):
    metrics = ['accuracy', 'f1', 'mcc']
    zs_values = [zs_metrics[m] for m in metrics]
    ft_values = [ft_metrics[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, zs_values, width, label='Zero-Shot', color='#1f77b4', edgecolor='black')
    bars2 = ax.bar(x + width/2, ft_values, width, label='Fine-Tuned', color='#ff7f0e', edgecolor='black')

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Zero-Shot vs Fine-Tuned Performance Comparison', fontsize=18, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'F1-Score', 'MCC'])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12)

    # Add value labels on top of bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    path = f"{output_dir}/metrics_comparison_bar_chart.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Bar chart saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", type=str, default="outputs/sst2_final")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--epochs", type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    formatter = PromptFormatter()
    dataset = load_dataset("stanfordnlp/sst2")
    eval_data = dataset["validation"]  # 872 examples

    print("\nZERO-SHOT EVALUATION")
    zs_metrics, _ = evaluate_model_safe(model, tokenizer, eval_data, formatter, device)
    print(f"Zero-Shot → Acc: {zs_metrics['accuracy']:.4f} | F1: {zs_metrics['f1']:.4f} | MCC: {zs_metrics['mcc']:.4f}")
    plot_confusion(np.array(zs_metrics["confusion_matrix"]), "Zero-Shot", f"{args.output_dir}/confusion_zero_shot.png")
    generate_all_explanations(model, tokenizer, formatter, eval_data, device, "Zero-Shot", args.output_dir)

    print("\nSTARTING LoRA FINE-TUNING...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    ))

    def tokenize_function(examples):
        prompts = [formatter.format(t) for t in examples["sentence"]]
        full_texts = [p + f" {l}" for p, l in zip(prompts, examples["label"])]
        tokenized = tokenizer(full_texts, truncation=True, max_length=256, padding=False)
        labels = []
        for i, text in enumerate(examples["sentence"]):
            prompt_len = len(tokenizer(formatter.format(text))["input_ids"])
            label_seq = tokenized["input_ids"][i][:]
            label_seq[:prompt_len] = [-100] * prompt_len
            labels.append(label_seq)
        tokenized["labels"] = labels
        return tokenized

    train_dataset = dataset["train"].shuffle(seed=42).select(range(args.train_size))
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
        ),
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    print("\nFINE-TUNED EVALUATION")
    ft_metrics, _ = evaluate_model_safe(model, tokenizer, eval_data, formatter, device)
    print(f"Fine-Tuned → Acc: {ft_metrics['accuracy']:.4f} | F1: {ft_metrics['f1']:.4f} | MCC: {ft_metrics['mcc']:.4f}")
    plot_confusion(np.array(ft_metrics["confusion_matrix"]), "Fine-Tuned", f"{args.output_dir}/confusion_fine_tuned.png")
    generate_all_explanations(model, tokenizer, formatter, eval_data, device, "Fine-Tuned", args.output_dir)

    # COMPARISON BAR CHART
    plot_metrics_comparison(zs_metrics, ft_metrics, args.output_dir)

    # SAVE RESULTS
    results = {
        "zero_shot": zs_metrics,
        "fine_tuned": ft_metrics
    }
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Pretty print in terminal
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Metric':<12} {'Zero-Shot':<12} {'Fine-Tuned':<12} {'Gain'}")
    print("-"*60)
    for m in ['accuracy', 'f1', 'mcc']:
        gain = ft_metrics[m] - zs_metrics[m]
        print(f"{m.capitalize():<12} {zs_metrics[m]:<12.4f} {ft_metrics[m]:<12.4f} +{gain:.4f}")
    print("="*60)
    print(f"All files saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
