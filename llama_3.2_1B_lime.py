# final_with_10_balanced_lime.py
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


class PromptFormatter:
    def __init__(self, label_space=(0, 1)):
        labels_str = ", ".join(str(l) for l in label_space)
        self.template = f"Classify the sentiment as {labels_str}.\nText: {{text}}\nSentiment:"

    def format(self, text: str) -> str:
        return self.template.format(text=text)


def prepare_masked_dataset(dataset, tokenizer, formatter, max_length=256, train_subset=None, eval_subset=None):
    def tokenize(examples):
        texts, labels = examples["text"], examples["label"]
        input_ids_list, attention_mask_list, labels_list = [], [], []

        for text, label in zip(texts, labels):
            prompt = formatter.format(text)
            full = prompt + f" {label}"
            tokenized = tokenizer(full, truncation=True, max_length=max_length, padding=False)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            prompt_tok = tokenizer(prompt, truncation=True, max_length=max_length)
            label_start = len(prompt_tok["input_ids"])
            lbl = [-100] * len(input_ids)
            if label_start < len(input_ids):
                lbl[label_start] = input_ids[label_start]
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(lbl)

        return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}

    processed = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    if train_subset:
        processed["train"] = processed["train"].shuffle(seed=42).select(range(min(train_subset, len(processed["train"]))))
    if eval_subset:
        processed["test"] = processed["test"].shuffle(seed=42).select(range(min(eval_subset, len(processed["test"]))))
    return processed


def evaluate_model(model, tokenizer, dataset, formatter, device):
    model.eval()
    token_0 = tokenizer(" 0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer(" 1", add_special_tokens=False)["input_ids"][0]

    texts = [ex["text"] for ex in dataset]
    labels = [ex["label"] for ex in dataset]

    batch = [formatter.format(t) for t in texts]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
    prob_1 = torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1].cpu().numpy()
    preds = (prob_1 > 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "accuracy": float(acc), "precision": float(p), "recall": float(r),
        "f1": float(f1), "mcc": float(mcc), "confusion_matrix": cm.tolist()
    }, preds, prob_1


def plot_confusion_matrix(cm, title, path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    plt.ylabel("True"); plt.xlabel("Predicted")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center",
                     color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def generate_lime_grid_balanced(model, tokenizer, formatter, dataset, device, title, path):
    explainer = LimeTextExplainer(class_names=["Negative", "Positive"], mode="classification")
    token_0 = tokenizer(" 0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer(" 1", add_special_tokens=False)["input_ids"][0]

    def predict_fn(texts):
        batch = [formatter.format(t) for t in texts]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        prob_1 = torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1].cpu().numpy()
        return np.stack([1 - prob_1, prob_1], axis=1)

    # Get predictions
    texts = [ex["text"] for ex in dataset]
    labels = [ex["label"] for ex in dataset]
    prob_1 = predict_fn(texts)[:, 1]
    preds = (prob_1 > 0.5).astype(int)

    correct_neg = [i for i, (l, p) in enumerate(zip(labels, preds)) if l == 0 and p == 0]
    correct_pos = [i for i, (l, p) in enumerate(zip(labels, preds)) if l == 1 and p == 1]

    selected_neg = random.sample(correct_neg, min(5, len(correct_neg)))
    selected_pos = random.sample(correct_pos, min(5, len(correct_pos)))
    indices = selected_neg + selected_pos
    random.shuffle(indices)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(title, fontsize=24, weight="bold")

    for ax, idx in zip(axes.flat, indices):
        ex = dataset[idx]
        exp = explainer.explain_instance(ex["text"], predict_fn, num_features=10, num_samples=1000)
        exp.as_pyplot_figure(label=ex["label"])
        lime_img = plt.gcf()
        ax.imshow(lime_img.canvas.renderer.buffer_rgba())
        true = "Positive" if ex["label"] == 1 else "Negative"
        ax.set_title(f"True: {true}\n\"{ex['text'][:70]}{'...' if len(ex['text'])>70 else ''}\"",
                     fontsize=10, pad=8)
        ax.axis("off")
        plt.close(lime_img)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", default="outputs/final_balanced_lime")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--eval-size", type=int, default=2000)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4") if not args.no_4bit else None

    base_model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant_config,
                                                      device_map="auto", torch_dtype=torch.bfloat16)
    if not args.no_4bit:
        base_model = prepare_model_for_kbit_training(base_model)

    formatter = PromptFormatter()
    raw = load_dataset("mteb/tweet_sentiment_extraction").filter(lambda x: x["label"] in [0, 1])
    eval_data = raw["test"].shuffle(seed=42).select(range(args.eval_size))

    print("Zero-shot evaluation...")
    zs_metrics, zs_preds, _ = evaluate_model(base_model, tokenizer, eval_data, formatter, device)
    plot_confusion_matrix(np.array(zs_metrics["confusion_matrix"]), "Zero-Shot", f"{args.output_dir}/confusion_zero_shot.png")
    generate_lime_grid_balanced(base_model, tokenizer, formatter, eval_data, device,
                                "LIME - Zero-Shot (5 Neg + 5 Pos Correct)", f"{args.output_dir}/lime_zero_shot.png")

    print("Fine-tuning...")
    peft_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                             task_type="CAUSAL_LM")
    model = get_peft_model(base_model, peft_config)

    dataset = prepare_masked_dataset(raw, tokenizer, formatter, train_subset=args.train_size, eval_subset=args.eval_size)

    trainer = Trainer(model=model,
                      args=TrainingArguments(output_dir=args.output_dir, num_train_epochs=args.epochs,
                                             per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=4,
                                             learning_rate=args.lr, weight_decay=0.01, warmup_ratio=0.1,
                                             logging_steps=10, evaluation_strategy="epoch", save_strategy="epoch",
                                             fp16=not args.no_4bit, bf16=args.no_4bit, report_to="none",
                                             load_best_model_at_end=True, remove_unused_columns=False),
                      train_dataset=dataset["train"], eval_dataset=dataset["test"],
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()

    print("Fine-tuned evaluation...")
    ft_metrics, ft_preds, _ = evaluate_model(model, tokenizer, eval_data, formatter, device)
    plot_confusion_matrix(np.array(ft_metrics["confusion_matrix"]), "Fine-Tuned", f"{args.output_dir}/confusion_fine_tuned.png")
    generate_lime_grid_balanced(model, tokenizer, formatter, eval_data, device,
                                "LIME - Fine-Tuned (5 Neg + 5 Pos Correct)", f"{args.output_dir}/lime_fine_tuned.png")

    results = {"zero_shot": zs_metrics, "fine_tuned": ft_metrics}
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDONE! Check:", args.output_dir)


if __name__ == "__main__":
    main()
