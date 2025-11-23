# masked_finetune_with_lime.py
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Sequence

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# LIME
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns


# === Prompt & Dataset Preparation (Masked Last-Token) ===
class PromptFormatter:
    def __init__(self, label_space: Sequence[int] = (0, 1)):
        self.label_space = sorted(label_space)
        labels_str = ", ".join(str(l) for l in self.label_space)
        self.template = f"Classify the sentiment as {labels_str}.\nText: {{text}}\nSentiment:"

    def format(self, text: str) -> str:
        return self.template.format(text=text)


def prepare_dataset(dataset: DatasetDict, tokenizer, formatter, max_length=256, train_subset=None, eval_subset=None):
    def tokenize_and_mask(examples):
        texts = examples["text"]
        labels = examples["label"]

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for text, label in zip(texts, labels):
            prompt = formatter.format(text)
            full_text = prompt + f" {label}"

            tokenized = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
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

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    processed = dataset.map(tokenize_and_mask, batched=True,
                           remove_columns=dataset["train"].column_names)

    if train_subset:
        processed["train"] = processed["train"].shuffle(seed=42).select(range(min(train_subset, len(processed["train"]))))
    if eval_subset:
        processed["test"] = processed["test"].shuffle(seed=42).select(range(min(eval_subset, len(processed["test"]))))

    return processed


# === LIME Predictor Function ===
def make_lime_predictor(model, tokenizer, formatter, device, label_tokens):
    model.eval()

    def predict_proba(texts):
        batch = [formatter.format(text) for text in texts]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]  # next-token logits

        probs_0 = torch.softmax(logits[:, label_tokens], dim=-1)[:, 0].cpu().numpy()
        probs_1 = 1 - probs_0
        return np.stack([probs_0, probs_1], axis=1)

    return predict_proba


# === Evaluation ===
def evaluate(model, tokenizer, dataset, formatter, device, label_space=(0, 1)):
    model.eval()
    preds, trues = [], []

    token_0 = tokenizer(" 0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer(" 1", add_special_tokens=False)["input_ids"][0]

    for ex in dataset:
        prompt = formatter.format(ex["text"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        pred = 1 if logits[token_1] > logits[token_0] else 0

        preds.append(pred)
        trues.append(ex["label"])

    acc = accuracy_score(trues, preds)
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(trues, preds)
    cm = confusion_matrix(trues, preds, labels=sorted(label_space))

    return {
        "accuracy": acc, "precision": p, "recall": r, "f1": f1, "mcc": mcc,
        "confusion_matrix": cm.tolist()
    }


# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", default="outputs/masked_lime")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--eval-size", type=int, default=2000)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lime-examples", type=int, default=10)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if not args.no_4bit:
        base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Dataset
    raw = load_dataset("mteb/tweet_sentiment_extraction")
    raw = raw.filter(lambda x: x["label"] in [0, 1])
    formatter = PromptFormatter((0, 1))

    dataset = prepare_dataset(raw, tokenizer, formatter,
                              train_subset=args.train_size, eval_subset=args.eval_size)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=not args.no_4bit,
        bf16=args.no_4bit,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=dataset["train"],
                      eval_dataset=dataset["test"],
                      data_collator=data_collator)

    print("Training with masked last-token supervision...")
    trainer.train()

    # Final evaluation
    print("Final evaluation...")
    metrics = evaluate(model, tokenizer, raw["test"].select(range(args.eval_size)),
                       formatter, device)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k.capitalize():10}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # === LIME EXPLANATIONS ===
    print(f"\nGenerating LIME explanations for {args.lime_examples} examples...")
    lime_explainer = LimeTextExplainer(class_names=["Negative", "Positive"], mode="classification")

    token_0 = tokenizer(" 0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer(" 1", add_special_tokens=False)["input_ids"][0]
    predict_fn = make_lime_predictor(model, tokenizer, formatter, device, [token_0, token_1])

    test_examples = raw["test"].shuffle(seed=42).select(range(args.eval_size))
    lime_samples = random.sample(range(len(test_examples)), args.lime_examples)

    os.makedirs(os.path.join(args.output_dir, "lime"), exist_ok=True)

    for idx in lime_samples:
        example = test_examples[idx]
        text = example["text"]
        true_label = "Positive" if example["label"] == 1 else "Negative"

        explanation = lime_explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_fn,
            num_features=10,
            num_samples=1000,
        )

        # Save HTML
        html_path = os.path.join(args.output_dir, "lime", f"ex_{idx:04d}_true_{true_label}.html")
        explanation.save_to_file(html_path)

        # Save PNG
        fig = explanation.as_pyplot_figure()
        png_path = os.path.join(args.output_dir, "lime", f"ex_{idx:04d}_true_{true_label}.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  → {png_path}")

    print(f"\nAll done! LIME visualizations saved to {args.output_dir}/lime/")
    print("   • Open HTML files in browser for interactive view")
    print("   • PNG files are static high-res versions")


if __name__ == "__main__":
    main()
