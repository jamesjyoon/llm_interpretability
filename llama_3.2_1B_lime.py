# final_working_v100.py
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
        self.template = "Classify the sentiment as 0, 1.\nText: {text}\nSentiment:"

    def format(self, text: str) -> str:
        return self.template.format(text=text)


def evaluate_model_safe(model, tokenizer, dataset, formatter, device, batch_size=8):
    model.eval()
    token_0 = tokenizer(" 0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer(" 1", add_special_tokens=False)["input_ids"][0]

    all_preds = []
    all_labels = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch_texts = [ex["text"] for ex in dataset[i:i+batch_size]]
        batch_labels = [ex["label"] for ex in dataset[i:i+batch_size]]
        prompts = [formatter.format(t) for t in batch_texts]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # next token

        probs_1 = torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1]
        preds = (probs_1 > 0.5).int().cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels)

        # Free memory
        del inputs, outputs, logits, probs_1
        torch.cuda.empty_cache()

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "mcc": float(mcc),
        "confusion_matrix": cm.tolist()
    }, np.array(all_preds)


def plot_confusion(cm, title, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(title, fontsize=14, pad=20)
    plt.colorbar()
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center",
                     color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=16)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_lime_grid(model, tokenizer, formatter, dataset, device, title, path):
    explainer = LimeTextExplainer(class_names=["Negative", "Positive"], mode="classification")

    token_0 = tokenizer(" 0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer(" 1", add_special_tokens=False)["input_ids"][0]

    def predict_fn(texts):
        prompts = [formatter.format(t) for t in texts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        prob_1 = torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1].cpu().numpy()
        return np.stack([1 - prob_1, prob_1], axis=1)

    # Sample 5 correct neg + 5 correct pos
    print("Selecting 10 balanced examples for LIME...")
    _, preds = evaluate_model_safe(model, tokenizer, dataset, formatter, device, batch_size=16)
    labels = [ex["label"] for ex in dataset]

    correct_neg = [i for i, (l, p) in enumerate(zip(labels, preds)) if l == 0 and p == 0]
    correct_pos = [i for i, (l, p) in enumerate(zip(labels, preds)) if l == 1 and p == 1]

    selected = random.sample(correct_neg, min(5, len(correct_neg))) + random.sample(correct_pos, min(5, len(correct_pos)))
    random.shuffle(selected)

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    fig.suptitle(title, fontsize=24, weight="bold")

    for idx, ax in zip(selected, axes.flat):
        ex = dataset[idx]
        exp = explainer.explain_instance(ex["text"], predict_fn, num_features=10, num_samples=1000)
        exp.as_pyplot_figure(label=ex["label"])
        lime_img = plt.gcf()
        ax.imshow(lime_img.canvas.buffer_rgba())
        true_label = "Positive" if ex["label"] == 1 else "Negative"
        ax.set_title(f"True: {true_label}\n\"{ex['text'][:70]}{'...' if len(ex['text'])>70 else ''}\"",
                     fontsize=10, pad=10)
        ax.axis("off")
        plt.close(lime_img)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", default="outputs/final_v100")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--eval-size", type=int, default=2000)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    formatter = PromptFormatter()
    raw = load_dataset("mteb/tweet_sentiment_extraction")
    raw = raw.filter(lambda x: x["label"] in [0, 1], batched=False)
    eval_data = raw["test"].shuffle(seed=42).select(range(args.eval_size))

    print("ZERO-SHOT EVALUATION (safe batching)...")
    zs_metrics, _ = evaluate_model_safe(model, tokenizer, eval_data, formatter, device, batch_size=8)
    print(f"Zero-shot F1: {zs_metrics['f1']:.4f}")

    plot_confusion_matrix(np.array(zs_metrics["confusion_matrix"]),
                          "Zero-Shot Confusion Matrix", f"{args.output_dir}/confusion_zero_shot.png")
    generate_lime_grid(model, tokenizer, formatter, eval_data, device,
                       "LIME - Zero-Shot (5 Neg + 5 Pos Correct)", f"{args.output_dir}/lime_zero_shot.png")

    print("\nSTARTING FINE-TUNING...")
    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # Prepare training data (masked)
    def prepare(examples):
        prompts = [formatter.format(t) + f" {l}" for t, l in zip(examples["text"], examples["label"])]
        tokenized = tokenizer(prompts, truncation=True, max_length=256, padding=False)
        labels = []
        for i, inp in enumerate(tokenized["input_ids"]):
            prompt_len = len(tokenizer(formatter.format(examples["text"][i]))["input_ids"])
            lbl = [-100] * len(inp)
            if prompt_len < len(inp):
                lbl[prompt_len] = inp[prompt_len]
            labels.append(lbl)
        tokenized["labels"] = labels
        return tokenized

    train_data = raw["train"].shuffle(seed=42).select(range(args.train_size)).map(prepare, batched=True, remove_columns=raw["train"].column_names)

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
            report_to="none",
        ),
        train_dataset=train_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    print("FINE-TUNED EVALUATION...")
    ft_metrics, _ = evaluate_model_safe(model, tokenizer, eval_data, formatter, device, batch_size=8)
    print(f"Fine-tuned F1: {ft_metrics['f1']:.4f} (+{ft_metrics['f1'] - zs_metrics['f1']:+.4f})")

    plot_confusion_matrix(np.array(ft_metrics["confusion_matrix"]),
                          "Fine-Tuned Confusion Matrix", f"{args.output_dir}/confusion_fine_tuned.png")
    generate_lime_grid(model, tokenizer, formatter, eval_data, device,
                       "LIME - Fine-Tuned (5 Neg + 5 Pos Correct)", f"{args.output_dir}/lime_fine_tuned.png")

    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump({"zero_shot": zs_metrics, "fine_tuned": ft_metrics}, f, indent=2)

    print(f"\nSUCCESS! All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
