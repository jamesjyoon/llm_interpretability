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


# Correct & safe predict function for LIME + evaluation
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
                logits = model(**inputs).logits[:, -1, :].float()          # .float() → fixes bfloat16 error
            prob_1 = torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1]
            probs.extend(prob_1.cpu().numpy())
            del inputs, logits
            torch.cuda.empty_cache()
        p = np.array(probs)
        return np.stack([1 - p, p], axis=1)
    return predict_fn


def evaluate_model_safe(model, tokenizer, dataset, formatter, device, batch_size=8):
    model.eval()
    token_0 = tokenizer("0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]

    all_preds, all_labels = [], []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        end = min(i + batch_size, len(dataset))
        batch = [dataset[j] for j in range(i, end)]
        texts = [ex["text"] for ex in batch]
        labels = [ex["label"] for ex in batch]

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
        "accuracy": float(acc), "precision": float(p), "recall": float(r),
        "f1": float(f1), "mcc": float(mcc), "confusion_matrix": cm.tolist()
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
    labels = [ex["label"] for ex in dataset]

    correct_neg = [i for i, (l, p) in enumerate(zip(labels, preds)) if l == 0 and p == 0]
    correct_pos = [i for i, (l, p) in enumerate(zip(labels, preds)) if l == 1 and p == 1]

    selected = (random.sample(correct_neg, min(5, len(correct_neg))) +
                random.sample(correct_pos, min(5, len(correct_pos))))
    random.shuffle(selected)

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    axes = axes.flatten()

    for ax, idx in zip(axes, selected):
        text = dataset[idx]["text"]
        true_label = "Pos" if dataset[idx]["label"] == 1 else "Neg"

        exp = explainer.explain_instance(text, predict_fn, num_features=10, num_samples=500)
        
        # Save individual LIME plot to temp file then load back
        temp_path = f"/tmp/lime_temp_{plot_idx}.png"
        exp.save_html(temp_path.replace(".png", ".html"))
        exp.as_pyplot_figure()
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        plt.close()

        img = plt.imread(temp_path)
        ax.imshow(img)
        ax.set_title(f"{true_label}: {text[:60]}...", fontsize=9)
        ax.axis("off")

    plt.suptitle(f"{title_prefix} - LIME Explanations", fontsize=26, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/lime_{title_prefix.lower().replace(' ', '_')}.png", dpi=200)
    plt.close('all')
    print(f"LIME grid saved: {title_prefix}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", type=str, default="outputs/final_run")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--eval-size", type=int, default=2000)
    parser.add_argument("--epochs", type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer & 4-bit model...")
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
    raw = load_dataset("mteb/tweet_sentiment_extraction").filter(lambda x: x["label"] in [0, 1])
    eval_data = raw["test"].shuffle(seed=42).select(range(args.eval_size))

    # Zero-shot
    print("Zero-shot evaluation...")
    zs_metrics, _ = evaluate_model_safe(model, tokenizer, eval_data, formatter, device)
    plot_confusion(np.array(zs_metrics["confusion_matrix"]), "Zero-Shot",
                   f"{args.output_dir}/confusion_zero_shot.png")
    generate_all_explanations(model, tokenizer, formatter, eval_data, device, "Zero-Shot", args.output_dir)

    # Fine-tuning
    print("Fine-tuning with LoRA...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    ))

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

    train_data = raw["train"].shuffle(seed=42).select(range(args.train_size)).map(prepare, batched=True)

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
        ),
        train_dataset=train_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    # Fine-tuned
    print("Fine-tuned evaluation...")
    ft_metrics, _ = evaluate_model_safe(model, tokenizer, eval_data, formatter, device)
    plot_confusion(np.array(ft_metrics["confusion_matrix"]), "Fine-Tuned",
                   f"{args.output_dir}/confusion_fine_tuned.png")
    generate_all_explanations(model, tokenizer, formatter, eval_data, device, "Fine-Tuned", args.output_dir)

    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump({"zero_shot": zs_metrics, "fine_tuned": ft_metrics}, f, indent=2)

    print("\nALL DONE! Check your output folder:")
    print(f"   {args.output_dir}/")


if __name__ == "__main__":
    main()
