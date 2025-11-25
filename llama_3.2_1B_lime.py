# llama_3.2_1B_sst2_FINAL_WORKING.py

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", type=str, default="outputs/final_run")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--finetune", action="store_true", default=True)
    parser.add_argument("--run-lime", action="store_true", default=True)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--huggingface-token", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.huggingface_token, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=args.huggingface_token,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ) if args.load_in_4bit else None,
    )

    class PromptFormatter:
        def format(self, text: str) -> str:
            return f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {text}\nSentiment:"

    formatter = PromptFormatter()
    dataset = load_dataset("stanfordnlp/sst2")
    eval_data = dataset["validation"]

    # Predict function
    def make_predict_fn(model, tokenizer, device):
        token_0 = tokenizer("0", add_special_tokens=False)["input_ids"][0]
        token_1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]
        def predict_fn(texts):
            if isinstance(texts, str): texts = [texts]
            probs = []
            for i in range(0, len(texts), 4):
                batch = texts[i:i+4]
                prompts = [formatter.format(t) for t in batch]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits[:, -1, :].float()
                prob_1 = torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1].cpu().numpy()
                probs.extend(prob_1)
                del inputs, logits; torch.cuda.empty_cache()
            return np.stack([1 - np.array(probs), np.array(probs)], axis=1)
        return predict_fn

    def evaluate(model, tokenizer, data, device):
        predict_fn = make_predict_fn(model, tokenizer, device)
        texts = data["sentence"]
        labels = data["label"]
        probs = []
        for i in tqdm(range(0, len(texts), 8), desc="Evaluating"):
            batch = texts[i:i+8]
            batch_probs = predict_fn(batch)[:, 1]
            probs.extend(batch_probs)
        preds = (np.array(probs) > 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        mcc = matthews_corrcoef(labels, preds)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        return {"accuracy": round(float(acc), 4), "f1": round(float(f1), 4), "mcc": round(float(mcc), 4), "confusion_matrix": cm.tolist()}, preds

    # Zero-shot
    print("\nZERO-SHOT")
    zs_metrics, _ = evaluate(model, tokenizer, eval_data, device)
    print(f"Zero-Shot → Acc: {zs_metrics['accuracy']:.4f}  F1: {zs_metrics['f1']:.4f}  MCC: {zs_metrics['mcc']:.4f}")

    plt.figure(figsize=(6,5))
    plt.imshow(np.array(zs_metrics["confusion_matrix"]), cmap="Blues")
    plt.title("Zero-Shot Confusion Matrix", fontsize=16)
    plt.xticks([0,1],["Neg","Pos"]); plt.yticks([0,1],["Neg","Pos"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(zs_metrics["confusion_matrix"][i][j]), ha="center", va="center", color="black", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/confusion_zero_shot.png", dpi=200)
    plt.close()

    results = {"zero_shot": zs_metrics}

    if args.run_lime:
        explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
        predict_fn = make_predict_fn(model, tokenizer, device)
        _, preds = evaluate(model, tokenizer, eval_data, device)
        correct = [i for i, (l, p) in enumerate(zip(eval_data["label"], preds)) if l == p]
        selected = random.sample(correct, min(10, len(correct)))

        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        for i, idx in enumerate(selected):
            text = eval_data[idx]["sentence"]
            exp = explainer.explain_instance(text, predict_fn, num_features=10, num_samples=500)
            temp_path = f"/tmp/lime_zs_{i}.png"
            exp.as_pyplot_figure()
            plt.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            img = plt.imread(temp_path)
            axes[i//5, i%5].imshow(img)
            axes[i//5, i%5].set_title(f"{text[:70]}...", fontsize=9)
            axes[i//5, i%5].axis("off")
            os.remove(temp_path)
        plt.suptitle("Zero-Shot - LIME", fontsize=28)
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/lime_zero_shot.png", dpi=200)
        plt.close()

    # Fine-tuning
    if args.finetune:
        print("\nFINE-TUNING...")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            task_type="CAUSAL_LM"
        ))

        def tokenize(examples):
            prompts = [formatter.format(t) for t in examples["sentence"]]
            full = [p + f" {l}" for p, l in zip(prompts, examples["label"])]
            tokenized = tokenizer(full, truncation=True, max_length=256, padding=False)
            # Let collator pad → no manual labels needed
            return tokenized

        train_ds = dataset["train"].shuffle(seed=42).select(range(args.train_size))
        train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
            train_dataset=train_ds,
            data_collator=data_collator,
        )
        trainer.train()

        print("\nFINE-TUNED")
        ft_metrics, _ = evaluate(model, tokenizer, eval_data, device)
        print(f"Fine-Tuned → Acc: {ft_metrics['accuracy']:.4f}  F1: {ft_metrics['f1']:.4f}  MCC: {ft_metrics['mcc']:.4f}")
        results["fine_tuned"] = ft_metrics

        plt.figure(figsize=(6,5))
        plt.imshow(np.array(ft_metrics["confusion_matrix"]), cmap="Blues")
        plt.title("Fine-Tuned Confusion Matrix", fontsize=16)
        plt.xticks([0,1],["Neg","Pos"]); plt.yticks([0,1],["Neg","Pos"])
        plt.xlabel("Predicted"); plt.ylabel("True")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(ft_metrics["confusion_matrix"][i][j]), ha="center", va="center", color="black", fontsize=18)
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/confusion_fine_tuned.png", dpi=200)
        plt.close()

        if args.run_lime:
            fig, axes = plt.subplots(2, 5, figsize=(25, 10))
            _, preds = evaluate(model, tokenizer, eval_data, device)
            correct = [i for i, (l, p) in enumerate(zip(eval_data["label"], preds)) if l == p]
            selected = random.sample(correct, min(10, len(correct)))
            for i, idx in enumerate(selected):
                text = eval_data[idx]["sentence"]
                exp = explainer.explain_instance(text, predict_fn, num_features=10, num_samples=500)
                temp_path = f"/tmp/lime_ft_{i}.png"
                exp.as_pyplot_figure()
                plt.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                img = plt.imread(temp_path)
                axes[i//5, i%5].imshow(img)
                axes[i//5, i%5].set_title(f"{text[:70]}...", fontsize=9)
                axes[i//5, i%5].axis("off")
                os.remove(temp_path)
            plt.suptitle("Fine-Tuned - LIME", fontsize=28)
            plt.tight_layout()
            plt.savefig(f"{args.output_dir}/lime_fine_tuned.png", dpi=200)
            plt.close()

    # Bar chart
    if "fine_tuned" in results:
        metrics = ['accuracy', 'f1', 'mcc']
        x = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, [results["zero_shot"][m] for m in metrics], width, label='Zero-Shot', color='#1f77b4')
        ax.bar(x + width/2, [results["fine_tuned"][m] for m in metrics], width, label='Fine-Tuned', color='#ff7f0e')
        ax.set_ylabel('Score'); ax.set_title('Zero-Shot vs Fine-Tuned')
        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.legend()
        for i, (v1, v2) in enumerate(zip([results["zero_shot"][m] for m in metrics], [results["fine_tuned"][m] for m in metrics])):
            ax.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center')
            ax.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center')
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/comparison_bar_chart.png", dpi=200)
        plt.close()

    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSUCCESS! All in {args.output_dir}/")


if __name__ == "__main__":
    main()
