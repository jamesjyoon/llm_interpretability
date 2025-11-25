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
    parser = argparse.ArgumentParser(description="Llama-3.2-1B Sentiment Analysis (SST-2) + LIME")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset-name", type=str, default="stanfordnlp/sst2")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--finetune", action="store_true", help="Run fine-tuning")
    parser.add_argument("--no-finetune", dest="finetune", action="store_false")
    parser.add_argument("--run-lime", action="store_true", default=True)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--output-dir", type=str, default="outputs/sst2_run")
    parser.add_argument("--huggingface-token", type=str, default=None)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--eval-size", type=int, default=872)   # SST-2 validation size
    parser.add_argument("--epochs", type=float, default=3.0)

    parser.set_defaults(finetune=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=args.huggingface_token,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=args.huggingface_token,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    class PromptFormatter:
        def format(self, text: str) -> str:
            return f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {text}\nSentiment:"

    formatter = PromptFormatter()

    dataset = load_dataset(args.dataset_name, args.dataset_config) if args.dataset_config else load_dataset(args.dataset_name)
    eval_data = dataset["validation"]

    # === Predict function ===
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
            batch_probs = predict_fn(texts[i:i+8])[:, 1]
            probs.extend(batch_probs)
        preds = (np.array(probs) > 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        mcc = matthews_corrcoef(labels, preds)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        return {"accuracy": round(acc, 4), "f1": round(f1, 4), "mcc": round(mcc, 4), "confusion_matrix": cm.tolist()}, preds

    # Zero-shot
    print("\nZERO-SHOT")
    zs_metrics, _ = evaluate(model, tokenizer, eval_data, device)
    print(f"Zero-Shot → Acc: {zs_metrics['accuracy']:.4f}  F1: {zs_metrics['f1']:.4f}  MCC: {zs_metrics['mcc']:.4f}")

    # Save zero-shot results
    plot_confusion(np.array(zs_metrics["confusion_matrix"]), "Zero-Shot", f"{args.output_dir}/confusion_zero_shot.png")

    if args.run_lime:
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
        predict_fn = make_predict_fn(model, tokenizer, device)
        _, preds = evaluate(model, tokenizer, eval_data, device)
        correct = [i for i, (l, p) in enumerate(zip(eval_data["label"], preds)) if l == p]
        selected = random.sample(correct, 10)

        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        for idx, ax_idx in enumerate(selected):
            text = eval_data[idx]["sentence"]
            exp = explainer.explain_instance(text, predict_fn, num_features=10, num_samples=500)
            temp_path = f"/tmp/lime_{idx}.png"
            exp.as_pyplot_figure()
            plt.savefig(temp_path, dpi=150, bbox_inches='tight'); plt.close()
            img = plt.imread(temp_path)
            axes[idx//5, idx%5].imshow(img)
            axes[idx//5, idx%5].set_title(f"{text[:70]}...", fontsize=9)
            axes[idx//5, idx%5].axis("off")
            os.remove(temp_path)
        plt.suptitle("Zero-Shot - LIME Explanations", fontsize=28)
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/lime_zero_shot.png", dpi=200)
        plt.close()

    results = {"zero_shot": zs_metrics}

    if args.finetune:
        print("\nFINE-TUNING WITH LoRA...")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            task_type="CAUSAL_LM"
        ))

        def tokenize(examples):
            prompts = [formatter.format(t) for t in examples["sentence"]]
            full = [p + f" {l}" for p, l in zip(prompts, examples["label"])]
            tok = tokenizer(full, truncation=True, max_length=256, padding=False)
            labels = []
            for i, s in enumerate(examples["sentence"]):
                plen = len(tokenizer(formatter.format(s))["input_ids"])
                lbl = tok["input_ids"][i][:]
                lbl[:plen] = [-100] * plen
                labels.append(lbl)
            tok["labels"] = labels
            return tok

        train_ds = dataset["train"].shuffle(seed=42).select(range(args.train_size))
        train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)

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
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()

        print("\nFINE-TUNED EVALUATION")
        ft_metrics, _ = evaluate(model, tokenizer, eval_data, device)
        print(f"Fine-Tuned → Acc: {ft_metrics['accuracy']:.4f}  F1: {ft_metrics['f1']:.4f}  MCC: {ft_metrics['mcc']:.4f}")
        results["fine_tuned"] = ft_metrics

        plot_confusion(np.array(ft_metrics["confusion_matrix"]), "Fine-Tuned", f"{args.output_dir}/confusion_fine_tuned.png")

        if args.run_lime:
            fig, axes = plt.subplots(2, 5, figsize=(25, 10))
            _, preds = evaluate(model, tokenizer, eval_data, device)
            correct = [i for i, (l, p) in enumerate(zip(eval_data["label"], preds)) if l == p]
            selected = random.sample(correct, 10)
            for idx, ax_idx in enumerate(selected):
                text = eval_data[idx]["sentence"]
                exp = explainer.explain_instance(text, predict_fn, num_features=10, num_samples=500)
                temp_path = f"/tmp/lime_ft_{idx}.png"
                exp.as_pyplot_figure(); plt.savefig(temp_path, dpi=150, bbox_inches='tight'); plt.close()
                img = plt.imread(temp_path)
                axes[idx//5, idx%5].imshow(img)
                axes[idx//5, idx%5].set_title(f"{text[:70]}...", fontsize=9)
                axes[idx//5, idx%5].axis("off")
                os.remove(temp_path)
            plt.suptitle("Fine-Tuned - LIME Explanations", fontsize=28)
            plt.tight_layout()
            plt.savefig(f"{args.output_dir}/lime_fine_tuned.png", dpi=200)
            plt.close()

    # Bar chart comparison
    if "fine_tuned" in results:
        metrics = ['accuracy', 'f1', 'mcc']
        zs = [results["zero_shot"][m] for m in metrics]
        ft = [results["fine_tuned"][m] for m in metrics]
        x = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, zs, width, label='Zero-Shot', color='#1f77b4')
        ax.bar(x + width/2, ft, width, label='Fine-Tuned', color='#ff7f0e')
        ax.set_ylabel('Score'); ax.set_title('Zero-Shot vs Fine-Tuned')
        ax.set_xticks(x); ax.set_xticklabels(['Accuracy', 'F1', 'MCC'])
        ax.legend()
        for i, (v1, v2) in enumerate(zip(zs, ft)):
            ax.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center')
            ax.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center')
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/comparison_bar_chart.png", dpi=200)
        plt.close()

    # Save JSON
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll done! Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
