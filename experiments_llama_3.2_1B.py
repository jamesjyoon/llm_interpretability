import argparse
import json
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from lime.lime_text import LimeTextExplainer
import shap
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", default="outputs/final_xai")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--finetune", action="store_true", default=True)
    parser.add_argument("--run-lime", action="store_true", default=True)
    parser.add_argument("--run-xai", action="store_true", default=True)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--huggingface-token", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.huggingface_token)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure model loading based on quantization
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
            quantization_config=quantization_config,
            device_map="auto",  # This handles device placement automatically
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            token=args.huggingface_token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    def format_prompt(text):
        return f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {text}\nSentiment:"

    token_0 = tokenizer("0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]

    def predict_proba(texts):
        if isinstance(texts, str): texts = [texts]
        probs = []
        # Determine the device dynamically from the model
        current_model_device = model.device # This will be the device where the model's first parameter is
        for i in range(0, len(texts), 4):
            batch = texts[i:i+4]
            prompts = [format_prompt(t) for t in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            
            # Move inputs to the model's actual device
            inputs = {k: v.to(current_model_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs).logits[:, -1, :].float()
            prob_1 = torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1].cpu().numpy()
            probs.extend(prob_1)
            del inputs, logits
            torch.cuda.empty_cache()
        return np.stack([1 - np.array(probs), np.array(probs)], axis=1)

    def evaluate(model, data):
        texts = data["sentence"]
        labels = data["label"]
        probs = [predict_proba([t])[0, 1] for t in tqdm(texts, desc="Evaluating")]
        preds = (np.array(probs) > 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        mcc = matthews_corrcoef(labels, preds)
        return {"accuracy": round(float(acc), 4), "f1": round(float(f1), 4), "mcc": round(float(mcc), 4)}, preds

    dataset = load_dataset("stanfordnlp/sst2")
    eval_data = dataset["validation"]

    # Zero-shot
    print("\nZERO-SHOT")
    zs_metrics, zs_preds = evaluate(model, eval_data)
    print(f"Zero-Shot → Acc: {zs_metrics['accuracy']:.4f} | F1: {zs_metrics['f1']:.4f} | MCC: {zs_metrics['mcc']:.4f}")

    results = {"zero_shot": zs_metrics}

    if args.finetune:
        print("\nFINE-TUNING...")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            task_type="CAUSAL_LM"))

        def tokenize(examples):
            prompts = [format_prompt(t) for t in examples["sentence"]]
            full = [p + f" {l}" for p, l in zip(prompts, examples["label"])]
            tok = tokenizer(full, truncation=True, max_length=256, padding=False)
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
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        trainer.train()

        print("\nFINE-TUNED")
        ft_metrics, ft_preds = evaluate(model, eval_data)
        print(f"Fine-Tuned → Acc: {ft_metrics['accuracy']:.4f} | F1: {ft_metrics['f1']:.4f} | MCC: {ft_metrics['mcc']:.4f}")
        results["fine_tuned"] = ft_metrics

    # XAI EVALUATION + PLOTS (NO OOM!)
    if args.run_xai:
        print("\nXAI EVALUATION + HEATMAPS...")
        preds_for_xai = ft_preds if args.finetune else zs_preds
        correct = [i for i, (l, p) in enumerate(zip(eval_data["label"], preds_for_xai)) if l == p]
        sample_idx = random.sample(correct, min(10, len(correct)))
        sample_texts = [eval_data[i]["sentence"] for i in sample_idx]

        # LIME
        explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
        for i, text in enumerate(sample_texts):
            exp = explainer.explain_instance(text, predict_proba, num_features=10, num_samples=500)
            exp.as_pyplot_figure()
            plt.savefig(f"{args.output_dir}/lime_{i}.png", dpi=150, bbox_inches='tight')
            plt.close()

        # KernelSHAP (safe)
        background = random.sample(list(eval_data["sentence"]), 10)
        kshap = shap.KernelExplainer(lambda x: predict_proba(x), background)
        for i, text in enumerate(sample_texts):
            shap_vals = kshap.shap_values([text], nsamples=100)
            shap.force_plot(kshap.expected_value[1], shap_vals[1][0], text, show=False, matplotlib=True)
            plt.savefig(f"{args.output_dir}/shap_{i}.png", dpi=150, bbox_inches='tight')
            plt.close()

        # Integrated Gradients (safe)
        lig = LayerIntegratedGradients(model, model.model.embed_tokens)
        def ig_forward(input_ids):
            # Ensure model is in eval mode if not already
            model.eval() 
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            return torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1]

        for i, text in enumerate(sample_texts):
            prompt = format_prompt(text)
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Move inputs to the model's actual device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            attr, _ = lig.attribute(inputs["input_ids"], target=0, return_convergence_delta=False, n_steps=20)
            attr = attr.sum(dim=-1).squeeze(0).cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            fig, _ = viz.visualize_text([viz.VisualizationDataRecord(attr, 0, 0, 0, 0, np.sum(attr), tokens, 0)])
            fig.savefig(f"{args.output_dir}/ig_{i}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

    # Bar chart
    if "fine_tuned" in results:
        metrics = ["accuracy", "f1", "mcc"]
        x = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, [results["zero_shot"][m] for m in metrics], width, label='Zero-Shot', color='#1f77b4')
        ax.bar(x + width/2, [results["fine_tuned"][m] for m in metrics], width, label='Fine-Tuned', color='#ff7f0e')
        ax.set_ylabel('Score'); ax.set_title('Zero-Shot vs Fine-Tuned')
        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/comparison_bar_chart.png", dpi=200)
        plt.close()

    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nALL DONE! Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
