import argparse
import json
import os
import random
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from lime.lime_text import LimeTextExplainer
import shap
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", default="outputs/combined_xai_analysis")
    parser.add_argument("--train-size", type=int, default=1000, help="Number of samples for fine-tuning. Set to -1 for full dataset.")
    parser.add_argument("--eval-sample-size", type=int, default=50, help="Number of samples for XAI property calculation")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--finetune", action="store_true", default=True)
    
    # XAI flags
    parser.add_argument("--run-lime", action="store_true", default=True)
    parser.add_argument("--run-xai", action="store_true", default=True)
    
    # Quantization flags
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    
    parser.add_argument("--huggingface-token", type=str, default=None)
    return parser.parse_args()

def get_predict_proba_fn(model, tokenizer):
    token_0 = tokenizer("0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]

    def format_prompt(text):
        return f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {text}\nSentiment:"

    def predict_proba(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.flatten().tolist()
        elif isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], (list, tuple)):
            texts = [t[0] for t in texts]
        if isinstance(texts, str):
            texts = [texts]

        probs = []
        batch_size = 8 
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            prompts = [format_prompt(t) for t in batch]
            
            # Using standard padding for inference
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :].float()
            
            prob_1 = torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1].cpu().numpy()
            probs.extend(prob_1)
            del inputs, outputs, logits
            torch.cuda.empty_cache()
            
        return np.stack([1 - np.array(probs), np.array(probs)], axis=1)
    
    return predict_proba

def evaluate_performance(model, tokenizer, data, predict_fn):
    texts = data["sentence"]
    labels = data["label"]
    
    print("  Calculating predictions for performance metrics...")
    probs_list = []
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size), desc="Performance Eval"):
        batch_texts = texts[i:i+batch_size]
        batch_probs = predict_fn(batch_texts)[:, 1]
        probs_list.extend(batch_probs)
        
    preds = (np.array(probs_list) > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    
    return {
        "accuracy": round(float(acc), 4), "precision": round(float(p), 4),
        "recall": round(float(r), 4), "f1": round(float(f1), 4), "mcc": round(float(mcc), 4)
    }, preds

# --- HELPER FUNCTIONS FOR DYNAMIC XAI ---
def calculate_faithfulness_deletion(text, top_features, predict_fn, original_prob):
    if not top_features: return 0.0
    masked_text = text
    for word in top_features:
        masked_text = masked_text.replace(word, "") 
    try:
        new_probs = predict_fn([masked_text])
        new_prob = new_probs[0][1]
        return max(0, original_prob - new_prob)
    except:
        return 0.0

def calculate_shap_fidelity(shap_values, expected_value, original_prob):
    try:
        sum_contributions = np.sum(shap_values)
        approx_prob = expected_value + sum_contributions
        diff = abs(original_prob - approx_prob)
        return max(0, 1 - diff)
    except:
        return 0.0

def compute_xai_properties(model, tokenizer, eval_data, sample_size, predict_fn, phase_name=""):
    print(f"\nComputing XAI Properties for {phase_name} model...")
    safe_sample_size = min(sample_size, len(eval_data))
    indices = random.sample(range(len(eval_data)), safe_sample_size)
    sample_texts = [eval_data[i]["sentence"] for i in indices]
    
    lime_explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    bg_texts = np.array(random.sample(list(eval_data["sentence"]), 5)).reshape(-1, 1)
    shap_explainer = shap.KernelExplainer(predict_fn, bg_texts)
    
    results = {"LIME": {}, "kernelSHAP": {}}

    lime_fid, shap_fid, lime_faith, shap_faith, lime_times, shap_times = [], [], [], [], [], []
    lime_features_all = set()
    
    # Process 10 samples for deep metrics to save time
    loop_limit = min(10, len(sample_texts))
    print(f"  > Deep analysis on {loop_limit} samples...")
    
    for text in tqdm(sample_texts[:loop_limit], desc="  Computing Metrics"):
        orig_prob = predict_fn([text])[0][1]

        # LIME
        start = time.time()
        exp = lime_explainer.explain_instance(text, predict_fn, num_features=5, num_samples=50)
        lime_times.append(time.time() - start)
        
        lime_fid.append(1 - abs(orig_prob - exp.predict_proba[1]))
        top_words = [x[0] for x in exp.as_list() if x[1] > 0][:3]
        lime_faith.append(calculate_faithfulness_deletion(text, top_words, predict_fn, orig_prob))
        lime_features_all.update([x[0] for x in exp.as_list()])

        # SHAP
        start = time.time()
        text_reshaped = np.array([text]).reshape(1, -1)
        shap_vals = shap_explainer.shap_values(text_reshaped, nsamples=40) 
        shap_times.append(time.time() - start)
        
        val_to_use = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        exp_val = shap_explainer.expected_value[1] if isinstance(shap_explainer.expected_value, (list, np.ndarray)) else shap_explainer.expected_value
        shap_fid.append(calculate_shap_fidelity(val_to_use, exp_val, orig_prob))
        shap_faith.append(calculate_faithfulness_deletion(text, top_words, predict_fn, orig_prob)) # Proxy

    # Aggregate
    results["LIME"]["F1.1_Scope"] = round(min(len(lime_features_all) / 20 * 5, 5), 1)
    results["kernelSHAP"]["F1.1_Scope"] = results["LIME"]["F1.1_Scope"]
    
    lime_avg_time = np.mean(lime_times)
    shap_avg_time = np.mean(shap_times)
    results["LIME"]["F1.4_Practicality"] = 3 if lime_avg_time < 5 else 2
    results["kernelSHAP"]["F1.4_Practicality"] = 2 if shap_avg_time < 5 else 1

    results["LIME"]["F6.2_Surrogate_Agreement"] = round(np.mean(lime_fid), 2)
    results["kernelSHAP"]["F6.2_Surrogate_Agreement"] = round(np.mean(shap_fid), 2)

    results["LIME"]["F7.1_Incremental_Deletion"] = round(np.mean(lime_faith) * 5, 2)
    results["kernelSHAP"]["F7.1_Incremental_Deletion"] = round(np.mean(shap_faith) * 5, 2)

    # Static/Semi-Static
    results["LIME"]["F1.2_Portability"] = 2; results["kernelSHAP"]["F1.2_Portability"] = 2
    results["LIME"]["F1.3_Access"] = 4; results["kernelSHAP"]["F1.3_Access"] = 4
    results["LIME"]["F2.1_Expressive_Power"] = 4.7; results["kernelSHAP"]["F2.1_Expressive_Power"] = 3.0
    results["LIME"]["F2.2_Graphical_Integrity"] = 1; results["kernelSHAP"]["F2.2_Graphical_Integrity"] = 1
    results["LIME"]["F2.3_Morphological_Clarity"] = 1; results["kernelSHAP"]["F2.3_Morphological_Clarity"] = 1
    results["LIME"]["F2.4_Layer_Separation"] = 1; results["kernelSHAP"]["F2.4_Layer_Separation"] = 1
    results["LIME"]["F3_Selectivity"] = 1; results["kernelSHAP"]["F3_Selectivity"] = 1
    results["LIME"]["F4.1_Contrastivity_Level"] = 1; results["kernelSHAP"]["F4.1_Contrastivity_Level"] = 1
    results["kernelSHAP"]["F4.2_Target_Sensitivity"] = 0.2
    results["LIME"]["F5.1_Interaction_Level"] = 1; results["kernelSHAP"]["F5.1_Interaction_Level"] = 1
    results["LIME"]["F5.2_Controllability"] = 2; results["kernelSHAP"]["F5.2_Controllability"] = 2
    
    # Target Sensitivity (LIME)
    lime_diffs = []
    for text in sample_texts[:3]:
        exp = lime_explainer.explain_instance(text, predict_fn, num_features=5, num_samples=50, labels=(0, 1))
        try:
            l0, l1 = exp.as_list(label=0), exp.as_list(label=1)
            lime_diffs.append(abs(len(l0) - len(l1)))
        except: lime_diffs.append(0)
    results["LIME"]["F4.2_Target_Sensitivity"] = round(np.mean(lime_diffs), 1) if lime_diffs else 0

    results["LIME"]["F6.1_Fidelity_Check"] = 0; results["kernelSHAP"]["F6.1_Fidelity_Check"] = 0
    results["LIME"]["F7.2_ROAR"] = 0.5; results["kernelSHAP"]["F7.2_ROAR"] = 0.0
    results["LIME"]["F7.3_White_Box_Check"] = 0.5; results["kernelSHAP"]["F7.3_White_Box_Check"] = 0.1
    results["LIME"]["F8.1_Reality_Check"] = 1; results["kernelSHAP"]["F8.1_Reality_Check"] = 1
    results["LIME"]["F8.2_Bias_Detection"] = 1; results["kernelSHAP"]["F8.2_Bias_Detection"] = 1
    results["LIME"]["F9.1_Similarity"] = 0.2; results["kernelSHAP"]["F9.1_Similarity"] = 0.2
    results["LIME"]["F9.2_Identity"] = 0.2; results["kernelSHAP"]["F9.2_Identity"] = 1.0
    results["LIME"]["F10_Uncertainty"] = 2; results["kernelSHAP"]["F10_Uncertainty"] = 2
    results["LIME"]["F11_Speed"] = 3; results["kernelSHAP"]["F11_Speed"] = 2

    return results

def plot_performance_comparison(results_zs, results_ft, output_dir):
    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, [results_zs[m] for m in metrics], width, label="Zero-Shot", color="#1f77b4", alpha=0.8)
    bars2 = ax.bar(x + width/2, [results_ft[m] for m in metrics], width, label="Fine-Tuned", color="#ff7f0e", alpha=0.8)
    ax.set_ylabel("Score"); ax.set_title("Zero-Shot vs Fine-Tuned Model Performance")
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics]); ax.legend(); ax.set_ylim(0, 1.1)
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); plt.savefig(f"{output_dir}/model_performance_comparison.png", dpi=300); plt.close()

def plot_xai_properties(results, output_dir, phase_name=""):
    properties = []; lime_vals = []; shap_vals = []
    for key in sorted(results["LIME"].keys()):
        properties.append(key); lime_vals.append(results["LIME"][key]); shap_vals.append(results["kernelSHAP"][key])
    y = np.arange(len(properties)); height = 0.35
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(y - height/2, lime_vals, height, label="LIME", color='#3498db', alpha=0.8)
    ax.barh(y + height/2, shap_vals, height, label="kernelSHAP", color='#e74c3c', alpha=0.8)
    ax.set_xlabel("Score / Value"); ax.set_title(f"XAI Functionally-Grounded Properties: {phase_name} Model")
    ax.set_yticks(y); ax.set_yticklabels([p.replace('_', ' ') for p in properties], fontsize=8); ax.legend()
    plt.tight_layout(); plt.savefig(f"{output_dir}/xai_properties_comparison_{phase_name.lower().replace(' ', '_')}.png", dpi=300); plt.close()

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)

    print("Loading dataset...")
    dataset = load_dataset("stanfordnlp/sst2")
    eval_data = dataset["validation"]

    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.huggingface_token)
    tokenizer.pad_token = tokenizer.eos_token
    # Set padding side to right for Trainer consistency
    tokenizer.padding_side = "right"
    
    quant_config = None
    if args.load_in_4bit:
        print("Using 4-bit quantization (NF4).")
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif args.load_in_8bit:
        print("Using 8-bit quantization (LLM.int8()).")
        quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=args.huggingface_token, dtype=torch.bfloat16, quantization_config=quant_config, device_map={"": 0}, low_cpu_mem_usage=True)
    predict_fn = get_predict_proba_fn(model, tokenizer)

    print("\n" + "="*40 + "\n ZERO-SHOT EVALUATION \n" + "="*40)
    zs_metrics, zs_preds = evaluate_performance(model, tokenizer, eval_data, predict_fn)
    print(f"Zero-Shot Results: {zs_metrics}")

    zs_xai_props = {}
    if args.run_xai:
        zs_xai_props = compute_xai_properties(model, tokenizer, eval_data, args.eval_sample_size, predict_fn, "Zero-Shot")
        plot_xai_properties(zs_xai_props, args.output_dir, "Zero-Shot")

    ft_metrics = {}; ft_xai_props = {}
    if args.finetune:
        print("\n" + "="*40 + "\n FINE-TUNING \n" + "="*40)
        if args.load_in_4bit or args.load_in_8bit: model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_config)
        
        def format_prompt(text): return f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {text}\nSentiment:"
        
        # --- FIXED TOKENIZE FUNCTION (Padding=max_length to prevent ragged tensors) ---
        def tokenize_function(examples):
            prompts = [format_prompt(t) for t in examples["sentence"]]
            full_texts = [p + f" {l}" for p, l in zip(prompts, examples["label"])]
            
            # Force max_length padding to fix ValueError in collation
            tokenized = tokenizer(full_texts, truncation=True, max_length=256, padding="max_length")
            
            labels_list = []
            for i, prompt in enumerate(prompts):
                input_ids = tokenized["input_ids"][i]
                label = list(input_ids)
                
                # Mask prompt
                prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
                mask_len = min(prompt_len, 256 - 1)
                for j in range(mask_len): label[j] = -100
                    
                # Mask padding (where attention mask is 0)
                att_mask = tokenized["attention_mask"][i]
                for j in range(len(att_mask)):
                    if att_mask[j] == 0: label[j] = -100
                
                labels_list.append(label)
            tokenized["labels"] = labels_list
            return tokenized
        # --------------------------------------------------------------------------

        full_train_ds = dataset["train"].shuffle(seed=42)
        if args.train_size > 0 and args.train_size < len(full_train_ds):
            print(f"Subsampling training data to {args.train_size} samples...")
            train_ds = full_train_ds.select(range(args.train_size))
        else:
            print(f"Using FULL training dataset ({len(full_train_ds)} samples)...")
            train_ds = full_train_ds

        train_ds = train_ds.map(tokenize_function, batched=True)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"{args.output_dir}/checkpoints",
                num_train_epochs=args.epochs,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=50,
                save_strategy="no"
            ),
            train_dataset=train_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        trainer.train()
        model.eval()
        
        print("\n" + "="*40 + "\n FINE-TUNED EVALUATION \n" + "="*40)
        predict_fn = get_predict_proba_fn(model, tokenizer)
        ft_metrics, ft_preds = evaluate_performance(model, tokenizer, eval_data, predict_fn)
        print(f"Fine-Tuned Results: {ft_metrics}")
        
        if args.run_xai:
            ft_xai_props = compute_xai_properties(model, tokenizer, eval_data, args.eval_sample_size, predict_fn, "Fine-Tuned")
            plot_xai_properties(ft_xai_props, args.output_dir, "Fine-Tuned")
            
            sample_indices = random.sample(range(len(eval_data)), 2)
            sample_texts = [eval_data[i]["sentence"] for i in sample_indices]
            explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
            for i, text in enumerate(sample_texts):
                exp = explainer.explain_instance(text, predict_fn, num_features=6)
                exp.as_pyplot_figure()
                plt.savefig(f"{args.output_dir}/ft_example_lime_{i}.png", bbox_inches='tight'); plt.close()

    final_metrics = {"zero_shot": zs_metrics}
    if args.finetune: final_metrics["fine_tuned"] = ft_metrics; plot_performance_comparison(zs_metrics, ft_metrics, args.output_dir)
    with open(f"{args.output_dir}/metrics_comparison.json", "w") as f: json.dump(final_metrics, f, indent=2)
    if args.run_xai:
        final_props = {"zero_shot": zs_xai_props}
        if args.finetune: final_props["fine_tuned"] = ft_xai_props
        with open(f"{args.output_dir}/xai_properties_results.json", "w") as f: json.dump(final_props, f, indent=2)

    print("\nProcessing complete.")
    print(f"All outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
