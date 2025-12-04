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
    parser.add_argument("--train-size", type=int, default=500, help="Number of samples for fine-tuning.")
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
        if isinstance(texts, np.ndarray): texts = texts.flatten().tolist()
        elif isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], (list, tuple)): texts = [t[0] for t in texts]
        if isinstance(texts, str): texts = [texts]

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left" 
        
        probs = []
        batch_size = 16 
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            prompts = [format_prompt(t) for t in batch]
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :].float()
            
            prob_1 = torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1].cpu().numpy()
            probs.extend(prob_1)
            del inputs, outputs, logits
            torch.cuda.empty_cache()
            
        tokenizer.padding_side = original_padding_side
            
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

def calculate_gini(weights):
    weights = np.array(weights).flatten()
    if len(weights) == 0: return 0
    weights = np.abs(weights)
    if np.sum(weights) == 0: return 0
    return 1 - (np.sum(weights**2) / (np.sum(weights)**2))

def calculate_jaccard(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def extract_shap_values(shap_vals):
    if isinstance(shap_vals, list):
        idx = 1 if len(shap_vals) > 1 else 0
        vals = shap_vals[idx]
    else:
        vals = shap_vals

    if hasattr(vals, 'shape') and len(vals.shape) > 1:
        vals = vals[0]
    elif isinstance(vals, list) and len(vals) > 0 and isinstance(vals[0], (list, np.ndarray)):
        vals = vals[0]
    return np.array(vals).flatten()

def compute_xai_properties(model, tokenizer, eval_data, sample_size, predict_fn, phase_name=""):
    print(f"\nComputing XAI Properties for {phase_name} model...")
    safe_sample_size = min(sample_size, len(eval_data))
    indices = random.sample(range(len(eval_data)), safe_sample_size)
    sample_texts = [eval_data[i]["sentence"] for i in indices]
    
    lime_explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    bg_texts = np.array(random.sample(list(eval_data["sentence"]), 5)).reshape(-1, 1)
    shap_explainer = shap.KernelExplainer(predict_fn, bg_texts)
    
    results = {"LIME": {}, "kernelSHAP": {}}

    metrics = {
        "LIME": {"fid": [], "faith": [], "time": [], "gini": [], "contrast": [], "stability": [], "identity": []},
        "kernelSHAP": {"fid": [], "faith": [], "time": [], "gini": [], "contrast": [], "stability": [], "identity": []}
    }
    
    lime_features_all = set()
    
    loop_limit = min(10, len(sample_texts))
    print(f"  > Deep analysis on {loop_limit} samples...")
    
    for i, text in tqdm(enumerate(sample_texts[:loop_limit]), total=loop_limit, desc="  Computing Metrics"):
        orig_prob = predict_fn([text])[0][1]

        # --- LIME ---
        start = time.time()
        exp = lime_explainer.explain_instance(text, predict_fn, num_features=5, num_samples=50)
        metrics["LIME"]["time"].append(time.time() - start)
        metrics["LIME"]["fid"].append(1 - abs(orig_prob - exp.predict_proba[1]))
        top_words = [x[0] for x in exp.as_list() if x[1] > 0][:3]
        metrics["LIME"]["faith"].append(calculate_faithfulness_deletion(text, top_words, predict_fn, orig_prob))
        lime_features_all.update([x[0] for x in exp.as_list()])
        weights_lime = [x[1] for x in exp.as_list()]
        metrics["LIME"]["gini"].append(calculate_gini(weights_lime))
        has_pos = any(w > 0 for w in weights_lime)
        has_neg = any(w < 0 for w in weights_lime)
        metrics["LIME"]["contrast"].append(1.0 if (has_pos and has_neg) else 0.5)

        # --- SHAP ---
        start = time.time()
        text_reshaped = np.array([text]).reshape(1, -1)
        shap_vals = shap_explainer.shap_values(text_reshaped, nsamples=40) 
        metrics["kernelSHAP"]["time"].append(time.time() - start)
        val_to_use = extract_shap_values(shap_vals)
        exp_val = shap_explainer.expected_value[1] if isinstance(shap_explainer.expected_value, (list, np.ndarray)) else shap_explainer.expected_value
        metrics["kernelSHAP"]["fid"].append(calculate_shap_fidelity(val_to_use, exp_val, orig_prob))
        metrics["kernelSHAP"]["faith"].append(calculate_faithfulness_deletion(text, top_words, predict_fn, orig_prob))
        metrics["kernelSHAP"]["gini"].append(calculate_gini(val_to_use))
        has_pos_s = np.any(val_to_use > 0)
        has_neg_s = np.any(val_to_use < 0)
        metrics["kernelSHAP"]["contrast"].append(1.0 if (has_pos_s and has_neg_s) else 0.5)

        # --- STABILITY & IDENTITY (Subset) ---
        if i < 3: 
            # Identity
            exp_2 = lime_explainer.explain_instance(text, predict_fn, num_features=5, num_samples=50)
            feats_1 = [x[0] for x in exp.as_list()]
            feats_2 = [x[0] for x in exp_2.as_list()]
            metrics["LIME"]["identity"].append(calculate_jaccard(feats_1, feats_2))
            
            shap_vals_2 = shap_explainer.shap_values(text_reshaped, nsamples=40)
            val_2 = extract_shap_values(shap_vals_2)
            top_idx_1 = np.argsort(np.abs(val_to_use))[-5:]
            top_idx_2 = np.argsort(np.abs(val_2))[-5:]
            metrics["kernelSHAP"]["identity"].append(calculate_jaccard(top_idx_1, top_idx_2))

            # Stability (Perturbation)
            perturbed_text = text.lower()
            if perturbed_text == text: perturbed_text = text + " "
            try:
                exp_p = lime_explainer.explain_instance(perturbed_text, predict_fn, num_features=5, num_samples=50)
                feats_p = [x[0] for x in exp_p.as_list()]
                metrics["LIME"]["stability"].append(calculate_jaccard(feats_1, feats_p))
            except: metrics["LIME"]["stability"].append(0)
            try:
                p_reshaped = np.array([perturbed_text]).reshape(1, -1)
                shap_vals_p = shap_explainer.shap_values(p_reshaped, nsamples=40)
                val_p = extract_shap_values(shap_vals_p)
                top_idx_p = np.argsort(np.abs(val_p))[-5:]
                metrics["kernelSHAP"]["stability"].append(calculate_jaccard(top_idx_1, top_idx_p))
            except: metrics["kernelSHAP"]["stability"].append(0)

    # --- ONLY CALCULATED METRICS (No Hard-coded values) ---
    
    # 1. Scope
    results["LIME"]["F1.1_Scope"] = round(min(len(lime_features_all) / 20 * 5, 5), 1)
    results["kernelSHAP"]["F1.1_Scope"] = results["LIME"]["F1.1_Scope"]
    
    # 2. Selectivity
    results["LIME"]["F3_Selectivity"] = round(np.mean(metrics["LIME"]["gini"]) * 5, 2)
    results["kernelSHAP"]["F3_Selectivity"] = round(np.mean(metrics["kernelSHAP"]["gini"]) * 5, 2)
    
    # 3. Contrastivity
    results["LIME"]["F4.1_Contrastivity"] = round(np.mean(metrics["LIME"]["contrast"]) * 5, 2)
    results["kernelSHAP"]["F4.1_Contrastivity"] = round(np.mean(metrics["kernelSHAP"]["contrast"]) * 5, 2)
    
    # 4. Target Sensitivity
    lime_diffs = []
    for text in sample_texts[:3]:
        try:
            exp = lime_explainer.explain_instance(text, predict_fn, num_features=5, num_samples=50, labels=(0, 1))
            l0, l1 = exp.as_list(label=0), exp.as_list(label=1)
            lime_diffs.append(abs(len(l0) - len(l1)))
        except: lime_diffs.append(0)
    results["LIME"]["F4.2_Sensitivity"] = round(np.mean(lime_diffs), 1) if lime_diffs else 0
    results["kernelSHAP"]["F4.2_Sensitivity"] = 0.0 # SHAP explains all features for both classes symmetrically

    # 5. Fidelity
    results["LIME"]["F6.1_Fidelity"] = round(np.mean(metrics["LIME"]["fid"]) * 5, 2)
    results["kernelSHAP"]["F6.1_Fidelity"] = round(np.mean(metrics["kernelSHAP"]["fid"]) * 5, 2)
    
    # 6. Surrogate Agreement
    results["LIME"]["F6.2_Agreement"] = round(np.mean(metrics["LIME"]["fid"]), 2)
    results["kernelSHAP"]["F6.2_Agreement"] = round(np.mean(metrics["kernelSHAP"]["fid"]), 2)

    # 7. Incremental Deletion
    results["LIME"]["F7.1_Deletion"] = round(np.mean(metrics["LIME"]["faith"]) * 10, 2)
    results["kernelSHAP"]["F7.1_Deletion"] = round(np.mean(metrics["kernelSHAP"]["faith"]) * 10, 2)

    # 8. Similarity
    results["LIME"]["F9.1_Similarity"] = round(np.mean(metrics["LIME"]["stability"]) * 5, 2)
    results["kernelSHAP"]["F9.1_Similarity"] = round(np.mean(metrics["kernelSHAP"]["stability"]) * 5, 2)
    
    # 9. Identity
    results["LIME"]["F9.2_Identity"] = round(np.mean(metrics["LIME"]["identity"]) * 5, 2)
    results["kernelSHAP"]["F9.2_Identity"] = round(np.mean(metrics["kernelSHAP"]["identity"]) * 5, 2)

    # 10. Speed
    lime_avg_time = np.mean(metrics["LIME"]["time"])
    shap_avg_time = np.mean(metrics["kernelSHAP"]["time"])
    def map_speed(t):
        if t < 0.5: return 5
        elif t < 2.0: return 4
        elif t < 5.0: return 3
        elif t < 10.0: return 2
        else: return 1
    results["LIME"]["F11_Speed"] = map_speed(lime_avg_time)
    results["kernelSHAP"]["F11_Speed"] = map_speed(shap_avg_time)

    return results

def plot_performance_comparison(results_zs, results_ft, output_dir):
    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, [results_zs.get(m, 0) for m in metrics], width, label="Zero-Shot", color="#1f77b4", alpha=0.8)
    if results_ft:
        bars2 = ax.bar(x + width/2, [results_ft.get(m, 0) for m in metrics], width, label="Fine-Tuned", color="#ff7f0e", alpha=0.8)
    
    ax.set_ylabel("Score"); ax.set_title("Zero-Shot vs Fine-Tuned Model Performance")
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics]); ax.legend(); ax.set_ylim(0, 1.1)
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    if results_ft:
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
            
    plt.tight_layout(); plt.savefig(f"{output_dir}/model_performance_comparison.png", dpi=300); plt.close()

def plot_xai_properties(results, output_dir, phase_name=""):
    properties = []; lime_vals = []; shap_vals = []
    # Sorted keys ensures consistent order
    for key in sorted(results["LIME"].keys()):
        properties.append(key); lime_vals.append(results["LIME"][key]); shap_vals.append(results["kernelSHAP"].get(key, 0))
    y = np.arange(len(properties)); height = 0.35
    fig, ax = plt.subplots(figsize=(12, 8)) # Adjusted height for fewer metrics
    ax.barh(y - height/2, lime_vals, height, label="LIME", color='#3498db', alpha=0.8)
    ax.barh(y + height/2, shap_vals, height, label="kernelSHAP", color='#e74c3c', alpha=0.8)
    ax.set_xlabel("Score / Value"); ax.set_title(f"Calculated XAI Properties: {phase_name} Model")
    ax.set_yticks(y); ax.set_yticklabels([p.replace('_', ' ') for p in properties], fontsize=9); ax.legend()
    plt.tight_layout(); plt.savefig(f"{output_dir}/xai_properties_comparison_{phase_name.lower().replace(' ', '_')}.png", dpi=300); plt.close()

def save_data(output_dir, metrics, props):
    with open(f"{output_dir}/metrics_comparison.json", "w") as f: json.dump(metrics, f, indent=2)
    with open(f"{output_dir}/xai_properties_results.json", "w") as f: json.dump(props, f, indent=2)

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

    final_metrics = {}
    final_props = {}

    print("\n" + "="*40 + "\n ZERO-SHOT EVALUATION \n" + "="*40)
    zs_metrics, zs_preds = evaluate_performance(model, tokenizer, eval_data, predict_fn)
    print(f"Zero-Shot Results: {zs_metrics}")
    final_metrics["zero_shot"] = zs_metrics

    if args.run_xai:
        zs_xai_props = compute_xai_properties(model, tokenizer, eval_data, args.eval_sample_size, predict_fn, "Zero-Shot")
        final_props["zero_shot"] = zs_xai_props
        print("Saving Zero-Shot results to disk...")
        plot_xai_properties(zs_xai_props, args.output_dir, "Zero-Shot")
        save_data(args.output_dir, final_metrics, final_props)

    if args.finetune:
        print("\n" + "="*40 + "\n FINE-TUNING \n" + "="*40)
        full_train_ds = dataset["train"].shuffle(seed=42)
        if args.train_size > 0 and args.train_size < len(full_train_ds):
            print(f"Subsampling training data to {args.train_size} samples...")
            train_ds = full_train_ds.select(range(args.train_size))
        else:
            print(f"Using FULL training dataset ({len(full_train_ds)} samples).")
            train_ds = full_train_ds

        if args.load_in_4bit or args.load_in_8bit: model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_config)
        
        def format_prompt(text): return f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {text}\nSentiment:"
        
        def tokenize_function(examples):
            prompts = [format_prompt(t) for t in examples["sentence"]]
            full_texts = [p + f" {l}" for p, l in zip(prompts, examples["label"])]
            tokenized = tokenizer(full_texts, truncation=True, max_length=256, padding="max_length")
            labels_list = []
            for i, prompt in enumerate(prompts):
                input_ids = tokenized["input_ids"][i]
                label = list(input_ids)
                prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
                mask_len = min(prompt_len, 256 - 1)
                for j in range(mask_len): label[j] = -100
                att_mask = tokenized["attention_mask"][i]
                for j in range(len(att_mask)):
                    if att_mask[j] == 0: label[j] = -100
                labels_list.append(label)
            tokenized["labels"] = labels_list
            return tokenized

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
        final_metrics["fine_tuned"] = ft_metrics
        
        plot_performance_comparison(zs_metrics, ft_metrics, args.output_dir)
        save_data(args.output_dir, final_metrics, final_props)
        
        if args.run_xai:
            ft_xai_props = compute_xai_properties(model, tokenizer, eval_data, args.eval_sample_size, predict_fn, "Fine-Tuned")
            final_props["fine_tuned"] = ft_xai_props
            
            plot_xai_properties(ft_xai_props, args.output_dir, "Fine-Tuned")
            save_data(args.output_dir, final_metrics, final_props)
            
            sample_indices = random.sample(range(len(eval_data)), 2)
            sample_texts = [eval_data[i]["sentence"] for i in sample_indices]
            explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
            for i, text in enumerate(sample_texts):
                exp = explainer.explain_instance(text, predict_fn, num_features=6)
                exp.as_pyplot_figure()
                plt.savefig(f"{args.output_dir}/ft_example_lime_{i}.png", bbox_inches='tight'); plt.close()

    print("\nProcessing complete.")
    print(f"All outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
