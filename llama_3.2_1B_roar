import argparse
import random
import numpy as np
import torch
import copy
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", default="outputs/roar_experiment")
    # ROAR requires explaining the TRAINING data, which is slow. 
    # Keep this small (e.g., 500-1000) for demonstration.
    parser.add_argument("--train-size", type=int, default=500) 
    parser.add_argument("--mask-percentage", type=float, default=0.2, help="Remove top 20% of words")
    parser.add_argument("--huggingface-token", type=str, default=None)
    return parser.parse_args()

def get_predict_fn(model, tokenizer):
    token_0 = tokenizer("0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]
    
    def predict_proba(texts):
        if isinstance(texts, str): texts = [texts]
        prompts = [f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {t}\nSentiment:" for t in texts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits[:, [token_0, token_1]], dim=-1)[:, 1].cpu().numpy()
        return np.stack([1-probs, probs], axis=1)
    return predict_proba

def create_modified_dataset(original_data, model, tokenizer, strategy="roar_lime", percentage=0.2):
    """
    Creates a modified dataset.
    strategy: 'roar_lime', 'roar_shap', or 'random'
    """
    print(f"\nGenerating modified dataset: STRATEGY = {strategy.upper()}...")
    predict_fn = get_predict_fn(model, tokenizer)
    
    # Initialize Explainers
    lime_explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    
    # For SHAP, we need a background dataset
    bg_data = [x['sentence'] for x in original_data.select(range(min(10, len(original_data))))]
    bg_data = np.array(bg_data).reshape(-1, 1) # Reshape for text
    shap_explainer = shap.KernelExplainer(predict_fn, bg_data)

    new_sentences = []
    new_labels = []
    
    for i in tqdm(range(len(original_data)), desc=f"Processing {strategy}"):
        text = original_data[i]['sentence']
        label = original_data[i]['label']
        words = text.split()
        num_to_mask = max(1, int(len(words) * percentage))
        
        words_to_mask = []
        
        if strategy == "roar_lime":
            exp = lime_explainer.explain_instance(text, predict_fn, num_features=num_to_mask, num_samples=30)
            words_to_mask = [x[0] for x in exp.as_list()]
            
        elif strategy == "roar_shap":
            # SHAP is slow, using low nsamples for demo
            text_reshaped = np.array([text]).reshape(1, -1)
            shap_vals = shap_explainer.shap_values(text_reshaped, nsamples=20)
            
            # Extract top words from SHAP values
            # (Note: KernelSHAP on text usually returns values per token/word)
            # We map values back to words roughly for this experiment
            vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            # Get indices of top values
            top_indices = np.argsort(-np.abs(vals))[:num_to_mask]
            
            # Map indices back to words (Approximation)
            # In a real rigorous setting, you would mask tokens, but here we mask words
            words_to_mask = []
            for idx in top_indices:
                if idx < len(words):
                    words_to_mask.append(words[idx])

        elif strategy == "random":
            words_to_mask = random.sample(words, min(len(words), num_to_mask))
            
        # --- MASKING LOGIC (Using Replacement) ---
        modified_text = text
        for w in words_to_mask:
            # Replace word with [UNK] to preserve structure but remove meaning
            # We use a distinct marker.
            modified_text = modified_text.replace(w, "[UNK]") 
            
        new_sentences.append(modified_text)
        new_labels.append(label)
        
    return Dataset.from_dict({"sentence": new_sentences, "label": new_labels})

def train_and_evaluate(dataset_name, train_data, eval_data, args):
    print(f"\n{'='*40}")
    print(f"TRAINING MODEL ON: {dataset_name}")
    print(f"{'='*40}")
    
    # Reload fresh model for every run
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.huggingface_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map={"":0}, token=args.huggingface_token
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    
    def tokenize(examples):
      prompts = [f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {t}\nSentiment:" for t in examples["sentence"]]
      full_texts = [p + f" {l}" for p, l in zip(prompts, examples["label"])]
      
      # Tokenize
      tokenized = tokenizer(full_texts, truncation=True, max_length=256, padding="max_length")
      
      labels_list = []
      for i, pt in enumerate(prompts):
          input_ids = tokenized["input_ids"][i]
          labels = list(input_ids)
          
          # 1. Mask Prompt
          # add_special_tokens=True ensures we account for BOS token if present
          prompt_len = len(tokenizer(pt, add_special_tokens=True)["input_ids"])
          labels[:prompt_len] = [-100] * prompt_len
          
          # 2. Mask Padding (New Addition)
          # Check attention_mask (0 means padding)
          att_mask = tokenized["attention_mask"][i]
          for j in range(len(att_mask)):
              if att_mask[j] == 0:
                  labels[j] = -100
                  
          labels_list.append(labels)
          
      tokenized["labels"] = labels_list
      return tokenized

    train_ds = train_data.map(tokenize, batched=True)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"{args.output_dir}/{dataset_name}", 
            num_train_epochs=2, 
            per_device_train_batch_size=4, 
            learning_rate=2e-4, 
            fp16=True, 
            logging_steps=50,
            report_to=[]
        ),
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    trainer.train()
    
    # Evaluate
    print(f"Evaluating {dataset_name} model...")
    predict_fn = get_predict_fn(model, tokenizer)
    preds = []
    # Evaluate on a subset of validation for speed
    eval_subset = eval_data.select(range(200)) 
    
    for text in tqdm(eval_subset["sentence"]):
        preds.append(int(predict_fn(text)[0][1] > 0.5))
        
    acc = accuracy_score(eval_subset["label"], preds)
    print(f"Result for {dataset_name}: Accuracy = {acc:.4f}")
    return acc

def main():
    args = parse_args()
    set_seed(42)
    
    # Load Data
    full_data = load_dataset("stanfordnlp/sst2")
    # Keep small for speed in this demo
    train_subset = full_data["train"].shuffle(seed=42).select(range(args.train_size))
    eval_data = full_data["validation"]
    
    # Load helper model
    print("Loading helper model for XAI generation...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.huggingface_token)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, load_in_4bit=True, device_map={"":0}, token=args.huggingface_token
    )
    
    # 1. Generate Datasets
    # Control Group
    random_dataset = create_modified_dataset(train_subset, base_model, tokenizer, strategy="random", percentage=args.mask_percentage)
    # LIME Group
    lime_dataset = create_modified_dataset(train_subset, base_model, tokenizer, strategy="roar_lime", percentage=args.mask_percentage)
    # SHAP Group
    shap_dataset = create_modified_dataset(train_subset, base_model, tokenizer, strategy="roar_shap", percentage=args.mask_percentage)
    
    del base_model
    torch.cuda.empty_cache()
    
    # 2. Train & Evaluate
    print("\nTraining Baseline (Original)...")
    acc_base = train_and_evaluate("Baseline", train_subset, eval_data, args)
    
    print("\nTraining Random (Control)...")
    acc_rand = train_and_evaluate("Random", random_dataset, eval_data, args)
    
    print("\nTraining LIME ROAR...")
    acc_lime = train_and_evaluate("ROAR_LIME", lime_dataset, eval_data, args)

    print("\nTraining SHAP ROAR...")
    acc_shap = train_and_evaluate("ROAR_SHAP", shap_dataset, eval_data, args)
    
    # 3. Final Calculation
    print("\n" + "="*50)
    print("FINAL ROAR SCORES (F7.2)")
    print("="*50)
    print(f"Accuracy Random: {acc_rand:.4f}")
    print(f"Accuracy LIME:   {acc_lime:.4f}")
    print(f"Accuracy SHAP:   {acc_shap:.4f}")
    print("-" * 30)
    
    # Calculate Score: (Random Acc - XAI Acc)
    # If XAI did a good job, XAI Acc should be LOWER than Random Acc.
    # We multiply by 10 to scale it similarly to other metrics (0-5 scale usually, or 0-1)
    
    lime_roar_score = max(0, acc_rand - acc_lime)
    shap_roar_score = max(0, acc_rand - acc_shap)
    
    print(f"LIME F7.2 ROAR Score: {lime_roar_score:.4f}  (>0 means successful)")
    print(f"SHAP F7.2 ROAR Score: {shap_roar_score:.4f}  (>0 means successful)")
    
    if lime_roar_score > shap_roar_score:
        print("\nWINNER: LIME is more faithful (caused more damage).")
    elif shap_roar_score > lime_roar_score:
        print("\nWINNER: SHAP is more faithful (caused more damage).")
    else:
        print("\nTIE or FAILURE (Both performed worse than or equal to random).")

if __name__ == "__main__":
    main()
