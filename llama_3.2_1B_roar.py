import argparse
import random
import numpy as np
import torch
import copy
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from lime.lime_text import LimeTextExplainer
import shap
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", default="outputs/roar_experiment")
    # ROAR requires explaining the TRAINING data.
    # 200 samples is a good balance for a 4-hour job.
    parser.add_argument("--train-size", type=int, default=200) 
    parser.add_argument("--mask-percentage", type=float, default=0.2, help="Remove top 20% of words")
    parser.add_argument("--huggingface-token", type=str, default=None)
    return parser.parse_args()

def get_predict_fn(model, tokenizer):
    token_0 = tokenizer("0", add_special_tokens=False)["input_ids"][0]
    token_1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]
    
    def predict_proba(texts):
        # Handle various input formats
        if isinstance(texts, np.ndarray): texts = texts.flatten().tolist()
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
    Creates a modified dataset by masking words with [UNK].
    strategy: 'roar_lime', 'roar_shap', or 'random'
    """
    print(f"\nGenerating modified dataset: STRATEGY = {strategy.upper()}...")
    predict_fn = get_predict_fn(model, tokenizer)
    
    # Initialize LIME Explainer (SHAP will use a custom wrapper)
    lime_explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    
    new_sentences = []
    new_labels = []
    
    for i in tqdm(range(len(original_data)), desc=f"Processing {strategy}"):
        text = original_data[i]['sentence']
        label = original_data[i]['label']
        words = text.split()
        
        # Determine how many words to mask
        num_to_mask = max(1, int(len(words) * percentage))
        words_to_mask = []
        
        if strategy == "roar_lime":
            # num_samples=30 is low for speed; increase to 100+ for better precision if time allows
            exp = lime_explainer.explain_instance(text, predict_fn, num_features=num_to_mask, num_samples=30)
            words_to_mask = [x[0] for x in exp.as_list()]
            
        elif strategy == "roar_shap":
            # --- CUSTOM SHAP WRAPPER FOR WORDS ---
            # KernelSHAP needs a function that maps binary masks to predictions
            # 1 = word present, 0 = word removed ([UNK])
            def shap_target_fn(masks):
                # masks shape: (nsamples, nwords)
                texts_batch = []
                for mask in masks:
                    # Reconstruct text based on mask
                    masked_words = [words[j] if mask[j] == 1 else "[UNK]" for j in range(len(words))]
                    texts_batch.append(" ".join(masked_words))
                return predict_fn(texts_batch)
            
            # Background: All zeros (all words masked)
            # Input: All ones (all words present)
            num_features = len(words)
            if num_features > 0:
                background = np.zeros((1, num_features))
                explainer = shap.KernelExplainer(shap_target_fn, background)
                
                # Run SHAP for the single instance (all words present)
                # nsamples=30 matches LIME effort
                shap_values = explainer.shap_values(np.ones((1, num_features)), nsamples=30, silent=True)
                
                # shap_values is list [class0_vals, class1_vals]. Take Class 1.
                vals = shap_values[1][0] # Shape (n_features,)
                
                # Get indices of top words (highest absolute importance)
                top_indices = np.argsort(-np.abs(vals))[:num_to_mask].flatten()
                
                for idx in top_indices:
                    words_to_mask.append(words[idx])

        elif strategy == "random":
            if len(words) > 0:
                words_to_mask = random.sample(words, min(len(words), num_to_mask))
            
        # --- MASKING LOGIC ---
        modified_text = text
        for w in words_to_mask:
            # Replace word with [UNK] marker
            # Using split/join is safer to avoid substring replacement issues, but simple replace works for ROAR proxy
            modified_text = modified_text.replace(w, "[UNK]") 
            
        new_sentences.append(modified_text)
        new_labels.append(label)
        
    return Dataset.from_dict({"sentence": new_sentences, "label": new_labels})

def train_and_evaluate(dataset_name, train_data, eval_data, args):
    print(f"\n{'='*40}")
    print(f"TRAINING MODEL ON: {dataset_name}")
    print(f"{'='*40}")
    
    # Reload fresh model/tokenizer for every run to ensure no leakage
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.huggingface_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit Quantization
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
    
    # Tokenization with PROMPT MASKING
    def tokenize(examples):
        prompts = [f"Classify the sentiment as 0 (negative) or 1 (positive).\nText: {t}\nSentiment:" for t in examples["sentence"]]
        full_texts = [p + f" {l}" for p, l in zip(prompts, examples["label"])]
        
        # Use max_length padding to avoid ragged tensor issues
        tokenized = tokenizer(full_texts, truncation=True, max_length=256, padding="max_length")
        
        labels_list = []
        for i, pt in enumerate(prompts):
            input_ids = tokenized["input_ids"][i]
            labels = list(input_ids)
            
            # 1. Mask Prompt (Instruction)
            prompt_len = len(tokenizer(pt, add_special_tokens=True)["input_ids"])
            labels[:prompt_len] = [-100] * prompt_len
            
            # 2. Mask Padding
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
            gradient_accumulation_steps=2,
            learning_rate=2e-4, 
            fp16=True, 
            logging_steps=50,
            report_to=[],
            save_strategy="no"
        ),
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    trainer.train()
    
    # Evaluate
    print(f"Evaluating {dataset_name} model...")
    predict_fn = get_predict_fn(model, tokenizer)
    preds = []
    # Evaluate on a subset of validation for speed (e.g., 200 samples)
    eval_subset = eval_data.select(range(min(200, len(eval_data)))) 
    
    for text in tqdm(eval_subset["sentence"]):
        # Simple thresholding at 0.5
        preds.append(int(predict_fn(text)[0][1] > 0.5))
        
    acc = accuracy_score(eval_subset["label"], preds)
    print(f"Result for {dataset_name}: Accuracy = {acc:.4f}")
    return acc

def main():
    args = parse_args()
    set_seed(42)
    
    # Load Data
    full_data = load_dataset("stanfordnlp/sst2")
    # Using small subset for speed. Increase to 500-1000 for better results if time permits.
    train_subset = full_data["train"].shuffle(seed=42).select(range(args.train_size))
    eval_data = full_data["validation"]
    
    # Load helper model for XAI generation
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
    
    # Clean up
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
    
    # 3. Final Comparison
    print("\n" + "="*50)
    print("FINAL ROAR SCORES (F7.2)")
    print("="*50)
    print(f"Accuracy Baseline: {acc_base:.4f}")
    print(f"Accuracy Random:   {acc_rand:.4f}")
    print(f"Accuracy LIME:     {acc_lime:.4f}")
    print(f"Accuracy SHAP:     {acc_shap:.4f}")
    print("-" * 30)
    
    # Calculate Score: (Random Acc - XAI Acc)
    lime_roar_score = max(0, acc_rand - acc_lime)
    shap_roar_score = max(0, acc_rand - acc_shap)
    
    print(f"LIME F7.2 ROAR Score: {lime_roar_score:.4f}  (Higher is better)")
    print(f"SHAP F7.2 ROAR Score: {shap_roar_score:.4f}  (Higher is better)")
    
    if lime_roar_score > shap_roar_score:
        print("\nWINNER: LIME identified more faithful features.")
    elif shap_roar_score > lime_roar_score:
        print("\nWINNER: SHAP identified more faithful features.")
    else:
        print("\nTIE or FAILURE (Both performed worse than or equal to random).")

if __name__ == "__main__":
    main()
