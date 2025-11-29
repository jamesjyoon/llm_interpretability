# LLM Interpretability: Zero-Shot vs. Fine-Tuned Comparison

## Overview

This repository contains a comprehensive framework for comparing the interpretability of large language models (LLMs) across zero-shot and task-specific fine-tuned configurations. Using the Llama 3.2 1B model, we perform token-level attributional analysis through SHAP (SHapley Additive exPlanations) and provide extensive evaluation metrics including model performance and interpretability statistics.

The implementation is designed to be accessible and reproducible, with Colab-optimized code that can be executed on free-tier GPU resources.

## Key Features

- **Binary Classification Experiments**: Token-level interpretability analysis on the [Tweet Sentiment Extraction dataset](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction)
- **Dual-Path Evaluation**: Direct comparison of zero-shot model behavior versus LoRA-adapted fine-tuned variants
- **SHAP-Based Attribution**: Token-level explanation extraction with statistical aggregation
- **Comprehensive Metrics**: Performance evaluation (accuracy, precision, recall, F1), interpretability metrics (sparsity, entropy, agreement), and effect size analysis
- **LoRA Adapter Support**: Lightweight fine-tuning with persistent adapter storage
- **Colab-Ready**: Optimized for cloud GPU environments with progress visualization and artifact export

## Interpretability Methods

This implementation references three major approaches for model explanation:

- **LIME (Local Interpretable Model-agnostic Explanations)**: A perturbation-based technique that trains local surrogate models to identify feature importance. While model-agnostic and easy to apply to text, it can be sensitive to sampling strategies.

- **KernelSHAP**: Approximates Shapley values using a weighted kernel that emphasizes coalitions near the original input. It inherits desirable properties (local accuracy, missingness, consistency) while remaining practical for deep networks through sampling.

- **TreeSHAP**: Provides exact and efficient Shapley value computation for tree ensembles by exploiting their structure. This is the preferred method for gradient-boosted trees and random forests.

For classification evaluation, we complement accuracy-based metrics with the **Matthews Correlation Coefficient (MCC)**, which provides a balanced score (−1 to +1) that remains informative under class imbalance. An MCC of +1 indicates perfect predictions, 0 matches random guessing, and −1 reflects complete disagreement.

## Getting Started

### Prerequisites
- Python 3.8+
- GPU with at least 8GB VRAM (16GB recommended for fine-tuning)
- Hugging Face account and authentication token (for gated models)

### Installation

```bash
git clone https://github.com/jamesjyoon/llm_interpretability.git
cd llm_interpretability
pip install -r requirements.txt
```

### Running on Google Colab

1. **Start a new Colab notebook** with GPU runtime enabled.

2. **Clone and install dependencies**:
   ```python
   !git clone https://github.com/jamesjyoon/llm_interpretability.git
   %cd llm_interpretability
   !pip install -r requirements.txt
   ```

3. **Authenticate with Hugging Face** (required for gated models):
   ```python
   # Option A: Set environment variable
   import os
   os.environ["HF_TOKEN"] = "<your_hf_token>"
   
   # Option B: Manual login
   from huggingface_hub import login
   login("<your_hf_token>")
   ```
   
   Alternatively, visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and request access to the [Llama 3.2 1B model](https://huggingface.co/meta-llama/Llama-3.2-1B).

4. **Launch the experiment**:
   ```python
   !python experiments_llama_3.2_1B.py --finetune --run-shap --output-dir outputs/tweet_sentiment_extraction
   ```

## Output Artifacts

The pipeline generates the following outputs in the specified `--output-dir`:

### Performance Metrics
- **`zero_shot_metrics.json`** / **`fine_tuned_metrics.json`**: Accuracy, precision, recall, F1 scores (per-class and aggregated), confusion matrix, and probability distributions
- **`metrics_comparison.png`**: Side-by-side visualization of zero-shot vs. fine-tuned performance

### Interpretability Analysis  
- **`zero_shot_shap.json`** / **`fine_tuned_shap.json`**: Serialized SHAP explanations for all evaluated examples
- **`zero_shot_shap_summary.png`** / **`fine_tuned_shap_summary.png`**: Aggregated token importance visualizations
- **`interpretability_metrics.json`**: Comparative statistics including:
  - Average absolute token importance and sparsity (Gini coefficients)
  - Entropy of attribution distributions
  - Top-5 and top-10 token overlap rates
  - Cosine and Spearman correlation of explanations
  - Per-example mean-importance deltas and Cohen's d effect size
  - Per-example SHAP summaries for downstream analysis

### Model Artifacts
- **`lora_adapter/`**: Trained LoRA adapter weights for reuse via `PeftModel.from_pretrained()`

## Configuration Options

### Model Selection
- `--model-name` (default: `meta-llama/Llama-3.2-1B`): Model checkpoint to use. Swap to an open-access alternative if preferred.

### Data Configuration
- `--train-subset` (default: 2000): Number of training examples to sample
- `--eval-subset` (default: 1000): Number of evaluation examples to sample
- `--train-split` (default: `train`): Training data split name
- `--eval-split` (default: `test`): Evaluation data split name
- `--text-field` (default: `text`): Column name for input text
- `--label-field` (default: `label`): Column name for labels
- `--label-space`: Explicitly specify numeric labels to include (e.g., to exclude neutral class)

### Experiment Options
- `--finetune`: Enable LoRA fine-tuning on the training subset
- `--run-shap`: Extract SHAP attributions (disable with `--no-run-shap` for faster runs)
- `--load-in-4bit` (default: True): Use 4-bit quantization. Disable with `--no-load-in-4bit` for full precision on high-memory GPUs
- `--output-dir`: Directory for output artifacts
- `--huggingface-token`: Hugging Face authentication token for gated models

## Notes

- **Model Access**: The default Llama 3.2 1B model is gated. [Request access](https://huggingface.co/meta-llama/Llama-3.2-1B) and authenticate via token before running.
- **Memory Efficiency**: By default, we sample 2,000 training and 1,000 evaluation examples to maintain Colab compatibility. Adjust `--train-subset` and `--eval-subset` as needed, or set to `None` for the full dataset.
- **Fine-Tuning Details**: During fine-tuning, instruction tokens are masked with `-100` so the loss only supervises the appended label token, keeping LoRA updates focused on classification decisions.
- **Output Visualization**: Charts render inline in Colab notebooks when available; otherwise they are saved as PNG files.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{yoon2025llm_interpretability,
  author = {Yoon, James},
  title = {LLM Interpretability: Zero-Shot vs. Fine-Tuned Comparison},
  year = {2025},
  url = {https://github.com/jamesjyoon/llm_interpretability}
}
```

## Related Work

This project builds on foundational research in model interpretability and attribution methods:

- **SHAP/Shapley Values**: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
- **LIME**: Ribeiro et al. (2016) "'Why Should I Trust You?': Explaining the Predictions of Any Classifier"
- **LoRA**: Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"

## License

MIT License. See LICENSE file for details.

## Contact

For questions or feedback, please open an issue or contact [jamesjyoon](https://github.com/jamesjyoon).
