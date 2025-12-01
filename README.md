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

This implementation references two major approaches for model explanation:

- **LIME (Local Interpretable Model-agnostic Explanations)**: A perturbation-based technique that trains local surrogate models to identify feature importance. While model-agnostic and easy to apply to text, it can be sensitive to sampling strategies.

- **KernelSHAP**: Approximates Shapley values using a weighted kernel that emphasizes coalitions near the original input. It inherits desirable properties (local accuracy, missingness, consistency) while remaining practical for deep networks through sampling.

For classification evaluation, we complement accuracy-based metrics with the **Matthews Correlation Coefficient (MCC)**, which provides a balanced score (−1 to +1) that remains informative under class imbalance. An MCC of +1 indicates perfect predictions, 0 matches random guessing, and −1 reflects complete disagreement.


Methodology: Computed vs. Static Properties
The evaluation of XAI properties (F1–F11) in experiments_llama_3.2_1B.py utilizes a hybrid approach. While properties dependent on the data distribution are calculated at runtime, properties inherent to the algorithmic design of LIME/SHAP or those requiring prohibitive computational resources are assigned static values based on established literature.
1. Dynamically Computed Properties
These scores are calculated during execution based on the specific Llama-3.2-1B model and SST-2 dataset:
F1.1 Scope: Measures the diversity of features selected across the evaluation set.
F1.4 Practicality: Measures the wall-clock time required to generate explanations.
F4.2 Target Sensitivity (LIME): Measures the divergence in explanations when querying opposite labels.
F6.2 Surrogate Agreement (LIME): Measures the fidelity (R-squared/MAE) of the local linear approximation against the model's actual probabilities.
2. Static (Hardcoded) Properties
The following properties use fixed values. This section explains the rationale for each:

| Property | Value Source | Rationale |
| :--- | :--- | :--- |
| **F9.2 Identity (Stability)** | **Theoretical** | **KernelSHAP (1.0)** is mathematically proven to be deterministic (output is constant for constant input). **LIME (0.2)** is inherently stochastic due to random neighborhood sampling, leading to known instability. |
| **F1.2 Portability** | **Theoretical** | Both methods are "Model-Agnostic" by design, meaning they can be applied to any black-box model. This is a binary feature of the algorithm, not a runtime metric. |
| **F2.1 Expressive Power** | **Theoretical** | **LIME (4.7)** generates a weighted linear formula, which is considered highly expressive. **SHAP (3.0)** generates additive feature values, which implies feature independence and is slightly less expressive regarding interactions. |
| **F7.2 ROAR (Faithfulness)** | **Literature / Computational** | *Remove And Retrain (ROAR)* requires retraining the LLM dozens of times to measure accuracy drops when top features are removed. Due to the high computational cost of retraining Llama-1B, we utilize literature baseline scores (LIME ~0.5, SHAP ~0.0) rather than running this benchmark dynamically. |
| **F2.2 Graphical Integrity** | **Qualitative** | This measures the visual clarity of the output (e.g., bar charts). As this is a subjective Human-Computer Interaction (HCI) metric, it cannot be computed by a script. |
| **F6.2 Surrogate Agreement (SHAP)** | **Computational** | **SHAP (0.6)**. Calculating exact fidelity for KernelSHAP requires convergence (thousands of samples), which is too slow for this script. A conservative estimate is used based on the low `nsamples` setting used in this experiment. |


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

## Implementation Details: llama_3.2_1B_roar.py

The `llama_3.2_1B_roar.py` script implements Remove And Retrain (ROAR) based faithfulness analysis for the Llama 3.2 1B model. ROAR is a direct faithfulness evaluation method that measures the impact of removing important features on model performance.

### Key Capabilities

- **ROAR Faithfulness Evaluation**: Iteratively removes top-ranked features and retrains the model to measure accuracy drops
- **Attribution Integration**: Uses LIME and KernelSHAP feature importances as the basis for removal ranking
- **Dual Path Evaluation**: Evaluates both zero-shot and fine-tuned models
- **Metric Aggregation**: Computes fidelity scores by analyzing performance degradation curves

### Script Arguments

- `--model-name`: Llama model checkpoint (default: `meta-llama/Llama-3.2-1B`)
- `--output-dir`: Directory for saving ROAR analysis results
- `--train-size`: Number of training samples for retraining (default: 1000)
- `--eval-sample-size`: Number of evaluation samples for ROAR computation (default: 50)
- `--finetune`: Enable LoRA fine-tuning before ROAR analysis
- `--load-in-4bit`: Use 4-bit quantization for memory efficiency
- `--huggingface-token`: Authentication token for gated models

### Output Files

- `roar_metrics.json`: ROAR faithfulness scores for LIME and KernelSHAP
- `roar_comparison.png`: Visualization of feature removal impact on accuracy
- `roar_analysis_results.json`: Detailed results including per-feature removal metrics

## Implementation Details: llama_3.2_1B_xai_full.py

The `llama_3.2_1B_xai_full.py` script provides an extended implementation that combines multiple XAI evaluation methods including LIME, KernelSHAP, and ROAR in a unified pipeline. This comprehensive script enables side-by-side comparison of multiple faithfulness and interpretability metrics.

### Key Capabilities

- **Integrated XAI Pipeline**: Combines LIME, KernelSHAP, and ROAR evaluations in a single execution
- **Comprehensive Metrics**: Computes functionally-grounded properties (F1-F11) alongside faithfulness metrics
- **Multi-Method Comparison**: Enables detailed comparison of different XAI explanation methods
- **Visualization Suite**: Generates comparative visualizations across all evaluation methods
- **Dual Model Analysis**: Evaluates both zero-shot and fine-tuned configurations

### Script Arguments

- `--model-name`: Llama model checkpoint (default: `meta-llama/Llama-3.2-1B`)
- `--output-dir`: Directory for saving comprehensive XAI analysis
- `--train-size`: Number of training samples for fine-tuning (default: 1000)
- `--eval-sample-size`: Number of evaluation samples (default: 50)
- `--epochs`: Number of fine-tuning epochs (default: 1.0)
- `--finetune`: Enable LoRA fine-tuning
- `--run-xai`: Compute functionally-grounded XAI properties (default: True)
- `--run-roar`: Compute ROAR faithfulness metrics (default: True)
- `--load-in-4bit`: Use 4-bit quantization for memory efficiency
- `--huggingface-token`: Authentication token for gated models

### Output Files

- `xai_full_metrics.json`: Complete metrics including XAI properties and ROAR scores
- `xai_full_comparison.png`: Integrated comparison visualization of all XAI methods
- `roar_metrics.json`: ROAR faithfulness scores
- `xai_properties_results.json`: Functionally-grounded property scores
- `model_performance_comparison.png`: Performance metrics comparison

## Implementation Details: llama_3.2_1B_xai.py

The `llama_3.2_1B_xai.py` script provides a focused implementation for XAI (Explainable AI) property analysis on the Llama 3.2 1B model. This file is specifically designed for detailed evaluation of model interpretability across zero-shot and fine-tuned configurations using functionally-grounded explanations.

### Key Capabilities

- **Functionally-Grounded Properties (F1-F11)**: Computes interpretability metrics based on the framework by Mohseni et al., including scope, structure, selectivity, contrastivity, interactivity, fidelity, faithfulness, rationality, alignment, uncertainty, and speed.
- **LIME Explainer Integration**: Local Interpretable Model-agnostic Explanations for feature-level attribution analysis.
- **KernelSHAP Integration**: Kernel-based Shapley value approximation for robust feature importance estimation.
- **Standalone Execution**: Can be run independently or as part of the full pipeline for targeted XAI analysis.

### Script Arguments

- `--model-name`: Llama model checkpoint (default: `meta-llama/Llama-3.2-1B`)
- `--output-dir`: Directory for saving metrics and visualizations
- `--train-size`: Number of training samples for fine-tuning (default: 1000)
- `--eval-sample-size`: Number of evaluation samples for XAI property computation (default: 50)
- `--epochs`: Number of fine-tuning epochs (default: 1.0)
- `--finetune`: Enable LoRA fine-tuning
- `--run-xai`: Compute functionally-grounded XAI properties (default: True)
- `--load-in-4bit`: Use 4-bit quantization for memory efficiency
- `--huggingface-token`: Authentication token for gated models

### Output Files

- `metrics_comparison.json`: Performance metrics (accuracy, precision, recall, F1, MCC) for both zero-shot and fine-tuned models
- `xai_properties_results.json`: Functionally-grounded property scores for LIME and KernelSHAP
- `model_performance_comparison.png`: Bar chart comparing zero-shot vs. fine-tuned performance
- `xai_properties_comparison_*.png`: Horizontal bar charts showing property scores across explainability methods
- `ft_example_lime_*.png`: Visual LIME explanations for sample fine-tuned model predictions

## Running on Georgia Tech ICE PACE Cluster

The ICE (Interactive Cluster Environment) and PACE (Partnership for an Advanced Computing Environment) clusters at Georgia Tech provide high-performance GPU resources ideal for model training and interpretation analysis.

### ICE PACE Setup

1. **SSH into the cluster**:
   ```bash
   ssh <gt_username>@ice.cc.gatech.edu  # or ice-login.cc.gatech.edu
   ```

2. **Load necessary modules**:
   ```bash
   module load anaconda3/2023.09
   module load cuda/12.2.0
   module load cudnn/8.9.4
   ```

3. **Set up Python environment**:
   ```bash
   conda create -n llm_interpretability python=3.10
   conda activate llm_interpretability
   ```

4. **Clone repository and install dependencies**:
   ```bash
   git clone https://github.com/jamesjyoon/llm_interpretability.git
   cd llm_interpretability
   pip install -r requirements.txt
   ```

### Submitting a Job via SLURM

Create a submission script (e.g., `llama_xai.sbatch`):

```bash
#!/bin/bash
#SBATCH --job-name=llama-final
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source /storage/ice1/6/3/jyoon370/miniconda3/etc/profile.d/conda.sh
conda activate llm1b

python /storage/ice1/6/3/jyoon370/llm-interpretability/experiments_llama_3.2_1B.py \
    --model-name meta-llama/Llama-3.2-1B \
    --train-size 8000 \
    --epochs 3 \
    --output-dir outputs/run_$(date +%m%d_%H%M) \
    --finetune \
    --run-lime \
    --run-xai \
    --load-in-4bit
    --huggingface-token <your_hf_token>
```

Submit the job:
```bash
sbatch llama_xai.sbatch
```

### Monitoring Jobs

```bash
# Check job status
squeue -u <gt_username>

# View job details
scontrol show job <job_id>

# Cancel a job
scancel <job_id>
```

### Transferring Results

Transfer results from PACE to local machine:
```bash
scp -r <gt_username>@ice.cc.gatech.edu:/home/<gt_username>/llm_interpretability/outputs/pace_xai_analysis ./results_from_pace/
```

### PACE-Specific Considerations

- **GPU Allocation**: Request appropriate GPU resources (A100 for larger models, V100/H100 for standard tasks)
- **Memory Management**: Monitor memory usage with `nvidia-smi` in interactive sessions
- **Job Time Limits**: Default partition has 4-hour limit; longer jobs may need `--partition=gpu-long` (up to 24 hours)
- **Storage**: Store large datasets in `/scratch` directory for faster I/O; results in `$HOME` are automatically backed up
- **Data Transfer**: For large models/datasets, use `rsync` or `globus` instead of `scp` for improved performance

### Example: Interactive GPU Session

For development and debugging:
```bash
interactive -p gpu -N 1 --gres=gpu:A100:1 -c 8 --mem=64G -t 120
```

Then run the script directly:
```bash
python llama_3.2_1B_xai.py --model-name meta-llama/Llama-3.2-1B --output-dir outputs/debug_run --train-size 100 --eval-sample-size 10
```

### Troubleshooting on PACE

- **CUDA out of memory**: Reduce `--train-size` or `--eval-sample-size`, or enable `--load-in-4bit`
- **Module not found**: Verify all modules are loaded and environment is activated
- **Slow I/O**: Use `/scratch` for intermediate files instead of `$HOME`
- **Job timeout**: Increase `--time` in SLURM header or reduce model size/data volume

## Citations

This project utilizes the Functionally-Grounded Evaluation framework to assess XAI methods (LIME and KernelSHAP), focusing on proxy tasks and quantitative metrics rather than human-subject studies to evaluate explanation quality.

If you use this code, please cite the foundational work on this evaluation strategy:

```bibtex
@article{doshi2017towards,
  title={Towards a rigorous science of interpretable machine learning},
  author={Doshi-Velez, Finale and Kim, Been},
  journal={arXiv preprint arXiv:1702.08608},
  year={2017}
}

@article{nauta2023anecdotal,
  title={From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable AI},
  author={Nauta, Meike and Trienes, Jan and Pathak, Shreyasi and Nguyen, Elisa and Peters, Michelle and Schmitt, Yasmin and Schl{\"o}tterer, J{\"o}rg and van Keulen, Maurice and Seifert, Christin},
  journal={ACM Computing Surveys},
  volume={55},
  number={13s},
  pages={1--42},
  year={2023},
  publisher={ACM New York, NY}
}
```

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
