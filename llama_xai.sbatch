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

# FIXED: Correct path to your NEW Miniconda
source /storage/ice1/6/3/jyoon370/miniconda3/etc/profile.d/conda.sh
conda activate llm1b

# Verify
echo "Python: $(which python)"
python -c "import torch; print('PyTorch OK:', torch.cuda.is_available())"

# Run your script
python /storage/ice1/6/3/jyoon370/llm-interpretability/experiments_llama_3.2_1B.py \
    --model-name meta-llama/Llama-3.2-1B \
    --train-size 8000 \
    --epochs 3 \
    --output-dir /storage/ice1/6/3/jyoon370/llm-interpretability/outputs/run_$(date +%m%d_%H%M) \
    --finetune \
    --run-lime \
    --load-in-4bit

