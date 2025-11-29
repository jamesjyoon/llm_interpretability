#!/bin/bash
#SBATCH --job-name=llama-1B-final
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3
conda activate llm1b

# Final safety install (won't hurt if already done)
pip install -r /storage/ice1/6/3/jyoon370/llm-interpretability/requirements.txt --quiet --no-cache-dir

python /storage/ice1/6/3/jyoon370/llm-interpretability/llama_3.2_1B_lime.py \
  --model-name meta-llama/Llama-3.2-1B \
  --train-size 8000 \
  --epochs 3 \
  --output-dir /storage/ice1/6/3/jyoon370/llm-interpretability/outputs/my_run_$(date +%m%d_%H%M) \
  --finetune \
  --run-lime \
  --load-in-4bit

