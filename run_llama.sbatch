#!/bin/bash
#SBATCH --job-name=llama-1B
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/llama-1B.%j.out

module purge
module load anaconda3
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate /storage/ice1/6/3/jyoon370/llm-interpretability/envs/llm-1B

cd /storage/ice1/6/3/jyoon370/llm-interpretability

python experiments_llama_3.2_1B_interpretability.py \
  --model_name meta-llama/Llama-3.2-1B \
  --output-dir outputs/llama_sentiment_1B \
  --run-shap \
  --finetune

