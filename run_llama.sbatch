#!/bin/bash
#SBATCH --job-name=llama-1B
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/llama-1B-%j.out

cd $SLURM_SUBMIT_DIR

module purge
module load anaconda3

# Proper conda activation inside a batch script
eval "$(conda shell.bash hook)"
conda activate $SLURM_SUBMIT_DIR/envs/llm-1B

python experiments_llama_3.2_1B_interpretability.py \
    --output-dir outputs/llama_sentiment_1B \
    --run-shap \
    --finetune

