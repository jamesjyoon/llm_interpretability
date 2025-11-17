#!/bin/bash
#SBATCH --job-name=llama-1B
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:V100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/llama-1B-%j.out

module purge
module load anaconda3
conda activate $SLURM_SUBMIT_DIR/envs/llm-1B

cd $SLURM_SUBMIT_DIR

python experiments_llama_3.2_1B.py \
    --output-dir outputs/sentiment_1B \
    --run-shap \
    --finetune

