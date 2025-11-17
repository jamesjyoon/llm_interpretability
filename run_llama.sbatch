#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:V100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Load modules
module purge
module load anaconda3
conda activate $SLURM_SUBMIT_DIR/envs/myenv

# Run your script from this repo
python experiments_llama_3.2_1B.py --arg1 value
