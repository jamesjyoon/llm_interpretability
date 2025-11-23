#!/bin/bash
#SBATCH --job-name=llama-1B
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/llama-1B.%j.out

echo "=== Job starting at $(date) on $(hostname) ==="

# 1. Load Anaconda and activate your env (by path)
module purge
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /storage/ice1/6/3/jyoon370/llm-interpretability/envs/llm-1B

echo "Python: $(which python)"
python -c "import torch; print('torch version:', torch.__version__); print('cuda?', torch.cuda.is_available())"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 2. Go to project directory
cd /storage/ice1/6/3/jyoon370/llm-interpretability

# 3. Run your experiment script (adjust args if needed)
python experiments_llama_3.2_1B.py \
  --model meta-llama/Llama-3.2-1B \
  --train-size 8000 \
  --epochs 3 \
  --output-dir outputs/llama_3.2_1B

echo "=== Job finished at $(date) ==="


