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

# 1. Clean modules and load Anaconda
module purge
module load anaconda3
# 2. Activate your env by path (no need for global 'conda' command)
# This uses the env you already created at /storage/ice1/6/3/jyoon370/llm-interpretability/envs/llm-1B
source /storage/ice1/6/3/jyoon370/llm-interpretability/envs/llm-1B/bin/activate

# 3. Load Hugging Face token (assuming you created ~/.hf_token as before)
if [ -f ~/.hf_token ]; then
    source ~/.hf_token
fi

echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 4. Go to project directory
cd /storage/ice1/6/3/jyoon370/llm-interpretability

# 5. Run experiment (note: correct arg names, no 'None' string, fixed backslashes)
python experiments_llama_3.2_1B_interpretability.py \
    --model-name meta-llama/Llama-3.2-1B \
    --dataset-name mteb/tweet_sentiment_extraction \
    --dataset-config default \
    --output-dir outputs/llama_sentiment_1B \

echo "=== Job finished at $(date) ==="


