# LLM Interpretability

## Binary classification experiment (Colab-ready)

The `experiments.py` script implements the workflow discussed in
our last meeting: it compares zero-shot and LoRA fine-tuned variants of a
TinyLLaMA model on a binary sentiment classification task, then captures token
level SHAP explanations for both models.

### Running on Google Colab

1. Start a new Colab notebook (a GPU runtime is strongly recommended).
2. Clone this repository and install the dependencies:

   ```python
   !git clone https://github.com/<your-org>/llm_interpretability.git
   %cd llm_interpretability
   !pip install -r requirements.txt
   ```

3. Launch the experiment. The command below downloads the TinyLLaMA 1.1B chat
   model, evaluates the zero-shot baseline on SST-2, fine-tunes a LoRA adapter,
   and generates SHAP attributions for the first 10 validation examples.

   ```python
   !python experiments.py --finetune --run-shap --output-dir outputs/sst2_tinyllama
   ```

4. Inspect the outputs stored under `outputs/sst2_tinyllama/`:
   - `zero_shot_metrics.json` and `fine_tuned_metrics.json` contain accuracy,
     precision, recall, and F1 scores.
   - `zero_shot_shap.json` and `fine_tuned_shap.json` store serialized SHAP
     explanations for later analysis.
   - `lora_adapter/` holds the trained adapter weights which can be reloaded via
     `PeftModel.from_pretrained`.

### Notes

- The script defaults to the public
  [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  checkpoint. Swap `--model-name` if you have access to other LLaMA-family
  models.
- By default we sample 2,000 training and 1,000 validation examples from the
  SST-2 dataset to keep runs Colab-friendly. Adjust `--train-subset` and
  `--eval-subset` as needed.
- Use `--no-run-shap` to skip SHAP generation when you want a faster pass, or
  `--no-load-in-4bit` to load the model in full precision on high-memory GPUs.
