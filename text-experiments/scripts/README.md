
# Baseline Data
bash scripts/eval.sh \
    Qwen/Qwen2.5-7B-Instruct \
    gsm_symbolicp//prompt_cot \
    http://localhost:8000/v1 \
    "--include_path tasks/ --n_samples 100 --output_path output/"

# Generate Paraphrased Data
bash scripts/eval.sh \
    Qwen/Qwen2.5-7B-Instruct \
    gsm_symbolic_generate_paraphrase \
    http://localhost:8000/v1 \
    "--include_path tasks/ --n_samples 100 --output_path tasks/data/ --run_name gsm_symbolic"

# Evaluate on Paraphrased Data
bash scripts/eval.sh \
    Qwen/Qwen2.5-7B-Instruct \
    gsm_symbolic_paraphrasedp//prompt_cot \
    http://localhost:8000/v1 \
    "--include_path tasks/ --n_samples 100 --output_path output/"