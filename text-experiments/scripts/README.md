
# Baseline Data
bash scripts/eval.sh \
    Qwen/Qwen2.5-7B-Instruct \
    gsm_symbolicp//prompt_cot \
    http://babel-6-9:8084/v1 \
    "--include_path tasks/ --n_samples 1000 --output_path output/ --run_name Qwen2.5-7B-Instruct--cot"

# Generate Paraphrased Data
bash scripts/eval.sh \
    Qwen/Qwen2.5-7B-Instruct \
    gsm_symbolic_generate_paraphrase_4 \
    http://babel-6-9:8084/v1 \
    "--include_path tasks/ --n_samples 1000 --output_path tasks/data/ --run_name gsm_symbolic_4"

# Evaluate on Paraphrased Data
bash scripts/eval.sh \
    Qwen/Qwen2.5-7B-Instruct \
    gsm_symbolic_paraphrasedp//prompt_cot \
    http://localhost:8000/v1 \
    "--include_path tasks/ --n_samples 100 --output_path output/"

# Generate Paraphrase Data with Feedback+History
sbatch scripts/create_feedback_data.sh \
    5 \
    gsm_plus \
    Qwen/Qwen2.5-3B-Instruct \
    http://babel-12-13:8084/v1 \
    output/feedback-with-history 