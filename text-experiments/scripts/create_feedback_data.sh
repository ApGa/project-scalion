#!/bin/bash
#SBATCH --job-name=feedback
#SBATCH --output=slurm-out/feedback-%j.out
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --time 2:00:00
#SBATCH --exclude=babel-13-37,babel-9-7,babel-0-27,babel-0-19,babel-4-37,babel-9-11,babel-13-33,babel-14-33,babel-8-17,babel-4-13,babel-14-1,babel-3-9,babel-4-9
#SBATCH --partition=general

# Currently written for GSM Symbolic

set -a 
source scripts/configs/.env
set +a

source ${MINICONDA_PATH}
conda init bash
conda activate ${ENV_NAME}

MAX_ITERATIONS=$1
MODEL=$2
API_BASE=$3
OUTPUT_PATH=$4
RUN_NAME=$5

# first evaluate on original questions
# yeval \
#     --model $MODEL \
#     --task gsm_symbolicp//prompt_cot \
#     --trust_remote_code \
#     --api_base $API_BASE \
#     --output_path $OUTPUT_PATH/0 \
#     --include_path tasks/ \
#     --run_name $RUN_NAME

mkdir -p $OUTPUT_PATH/paraphrase_input_data
mkdir -p $OUTPUT_PATH/paraphrase_questions
for i in $(seq 1 $MAX_ITERATIONS); do
    model_output_path=$OUTPUT_PATH/$((i-1))/$RUN_NAME/output.jsonl
    save_filepath=$OUTPUT_PATH/paraphrase_input_data/$i.jsonl
    # filter and create inputs for paraphrase model
    python scripts/feedback_data.py \
        --model_output_filepath $model_output_path \
        --save_filepath $save_filepath

    # if no incorrect answers, exit
    exit_code=$?
    if [ $exit_code -eq 2 ]; then
        echo "No incorrect answers, exiting."
        break
    fi

    # paraphrase questions
    data_kwargs='{"data_files": "'$save_filepath'"}'
    output_path=$OUTPUT_PATH/$i/paraphrase_questions
    echo $data_kwargs
    yeval \
        --model $MODEL \
        --task generate_paraphrase_with_feedback \
        --data_kwargs "$data_kwargs" \
        --trust_remote_code \
        --api_base $API_BASE \
        --output_path $output_path \
        --include_path tasks/

    # then evaluate the paraphrased questions    
    data_kwargs='{"data_files": "'$output_path/output.jsonl'"}'
    yeval \
        --model $MODEL \
        --task score_paraphrase//prompt_cot \
        --data_kwargs "$data_kwargs" \
        --trust_remote_code \
        --api_base $API_BASE \
        --output_path $OUTPUT_PATH/$i \
        --include_path tasks/ \
        --run_name $RUN_NAME
done