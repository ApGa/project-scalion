#!/bin/bash
#SBATCH --job-name=feedback
#SBATCH --output=slurm-out/feedback-%j.out
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --time 12:00:00
#SBATCH --exclude=babel-13-37,babel-9-7,babel-0-27,babel-0-19,babel-4-37,babel-9-11,babel-13-33,babel-14-33,babel-8-17,babel-4-13,babel-14-1,babel-3-9,babel-4-9,babel-11-17
#SBATCH --partition=general

set -a 
source scripts/configs/.env
set +a

source ${MINICONDA_PATH}
conda init bash
conda activate ${ENV_NAME}

MAX_ITERATIONS=$1
TASK=$2
MODEL=$3
API_BASE=$4
OUTPUT_PATH=$5
RUN_NAME=$MODEL-$TASK

mkdir -p ${OUTPUT_PATH}/paraphrase_input_data
# first evaluate on original questions
yeval \
    --model $MODEL \
    --task ${TASK}p//prompt_cot \
    --trust_remote_code \
    --api_base $API_BASE \
    --output_path $OUTPUT_PATH/0 \
    --include_path tasks/ \
    --run_name $RUN_NAME

for i in $(seq 1 $MAX_ITERATIONS); do
    model_output_path=$OUTPUT_PATH/$((i-1))/$RUN_NAME/output.jsonl
    save_filepath=$OUTPUT_PATH/paraphrase_input_data/$i.jsonl
    # filter and create inputs for paraphrase model
    if [ i==1 ]; then
        # first iteration, no previous paraphrased questions
        python scripts/feedback_data.py \
        --model_output_filepath $model_output_path \
        --save_filepath $save_filepath
    fi
    else
        python scripts/feedback_data.py \
        --model_output_filepath $model_output_path \
        --prev_paraphrase_questions $prev_paraphrase_questions \
        --save_filepath $save_filepath
    fi

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
        --task generate_paraphrase_with_feedback_history \
        --data_kwargs "$data_kwargs" \
        --trust_remote_code \
        --api_base $API_BASE \
        --output_path $output_path \
        --include_path tasks/ \
        --run_name $RUN_NAME
    paraphrase_questions=$output_path/$RUN_NAME/output.jsonl

    # then evaluate the paraphrased questions    
    data_kwargs='{"data_files": "'$paraphrase_questions'"}'
    yeval \
        --model $MODEL \
        --task score_paraphrasep//prompt_cot \
        --data_kwargs "$data_kwargs" \
        --trust_remote_code \
        --api_base $API_BASE \
        --output_path $OUTPUT_PATH/$i \
        --include_path tasks/ \
        --run_name $RUN_NAME
    
    prev_paraphrase_questions=$paraphrase_questions
done