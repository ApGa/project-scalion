#!/bin/bash
#SBATCH --job-name=scln
#SBATCH --output=/home/lsutawik/scln-%j.out
#SBATCH --error=/home/lsutawik/scln-%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G

TASK=$1
P_MODEL=$2
E_MODEL=$3
OUTPUT_MODEL_PATH=$4
export NCCL_P2P_DISABLE=1
set -a 
source scripts/configs/.env
set +a

source ${MINICONDA_PATH}
conda init bash
conda activate ${ENV_NAME}

PORT=4512

for I in 1
do

    # 1. Generate Paraphrased data by model p1
    vllm serve $P_MODEL --port ${PORT} --max_model_len 2048 > ${TMP_DIR}o.txt &
    # for NUM in {1..4}
    for NUM in {1..2}
    do
        RUN_NAME=${TASK}_${NUM}
        yeval \
            --model $P_MODEL \
            --task ${TASK}t//generate_paraphrase_${NUM} \
            --n_samples 500 \
            --api_base "http://localhost:${PORT}/v1" \
            --run_name ${RUN_NAME} \
            --output_path tasks/data/paraphrased_train/${TASK}/${P_MODEL}/ \
            --include_path tasks/
    done

    pkill vllm
    sleep 120

    # 2. Evaluate how well the paraphrased data by e1,..eN
    vllm serve $E_MODEL --port ${PORT} --max_model_len 2048 > ${TMP_DIR}o.txt &
    # for NUM in {1..4}
    for NUM in {1..2}
    do
        RUN_NAME=${TASK}_${NUM}
        yeval \
            --model $E_MODEL \
            --task score_paraphrase \
            --include_path tasks/ \
            --data_kwargs "{'data_files': 'tasks/data/paraphrased_train/${TASK}/${P_MODEL}/${RUN_NAME}/output.jsonl'}" \
            --api_base "http://localhost:${PORT}/v1" \
            --run_name ${RUN_NAME} \
            --output_path data/raw/${TASK}/${E_MODEL}
    done

    pkill vllm
    sleep 120

    # 3. Generate preference data
    python scripts/generate_preference_dataset.py \
        --input_path data/raw/${TASK}/${E_MODEL} \
        --output_path data/preference/${TASK}/${E_MODEL} \
        --source_path tasks/data/paraphrased_train/${TASK}/${P_MODEL}/

    # 4. DPO Training
    bash scripts/train-dpo.sh \
        ${E_MODEL} \
        data/preference/${TASK}/${E_MODEL} \
        ${OUTPUT_MODEL_PATH}

    # 6. Trained P Model generates paraphrased data
    TRAINED_MODEL=${OUTPUT_MODEL_PATH}/model/
    vllm serve $TRAINED_MODEL --port ${PORT} --max_model_len 2048 > ${TMP_DIR}o.txt &
    # for NUM in {1..4}
    for NUM in {1..2}
    do
        RUN_NAME=${TASK}_${NUM}
        yeval \
            --model $TRAINED_MODEL \
            --task test_${TASK}t//generate_paraphrase_${NUM} \
            --sample_args "n=1" \
            --run_name ${TASK}_${NUM} \
            --api_base "http://localhost:${PORT}/v1" \
            --output_path tasks/data/paraphrased_test/${TASK}/${P_MODEL}/ \
            --include_path tasks/
    done

    pkill vllm
    sleep 120

    # 7. Evaluate how well Trained P Model is 
    vllm serve $E_MODEL --port ${PORT} --max_model_len 2048 > ${TMP_DIR}o.txt &
    # for TNUM in {1..4}
    for NUM in {1..2}
    do
        RUN_NAME=${TASK}_${TNUM}
        for NUM in {1..3}
        do
            for VARIANT in A B C
            do
            yeval \
                --model $E_MODEL \
                --task score_paraphrasep//reason${NUM}${VARIANT} \
                --include_path tasks/ \
                --data_kwargs "{'data_files': 'tasks/data/paraphrased_test/${TASK}/${P_MODEL}/${RUN_NAME}/output.jsonl'}" \
                --api_base "http://localhost:${PORT}/v1" \
                --run_name $E_MODEL:${RUN_NAME}:${NUM}:${VARIANT} \
                --output_path data/runs/${P_MODEL}/ 
            done
        done
    done

    pkill vllm
    sleep 120

done
