#!/bin/bash
#SBATCH --job-name=scln
#SBATCH --output=/home/lsutawik/scln-%j.out
#SBATCH --error=/home/lsutawik/scln-%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:A100_40GB:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G

TASK=$1
P_MODEL=$2
E_MODEL=$3
OUTPUT_PATH=$4
export NCCL_P2P_DISABLE=1
set -a 
source scripts/configs/.env
set +a

source ${MINICONDA_PATH}
conda init bash
conda activate ${ENV_NAME}

PORT=5423
SAVE_P_MODEL="${P_MODEL//\//-}"
SAVE_E_MODEL="${E_MODEL//\//-}"

for I in {1..4}
do

    OUTPUT_MODEL_PATH=${OUTPUT_PATH}/${SAVE_P_MODEL}-DPO-${I}

    # 1. Generate Paraphrased data by model p1
    vllm serve $P_MODEL --port ${PORT} --max_model_len 4096 > ${TMP_DIR}o.txt &
    for NUM in {1..4}
    # for NUM in {1..2}
    do
        yeval \
            --model $P_MODEL \
            --task ${TASK}t//generate_paraphrase_${NUM} \
            --n_samples 250 \
            --api_base "http://localhost:${PORT}/v1" \
            --run_name ${NUM} \
            --output_path tasks/data/paraphrased_train/${TASK}/${SAVE_P_MODEL}-DPO-${I}/ \
            --include_path tasks/
    done

    pkill vllm
    sleep 120

    # 2. Evaluate how well the paraphrased data by e1,..eN
    vllm serve $E_MODEL --port ${PORT} --max_model_len 4096 > ${TMP_DIR}o.txt &
    for NUM in {1..4}
    # for NUM in {1..2}
    do
        yeval \
            --model $E_MODEL \
            --task score_paraphrase \
            --include_path tasks/ \
            --data_kwargs "{'data_files': 'tasks/data/paraphrased_train/${TASK}/${SAVE_P_MODEL}-DPO-${I}/${NUM}/output.jsonl'}" \
            --api_base "http://localhost:${PORT}/v1" \
            --run_name ${NUM} \
            --output_path data/raw/${TASK}/${SAVE_E_MODEL}-DPO-${I}/
    done

    pkill vllm
    sleep 120

    # 3. Generate preference data
    python scripts/generate_preference_dataset.py \
        --input_path data/raw/${TASK}/${SAVE_E_MODEL}-DPO-${I}/ \
        --output_path data/preference/${TASK}/${SAVE_E_MODEL}-DPO-${I}/ \
        --source_path tasks/data/paraphrased_train/${TASK}/${SAVE_P_MODEL}-DPO-${I}/

    # 4. DPO Training
    bash scripts/train-dpo.sh \
        ${P_MODEL} \
        data/preference/${TASK}/${SAVE_E_MODEL}-DPO-${I}/ \
        ${OUTPUT_MODEL_PATH}

    # 6. Trained P Model generates paraphrased data
    TRAINED_MODEL=${OUTPUT_MODEL_PATH}/model/
    vllm serve $TRAINED_MODEL --port ${PORT} --max_model_len 4096 > ${TMP_DIR}o.txt &
    for NUM in {1..4}
    # for NUM in {1..2}
    do
        RUN_NAME=${TASK}_${NUM}
        yeval \
            --model $TRAINED_MODEL \
            --task test_${TASK}t//generate_paraphrase_${NUM} \
            --sample_args "n=1" \
            --run_name ${NUM} \
            --api_base "http://localhost:${PORT}/v1" \
            --output_path tasks/data/paraphrased_test/${TASK}/${SAVE_P_MODEL}-DPO-${I}/ \
            --include_path tasks/
    done

    pkill vllm
    sleep 120

    # 7. Evaluate how well Trained P Model is 
    vllm serve $E_MODEL --port ${PORT} --max_model_len 4096 > ${TMP_DIR}o.txt &
    # for RUN in {1..4}
    for NUM in 1
    do
        for NUM in {1..2}
        do
            for VARIANT in A B
            do
            yeval \
                --model $E_MODEL \
                --task score_paraphrasep//reason${NUM}${VARIANT} \
                --include_path tasks/ \
                --data_kwargs "{'data_files': 'tasks/data/paraphrased_test/${TASK}/${SAVE_P_MODEL}-DPO-${I}/${RUN}/output.jsonl'}" \
                --api_base "http://localhost:${PORT}/v1" \
                --run_name $E_MODEL:test_${TASK}:${RUN}:${NUM}:${VARIANT} \
                --output_path data/runs/${SAVE_P_MODEL}-DPO-${I}/ 
            done
        done
    done

    pkill vllm
    sleep 120
    P_MODEL=${TRAINED_MODEL}
done
