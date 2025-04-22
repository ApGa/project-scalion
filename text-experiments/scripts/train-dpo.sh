#!/bin/bash
#SBATCH --job-name=dpo
#SBATCH --output=/home/lsutawik/dpo-%j.out
#SBATCH --error=/home/lsutawik/dpo-%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:A100_40GB:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G

MODEL_PATH=$1
DATA_PATH=$2
OUTPUT_PATH=$3

set -a 
source scripts/configs/.env
set +a

source ${MINICONDA_PATH}
conda init bash
conda activate ${ENV_NAME}

# export NCCL_P2P_DISABLE=1

# DPO Training
deepspeed --module openrlhf.cli.train_dpo \
    --save_path ${OUTPUT_PATH}/model/ \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size 64 \
    --micro_train_batch_size 32 \
    --pretrain ${MODEL_PATH} \
    --save_hf_ckpt \
    --bf16 \
    --max_samples 6400 \
    --max_epochs 1 \
    --max_len 2048 \
    --zero_stage 3 \
    --ref_offload \
    --learning_rate 3e-7 \
    --l2 0.05 \
    --beta 0.05 \
    --dataset json@${DATA_PATH} \
    --apply_chat_template \
    --prompt_key question \
    --chosen_key response_i \
    --rejected_key response_j \
    --flash_attn \
    --gradient_checkpointing \
    --adam_offload --use_liger_kernel --packing_samples
    # --train_split train \
    # --eval_split test \