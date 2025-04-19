#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=slurm-out/eval-%j.out
#SBATCH --nodes=1
#SBATCH --mem=48GB
#SBATCH --time 1:00:00
#SBATCH --exclude=babel-13-37,babel-9-7,babel-0-27,babel-0-19,babel-4-37,babel-9-11,babel-13-33,babel-14-33,babel-8-17,babel-4-13,babel-14-1,babel-3-9,babel-4-9
#SBATCH --partition=general

source ${MINICONDA_PATH}
conda init bash
conda activate ${ENV_NAME}

MODEL=$1
TASK=$2
API_BASE=$3
OTHER_ARGS=$4

yeval \
    --model $MODEL \
    --task $TASK \
    --trust_remote_code \
    --api_base $API_BASE \
    $OTHER_ARGS
    # --output_path ./ 
