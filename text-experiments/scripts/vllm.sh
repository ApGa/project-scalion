#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --output=text-experiments/slurm-out/vllm/host-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=48GB
#SBATCH --time 2-00:00:00
#SBATCH --exclude=babel-13-37,babel-9-7,babel-0-27,babel-0-19,babel-4-37,babel-9-11,babel-13-33,babel-14-33,babel-8-17,babel-4-13,babel-14-1,babel-3-9,babel-4-9
#SBATCH --partition=general

usage() {
  echo "Usage: $0 [--model_id=STRING] [--port=NUMBER] [--tensor_parallel_size=NUMBER] [--gpu_memory_utilization=NUMBER] [--help]"
  echo
  echo "Options:"
  echo "  --model_id=STRING  Huggingface model ID (e.g., allenai/OLMo-2-1124-13B-SFT)"
  echo "  --port=NUMBER      Port number for the server (default: 8084)"
  echo "  --tensor_parallel_size=NUMBER Number of GPUs for tensor parallelism (default: 2)"
  echo "  --gpu_memory_utilization=NUMBER GPU memory utilization (default: 0.8)"
  echo "  --help            Display this help message"
  exit 1
}

# Default values
PORT=8084
tensor_parallel_size=2
gpu_memory_utilization=0.8

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_id=*)
      model_id="${1#*=}"
      shift
      ;;
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --tensor_parallel_size=*)
      tensor_parallel_size="${1#*=}"
      shift
      ;;
    --gpu_memory_utilization=*)
      gpu_memory_utilization="${1#*=}"
      shift
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Validate required arguments
if [ -z "$model_id" ]; then
  echo "Error: --model_id is required"
  usage
fi

echo "Model ID: $model_id"
echo "Port: $PORT"
echo "Tensor parallel size: $tensor_parallel_size"
echo "GPU memory utilization: $gpu_memory_utilization"

set -a 
source text-experiments/scripts/configs/.env
set +a

source ${MINICONDA_PATH}
conda init bash
conda activate ${ENV_NAME}

huggingface-cli login --token ${HF_TOKEN}

export VLLM_LOGGING_LEVEL=ERROR
export NCCL_P2P_DISABLE=1

echo "$HOSTNAME:$PORT"

# lowering GPU utilizaition to test
if ss -tulwn | grep -q ":$PORT "; then
    echo "Port $PORT is already in use. Exiting..."
    exit 1
else
    python -m vllm.entrypoints.openai.api_server \
        --gpu_memory_utilization $gpu_memory_utilization \
        --model $model_id \
        --port $PORT \
        --tensor-parallel-size $tensor_parallel_size \
        --device cuda \
        --enable-chunked-prefill \
        --disable-log-requests \
        --download-dir ${HF_HOME} # Either shared model cache on babel or your own directory
fi
echo $PORT