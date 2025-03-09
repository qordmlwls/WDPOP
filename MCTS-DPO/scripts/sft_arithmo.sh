#!/usr/bin/env bash

##############################################################################
# Example script to run Supervised Fine-Tuning (SFT) on Mistral-7B with DeepSpeed
# referencing the 'mcts_rl.finetune.deepspeed' entry-point and the code you shared.
#-----------------------------------------------------------------------------
# USAGE:
#   bash run_sft.sh [GPU_IDS]
#
# Example:
#   bash run_sft.sh 0,1,2,3
#
# (Make sure you have the environment correctly set up, with the codebase in PYTHONPATH.)
##############################################################################

# Fail on any error
set -e

# Print executed commands
set -x

# Which GPUs to use; default is "0,1,2,3" if not given
GPU_IDS="${1:-0,1,2,3}"

# Number of total processes = number of GPUs you want to use
NUM_GPUS=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')

# Where to store output (checkpoints, logs, etc.)
OUTPUT_DIR="outputs/sft_mistral7B"
mkdir -p "${OUTPUT_DIR}"

WANDB_API_KEY="8f05ac03be73a9566c05daa8a2dd7a0dc5720534"
# If you use wandb and have WANDB_API_KEY exported, it will run in online mode;
# otherwise, it falls back to offline.
if [[ -z "${WANDB_API_KEY}" ]]; then
  export WANDB_MODE="offline"
else
  export WANDB_MODE="online"
fi

# Some people prefer to randomize the master port for distributed runs:
MASTER_PORT=$((10000 + $RANDOM % 55535))

# Basic training hyperparameters for SFT (from your paper/appendix)
LR=5e-6
BATCH_PER_GPU=32       # so 32 × 4 GPUs = 128 total
EPOCHS=1               # set as needed
MAX_LEN=512
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.03
WEIGHT_DECAY=1e-6

# If you prefer half-precision or BF16, set these flags accordingly.
FP16="true"
BF16="false"

# Construct the DeepSpeed command:
deepspeed --include="localhost:${GPU_IDS}" \
  --master_port "${MASTER_PORT}" \
  --module mcts_rl.finetune.deepspeed \
  --model_name_or_path "mistralai/Mistral-7B-v0.1" \
  --max_length "${MAX_LEN}" \
  --trust_remote_code false \
  \
  --train_datasets "Arithmo" \
  --eval_datasets "some_eval_dataset" \
  --eval_strategy "epoch" \
  --need_eval \
  \
  --per_device_train_batch_size "${BATCH_PER_GPU}" \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --epochs "${EPOCHS}" \
  \
  --lr "${LR}" \
  --lr_scheduler_type "${LR_SCHEDULER}" \
  --lr_warmup_ratio "${WARMUP_RATIO}" \
  --weight_decay "${WEIGHT_DECAY}" \
  \
  --seed 42 \
  --fp16 "${FP16}" \
  --bf16 "${BF16}" \
  --tf32 true \
  \
  --zero_stage 0 \
  --offload "none" \
  \
  --output_dir "${OUTPUT_DIR}" \
  --save_interval 5000 \
  \
  --log_type "wandb" \
  --log_project "mistral_sft_project" \
  --log_run_name "sft_mistral7B_run" \
  \
  --deepspeed_config "deepspeed_config.json"