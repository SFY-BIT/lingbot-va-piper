#!/bin/bash
#SBATCH --job-name=libero_finetune
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/group1/junjie008/lingbot-va/libero_finetune_%j.out
#SBATCH --error=/data/group1/junjie008/lingbot-va/libero_finetune_%j.err

set -euo pipefail
set -x

ROOT=/data/group1/junjie008/lingbot-va
RESULT_ROOT=/data/group1/junjie008/lingbot-va/libero_finetune_${SLURM_JOB_ID}

# Remote paths on junjie008.
# NOTE: wan_va/train.py will append "/transformer" internally, so MODEL_PATH
# should point to the checkpoint root, not the transformer subdirectory itself.
MODEL_PATH=/data/group1/junjie008/lingbot-finetune-ckpt/checkpoint_step_800
DATASET_PATH=/data/group1/junjie008/datasets/libero_10/0.0.0/libero_10_0.0.0_lerobot_part_0

# Training config from wan_va/configs/__init__.py
# Use the smoke config first to validate dataset initialization cheaply.
CONFIG_NAME=libero_smoke_train
MASTER_PORT=29501
NGPU=1

mkdir -p "${RESULT_ROOT}"

cd "${ROOT}"

echo "job id: ${SLURM_JOB_ID}"
hostname
pwd

set +u
source /data/group1/junjie008/miniconda3/bin/activate base
conda activate lingbot_pt
set -u

export TORCH_SHOW_CPP_STACKTRACES=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

# These environment variables are consumed by wan_va/configs/va_libero_cfg.py
# and wan_va/configs/va_libero_train_cfg.py.
export LINGBOT_VA_MODEL_PATH="${MODEL_PATH}"
export LIBERO_DATASET_PATH="${DATASET_PATH}"

export WANDB_API_KEY="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"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_TEAM_NAME="songfeiyang361-"
export WANDB_PROJECT="lingbot-va-libero"

which python || true
nvidia-smi || true

if [ ! -d "${LINGBOT_VA_MODEL_PATH}/transformer" ]; then
  echo "Model path is invalid: ${LINGBOT_VA_MODEL_PATH}" >&2
  exit 1
fi

if [ ! -d "${LIBERO_DATASET_PATH}" ]; then
  echo "Dataset path is invalid: ${LIBERO_DATASET_PATH}" >&2
  exit 1
fi

python -m torch.distributed.run \
  --nproc_per_node="${NGPU}" \
  --master_port "${MASTER_PORT}" \
  -m wan_va.train \
  --config-name "${CONFIG_NAME}" \
  --save-root "${RESULT_ROOT}" \
  > "${RESULT_ROOT}/train.log" 2>&1

echo "LingBot finetune finished."
echo "Results saved under: ${RESULT_ROOT}"
