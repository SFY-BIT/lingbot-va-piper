#!/usr/bin/bash

set -euo pipefail
set -x

umask 007

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# GPU 数量。默认单卡，迁移到新机器后可通过环境变量覆盖。
NGPU=${NGPU:-"1"}
# 分布式训练通信端口。
MASTER_PORT=${MASTER_PORT:-"29501"}
# 只打印哪个 rank 的日志；设为 all 可打印所有 rank。
LOG_RANK=${LOG_RANK:-"0"}
# torchft 相关地址，保留原脚本默认值。
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
# LIBERO 完整训练默认配置名。
CONFIG_NAME=${CONFIG_NAME:-"libero_train"}
# 模型和数据集路径允许在新机器上通过环境变量覆盖。
LINGBOT_VA_MODEL_PATH=${LINGBOT_VA_MODEL_PATH:-"/mnt/hdd/sfy/models/lingbot-va-base"}
LIBERO_DATASET_PATH=${LIBERO_DATASET_PATH:-"/mnt/hdd/sfy/datasets/libero_spatial_lerobot"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# 只有在真的需要 WandB 时再填写这些值。
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_TEAM_NAME="${WANDB_TEAM_NAME:-songfeiyang361-}"
export WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-libero}"
export LINGBOT_VA_MODEL_PATH
export LIBERO_DATASET_PATH

if [ ! -d "${LINGBOT_VA_MODEL_PATH}/transformer" ]; then
    echo "Model path is invalid: ${LINGBOT_VA_MODEL_PATH}" >&2
    exit 1
fi

if [ ! -d "${LIBERO_DATASET_PATH}" ]; then
    echo "Dataset path is invalid: ${LIBERO_DATASET_PATH}" >&2
    exit 1
fi

num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

export TOKENIZERS_PARALLELISM=false

rank_filter_args=()
if [ "${log_rank}" != "all" ]; then
    rank_filter_args+=(--local-ranks-filter="${log_rank}")
fi

# 这里仍然调用 train.py，真正决定数据集路径的是对应 config。
PYTORCH_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    "${rank_filter_args[@]}" \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
