# !/bin/bash
set -x

NV_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPUS=${GPUS:-${NV_GPUS}}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-55565}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
HOSTE_NODE_ADDR=${MASTER_ADDR}:${PORT}
TIMESTAMP=$(date +%Y_%m_%d-%H_%M_%S)

TORCH_RUN_PATH=${TORCH_RUN_PATH:-torchrun}
export TOKENIZERS_PARALLELISM=true
# export TORCH_HOME=${ROOT}


${TORCH_RUN_PATH} \
--nnodes=$NNODES --nproc_per_node=$GPUS --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=$PORT \
autoregressive/sample/sample_c2i_cfg_search.py "$@"