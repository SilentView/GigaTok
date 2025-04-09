# !/bin/bash
set -x

TORCH_RUN_PATH=${TORCH_RUN_PATH}

NV_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPUS=${GPUS:-${NV_GPUS}}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-55565}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
HOSTE_NODE_ADDR=${MASTER_ADDR}:${PORT}

$TORCH_RUN_PATH \
--nnodes=$NNODES --nproc_per_node=$GPUS --node_rank=$NODE_RANK \
--master_port=$PORT --master_addr=$MASTER_ADDR \
tokenizer/tokenizer_image/reconstruction_ddp.py \
"$@"