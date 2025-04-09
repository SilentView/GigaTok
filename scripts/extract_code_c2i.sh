# !/bin/bash
set -x
TORCH_RUN_PATH=${TORCH_RUN_PATH}

# The root for imagnet dataset
IMGNET_ROOT=${IMGNET_ROOT}
PROJECT_ROOT=${PROJECT_ROOT}

NV_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPUS=${GPUS:-${NV_GPUS}}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-55565}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
HOSTE_NODE_ADDR=${MASTER_ADDR}:${PORT}

BSZ=${BSZ:-1}

DEBUG=${DEBUG:-False}
DEBUG_FLAG=""
if [[ $DEBUG != False ]]; then
    DEBUG_FLAG="--debug"
fi

VQ_CKPT=${VQ_CKPT}
TOK_CONFIG=${TOK_CONFIG}

DATA_PATH="${IMGNET_ROOT}/ILSVRC2012_img_train/"
# the path must match */ten_crop/*
CODE_PATH=${CODE_PATH}

$TORCH_RUN_PATH \
--nnodes=$NNODES --nproc_per_node=$GPUS --node_rank=$NODE_RANK \
--master_port=$PORT --master_addr=$MASTER_ADDR \
autoregressive/train/extract_codes_c2i.py \
--data-path $DATA_PATH \
--code-path $CODE_PATH \
--vq-ckpt $VQ_CKPT_PATH \
--batch-size $BSZ \
--model-config ${TOK_CONFIG} \
--ten-crop \
--crop-range 1.1 \
--resume \
$DEBUG_FLAG \
"$@"