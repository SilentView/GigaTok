# !/bin/bash
set -x

# path for the root of the dataset folder
DATA_ROOT=${DATA_ROOT}

# the absolute path for the torchrun, for example, .../miniconda3/envs/gigatok/bin/torchrun
# or it can be just torchrun if the PATH is set or the virtual env is activated
TORCH_RUN_PATH=${TORCH_RUN_PATH:-torchrun}

# The root for imagnet dataset
IMGNET_ROOT=${IMGNET_ROOT}

PROJECT_ROOT="/mnt/bn/data-aigc-video/tianwei/code/Tokenizer1D"
PYPATH=${ROOT}/miniconda3/envs/1d-tok/bin/python

NV_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPUS=${GPUS:-${NV_GPUS}}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-55565}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
HOSTE_NODE_ADDR=${MASTER_ADDR}:${PORT}

# the name of the folder where the vq tokenizer training related results are stored
# note that this is just name, not path
VQ_EXP_DIR=${VQ_EXP_DIR}

# the name of the config files for the vq tokenizer (not path)
TOK_CONFIG=${TOK_CONFIG}

# specify the training iterations of the tokenizer
TOK_ITER=${TOK_ITER}


BSZ=${BSZ:-1}
DEBUG=${DEBUG:-False}

printf -v TOK_ITER "%07d" "$TOK_ITER"
echo "$TOK_ITER"

# currently there must be flip and ten_crop in the code_path
# this may be improved latter


# $TORCH_RUN_PATH \
# --nnodes=$NNODES --nproc_per_node=$GPUS --node_rank=$NODE_RANK \
# --master_port=$PORT --master_addr=$MASTER_ADDR \


$PYPATH \
autoregressive/train/check_codes_c2i.py \
--data-path ${IMGNET_ROOT}/ILSVRC2012_img_train/ \
--code-path ${DATA_ROOT}/imgnet_code/${VQ_EXP_DIR}/ten_crop/ \
--save-path ${DATA_ROOT}/imgnet_code/${VQ_EXP_DIR}/ten_crop/debug \
--vq-ckpt ${PROJECT_ROOT}/results/tokenizers/vq/${VQ_EXP_DIR}/checkpoints/${TOK_ITER}.pt \
--batch-size $BSZ \
--tok-config ${TOK_CONFIG} \
"$@"