#!/bin/bash

# This script will handle training and evaluation for the linear probing of LM models


########################################
# Tokenizer related parameters
########################################
# the absolute path to the checkpoint of the tokenizer model
VQ_CKPT=${VQ_CKPT:-"None"}

VQ_EXP_DIR=${VQ_EXP_DIR}
TOK_CONFIG=${TOK_CONFIG}
TOK_EPOCH=${TOK_EPOCH:-"100"}
TOK_BSZ=${TOK_BSZ:-"128"}
TOK_EARLY_STOP_ITER=${TOK_EARLY_STOP_ITER:-"None"}



########################################
# LM related parameters
########################################
GPT_CKPT=${GPT_CKPT}
LM_EXP_DIR=${LM_EXP_DIR}
LM_EPOCH=${LM_EPOCH:-300}
LM_BSZ=${LM_BSZ:-256}
GPT_MODEL=${GPT_MODEL:-"GPT-B"}
PRECISION=${PRECISION:-"none"}
LM_PRECISION=${LM_PRECISION:-${PRECISION}}
TOK_PRECISION=${TOK_PRECISION:-${PRECISION}}
LM_FRACT_DECAY=${LM_FRACT_DECAY:-0.2}

USE_QK_NORM=${USE_QK_NORM:-"False"}
if [[ "${USE_QK_NORM}" == "True" ]]; then
    QK_NORM_FLAG="--qk-norm"
    echo "Using QK Norm for GPT training"
else
    QK_NORM_FLAG=""
fi


###############################
# Linear Probe related
###############################
LM_LIN_EXP_DIR=${LM_LIN_EXP_DIR}
TOK_LIN_EXP_DIR=${TOK_LIN_EXP_DIR}
NUM_CODE=${NUM_CODE:-256}

# we mostly use 128*64 global batch size
CKPT_EPOCH=${CKPT_EPOCH:-5}
LIN_BSZ=${LIN_BSZ:-128}

PROJECT_ROOT=${PROJECT_ROOT}

# The dir for imgnet dataset, like .../imagenet/
IMGNET_ROOT=${IMGNET_ROOT}

if [[ $VQ_CKPT == "None" ]]; then
    if [ "$TOK_EARLY_STOP_ITER" != "None" ];then
        TOK_ITER=$TOK_EARLY_STOP_ITER
    else
        EPOCH_TO_ITER_FACTOR=$((256 * 5000 / TOK_BSZ))
        TOK_ITER=$((TOK_EPOCH * EPOCH_TO_ITER_FACTOR))
    fi

    printf -v TOK_ITER "%07d" "$TOK_ITER"
    echo "$TOK_ITER"
    VQ_CKPT=${PROJECT_ROOT}/results/tokenizers/vq/${VQ_EXP_DIR}/checkpoints/${TOK_ITER}.pt
fi


if [[ $GPT_CKPT == "None" ]]; then
    # WSD training management for LM, the decay ratio is fixed as 0.2
    BSZ_FACTOR=5000 # The iteraions for 1 epoch when trained with 256 batch size
    LM_CONST_EPOCH=$((LM_EPOCH * 4 / 5))
    EPOCH_TO_ITER_FACTOR=$((256 * BSZ_FACTOR / LM_BSZ ))
    # 256 * BSZ_FACTOR / batch size  * epoch = total_steps for ImageNet
    LM_ITER=$((LM_EPOCH * EPOCH_TO_ITER_FACTOR))
    LM_STOP_ITER=${LM_STOP_ITER:-$LM_ITER}

    # Format TOK_ITER and store it back in TOK_ITER
    printf -v LM_ITER "%07d" "$LM_ITER"
    echo "$LM_ITER"

    printf -v LM_STOP_ITER "%07d" "$LM_STOP_ITER"
    echo "LM_STOP_ITER:${LM_STOP_ITER}"

    if [[ $LM_STOP_ITER < $LM_ITER ]]; then
        # early stop at constant learning rate stage
        GPT_CKPT=results/gpt/${LM_EXP_DIR}/checkpoints/${LM_STOP_ITER}.pt
    else
        GPT_CKPT=results/gpt/${LM_EXP_DIR}/cd_records/cd_fract_${LM_FRACT_DECAY}_to_${LM_ITER}/checkpoints/${LM_ITER}.pt
    fi
fi



bash scripts/train_lin_probe_gpt.sh \
    --save-path ${PROJECT_ROOT}/results/lin_probe_gpt/ \
    --data-root ${IMGNET_ROOT} \
    --image-size 256 \
    --model-config ${TOK_CONFIG} \
    --gpt-model ${GPT_MODEL} \
    --epochs 90 \
    --blr 0.1 \
    --sub-exp-dir ${LM_LIN_EXP_DIR} \
    --batch-size $LIN_BSZ \
    --mixed-precision ${LM_PRECISION} \
    --num-code ${NUM_CODE} \
    --vq-ckpt $VQ_CKPT \
    --ckpt-epoch ${CKPT_EPOCH} \
    --gpt-ckpt ${GPT_CKPT} \
    $QK_NORM_FLAG

bash scripts/train_lin_probe_gpt.sh \
    --eval \
    --save-path ${PROJECT_ROOT}/results/lin_probe_gpt/ \
    --data-root ${IMGNET_ROOT} \
    --image-size 256 \
    --model-config ${TOK_CONFIG} \
    --gpt-model ${GPT_MODEL} \
    --epochs 90 \
    --blr 0.1 \
    --sub-exp-dir ${LM_LIN_EXP_DIR} \
    --batch-size ${LIN_BSZ} \
    --mixed-precision ${LM_PRECISION} \
    --num-code ${NUM_CODE} \
    --vq-ckpt ${VQ_CKPT} \
    --gpt-ckpt ${GPT_CKPT} \
    --ckpt-epoch ${CKPT_EPOCH} \
    --lin-probe-ckpt ${PROJECT_ROOT}/results/lin_probe_gpt/${LM_LIN_EXP_DIR}/checkpoints/0089.pt \
    $QK_NORM_FLAG