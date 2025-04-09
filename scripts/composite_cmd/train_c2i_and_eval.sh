#!/bin/bash

# This is the script for training and evaluating a LM
# the evaluation include:
#   - validation gFID
#   - validation cross entropy loss

########################################
# Tokenizer related parameters
########################################
# the absolute path to the checkpoint of the tokenizer model
VQ_CKPT=${VQ_CKPT:-"None"}


VQ_EXP_DIR=${VQ_EXP_DIR}
TOK_CONFIG=${TOK_CONFIG}
TOK_EPOCH=${TOK_EPOCH:-"50"}
TOK_BSZ=${TOK_BSZ:-"128"}
TOK_EARLY_STOP_ITER=${TOK_EARLY_STOP_ITER:-"None"}


########################################
# LM related parameters
########################################
GPT_2D=${GPT_2D:-"False"}
LM_EXP_DIR=${LM_EXP_DIR}
LM_EPOCH=${LM_EPOCH:-300}
LM_BSZ=${LM_BSZ:-256}
FRACT_DECAY=${FRACT_DECAY:-0.2}
GPT_MODEL=${GPT_MODEL:-"GPT-B"}
WARM_ITER=${WARM_ITER:-5000}
LR=${LR:-"1e-4"}
PRECISION=${PRECISION:-"fp16"}
CFG_SCHEDULE=${CFG_SCHEDULE:-"step"}
EVAL_BATCH_PER_GPU=${EVAL_BATCH_PER_GPU:-32}
DATASET=${DATASET:-"imagenet"}
CKPT_EVERY=${CKPT_EVERY:-5000}



USE_QK_NORM=${USE_QK_NORM:-"False"}
USE_ADALN=${USE_ADALN:-"False"}

if [[ "${USE_QK_NORM}" == "True" ]]; then
    QK_NORM_FLAG="--qk-norm"
    echo "Using QK Norm for GPT training"
else
    QK_NORM_FLAG=""
fi


if [[ ${GPT_2D} != "False" ]]; then
    GPT_2D_FLAG="--gpt-2d"
    echo "Using GPT 2D (2d rope) for GPT training"
else
    GPT_2D_FLAG=""
fi

if [[ ${USE_ADALN} != "False" ]]; then
    echo $USE_ADALN
    USE_ADALN_FLAG="--adaLN"
    echo "Using AdaLN for GPT training"
else
    USE_ADALN_FLAG=""
fi


PROJECT_ROOT=${PROJECT_ROOT}
# The dir for imgnet dataset, like .../imagenet/
IMGNET_ROOT=${IMGNET_ROOT}
VAL_PATH=${VAL_PATH:}



if [[ $VQ_CKPT == "None" ]]; then
    # we use the total training epoch to manage the training process for ImageNet
    EPOCH_TO_ITER_FACTOR=$((256 * 5000 / TOK_BSZ))
    TOK_ITER=$((TOK_EPOCH * EPOCH_TO_ITER_FACTOR))

    if [ "$TOK_EARLY_STOP_ITER" != "None" ];then
        TOK_ITER=$TOK_EARLY_STOP_ITER
    fi

    # Format TOK_ITER and store it back in TOK_ITER
    printf -v TOK_ITER "%07d" "$TOK_ITER"
    echo "$TOK_ITER"
    VQ_CKPT=results/tokenizers/vq/${VQ_EXP_DIR}/checkpoints/${TOK_ITER}.pt \
else
    VQ_CKPT=${VQ_CKPT}
fi

# For LM traning. Currently for fixed WSD leraning rate with 0.2 decay ratio
# TODO: support other learning rate scheduler
BSZ_FACTOR=5000  # The iteraions for 1 epoch when trained with 256 batch size
LM_CONST_EPOCH=$((LM_EPOCH * 4 / 5))
EPOCH_TO_ITER_FACTOR=$((256 * BSZ_FACTOR / LM_BSZ))
# 256 * BSZ_FACTOR / batch size  * epoch = total_steps for ImageNet
LM_ITER=$((LM_EPOCH * EPOCH_TO_ITER_FACTOR))
# The early stop iteration. If not specified, same as total training iteration
LM_STOP_ITER=${LM_STOP_ITER:-$LM_ITER}
CFG_START_RATIO=${CFG_START_RATIO:-0.18}

# Format TOK_ITER and store it back in TOK_ITER
printf -v LM_ITER "%07d" "$LM_ITER"
echo "$LM_ITER"

printf -v LM_STOP_ITER "%07d" "$LM_STOP_ITER"
echo "LM_STOP_ITER:${LM_STOP_ITER}"

DATAPATH=${DATAPATH:-${IMGNET_ROOT}/ILSVRC2012_img_train/}
# The path to pre-cached latent codes.
# can be something like ..../ten_crop/
DATASET=${DATASET:-imagenet}
CODEPATH=${CODEPATH:-None}


## Training Script
bash scripts/train_c2i.sh \
--save-path ${PROJECT_ROOT}/results/gpt/ \
--data-path $DATAPATH \
--code-path $CODEPATH \
--dataset ${DATASET} \
--image-size 256 \
--tok-config ${TOK_CONFIG} \
--mixed-precision ${PRECISION} \
--gpt-model ${GPT_MODEL} \
--vq-ckpt ${VQ_CKPT} \
--sub-exp-dir ${LM_EXP_DIR} \
--lr-scheduler wsd \
--warmup ${WARM_ITER} \
--lr ${LR} \
--ckpt-every ${CKPT_EVERY} \
--global-batch-size ${LM_BSZ} \
--fract-decay ${FRACT_DECAY} \
--iterations $LM_ITER \
--early-stop-iter $LM_STOP_ITER \
$QK_NORM_FLAG \
$GPT_2D_FLAG \
$USE_ADALN_FLAG

if [[ $LM_STOP_ITER < $LM_ITER ]]; then
    # early stop at constant learning rate stage
    GPT_CKPT=results/gpt/${LM_EXP_DIR}/checkpoints/${LM_STOP_ITER}.pt
    LM_ITER=$LM_STOP_ITER
else
    GPT_CKPT=results/gpt/${LM_EXP_DIR}/cd_records/cd_fract_${FRACT_DECAY}_to_${LM_ITER}/checkpoints/${LM_ITER}.pt
fi

## Evaluation Script for gFID
bash scripts/sample_c2i_search_cfg.sh \
--search \
--quant-way=vq \
--image-size=256 \
--sample-dir=${PROJECT_ROOT}/results/gpt/quan_eval/${LM_EXP_DIR}/${LM_ITER} \
--vq-ckpt $VQ_CKPT \
--tok-config ${TOK_CONFIG} \
--gpt-model ${GPT_MODEL} \
--cfg-schedule $CFG_SCHEDULE \
--step-start-ratio $CFG_START_RATIO \
--gpt-ckpt $GPT_CKPT \
--per-proc-batch-size $EVAL_BATCH_PER_GPU \
--precision ${PRECISION} \
--clear-cache \
$QK_NORM_FLAG \
$GPT_2D_FLAG \
$USE_ADALN_FLAG

## Evaluation script for validation loss
bash scripts/val_loss_c2i.sh \
--data-path=${IMGNET_ROOT}/ILSVRC2012_img_val/ \
--quant-way=vq \
--image-size=256 \
--vq-ckpt $VQ_CKPT \
--tok-config ${TOK_CONFIG} \
--gpt-model ${GPT_MODEL} \
--per-proc-batch-size $EVAL_BATCH_PER_GPU \
--gpt-ckpt $GPT_CKPT \
--precision ${PRECISION} \
$QK_NORM_FLAG \
$GPT_2D_FLAG \
$USE_ADALN_FLAG