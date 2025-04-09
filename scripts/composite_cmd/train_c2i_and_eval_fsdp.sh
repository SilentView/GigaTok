# This is the script for training and evaluating a LM using FSDP/SDP
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


########################################
# LM related parameters
########################################
LM_EXP_DIR=${LM_EXP_DIR}
LM_EPOCH=${LM_EPOCH:-300}
LM_BSZ=${LM_BSZ:-256}
GPT_MODEL=${GPT_MODEL:-"GPT-B"}
WARM_ITER=${WARM_ITER:-5000}
LR=${LR:-"1e-4"}
PRECISION=${PRECISION:-"fp16"}
EVAL_BATCH_PER_GPU=${EVAL_BATCH_PER_GPU:-16}
FRACT_DECAY=${FRACT_DECAY:-0.2}
CKPT_EVERY=${CKPT_EVERY:-5000}


DATA_PARALLEL=${DATA_PARALLEL:-sdp}

USE_QK_NORM=${USE_QK_NORM:-"False"}

if [[ "${USE_QK_NORM}" == "True" ]]; then
    QK_NORM_FLAG="--qk-norm"
else
    QK_NORM_FLAG=""
fi


PROJECT_ROOT=${PROJECT_ROOT}

# The dir for imgnet dataset, like .../imagenet/
IMGNET_ROOT=${IMGNET_ROOT}


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
fi


# For LM traning. Currently for fixed WSD leraning rate with 0.2 decay ratio
# TODO: support other learning rate scheduler
BSZ_FACTOR=5000 # The iteraions for 1 epoch when trained with 256 batch size
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
bash scripts/train_c2i_fsdp.sh \
--save-path ${PROJECT_ROOT}/results/gpt/ \
--data-path $DATAPATH \
--code-path $CODEPATH \
--dataset ${DATASET} \
--image-size 256 \
--tok-config ${TOK_CONFIG} \
--mixed-precision ${PRECISION} \
--gpt-model ${GPT_MODEL} \
--vq-ckpt $VQ_CKPT \
--sub-exp-dir ${LM_EXP_DIR} \
--lr-scheduler wsd \
--fract-decay ${FRACT_DECAY} \
--warmup ${WARM_ITER} \
--lr ${LR} \
--ckpt-every ${CKPT_EVERY} \
--global-batch-size ${LM_BSZ} \
--iterations $LM_ITER \
--early-stop-iter $LM_STOP_ITER \
--data-parallel $DATA_PARALLEL \
$QK_NORM_FLAG


if [[ $LM_STOP_ITER < $LM_ITER ]]; then
    # early stop at constant learning rate stage
    GPT_CKPT=results/gpt/${LM_EXP_DIR}/checkpoints/${LM_STOP_ITER}.pt
    LM_ITER=$LM_STOP_ITER
else
    GPT_CKPT=results/gpt/${LM_EXP_DIR}/cd_records/cd_fract_${FRACT_DECAY}_to_${LM_ITER}/checkpoints/${LM_ITER}.pt
fi


VAL_PATH=${VAL_PATH:-${PROJECT_ROOT}/results/reconstructions/img_data}
GT_NPZ_PATH=${GT_NPZ_PATH:-${PROJECT_ROOT}/VIRTUAL_imagenet256_labeled.npz}
# should use the specific evaluation environment from ADM
EVAL_PYTHON_PATH=${EVAL_PYTHON_PATH:-python}

## Evaluation Script for gFID
bash scripts/sample_c2i_search_cfg.sh \
--quant-way=vq \
--image-size=256 \
--sample-dir=${PROJECT_ROOT}/results/gpt/quan_eval/${LM_EXP_DIR}/${LM_ITER} \
--eval-wandb-dir ${PROJECT_ROOT}/results/gpt/quan_eval/${LM_EXP_DIR}/ \
--vq-ckpt $VQ_CKPT \
--tok-config ${TOK_CONFIG} \
--gpt-model ${GPT_MODEL} \
--cfg-schedule step \
--step-start-ratio $CFG_START_RATIO \
--search \
--gpt-ckpt $GPT_CKPT \
--per-proc-batch-size $EVAL_BATCH_PER_GPU \
--precision ${PRECISION} \
--wandb \
--clear-cache \
--eval-python-path ${EVAL_PYTHON_PATH} \
--gt-npz-path ${GT_NPZ_PATH} \
$QK_NORM_FLAG

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
$QK_NORM_FLAG