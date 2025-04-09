########################################
# Tokenizer related parameters
########################################
# the absolute path to the checkpoint of the tokenizer model
VQ_CKPT=${VQ_CKPT:-"None"}

VQ_EXP_DIR=${VQ_EXP_DIR}
TOK_CONFIG=${TOK_CONFIG}
TOK_EPOCH=${TOK_EPOCH:-"50"}
TOK_BSZ=${TOK_BSZ:-"128"}
FIX_DIM=${FIX_DIM:-"False"}
FROM_DECODER=${FROM_DECODER:-"False"}
TOK_EARLY_STOP_ITER=${TOK_EARLY_STOP_ITER:-"None"}


###############################
# Linear Probe related
###############################
PRECISION=${PRECISION:-"none"}
TOK_PRECISION=${TOK_PRECISION:-${PRECISION}}
TOK_LIN_EXP_DIR=${TOK_LIN_EXP_DIR}
NUM_CODE=${NUM_CODE:-256}
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
    VQ_CKPT=${PROJECT_ROOT}/results/tokenizers/vq/${VQ_EXP_DIR}/checkpoints/${TOK_ITER}.pt \
fi


if [[ "${FIX_DIM}" == "True" ]]; then
    FIX_DIM_FLAG="--fix-dim"
    echo "Using Fixed Dimenstion for tokenizer linear probe"
else
    FIX_DIM_FLAG=""
fi


if [[ "${FROM_DECODER}" == "True" ]]; then
    FROM_DECODER_FLAG="--from-decoder"
    echo "Using from decoder for tokenizer linear probe"
else
    FROM_DECODER_FLAG=""
fi



## For VQ
bash scripts/train_lin_probe.sh \
    --save-path ${PROJECT_ROOT}/results/lin_probe/ \
    --data-root ${IMGNET_ROOT} \
    --image-size 256 \
    --model-config ${TOK_CONFIG} \
    --epochs 90 \
    --blr 0.1 \
    --sub-exp-dir ${TOK_LIN_EXP_DIR} \
    --batch-size ${LIN_BSZ} \
    --mixed-precision ${TOK_PRECISION} \
    --num-code ${NUM_CODE} \
    --ckpt-epoch ${CKPT_EPOCH} \
    --vq-ckpt $VQ_CKPT \
    $FIX_DIM_FLAG \
    $FROM_DECODER_FLAG


bash scripts/train_lin_probe.sh \
    --save-path ${PROJECT_ROOT}/results/lin_probe/ \
    --data-root ${IMGNET_ROOT} \
    --image-size 256 \
    --model-config ${TOK_CONFIG} \
    --epochs 90 \
    --blr 0.1 \
    --sub-exp-dir ${TOK_LIN_EXP_DIR} \
    --batch-size ${LIN_BSZ} \
    --mixed-precision ${TOK_PRECISION} \
    --num-code ${NUM_CODE} \
    --vq-ckpt $VQ_CKPT \
    --ckpt-epoch ${CKPT_EPOCH} \
    --eval \
    --lin-probe-ckpt ${PROJECT_ROOT}/results/lin_probe/${TOK_LIN_EXP_DIR}/checkpoints/0089.pt \
    $FIX_DIM_FLAG \
    $FROM_DECODER_FLAG
