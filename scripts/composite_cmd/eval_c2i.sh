# This is the script for evaluating a LM
# If the gpt checkpoint is not specified, 
#   then the path of ckpt will be decided assuming training uses WSD learning rate
# the evaluation include:
#   - validation gFID (with fixed given cfg setting)
#   - validation cross entropy loss


########################################
# Tokenizer related parameters
########################################

# the absolute path to the checkpoint of the tokenizer model
VQ_CKPT=${VQ_CKPT:-"None"}
# The absolute path to the GPT model checkpoint.
GPT_CKPT=${GPT_CKPT:-"None"}

# The name of the experiment directory for the VQ model.
VQ_EXP_DIR=${VQ_EXP_DIR}
# The configuration file for the tokenizer.
TOK_CONFIG=${TOK_CONFIG}
# The total number of training epochs for the tokenizer.
TOK_EPOCH=${TOK_EPOCH:-"100"}
# The batch size for the tokenizer training.
TOK_BSZ=${TOK_BSZ:-"128"}
# The iteration number at which to stop the tokenizer training early and start the evaluation.
TOK_EARLY_STOP_ITER=${TOK_EARLY_STOP_ITER:-"None"}
# The frequency (in iterations) at which to save checkpoints.
CKPT_EVERY=${CKPT_EVERY:-5000}


########################################
# LM related parameters
########################################

# The name of the experiment directory for the LM model.
LM_EXP_DIR=${LM_EXP_DIR}
# The total number of training epochs for the LM.
LM_EPOCH=${LM_EPOCH:-300}
# The batch size for the LM training.
LM_BSZ=${LM_BSZ:-256}
# The fraction decay value for the LM training.
FRACT_DECAY=${FRACT_DECAY:-0.2}
# The name of the GPT model.
GPT_MODEL=${GPT_MODEL:-"GPT-B"}
# The number of warm-up iterations for the learning rate scheduler.
WARM_ITER=${WARM_ITER:-5000}
# The learning rate for the LM training.
LR=${LR:-"1e-4"}
# The precision for the training and evaluation.
PRECISION=${PRECISION:-"fp16"}
# The configuration schedule for the guidance scale.
CFG_SCHEDULE=${CFG_SCHEDULE:-"step"}
# The configuration scale for the guidance.
CFG_SCALE=${CFG_SCALE:-1.0}
# the starting ratio for the step function for CFG scheduling
CFG_START_RATIO=${CFG_START_RATIO:-0.18}
# The top-k value for sampling.
TOP_K=${TOP_K:-0}
# The batch size per GPU for evaluation.
EVAL_BATCH_PER_GPU=${EVAL_BATCH_PER_GPU:-16}
# Whether to only perform validation loss calculation.
VAL_ONLY=${VAL_ONLY:-"False"}

USE_QK_NORM=${USE_QK_NORM:-"False"}
if [[ "${USE_QK_NORM}" == "True" ]]; then
    QK_NORM_FLAG="--qk-norm"
else
    QK_NORM_FLAG=""
fi

# path to the project. Like .../GigaTok
PROJECT_ROOT=${PROJECT_ROOT}
# path for the imagnet 50k validation dataset. Like .../ILSVRC2012_img_val
VAL_PATH=${VAL_PATH}

if [[ $VQ_CKPT == "None" ]]; then
    if [ "$TOK_EARLY_STOP_ITER" != "None" ];then
        TOK_ITER=$TOK_EARLY_STOP_ITER
    else
        # we use the total training epoch to manage the training process for ImageNet
        # But for total training iteration, we will accurately control with iteration numbers.
        EPOCH_TO_ITER_FACTOR=$((256 * 5000 / TOK_BSZ))
        TOK_ITER=$((TOK_EPOCH * EPOCH_TO_ITER_FACTOR))
        # Format TOK_ITER and store it back in TOK_ITER
        printf -v TOK_ITER "%07d" "$TOK_ITER"
        echo "TOK_ITER: $TOK_ITER"
    fi
    VQ_CKPT=results/tokenizers/vq/${VQ_EXP_DIR}/checkpoints/${TOK_ITER}.pt \
fi





if [[ $GPT_CKPT == "None" ]]; then
    # WSD training management for LM, the decay ratio is fixed as 0.2
    BSZ_FACTOR=5000
    LM_CONST_EPOCH=$((LM_EPOCH * 4 / 5))
    EPOCH_TO_ITER_FACTOR=$((256 * 5000 / LM_BSZ))
    # 256 * 5000 / batch size  * epoch = total_steps for ImageNet
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
        LM_ITER=$LM_STOP_ITER
    else
        GPT_CKPT=results/gpt/${LM_EXP_DIR}/cd_records/cd_fract_${FRACT_DECAY}_to_${LM_ITER}/checkpoints/${LM_ITER}.pt
    fi
fi



if [[ $VAL_ONLY == "False" ]]; then
## Evaluation Script for gFID
## But there will be no searching for this setting
bash scripts/sample_c2i_search_cfg.sh \
--quant-way=vq \
--image-size=256 \
--sample-dir=${PROJECT_ROOT}/results/gpt/v4/quan_eval/${LM_EXP_DIR}/${LM_ITER} \
--eval-wandb-dir ${PROJECT_ROOT}/results/gpt/quan_eval/${LM_EXP_DIR}/ \
--vq-ckpt ${VQ_CKPT} \
--tok-config ${TOK_CONFIG} \
--gpt-model ${GPT_MODEL} \
--cfg-schedule $CFG_SCHEDULE \
--cfg-scale $CFG_SCALE \
--top-k ${TOP_K} \
--step-start-ratio $CFG_START_RATIO \
--gpt-ckpt $GPT_CKPT \
--per-proc-batch-size $EVAL_BATCH_PER_GPU \
--precision ${PRECISION} \
--wandb \
--clear-cache \
$QK_NORM_FLAG \

fi


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