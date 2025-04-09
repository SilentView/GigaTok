#!/bin/bash


# This is the entry script for training and evaluate a Tokenizer
# the evaluation include:
#   - validation rFID, LPIPS, PNSR, ...
#   - small GPT model trained for 50 epochs with constant learning rate
#       - gFID (constant cfg schedule)
#       - val cross entropy loss
VQ_EXP_DIR=${VQ_EXP_DIR}
TOK_CONFIG=${TOK_CONFIG}
TOK_EPOCH=${TOK_EPOCH:-"50"}
TOK_BSZ=${TOK_BSZ:-"128"}
CKPT_EVERY=${CKPT_EVERY:-5000}
FRACT_DECAY=${FRACT_DECAY:-0.2}

JSON_PATH=${JSON_PATH:-"None"}  # (deprecated)
DATASET=${DATASET:-"imagenet"}


LM_EXP_DIR=${LM_EXP_DIR}
PRECISION=${PRECISION:-"fp16"}
LM_PRECISION=${LM_PRECISION:-"fp16"}


PROJECT_ROOT=${PROJECT_ROOT}
# The dir for imgnet dataset, like .../imagenet/
IMGNET_ROOT=${IMGNET_ROOT}

# we use the total training epoch to manage the training process for ImageNet
# But for total training iteration, we will accurately control with iteration numbers, 
BSZ_FACTOR=5000  # The iteraions for 1 epoch when trained with 256 batch size
# note bsz is decided inside the configuration file
EPOCH_TO_ITER_FACTOR=$((BSZ_FACTOR * 256 / TOK_BSZ))
TOK_ITER=$((TOK_EPOCH * EPOCH_TO_ITER_FACTOR))
# Format TOK_ITER and store it back in TOK_ITER
printf -v TOK_ITER "%07d" "$TOK_ITER"
echo "$TOK_ITER"


# Training script for tokenizer
bash scripts/train_vq_fsdp.sh \
--save-path ${PROJECT_ROOT}/results/tokenizers/vq/ \
--data-path ${IMGNET_ROOT}/ILSVRC2012_img_train/ \
--image-size 256 \
--model-config ${TOK_CONFIG} \
--mixed-precision ${PRECISION} \
--grad-precision ${PRECISION} \
--iteration ${TOK_ITER} \
--global-batch-size ${TOK_BSZ} \
--dataset ${DATASET} \
--json-path ${JSON_PATH} \
--fract-decay ${FRACT_DECAY} \
--ckpt-every ${CKPT_EVERY} \
--sub-exp-dir ${VQ_EXP_DIR}


VAL_PATH=${VAL_PATH:-${PROJECT_ROOT}/results/reconstructions/img_data}
GT_VAL_NPZ_PATH=${GT_VAL_NPZ_PATH:-${PROJECT_ROOT}/results/reconstructions/val_imagenet.npz}
# should use the specific evaluation environment from ADM
EVAL_PYTHON_PATH=${EVAL_PYTHON_PATH}


# reconstruction Evaluation script
bash scripts/reconstruction.sh \
--data-path=${PROJECT_ROOT}/results/reconstructions/img_data \
--image-size 256 \
--quant-way vq \
--sample-dir ${PROJECT_ROOT}/results/reconstructions/vq/${VQ_EXP_DIR}/${TOK_ITER} \
--vq-ckpt ${PROJECT_ROOT}/results/tokenizers/vq/${VQ_EXP_DIR}/checkpoints/${TOK_ITER}.pt \
--model-config ${TOK_CONFIG} \
--wandb \
--wandb-project vqvit_imagenet_train \
--eval-wandb-dir ${PROJECT_ROOT}/results/reconstructions/vq/${VQ_EXP_DIR} \
--clear-cache \
--eval-python-path ${EVAL_PYTHON_PATH} \
--gt-npz-path ${GT_NPZ_PATH}


# Base LM Training recipe

VQ_EXP_DIR=$VQ_EXP_DIR \
TOK_CONFIG=$TOK_CONFIG \
TOK_EPOCH=$TOK_EPOCH \
TOK_BSZ=$TOK_BSZ \
\
LM_EXP_DIR=${LM_EXP_DIR} \
LM_EPOCH=300 \
LM_BSZ=256 \
GPT_MODEL="GPT-B" \
WARM_ITER=0 \
LR="1e-4" \
PRECISION=${LM_PRECISION} \
CFG_SCHEDULE="constant" \
LM_STOP_ITER=250000 \
bash scripts/train_c2i_and_eval.sh