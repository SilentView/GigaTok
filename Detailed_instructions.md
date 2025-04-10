# Evaluation

## Quantitative Reconstruction Evaluation

### 1. ADM FID evaluation environment
From https://github.com/FoundationVision/LlamaGen/blob/main/evaluations/c2i/README.md

For evaluation environment:
```shell
# use a seperate environment for evaluation!

# This is the environment for evaluation. Correspond to EVAL_PYTHON_PATH in set_env_vars.sh
conda create -n tok_eval python=3.9
conda activate tok_eval

# cuda 12.1
pip install tensorflow
pip install numpy==1.23.5
pip install scipy


# cuda 12.2
pip install tensorflow
pip install numpy==1.26.2
pip install scipy
```
### 2. Extract validation data/features
Sample 50k images from ImageNet validation set. This will create ```results/reconstructions/val_imagenet.npz```, and also `results/reconstructions/img_data/` for the 256x256 val images.

```shell
. set_env_vars.sh    # only need to set TORCH_RUN_PATH
bash scripts/val.sh \
--data-path ${IMGNET_ROOT}/ILSVRC2012_img_val/ \
--sample-dir results/reconstructions/img_data/
```

**Set VAL_PATH** according to  `--sample-dir results/reconstructions/img_data/` in set_env_vars.sh.


### 3. Evalute reconstructed samples/features
```shell
# remember to set EVAL_PYTHON_PATH
. set_env_vars.sh

export VAL_PATH=results/reconstructions/img_data/
# use absolute path
export GT_VAL_NPZ_PATH=${PROJECT_ROOT}/results/reconstructions/val_imagenet.npz

export VQ_CKPT=results/ckpts/VQ_XLXXL256_e300.pt
export TOK_CONFIG="configs/vq/VQ_XLXXL256.yaml"
export SAMPLE_DIR=results/vq/VQ_XLXXL256

bash scripts/reconstruction.sh \
--data-path $VAL_PATH \
--image-size 256 \
--quant-way vq \
--sample-dir $SAMPLE_DIR \
--vq-ckpt $VQ_CKPT \
--model-config ${TOK_CONFIG} \
--clear-cache \
--eval-python-path ${EVAL_PYTHON_PATH} \
--gt-npz-path ${GT_VAL_NPZ_PATH}
```



## Evaluate Generation Quantitatively

**Make sure the reference batch for gFID is downloaded. Also make sure the AMD FID calculation environment is set up.**
```shell
# download reference batch for gFID calculation
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

**Calculating gFID and validation loss**
```shell
. set_env_vars.sh

export VQ_CKPT=${PROJECT_ROOT}/results/ckpts/VQ_XLXXL256_e300.pt
export TOK_CONFIG="configs/vq/VQ_XLXXL256.yaml"

export GPT_CKPT=results/ckpts/GPT_B256_e300_VQ_XLXXL.pt
export GPT_MODEL="GPT-B"
export SAMPLE_DIR=${PROJECT_ROOT}/results/gpt/quan_eval/GPT_B256_e300_VQ_XLXXL/
export PRECISION="none" # fp32

export CFG_SCHEDULE="step"
export CFG_START_RATIO="0.18"
export CFG_SCALE=1.75
export EVAL_BATCH_PER_GPU=16
export SEED=0

# use absolute path, make sure it is downloaded
export GT_NPZ_PATH=${PROJECT_ROOT}/VIRTUAL_imagenet256_labeled.npz

# If you want to search for the best cfg scale, add `--search`
bash scripts/sample_c2i_search_cfg.sh \
--quant-way=vq \
--image-size=256 \
--sample-dir=${SAMPLE_DIR} \
--cfg-scale ${CFG_SCALE} \
--vq-ckpt $VQ_CKPT \
--tok-config ${TOK_CONFIG} \
--gpt-model ${GPT_MODEL} \
--cfg-schedule $CFG_SCHEDULE \
--step-start-ratio $CFG_START_RATIO \
--gpt-ckpt $GPT_CKPT \
--per-proc-batch-size $EVAL_BATCH_PER_GPU \
--precision ${PRECISION} \
--eval-python-path ${EVAL_PYTHON_PATH} \
--gt-npz-path ${GT_NPZ_PATH} \
--global-seed ${SEED} \
--clear-cache

## Evaluation script for validation loss according
bash scripts/val_loss_c2i.sh \
--data-path=${IMGNET_ROOT}/ILSVRC2012_img_val/ \
--quant-way=vq \
--image-size=256 \
--vq-ckpt $VQ_CKPT \
--tok-config ${TOK_CONFIG} \
--gpt-model ${GPT_MODEL} \
--per-proc-batch-size $EVAL_BATCH_PER_GPU \
--gpt-ckpt $GPT_CKPT \
--precision ${PRECISION}
```


## Linear Probing Evaluation


### For Autoregressive Models
```shell
. set_env_vars.sh

export TOK_CONFIG="configs/vq/VQ_XLXXL256.yaml"
export VQ_CKPT=${PROJECT_ROOT}/results/ckpts/VQ_XLXXL256_e300.pt

export GPT_CKPT=${PROJECT_ROOT}/results/ckpts/GPT_B256_e300_VQ_XLXXL.pt
export GPT_MODEL="GPT-B"
# the precision of the LM
export LM_PRECISION="fp16"

# directory for saving the linear probing results
export SAVE_DIR=${PROJECT_ROOT}/results/lin_probe_gpt/
# the final results are in ${SAVE_DIR}/${LM_LIN_EXP_DIR}
export LM_LIN_EXP_DIR=lin_GPT_B256_e300_VQ_XLXXL
# the batch size of the linear probing training
export LIN_BSZ=128

bash scripts/composite_cmd/train_lm_lin_probe_and_eval.sh
```

### For Tokenizers

**Tokenizer Linear Probing Acc. does not necessarily reflect downstream AR performance, therefore this evaluation is not recommended.**

```shell
. set_env_vars.sh
export TOK_CONFIG="configs/vq/VQ_BL56.yaml"
export VQ_CKPT=${PROJECT_ROOT}/results/ckpts/VQ_BL256_e200.pt
# the precision of the tokenizer
export TOK_PRECISION="none" # fp32

# directory for saving the linear probing results
export SAVE_DIR=${PROJECT_ROOT}/results/lin_probe/
# the final results are in ${SAVE_DIR}/${LM_LIN_EXP_DIR}
export TOK_LIN_EXP_DIR=lin_VQ_BL256_e200
# the batch size of the linear probing training
export LIN_BSZ=128

bash scripts/composite_cmd/train_tok_lin_probe_and_eval.sh
```


# Training

## VQ Tokenizer Training and Evaluation (w/ AR Probing)
**Training smaller tokenizers with DDP**
```shell
. set_env_vars.sh
# Set the experiment directory for the VQ tokenizer
export VQ_EXP_DIR="VQ_BL256_e200"
export TOK_CONFIG="configs/vq/VQ_BL256.yaml"
export TOK_EPOCH=200
export TOK_BSZ=256
# Set the experiment directory for the AR Probing model
export LM_EXP_DIR="GPT_B256_VQ_BL_e200" 
export PRECISION="fp16"

# Set the env variables for the evaluation for reconstruction and generaation 
# For reconstruction evaluation
export VAL_PATH=results/reconstructions/img_data
export GT_VAL_NPZ_PATH=results/reconstructions/val_imagenet.npz
# For generation evaluation
export GT_NPZ_PATH=${PROJECT_ROOT}/VIRTUAL_imagenet256_labeled.npz

bash scripts/composite_cmd/train_vq_and_eval.sh
```

**Training large tokenizers with FSDP**

```shell
. set_env_vars.sh
# Set the experiment directory for the VQ tokenizer
export VQ_EXP_DIR="VQ_XLXXL256_e300"
export TOK_CONFIG="configs/vq/VQ_XLXXL256.yaml"
export TOK_EPOCH=300
export TOK_BSZ=256
# Set the experiment directory for the AR Probing model
export LM_EXP_DIR="GPT_B256_VQ_XLXXL_e300" 
export PRECISION="bf16"

# Set the env variables for the evaluation for reconstruction and generaation 
# For reconstruction evaluation
export VAL_PATH=results/reconstructions/img_data
# The .npz files should use absolute path
export GT_VAL_NPZ_PATH=${PROJECT_ROOT}/results/reconstructions/val_imagenet.npz
# For generation evaluation
export GT_NPZ_PATH=${PROJECT_ROOT}/VIRTUAL_imagenet256_labeled.npz

bash scripts/composite_cmd/train_vq_and_eval_fsdp.sh
```


**While we use rotation trick for all our released tokenizers, users can toggle it off when training GigaTok from scratch.** This can be achieved by setting `rot: False` in the model config file. The performance will not degrade. Further, we suggest a simpler inital training setting by setting `aux_loss_end=0` (it toggles off the shortcut reconstruction and feature reconstruction loss for initial stage.)

To check the suggested simplified tokenizer training config, see [configs/vq/VQ_BL256_train_new.yaml](configs/vq/VQ_BL256_train_new.yaml).



## AR Model Training

### (Optional) Pre-computing indices for compressed images

```shell
. set_env_vars.sh
export VQ_CKPT=${PROJECT_ROOT}/results/ckpts/VQ_XLXXL256_e300.pt
export TOK_CONFIG="configs/vq/VQ_XLXXL256.yaml"
export CODE_PATH=${PROJECT_ROOT}/dataset/imgnet_code/VQ_XLXXL256_e300/ten_crop/

bash scripts/extract_code_c2i.sh
```

### AR Training Scripts 

**Training an AR model using WSD Learning rate**
```shell
. set_env_vars.sh

export VQ_CKPT=${PROJECT_ROOT}/results/ckpts/VQ_XLXXL256_e300.pt
export TOK_CONFIG="configs/vq/VQ_XLXXL256.yaml"

export LM_EXP_DIR=results/gpt/GPT_B256_VQ_XLXXL_e300/
export LM_EPOCH=300
export LM_BSZ=256
export GPT_MODEL="GPT-B"
export WARM_ITER=5000
export LR="1e-4"
export FRACT_DECAY=0.2

export GT_NPZ_PATH=${PROJECT_ROOT}/VIRTUAL_imagenet256_labeled.npz

# Uncomment for using pre-computed indices for acceleration
# export DATASET=imagenet_code
# export CODEPATH=YOUR_CODEPATH
bash scripts/composite_cmd/train_c2i_and_eval.sh
```

**Training 1.4B AR model on XL-XXL tokenizers using FSDP/SDP**
```shell
. set_env_vars.sh

export VQ_CKPT=${PROJECT_ROOT}/results/ckpts/VQ_XLXXL256_e300.pt
export TOK_CONFIG="configs/vq/VQ_XLXXL256.yaml"

export LM_EXP_DIR=results/gpt/GPT_B256_VQ_XLXXL_e300/
export LM_EPOCH=300
export LM_BSZ=512
export GPT_MODEL="GPT-XXL"
export WARM_ITER=200
export LR="1e-4"
export FRACT_DECAY=0.2

export GT_NPZ_PATH=${PROJECT_ROOT}/VIRTUAL_imagenet256_labeled.npz

# Uncomment for using pre-computed indices for acceleration
# We suggest using pre-computed indices when training large AR models
# export DATASET=imagenet_code
# export CODEPATH=YOUR_CODEPATH
bash scripts/composite_cmd/train_c2i_and_eval_fsdp.sh
```

## Training Result File Structure

```
results/
├── gpt/
├── lin_probe/
├── lin_probe_gpt/
├── reconstructions/
└── tokenizers/
    └── vq/
        ├── exp_name_1/
        │   ├── checkpoints/
        │   ├── config.json   (wandb_config)
        │   └── log.txt
        └── exp_name_2/
```

### For train_c2i

#### Without Constant+Cool Down
```
exp_name_1/
├── checkpoints/
├── config.json   (wandb_config)
├── vq_config.yaml
├── vq_ckpt.txt
└── log.txt
```

#### With Constant+Cool Down
```
exp_name_1/
├── checkpoints/  (for constant lr part)
├── config.json   (wandb_config for constant part)
├── log.txt
├── vq_config.yaml
├── vq_ckpt.txt
└── cd_records/   (cd stands for "cool down")
    └── cd_fract_0.2_from_N/   (N stands for the end iteration of constant part)
        ├── checkpoints/
        │   └── {iteration}.pt
        ├── config.json   (wandb_config for cool down part)
        └── log.txt
```

### For train_c2i_fsdp

```
exp_name_cd/
├── checkpoints/  (for constant lr part)
│   └── {CUR_n}.pt
├── optim_checkpoints/  (for cool down part)
│   ├── {ITER_1}/
│   │   ├── optimizer.{rank1:05d}-of-{world_size:05d}.pt    (sharded optimizer for FSDP)
│   │   ├── optimizer.{rank2:05d}-of-{world_size:05d}.pt
│   │   └── ...
│   ├── {ITER_2}/
│   └── ...
├── config.json   (wandb_config for constant part)
├── log.txt
├── vq_config.yaml
├── vq_ckpt.txt
└── cd_records/
    └── cd_fract_0.2_from_N/
        ├── checkpoints/
        │   └── {CUR_n}.pt
        ├── optim_checkpoints/
        │   ├── {ITER_1}/
        │   │   ├── optimizer.{rank1:05d}-of-{world_size:05d}.pt
        │   │   ├── optimizer.{rank2:05d}-of-{world_size:05d}.pt
        │   │   └── ...
        │   ├── {ITER_2}/
        │   └── ...
        ├── config.json   (wandb_config for cool down part)
        └── log.txt