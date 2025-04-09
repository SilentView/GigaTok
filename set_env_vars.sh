# ----------------------------------------------------------------------------------------
# setup environment variables
# disable TF verbose logging
TF_CPP_MIN_LOG_LEVEL=2
# fix known issues for pytorch-1.5.1 accroding to 
# https://blog.exxactcorp.com/pytorch-1-5-1-bug-fix-release/
MKL_THREADING_LAYER=GNU
# set NCCL envs for disributed communication
NCCL_IB_GID_INDEX=3
NCCL_IB_DISABLE=0
NCCL_DEBUG=INFO
ARNOLD_FRAMEWORK=pytorch

# get distributed training parameters 
NV_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


###############################
### Set the path and environment related variables
###############################
## path and keys
# (Optional) wandb setting
export WANDB_API_KEY=YOUR_WANDB_KEY
# The absolute dir for this project , like .../GigaTok/
export PROJECT_ROOT=YOUR_DIR/GigaTok
# The absolute dir for imgnet dataset, like .../imagenet/
export IMGNET_ROOT="YOUR_DIR/imagenet/"
# The python path for the specific env used for ADM FID evaluation, like .../miniconda/.../python
export EVAL_PYTHON_PATH=YOUR_PATH/python
# The path for the 256x256 ImageNet 50k validation data (generated from provided scripts)
export VAL_PATH=YOUR_PATH/img_data
# The torchrun path, like .../miniconda/.../torchrun
export TORCH_RUN_PATH=YOUR_PATH/torchrun

###############################
### Set the torchrun related parameters. The default is for a single node.
###############################
export GPUS=$NV_GPUS
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PORT=${PORT:-3343}
