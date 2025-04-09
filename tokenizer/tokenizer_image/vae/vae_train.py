# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py

"""
This training script will use config file for training
Further, we implement the a little complex constant + cooldown experiment mangagement machenism
to help find the optimal FLOPS for training the same model with the same hyperparameters

About the constant + cooldown experiment mangagement mechanism:
1. The file structure:
# the first kind (w/o constant+cool down)
- exp_name_1
    - checkpoints
    config.json   (wandb_config)
    log.txt
- exp_name_2
...

# the second kind (w/ constant+cool down)
- exp_name_cd 
    - checkpoints (for constant lr part)
    config.json   (wandb_config for constant part)
    log.txt
    - cd_records
        - cd_fract_0.2_from_N
            - checkpoints
                {CUR_n}.pt
            config.json   (wandb_config for cool down part)
            log.txt
- exp_name_cd_2
...

2. How to resume for constant + cooldown experiments:
First, the script will check the latest state:
 1. Is the constant training already ended?
    - If ended resume from the checkpoints in cd_records
    _ If not ended resume from the latest checkpoint in checkpoints
    - wandb config (run_id) will be different for the two cases

Then, during training, when saving checkpoints:
1. Is the constant training already ended?
    - If ended, save the checkpoints in cd_records
    - If not ended, save the checkpoints in checkpoints
    - This will allow sharing the constant stage ckpts for all experiments

Additional Features for beta version:

- accurate resuming
Not only saving iteration and epoch, but also resume from the latest step.
Then idea is stated in https://discuss.pytorch.org/t/save-checkpoint-every-step-instead-of-epoch/122600/5
We will first implement this by using empty loop, although there will be meaningless data loading with one epoch
When the dataset gets larger, we will consider resume the 


ensured by saving the model every 1 epcohs
Then can resume from the start of one epoch. Besides, all the ckpts will be named according
to training epochs instead of training iterations.

- stochastic weight averaging
Currently working on epoch management and saving root mangagement
"""

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F

import PIL

import os
import time
import argparse
from math import ceil
from glob import glob
from copy import deepcopy
from itertools import islice

import random

import numpy as np

from utils.logger import create_logger
import logging
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from utils.model_init import load_model_from_config, custom_load

from utils.resume_log import (
    init_wandb, upload_wandb_cache, wandb_cache_file_append,
    manage_ckpt_num, get_int_prefix_value, wsd_find_newest_ckpt
)
from utils.model_init import load_encoders
import wandb
import yaml


from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vae.vae_loss import VAELossSDAlign
from tokenizer.tokenizer_image.scheduler import cosine_lr, const_lr, cosine_schedule_with_warmup_v2, wsd_lr

from torchvision.transforms import Normalize
import warnings
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
warnings.filterwarnings('ignore')


wandb_project_name = "vaevit_imagenet_train"


CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

SIGLIP_DEFAULT_MEAN = (0.5, 0.5, 0.5)
SIGLIP_DEFAULT_STD = (0.5, 0.5, 0.5)

def preprocess_raw_image(x, enc_type):
    """
    x range is [-1 , 1]
    from https://github.com/sihyun-yu/REPA/blob/main/train.py
    """
    if 'clip' in enc_type:
        x = (x + 1) / 2.
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'siglip' in enc_type:
        x = (x + 1) / 2.
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
        x = Normalize(SIGLIP_DEFAULT_MEAN, SIGLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = (x + 1) / 2.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = (x + 1) / 2.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
    elif 'dinov1' in enc_type:
        x = (x + 1) / 2.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = (x + 1) / 2.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')

    return x





#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)


    # setting the constants used during training
    

    # the causal settings are the attributes of a model, and also necessary for training process
    causal_type = config["model"]["causal_settings"]["causal_type"]
    dynamic_length_train = config["model"]["causal_settings"]["dynamic_length_train"]
    min_level = config["model"]["causal_settings"]["min_level"]
    dynamic_level_range = config["model"]["causal_settings"].get("dynamic_level_range", None)
    power_sample_T = config["model"]["causal_settings"].get("power_sample_T", None)


    if args.sub_exp_dir is not None:
        experiment_dir = f"{args.save_path}/{args.sub_exp_dir}"  # Create an experiment folder
    else:
        raise ValueError("Please specify a sub_exp_dir")

    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints

    # configs for constant + cooldown training
    if config["trainer"]["lr_scheduler"] == "wsd":
        use_wsd = True
        fract_decay = args.fract_decay if args.fract_decay is not None \
                      else config["trainer"].get("fract_decay", 0.2)

        cd_sub_dir = experiment_dir + "/cd_records" + f"/cd_fract_{fract_decay}_to_{args.iterations:07d}"
        cd_checkpoint_dir = f"{cd_sub_dir}/checkpoints"

        # add extra wsd setting for trial_name
        # trial_name = args.model_config.split("/")[-1].replace(".yaml", "") + f"_cd_fract_{fract_decay}_to_{args.iterations:07d}"
        trial_name = args.sub_exp_dir
    else:
        # trial_name = args.model_config.split("/")[-1].replace(".yaml", "")
        trial_name = args.sub_exp_dir
        use_wsd = False
        fract_decay = 0


    args.global_batch_size = args.global_batch_size if args.global_batch_size is not None else config["trainer"]["global_batch_size"]


    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    node_rank = int(os.environ.get('NODE_RANK', 0))
    print("rank", rank)
    print("node_rank", node_rank)
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)



    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )


    total_steps = args.iterations
    if use_wsd:
        _ , const_end_flag = wsd_find_newest_ckpt(
                                                config, 
                                                const_ckpt_dir=checkpoint_dir, 
                                                cd_sub_dir=cd_sub_dir, 
                                                total_steps=total_steps,
                                                fract_decay=fract_decay,
                                                )
    else:
        const_end_flag = False

    exp_dir = cd_sub_dir if (use_wsd and const_end_flag) else experiment_dir
    # Setup an experiment folder:
    if rank == 0 and node_rank == 0:

        os.makedirs(args.save_path, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        if use_wsd:
            os.makedirs(cd_checkpoint_dir, exist_ok=True)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")

        # wandb
       
        init_wandb(
            project_name=wandb_project_name,
            config={"dataset":"imagenet"},
            exp_dir=exp_dir,
            name=trial_name,
            )
    else:
        logger = create_logger(None)


    # training args
    logger.info(f"{args}")
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")


    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    ######################################################################
    ## Load Teacher model for distillation
    ######################################################################
    if config["trainer"].get("distill_loss", False) is True:
        model_out_dim_dict = {
            "siglip-sovit-400m": 1152,
            "clip_dfn-vit-l/14": 1024
        }
        distill_model = config["trainer"].get("distill_model", "dinov2-vit-b")
        distill_encoder, encoder_type, architecture = load_encoders(distill_model, device)
        if "dinov2" in distill_model:
            config["model"]["init_args"]["out_inner_dim"] = distill_encoder.embed_dim
        else:
            # for open clip model, the output dim is different from the input dim
            try:
                config["model"]["init_args"]["out_inner_dim"] = distill_encoder.visual.output_dim
            except:
                # if cannot get from attributes, then refer to a hardcoded dict
                config["model"]["init_args"]["out_inner_dim"] = model_out_dim_dict[distill_model]

 
    vae_model = load_model_from_config(config)

    # create and load model
    logger.info(f"VAE Model Parameters(training): {sum(p.numel() for p in vae_model.parameters()):,}")    # logger.info(f"VAE Model Transformer Encoder Parameters: {sum(p.numel() for p in vae_model.s2to1encoder.parameters()):,}")
    # logger.info(f"VAE Model Transformer Decoder Parameters: {sum(p.numel() for p in vae_model.s1to2decoder.parameters()):,}")
    if args.ema:
        ema = deepcopy(vae_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"VAE Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vae_model = vae_model.to(device)


    logger.info(str(config))
    # Ensure all default values are set in the config dictionary beforehand or assume they are already set
    # vae_loss = VAELoss(
    #     disc_start=config["loss"]["params"]["disc_start"],
    #     disc_weight=config["loss"]["params"]["disc_weight"],
    #     disc_dim=config["loss"]["params"]["disc_dim"],
    #     disc_num_layers=config["loss"]["params"].get("disc_num_layers", 3),
    #     disc_type=config["loss"]["params"]["disc_type"],
    #     disc_loss=config["loss"]["params"]["disc_loss"],
    #     gen_adv_loss=config["loss"]["params"]["gen_adv_loss"],
    #     image_size=config["loss"]["params"]["image_size"],  # Assuming image_size is moved to config
    #     perceptual_weight=config["loss"]["params"]["perceptual_weight"],
    #     reconstruction_weight=config["loss"]["params"]["reconstruction_weight"],
    #     reconstruction_loss=config["loss"]["params"]["reconstruction_loss"],
    #     aux_loss_end=config["loss"]["params"]["aux_loss_end"],
    #     norm=config["loss"]["params"]["norm"],
    #     use_direct_rec_loss=config["loss"]["params"].get("use_direct_rec_loss", False),
    #     kw=config["loss"]["params"]["kw"],
    #     blur_ds=config["loss"]["params"].get("blur_ds", False),
    #     lecam=config["loss"]["params"].get("lecam", False),
    #     proj_weight=config["loss"]["params"].get("proj_weight", 0.5),
    #     disc_semantic_type=config["loss"]["params"].get("disc_semantic_type", "local"),
    #     use_semantic_input=config["loss"]["params"].get("use_semantic_input", False),
    #     perceptual_model=config["loss"]["params"].get("perceptual_model", "vgg"),
    #     gamma=config["loss"]["params"].get("gamma", 15),
    #     kl_weight=float(config["loss"]["params"].get("kl_weight", 1e-6)),
    # ).to(device)


    vae_loss = VAELossSDAlign(
        disc_start=config["loss"]["params"]["disc_start"],
        disc_weight=config["loss"]["params"]["disc_weight"],
        disc_factor=config["loss"]["params"].get("disc_factor", 1.0),
        disc_dim=config["loss"]["params"]["disc_dim"],
        disc_num_layers=config["loss"]["params"].get("disc_num_layers", 3),
        disc_type=config["loss"]["params"]["disc_type"],
        disc_loss=config["loss"]["params"]["disc_loss"],
        disc_adaptive_weight=config["loss"]["params"].get("disc_adaptive_weight", False),
        gen_adv_loss=config["loss"]["params"]["gen_adv_loss"],
        image_size=config["loss"]["params"]["image_size"],  # Assuming image_size is moved to config
        perceptual_weight=config["loss"]["params"]["perceptual_weight"],
        reconstruction_weight=config["loss"]["params"]["reconstruction_weight"],
        reconstruction_loss=config["loss"]["params"]["reconstruction_loss"],
        norm=config["loss"]["params"]["norm"],
        kw=config["loss"]["params"]["kw"],
        blur_ds=config["loss"]["params"].get("blur_ds", False),
        lecam=config["loss"]["params"].get("lecam", False),
        proj_weight=config["loss"]["params"].get("proj_weight", 0.5),
        sem_adaptive_weight=config["loss"]["params"].get("sem_adaptive_weight", False),
        sem_weight_upper_bound=float(config["loss"]["params"].get("sem_weight_upper_bound", 1e4)),
        sem_reg_start=config["loss"]["params"].get("sem_reg_start", 0),
        perceptual_model=config["loss"]["params"].get("perceptual_model", "vgg"),
        gamma=config["loss"]["params"].get("gamma", 15),
        kl_weight=float(config["loss"]["params"].get("kl_weight", 1e-6)),
    ).to(device)


    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vae_loss.discriminator.parameters()):,}")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))

    ###################################
    # Setup optimizer
    ###################################
    model_optimizer_groups = vae_model.parameters()


    if config["trainer"].get("optimizer", None) == "Adam":
        optimizer = torch.optim.Adam(
            model_optimizer_groups,
            lr=float(config["trainer"]["lr"]), 
            betas=(config["trainer"]["beta1"], config["trainer"]["beta2"]),
            weight_decay=float(config["trainer"]["weight_decay"])
        )

        optimizer_disc = torch.optim.Adam(
            vae_loss.discriminator.parameters(), 
            lr=float(config["trainer"]["lr"]), 
            betas=(config["trainer"]["beta1"], config["trainer"]["beta2"]),
            weight_decay=float(config["trainer"]["weight_decay"])
        )
    elif config["trainer"].get("optimizer", None) == "AdamW":
        optimizer = torch.optim.AdamW(
            model_optimizer_groups,
            lr=float(config["trainer"]["lr"]), 
            betas=(config["trainer"]["beta1"], config["trainer"]["beta2"]),
            weight_decay=float(config["trainer"]["weight_decay"])
        )

        optimizer_disc = torch.optim.AdamW(
                            vae_loss.discriminator.parameters(), 
                            lr=float(config["trainer"]["lr"]), 
                            betas=(config["trainer"]["beta1"], config["trainer"]["beta2"]),
                            weight_decay=float(config["trainer"]["weight_decay"])
                            )
    else:
        raise ValueError(f"Optimizer {config['trainer'].get('optimizer', 'Adam')} not supported.")


    

    ######################################
    # Prepare models for training:
    ######################################
    if args.vae_ckpt:
        checkpoint = torch.load(args.vae_ckpt, map_location="cpu")
        custom_load(vae_model, checkpoint["model"])
        if args.ema:
            # ema.load_state_dict(checkpoint["ema"])
            custom_load(ema, checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        vae_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vae_ckpt.split('/')[-1].split('.')[0])
            start_epoch = train_steps // (len(dataset) // args.global_batch_size)
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vae_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")

    # auto resume accurately
    elif len(glob(f"{checkpoint_dir}/*.pt")) != 0:

        if config["trainer"]["lr_scheduler"] == "wsd":
            # Is the constant training already ended?
            resume_checkpoint, const_end_flag = wsd_find_newest_ckpt(
                                                    config, 
                                                    const_ckpt_dir=checkpoint_dir, 
                                                    cd_sub_dir=cd_sub_dir, 
                                                    total_steps=total_steps,
                                                    fract_decay=fract_decay
                                                 )

            latest_checkpoint = resume_checkpoint
            checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        else:
            latest_checkpoint = max(glob(f"{checkpoint_dir}/*.pt"), key=get_int_prefix_value)
            checkpoint = torch.load(latest_checkpoint, map_location="cpu")

        custom_load(vae_model, checkpoint["model"])
        # vq_model.load_state_dict(checkpoint["model"])
        if args.ema:
            # ema.load_state_dict(checkpoint["ema"])
            custom_load(ema, checkpoint["ema"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            if rank == 0 and node_rank == 0:
                print("Optimizer state_dict not found in the checkpoint")
                # print(optimizer.state_dict())
                print(checkpoint["optimizer"].keys())
                print(len(optimizer.state_dict()["param_groups"][0]["params"]))
                print(len(optimizer.state_dict()["param_groups"][1]["params"]))
                print(len(checkpoint["optimizer"]["param_groups"][0]["params"]))
                print(checkpoint["optimizer"]["param_groups"])
                print(optimizer.state_dict()["param_groups"])
            exit()
        try:
            vae_loss.discriminator.load_state_dict(checkpoint["discriminator"])
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        except:
            print("You are using a unmatching discriminator")
            print("Assuming you are using a pretrained model with a different discriminator")
            print("reinitialize related model and optimizer")

        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vae_ckpt.split('/')[-1].split('.')[0])
            start_epoch = train_steps // (len(dataset) // args.global_batch_size)
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {latest_checkpoint}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
        with open(os.path.join(experiment_dir, "message.txt"), "w") as f:
            f.write(f"Resume training from checkpoint: {latest_checkpoint}")
    
    elif config["model"].get("init_ckpt", None) is not None:
        # the starting ckpt is given according to the config
        # init only when no existing ckpts are found
        init_ckpt = config["model"]["init_ckpt"]
        checkpoint = torch.load(init_ckpt, map_location="cpu")
        custom_load(vae_model, checkpoint["model"])
        # vq_model.load_state_dict(checkpoint["model"])
        if args.ema:
            # ema.load_state_dict(checkpoint["ema"])
            custom_load(ema, checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            vae_loss.discriminator.load_state_dict(checkpoint["discriminator"])
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        except:
            # Using a pretrained model before starting discriminator training, 
            # reinitialize discriminator
            pass
        train_steps = 0
        start_epoch = 0           
        del checkpoint
        logger.info(f"Init training from checkpoint: {init_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
        with open(os.path.join(experiment_dir, "message.txt"), "w") as f:
            f.write(f"Resume training from checkpoint: {init_ckpt}")
            f.write(f"Initial state: steps={train_steps}, epochs={start_epoch}")
        
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vae_model, decay=0)  # Ensure EMA is initialized with synced weights
    


    # decide the causal token keeping schedule
    if causal_type == "per-level":
        max_num_1d_tokens = vae_model.config.num_latent_tokens
        if dynamic_level_range is None:
            n_level = int(np.round(np.log2(max_num_1d_tokens)))
            choices = [i for i in range(min_level, n_level + 1)]
        else:
            choices = [i for i in range(dynamic_level_range[0], dynamic_level_range[1] + 1, dynamic_level_range[0])]
        probs = [1] * len(choices)
    elif causal_type == "per-token":
        max_num_1d_tokens = vae_model.config.num_latent_tokens
        if dynamic_level_range is None:
            n_level = max_num_1d_tokens
            choices = [i for i in range(min_level, n_level + 1)]
        else:
            if len(dynamic_level_range) > 2:
                choices = dynamic_level_range
            else:
                choices = [i for i in range(dynamic_level_range[0], dynamic_level_range[1] + 1, dynamic_level_range[0])]
        probs = [1] * len(choices)
    else:
        assert causal_type is None
    
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vae_model = torch.compile(vae_model) # requires PyTorch 2.0        
    
    vae_model = DDP(vae_model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    vae_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    vae_loss = DDP(vae_loss.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    vae_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    total_steps = args.iterations

    if config["trainer"]["lr_scheduler"] == "cosine":
        scheduler = cosine_lr(
                        optimizer, 
                        float(config["trainer"]["lr"]), 
                        int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup, 
                        total_steps)
        scheduler_disc = cosine_lr(
                            optimizer_disc, 
                            float(config["trainer"]["lr"]), 
                            int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup, 
                            total_steps)
    elif config["trainer"]["lr_scheduler"] == "const":
        scheduler = const_lr(
                        optimizer, 
                        float(config["trainer"]["lr"]), 
                        int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup, 
                        total_steps)
        scheduler_disc = const_lr(
                            optimizer_disc, 
                            float(config["trainer"]["lr"]), 
                            int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup, 
                            total_steps)
    elif config["trainer"]["lr_scheduler"] == "cosine_v2":
        scheduler = cosine_schedule_with_warmup_v2(
                        optimizer,
                        float(config["trainer"]["lr"]),
                        int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup,
                        total_steps,
                        end_lr=float(config["trainer"].get("end_lr", 1e-5))
                    )
        scheduler_disc = cosine_schedule_with_warmup_v2(
                            optimizer_disc,
                            float(config["trainer"]["lr"]),
                            int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup,
                            total_steps,
                            end_lr=float(config["trainer"].get("end_lr", 1e-5))
                        )
    elif config["trainer"]["lr_scheduler"] == "wsd":
        scheduler = wsd_lr(
                        optimizer,
                        base_lr=float(config["trainer"]["lr"]),
                        warmup_length=int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup,
                        steps=total_steps,
                        fract_decay=float(fract_decay),
                    )
        

        scheduler_disc = wsd_lr(
                            optimizer_disc,
                            base_lr=float(config["trainer"]["lr"]),
                            warmup_length=int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup,
                            steps=total_steps,
                            fract_decay=float(fract_decay),
                        )
    else:
        logging.error(
            f'Unknown scheduler, {config["trainer"]["lr_scheduler"]}. Available options are: cosine, const, const-cooldown.')
        exit(1)


    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_step_one_epoch = train_steps % (len(dataset) // args.global_batch_size)
    start_time = time.time()

    wandb_updates = []
    epochs = ceil(total_steps / (len(dataset) // args.global_batch_size))

    logger.info(f"Training for {epochs} epochs...")
    if use_wsd:
        hold_iterations = int(total_steps * (1 - fract_decay))
    breaking_flag = False
    for epoch in range(start_epoch, epochs):
        if args.iterations is not None and args.iterations <= train_steps:
            breaking_flag = True
            break

        if args.early_stop_iter is not None and args.early_stop_iter <= train_steps:
            breaking_flag = True
            break

        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        if epoch == start_epoch:
            loader_iter = iter(loader)
            loader_iter = islice(loader_iter, start_step_one_epoch, None)
        else:
            loader_iter = iter(loader)

        for x, y in loader_iter:
            if args.early_stop_iter is not None and args.early_stop_iter <= train_steps:
                breaking_flag = True
                break

            imgs = x.to(device, non_blocking=True)

            # dynamic token length training setting, depracated
            if causal_type is not None and dynamic_length_train:
                random.seed(train_steps)
                if power_sample_T is not None:
                    probs = [(l/max(choices))**(epoch / power_sample_T) for l in choices]
                num_en_q_level = random.choices(choices, weights=probs)[0]
                # np.random.choice(choices, p=probs, seed=train_steps)
            else:
                num_en_q_level = None

            # generator training
            optimizer.zero_grad()
            # ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]


            if config["trainer"].get("freeze_but_decoder_iter", None) is not None:
                if train_steps >= config["trainer"]["freeze_but_decoder_iter"] \
                    and not vae_model.module.freeze_but_2d_decoder_flag:
                    vae_model.module.freeze_but_2d_decoder()
                    
                    # disable distill_loss
                    config["trainer"]["distill_loss"] = False

            # prepare the distillive features
            if config["trainer"].get("distill_loss", False) is True:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        # if preprocess is defined as a local variable then use it
                           # use REPA style code
                        raw_image_ = preprocess_raw_image(imgs, encoder_type)
                        
                        # encode image
                        if "dinov2" in encoder_type:
                            z = distill_encoder.forward_features(raw_image_)
                            z = z['x_norm_patchtokens']
                        elif "siglip" in encoder_type:
                            outputs = distill_encoder(pixel_values=raw_image_)
                            z = outputs.last_hidden_state
                        elif "clip" in encoder_type:
                            z = distill_encoder(raw_image_)[-1]
            else:
                z = None


            with torch.cuda.amp.autocast(dtype=ptdtype):  
                recons_imgs, posteriors, inner_feat = vae_model(
                                                    input=imgs,
                                                    ret_posteriors=True, 
                                                    ret_inner_feat=config["trainer"].get("distill_loss", False),
                                                    num_en_q_level=num_en_q_level, 
                                                    causal_type=causal_type,
                                                )

                loss_gen = vae_loss(imgs, recons_imgs, posteriors=posteriors,
                                   exp_dir=exp_dir, optimizer_idx=0, global_step=train_steps+1, 
                                   logger=logger, log_every=args.log_every, ckpt_every=args.ckpt_every,
                                   causal_type=causal_type, num_en_q_level=num_en_q_level,
                                   inner_feat=inner_feat,
                                   sem_enc_feat=z,
                                   last_layer=vae_model.module.get_last_layer() if config["loss"]["params"].get("disc_adaptive_weight", False) else None,
                                   encoder_last_layer=vae_model.module.get_encoder_last_layer() if config["loss"]["params"].get("sem_adaptive_weight", False) else None,
                                   )
            
            if train_steps + 1 >= int(config["loss"]["params"].get("gen_start", 0)):
                scaler.scale(loss_gen).backward()
                if config["trainer"]["max_grad_norm"] != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(vae_model.parameters(), config["trainer"]["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                current_lr = scheduler(train_steps)
                if args.ema:
                    update_ema(ema, vae_model.module._orig_mod if args.compile else vae_model.module)
            else:
                loss_gen = torch.tensor(0.0, device=device)
                current_lr = scheduler(train_steps)

            # discriminator training            
            if train_steps + 1 >= int(config["loss"]["params"]["disc_start"]):
                optimizer_disc.zero_grad()
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    loss_disc = vae_loss(
                                    imgs, recons_imgs, posteriors=posteriors,
                                    optimizer_idx=1, global_step=train_steps+1, exp_dir=exp_dir,
                                    logger=logger, log_every=args.log_every, ckpt_every=args.ckpt_every, 
                                    sem_enc_feat=z)
                scaler_disc.scale(loss_disc).backward()
                if config["trainer"]["max_grad_norm"] != 0.0:
                    scaler_disc.unscale_(optimizer_disc)
                    torch.nn.utils.clip_grad_norm_(vae_loss.module.discriminator.parameters(), config["trainer"]["max_grad_norm"])
                scaler_disc.step(optimizer_disc)
                scaler_disc.update()
                current_lr_disc = scheduler_disc(train_steps)
            else:
                loss_disc = torch.tensor(0.0, device=device)
                current_lr_disc = 0
                
            # # Log loss values:
            running_loss += loss_gen.item() + loss_disc.item()


            if use_wsd:
                # update the wsd state for constant stage
                # for the last constant iteration, constant_end_flag is False
                # so that the ckpts will be saved in the constant dir
                const_end_flag = train_steps >= hold_iterations
                # also update the exp_dir
                exp_dir = cd_sub_dir if (use_wsd and const_end_flag) else experiment_dir
            
            log_steps += 1
            train_steps += 1



            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR: {current_lr:.4e}, LR_disc: {current_lr_disc:.4e}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

                wandb_updates.append(
                    {
                        "train_loss": avg_loss,
                        "train_steps_per_sec": steps_per_sec,
                        "iteration": train_steps,
                        "epoch": epoch,
                        "lr": current_lr,
                        "lr_disc": current_lr_disc,
                    }
                )

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0 and node_rank == 0:
                    if args.compile:
                        model_weight = vae_model.module._orig_mod.state_dict()
                    else:
                        model_weight = vae_model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "discriminator": vae_loss.module.discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    
                    save_ckpt_dir = cd_checkpoint_dir if (use_wsd and const_end_flag) else checkpoint_dir
                    checkpoint_path = f"{save_ckpt_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    wandb_cache_file_append(wandb_updates, exp_dir=exp_dir)
                    upload_wandb_cache(exp_dir=exp_dir)
                    wandb.finish()
                    init_wandb(
                        project_name=wandb_project_name,
                        config={"dataset":"imagenet"},
                        name=trial_name,
                        exp_dir=exp_dir,
                        )
                    
                    manage_ckpt_num(
                        save_ckpt_dir,
                        milestone_step=args.milestone_step,
                        milestone_start=args.milestone_start,
                        max_milestone_num=args.max_milestone_num
                    )

                wandb_updates = [] 
                dist.barrier()

            if args.iterations is not None and args.iterations <= train_steps:
                breaking_flag = True
                break

            if args.early_stop_iter is not None and args.early_stop_iter <= train_steps:
                breaking_flag = True
                break
        
        if breaking_flag:
            break

    dist.barrier()
    vae_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    if rank == 0 and node_rank == 0:
        wandb.finish()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--vae-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")

    # loss configs
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")

    parser.add_argument("--compile", action='store_true', default=False)

    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--json-path", type=str,
                        help="When given json path for dataset, all the images in the json will be used for training.")

    # training configs
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--aux-loss-end", type=int, default=None, help="iteration to stop using auxiliary loss")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--fract-decay", type=float, default=0.2, 
                        help="fraction of the total iterations to decay the learning rate, \
                        used when lr-scheduler=wsd")


    # log and ckpts
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--sub-exp-dir", type=str, default=None, help="sub experiment dir")
    parser.add_argument("--milestone-step", type=int, default=50_000, help="milestone step for checkpoint saving")
    parser.add_argument("--milestone-start", type=int, default=50_000, help="milestone start for checkpoint saving")
    parser.add_argument("--max-milestone-num", type=int, default=30, help="max milestone num for checkpoint saving")
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")

    to_int_or_none = lambda x: int(x) if x != 'None' else None
    parser.add_argument("--early-stop-iter", type=to_int_or_none, default=None)

    parser.add_argument("--model-config", type=str, required=True)
    # parser.add_argument("--power-sample-T", type=float, default=None, help="temperature for power sampling")
    # probs = [(l/max(choices))**(epoch / T) for l in choices]

    args = parser.parse_args()

    wandb_project_name = args.wandb_project if args.wandb_project is not None else wandb_project_name
    main(args)