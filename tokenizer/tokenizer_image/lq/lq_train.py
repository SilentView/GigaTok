# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
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

import os
import time
import argparse
from glob import glob
from copy import deepcopy

from utils.logger import create_logger
import logging
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad

from utils.resume_log import init_wandb, upload_wandb_cache, manage_ckpt_num, get_int_prefix_value, wandb_cache_file_append
import wandb


from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.lq.lq_model import LQ_models
from tokenizer.tokenizer_image.lq.lq_loss import LQLoss
from tokenizer.tokenizer_image.scheduler import cosine_lr, const_lr

import warnings
warnings.filterwarnings('ignore')


wandb_project_name = "lqvit_imagenet_train"

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
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


    if args.sub_exp_dir is not None:
        experiment_dir = f"{args.save_path}/{args.sub_exp_dir}"  # Create an experiment folder
    else:
        experiment_index = len(glob(f"{args.save_path}/*"))
        model_string_name = args.lq_model.replace("/", "-")
        experiment_dir = f"{args.save_path}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints

    # Setup an experiment folder:
    if rank == 0 and node_rank == 0:
        os.makedirs(args.save_path, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # wandb
        init_wandb(
            project_name=wandb_project_name,
            config={"dataset":"imagenet"},
            exp_dir=experiment_dir,
            )
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    lq_model = LQ_models[args.lq_model](
        levels=args.levels,
        commit_loss_beta=args.commit_loss_beta,
        # entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
    )
    logger.info(f"LQ Model Parameters: {sum(p.numel() for p in lq_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(lq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"LQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    lq_model = lq_model.to(device)

    lq_loss = LQLoss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight, 
        aux_loss_end=args.aux_loss_end,
    ).to(device)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in lq_loss.discriminator.parameters()):,}")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Setup optimizer
    optimizer = torch.optim.Adam(lq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(lq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

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
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    

    # Prepare models for training:
    if args.lq_ckpt:
        checkpoint = torch.load(args.lq_ckpt, map_location="cpu")
        lq_model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.lq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.lq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    
    # auto resume
    elif len(glob(f"{checkpoint_dir}/*.pt")) != 0:
        latest_checkpoint = max(glob(f"{checkpoint_dir}/*.pt"), key=get_int_prefix_value)
        checkpoint = torch.load(latest_checkpoint, map_location="cpu")
        lq_model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(latest_checkpoint.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {latest_checkpoint}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
        with open(os.path.join(experiment_dir, "message.txt"), "w") as f:
            f.write(f"Resume training from checkpoint: {latest_checkpoint}")
            f.write(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, lq_model, decay=0)  # Ensure EMA is initialized with synced weights
    
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        lq_model = torch.compile(lq_model) # requires PyTorch 2.0        
    
    lq_model = DDP(lq_model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    lq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    lq_loss = DDP(lq_loss.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    lq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    total_steps = int(len(dataset) / args.global_batch_size)  * args.epochs
    if args.lr_scheduler == "cosine":
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        scheduler_disc = cosine_lr(optimizer_disc, args.lr, args.warmup, total_steps)
    elif args.lr_scheduler == "const":
        scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        scheduler_disc = const_lr(optimizer_disc, args.lr, args.warmup, total_steps)
    else:
        logging.error(
            f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
        exit(1)


    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    wandb_updates = []

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            imgs = x.to(device, non_blocking=True)

            # generator training
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                recons_imgs, codebook_loss = lq_model(imgs)
                loss_gen = lq_loss(codebook_loss, imgs, recons_imgs, exp_dir=experiment_dir, optimizer_idx=0, global_step=train_steps+1, 
                                   last_layer=lq_model.module.decoder.last_layer, 
                                   logger=logger, log_every=args.log_every, ckpt_every=args.ckpt_every)
            scaler.scale(loss_gen).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lq_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            current_lr = scheduler(train_steps)
            if args.ema:
                update_ema(ema, lq_model.module._orig_mod if args.compile else lq_model.module)

            # discriminator training            
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss_disc = lq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps+1, exp_dir=experiment_dir,
                                    logger=logger, log_every=args.log_every, ckpt_every=args.ckpt_every)
            scaler_disc.scale(loss_disc).backward()
            if args.max_grad_norm != 0.0:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(lq_loss.module.discriminator.parameters(), args.max_grad_norm)
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()
            current_lr_disc = scheduler_disc(train_steps)
            
            # # Log loss values:
            running_loss += loss_gen.item() + loss_disc.item()
            
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
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR: {current_lr:.6f}, LR_disc: {current_lr_disc:.6f}")
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
                        model_weight = lq_model.module._orig_mod.state_dict()
                    else:
                        model_weight = lq_model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "discriminator": lq_loss.module.discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()

                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    wandb_cache_file_append(wandb_updates, exp_dir=experiment_dir)
                    upload_wandb_cache(exp_dir=experiment_dir)
                    wandb.finish()
                    init_wandb(
                        project_name=wandb_project_name,
                        config={"dataset":"imagenet"},
                        exp_dir=experiment_dir,
                        )
                    
                    manage_ckpt_num(
                        checkpoint_dir,
                        milestone_step=args.milestone_step,
                        milestone_start=args.milestone_start,
                        max_milestone_num=args.max_milestone_num
                    )

                wandb_updates = [] 
                dist.barrier()

    lq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    if rank == 0 and node_rank == 0:
        wandb.finish()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve lq model")
    parser.add_argument("--save-path", type=str, default="results_tokenizer_image")
    parser.add_argument("--lq-model", type=str, choices=list(LQ_models.keys()), default="LQ-16")
    parser.add_argument("--lq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained lq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")

    str2list = lambda x: [int(item) for item in x.split(',')]
    parser.add_argument("--levels", type=str2list, default=[8, 8, 8, 6, 5], help="levels for fsq quantization")  # default 2**14

    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
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
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, choices=['cosine', 'const'], default="const")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--aux-loss-end", type=int, default=40000, help="iteration to stop using auxiliary loss")

    parser.add_argument("--sub-exp-dir", type=str, default=None, help="sub experiment dir")
    parser.add_argument("--milestone-step", type=int, default=50_000, help="milestone step for checkpoint saving")
    parser.add_argument("--milestone-start", type=int, default=50_000, help="milestone start for checkpoint saving")
    parser.add_argument("--max-milestone-num", type=int, default=10, help="max milestone num for checkpoint saving")
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    args = parser.parse_args()

    wandb_project_name = args.wandb_project if args.wandb_project is not None else wandb_project_name
    main(args)
