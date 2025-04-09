
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py

"""
This training script will use config file for training
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

import math
from itertools import islice
from tqdm import tqdm

import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import torch.nn as nn
import torch.nn.functional as F



import os
import time
import argparse
from glob import glob
from copy import deepcopy

import random

import numpy as np

from utils.logger import create_logger
import logging
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from utils.model_init import load_model_from_config

from utils.resume_log import init_wandb, upload_wandb_cache, manage_ckpt_num, get_int_prefix_value, wandb_cache_file_append
import wandb
import yaml


from dataset.augmentation import random_crop_arr, center_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq.vq_loss import VQLoss
from tokenizer.tokenizer_image.scheduler import cosine_lr, const_lr
from tokenizer.tokenizer_image.vq.vq_vit_model import VQVitModelPlus, VQVitModel2DPlus

import warnings
warnings.filterwarnings('ignore')


#########################
# Model and training modules from mae https://github.com/facebookresearch/mae/blob/main/main_linprobe.py
#########################

from tokenizer.lin_probe.misc import LARS
from tokenizer.lin_probe.misc import NativeScalerWithGradNormCount as NativeScaler
from tokenizer.lin_probe.misc import RandomResizedCrop
from tokenizer.lin_probe.misc import adjust_learning_rate


class LinProb(nn.Module):
    def __init__(self, in_dim=8, num_classes=1000):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.bn = torch.nn.BatchNorm1d(self.in_dim, affine=False, eps=1e-6)
        self.head = torch.nn.Linear(self.in_dim, self.num_classes)
        # manually initialize fc layer: following MoCo v3
        trunc_normal_(self.head.weight, std=0.01)
    
    def forward(self, z):
        return self.head(self.bn(z))



def evaluate(data_loader, vq_model, probe, device, causal_type, num_code):
    # switch to evaluation mode
    vq_model.eval()
    probe.eval()

    correct_preds_num = 0
    total_sum = len(data_loader.dataset)

    for batch in tqdm(data_loader, desc='Evaluating', disable=dist.get_rank() == 0):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                if args.from_decoder:
                    quant, diff, spatial = vq_model.encode(
                                                images, 
                                                return_feat=False, 
                                                return_fix_dim_feat=False,
                                                return_code=False, 
                                                causal_type=causal_type,
                                                num_en_q_level=num_code,
                                                )
                    # get the feature for linear probe
                    _, _, feat= vq_model.decode(
                                            quant, 
                                            ret_inner_feat=False,
                                            return_feat=True
                                            )
                    # avg pool, for B ,L, C 
                    feat = feat.mean(dim=1)
                else:
                    feat, _, _ = vq_model.encode(
                                    images, 
                                    return_feat=not args.fix_dim, 
                                    return_fix_dim_feat=args.fix_dim,
                                    return_code=False, 
                                    causal_type=causal_type,
                                    num_en_q_level=num_code,
                                    )
                    # avg poll
                    feat = feat.reshape(feat.shape[0], feat.shape[1], -1).mean(dim=-1)
            outputs = probe(feat)
        
        # calculate the true number
        try:
            correct_preds_num += torch.sum(outputs.argmax(dim=1) == target)
        except Exception as e:
            print("feat shape:", feat.shape)
            print("outputs shape:", outputs.shape)
            print("target shape:", target.shape)
            print("images shape:", images.shape)
            raise e

    return correct_preds_num, total_sum


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
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."

    global_batch_size = dist.get_world_size() * args.batch_size
    rank = dist.get_rank()
    node_rank = int(os.environ.get('NODE_RANK', 0))
    ngpus = int(dist.get_world_size())
    print("rank", rank)
    print("node_rank", node_rank)
    print("ngpus", ngpus)
    print("global batch size:", global_batch_size)
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)


    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)
    # the causal settings are the attributes of a model, and also necessary for training process
    causal_type = config["model"]["causal_settings"]["causal_type"]
    dynamic_length_train = config["model"]["causal_settings"]["dynamic_length_train"]
    min_level = config["model"]["causal_settings"]["min_level"]
    dynamic_level_range = config["model"]["causal_settings"].get("dynamic_level_range", None)


    if args.sub_exp_dir is not None:
        experiment_dir = f"{args.save_path}/{args.sub_exp_dir}"  # Create an experiment folder
    else:
        raise ValueError("Please specify a sub_exp_dir")
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints


    final_checkpoint_path = f"{checkpoint_dir}/{args.epochs - 1:04d}.pt"
    if not args.eval and os.path.exists(final_checkpoint_path):
        print("Training already finished.")
        dist.barrier()
        dist.destroy_process_group()
        return

    if args.eval and os.path.exists(os.path.join(experiment_dir, 'lin_probe_acc.txt')):
        # print the file
        with open(os.path.join(experiment_dir, 'lin_probe_acc.txt'), 'r') as f:
            print(f.read())
        dist.barrier()
        dist.destroy_process_group()
        return

    # Setup an experiment folder:
    if rank == 0 and node_rank == 0:
        os.makedirs(args.save_path, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # # wandb
        # init_wandb(
        #     project_name=wandb_project_name,
        #     config={"dataset":"imagenet"},
        #     exp_dir=experiment_dir,
        #     )
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    vq_model = load_model_from_config(config)

    # create and load model
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")

    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")

    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    vq_model.load_state_dict(model_weight)


    if config["model"]["model_cls"] == "VQVitModelPlus":
        is_2d = False
    elif config["model"]["model_cls"] == "VQVitModel":
        is_2d = False
    elif config["model"]["model_cls"] == "VQVitModel2DPlus":
        is_2d = True
    else:
        raise NotImplementedError()

    # check the class of the vq_model
    # if isinstance(vq_model, VQVitModelPlus):
    #     print("please check the class of the vq_model, it is recognized as VQVitModelPlus now")
    #     is_2d = False
    # elif isinstance(vq_model, VQVitModel2DPlus):
    #     is_2d = True
    # else:
    #     print("please check the class of the vq_model")
    #     print(type(vq_model))
    #     is_2d = False

    del checkpoint
    if args.from_decoder:
        pass
    else:
        del vq_model.decoder
        if is_2d:
            del vq_model.s2ddecoder
        else:
            del vq_model.s1to2decoder
    torch.cuda.empty_cache()
    vq_model.to(device)
    vq_model.eval()
    # set vq_model require_grad as False
    for _, p in vq_model.named_parameters():
        p.requires_grad = False


    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # linear probe: weak augmentation
    # transform_train = transforms.Compose([
    #         RandomResizedCrop(224, interpolation=3),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # transform_val = transforms.Compose([
    #         transforms.Resize(256, interpolation=3),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = ImageFolder(os.path.join(args.data_root, 'ILSVRC2012_img_train'), transform=transform_train)
    dataset_val = ImageFolder(os.path.join(args.data_root, 'ILSVRC2012_img_val'), transform=transform_val)
    print(dataset_train)
    print(dataset_val)

    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )

    logger.info(f"Dataset contains {len(dataset_train):,} images ({args.data_root})")
    if args.dist_eval:
        # if len(dataset_val) % num_tasks != 0:
        #     print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
        #             'This will slightly alter validation results as extra duplicate entries are added to achieve '
        #             'equal num of samples per-process.')
        sampler_val = DistributedSampler(
            dataset_val, 
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True
            )  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # loading linear probe
    if args.fix_dim:
        # use the prev conv dimension
        if is_2d:
            probe = LinProb(in_dim=vq_model.s2dencoder.width).to(device)
        else:
            probe = LinProb(in_dim=vq_model.s2to1encoder.token_size).to(device)
    elif args.from_decoder:
        if is_2d:
            probe = LinProb(in_dim=vq_model.s2ddecoder.width).to(device)
        else:
            print("inner dimension:", vq_model.s1to2decoder.width)
            probe = LinProb(in_dim=vq_model.s1to2decoder.width).to(device)
    else:
        if is_2d:
            probe = LinProb(in_dim=vq_model.s2dencoder.width).to(device)
        else:
            probe = LinProb(in_dim=vq_model.s2to1encoder.width).to(device)
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size * ngpus / 256

    optimizer = LARS(
        probe.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay)


    if args.eval:
        # load checkpoint
        checkpoint = torch.load(args.lin_probe_ckpt, map_location="cpu")
        model_weight = checkpoint["model"]
        probe.load_state_dict(model_weight)
        probe.to(device)
        probe.eval()
        del checkpoint
    else:
        if len(glob(f"{checkpoint_dir}/*.pt")) != 0:
            latest_checkpoint = max(glob(f"{checkpoint_dir}/*.pt"), key=get_int_prefix_value)
            checkpoint = torch.load(latest_checkpoint, map_location="cpu")
            model_weight = checkpoint["model"]
            probe.load_state_dict(model_weight)
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = train_steps // (len(dataset_train) // global_batch_size)
            logger.info(f"Resume training from checkpoint: {latest_checkpoint}")
            logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            train_steps = 0
            start_epoch = 0           

        # auto resume accurately
        probe = DDP(probe.to(device), device_ids=[args.gpu])
        probe.train()


        print(optimizer)

    if args.eval:
        assert args.lin_probe_ckpt is not None
        # load the probe model


        corect_pred_num, totoal_num = evaluate(data_loader_val, vq_model, probe, device, causal_type, args.num_code)
        torch.cuda.synchronize()
        corect_pred_num = torch.tensor(corect_pred_num, device=device)
        totoal_num = torch.tensor(totoal_num, device=device)
        
        dist.all_reduce(corect_pred_num, op=dist.ReduceOp.SUM)
        dist.all_reduce(totoal_num, op=dist.ReduceOp.SUM)
        accuracy = corect_pred_num.item() / totoal_num.item()

        if rank == 0 and node_rank == 0:
            print(f"Tokenizer: Accuracy of the network on the {len(dataset_val)} test images: {accuracy:.4f}")
            print(f"from decoder: {args.from_decoder}")
            print(f"fixed dim: {args.fix_dim}")
            with open(os.path.join(experiment_dir, 'lin_probe_acc.txt'), 'a') as f:
                print(f"Tokenizer: Accuracy of the network on the {len(dataset_val)} test images: {accuracy:.4f}", file=f)
                print(f"from decoder: {args.from_decoder}")
                print(f"fixed dim: {args.fix_dim}")

        dist.barrier()
        dist.destroy_process_group()
        return
  
    # total_steps = int(len(dataset_train) / (args.batch_size * ) )  * args.epochs


    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    wandb_updates = []

    # train_steps = 0
    start_step_one_epoch = train_steps % (len(dataset_train) // global_batch_size)
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler_train.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        if epoch == start_epoch:
            loader_iter = iter(data_loader_train)
            loader_iter = islice(data_loader_train, start_step_one_epoch, None)
        else:
            loader_iter = iter(data_loader_train)

        for x, y in loader_iter:
            adjust_learning_rate(optimizer, train_steps / len(data_loader_train) , args)
            imgs = x.to(device, non_blocking=True)
            targets = y.to(device, non_blocking=True)
            # generator training
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype): 
                with torch.no_grad():
                # print(imgs.shape)
                    if args.from_decoder:
                        quant, diff, spatial = vq_model.encode(
                                                    imgs, 
                                                    return_feat=False, 
                                                    return_fix_dim_feat=False,
                                                    return_code=False, 
                                                    causal_type=causal_type,
                                                    num_en_q_level=args.num_code,
                                                    )
                        # get the feature for linear probe
                        _, _, feat= vq_model.decode(
                                                quant, 
                                                ret_inner_feat=False,
                                                return_feat=True
                                                )
                        # avg pool, for B ,L, C 
                        feat = feat.mean(dim=1)
                    else:
                        feat, _, _ = vq_model.encode(
                                        imgs, 
                                        return_feat=not args.fix_dim, 
                                        return_fix_dim_feat=args.fix_dim,
                                        return_code=False, 
                                        causal_type=causal_type,
                                        num_en_q_level=args.num_code,
                                        )
                        # avg pool, for B, C, H, W
                        feat = feat.reshape(feat.shape[0], feat.shape[1], -1).mean(dim=-1)
                outputs = probe(feat)
                loss = criterion(outputs, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                exit()

            loss_scaler(loss, optimizer, clip_grad=None,
                        parameters=probe.parameters(), create_graph=False,
                        update_grad=True)
            # optimizer.zero_grad()

            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}")
                running_loss = 0
                log_steps = 0

            # Save checkpoint:
        if (epoch % args.ckpt_epoch == 0 and epoch > 0) or epoch == args.epochs - 1:
            if rank == 0 and node_rank == 0:
                model_weight = probe.module.state_dict()  
                checkpoint = {
                    "model": model_weight,
                    "optimizer": optimizer.state_dict(),
                    "steps": train_steps,
                    "epoch": epoch,
                    "args": args
                }

                checkpoint_path = f"{checkpoint_dir}/{epoch:04d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

                # wandb_cache_file_append(wandb_updates, exp_dir=experiment_dir)
                # upload_wandb_cache(exp_dir=experiment_dir)
                # wandb.finish()
                # init_wandb(
                #     project_name=wandb_project_name,
                #     config={"dataset":"imagenet"},
                #     exp_dir=experiment_dir,
                #     )
                
                # manage_ckpt_num(
                #     checkpoint_dir,
                #     milestone_step=args.milestone_step,
                #     milestone_start=args.milestone_start,
                #     max_milestone_num=args.max_milestone_num
                # )

            # wandb_updates = [] 
            dist.barrier()

    vq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    # if rank == 0 and node_rank == 0:
    #     wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--save-path", type=str, default="results_lin_prob")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path")

    # loss configs
    parser.add_argument("--dataset", type=str, default='imagenet')

    # training configs
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--blr", type=float, default=0.1)  # lr = blr * effective batch size / 256.
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument('--batch-size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--min-lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    # parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='none', choices=["none", "fp16", "bf16"]) 

    # log and ckpts
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-epoch", type=int, default=10)
    parser.add_argument("--sub-exp-dir", type=str, default=None, help="sub experiment dir")
    # parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--num-code", type=int, default=None)
    parser.add_argument("--fix-dim",action='store_true', help="whether to fix the dim of the linear probe")
    parser.add_argument("--from-decoder", action='store_true', help="whether to use the decoder feature for the linear probe")

    # lin prob config
    parser.add_argument("--dist-eval", action='store_true', help="whether to use distributed evaluation")
    parser.add_argument("--lin-probe-ckpt", type=str, default=None, help="ckpt path")
    parser.add_argument("--eval", action='store_true', help="whether to use distributed evaluation")


    args = parser.parse_args()

    # wandb_project_name = args.wandb_project if args.wandb_project is not None else wandb_project_name
    main(args)