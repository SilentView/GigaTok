# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, size_based_auto_wrap_policy

from torchvision import transforms
from glob import glob
import json
from copy import deepcopy
import os
import shutil
import time
import inspect
import functools
import argparse
from itertools import islice
import contextlib
from math import ceil


from utils.logger import create_logger
from utils.resume_log import (
    init_wandb, upload_wandb_cache, wandb_cache_file_append,
    manage_ckpt_num, get_int_prefix_value, wsd_find_newest_ckpt
)
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt_1d import GPT_models
from tokenizer.tokenizer_image.fsq.fsq_model import FSQModel
from tokenizer.tokenizer_image.fsq.fsq_model import FSQ_models

from dataset.augmentation import random_crop_arr
from utils.model_init import load_model_from_config

from tokenizer.tokenizer_image.scheduler import const_lr, cosine_schedule_with_warmup_v2, wsd_lr
from tokenizer.tokenizer_image.vq.vq_train import wsd_find_newest_ckpt

import yaml


import wandb

wandb_project_name = "gpt_vqvit_imagenet_train"

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def create_optimizer_by_name(model, weight_decay, learning_rate, betas, rank, logger):
    # start with all of the candidate parameters
    all_param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in all_param_dict.items() if p.requires_grad}
    
    # create optim groups. 
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    
    # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    # model params are flatten by fsdp, we need to set the params by its name
    decay_params = [p for n, p in param_dict.items() if 'norm' not in n]
    nodecay_params = [p for n, p in param_dict.items() if 'norm' in n]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"(rank {rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"(rank {rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    print(f"(rank {rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"(rank {rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer



def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace, device) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        # auto_wrap_policy=size_based_auto_wrap_policy,
        # process_group=fs_init.get_data_parallel_group(),
        device_id=device,
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
                "none": torch.float,
            }[args.mixed_precision],
            reduce_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
                "none": torch.float,
            }[args.grad_precision or args.mixed_precision],
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=True,
    )

    torch.cuda.synchronize()

    return model



def codebook_size_to_levels(codebook_size:str) -> list:
    """
    The input codebook size must be a string like "a^b"
    examples:
    8^8 -> [8, 8, 8, 8, 8, 8, 8, 8]
    """
    base = codebook_size.split("^")[0]
    exponent = codebook_size.split("^")[1]
    levels = [int(base)] * int(exponent)
    return levels


def create_local_tokenizer(args, device):
    """
    Create a tokenizer that is local to each rank.
    TODO: Change the input to clearly defined configs.
    """
    # rank = dist.get_rank()
    # print("rank", rank)
    # device = rank % torch.cuda.device_count()
    # device = "cuda:" + str(dist.get_rank())
    # create and load model
    if args.quant_way == 'vq':
        tokenizer_model = load_model_from_config(args.tok_config)
    elif args.quant_way == 'fsq':
        raise NotImplementedError
        levels = codebook_size_to_levels(args.codebook_size_fsq)
        tokenizer_model = FSQ_models[args.fsq_model](
            levels=levels
        ).to(device)
    else:
        raise ValueError("please check quant way")

    if args.quant_way == 'vq':
        ckpt_path = args.vq_ckpt
    elif args.quant_way == 'fsq':
        ckpt_path = args.fsq_ckpt
    else:
        raise ValueError("please check quant way")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    tokenizer_model.load_state_dict(model_weight)

    del checkpoint
    del tokenizer_model.decoder
    if hasattr(tokenizer_model, "s1to2decoder"):
        del tokenizer_model.s1to2decoder
    torch.cuda.empty_cache()
    tokenizer_model.to(device)
    tokenizer_model.eval()

    return tokenizer_model



#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert args.gpt_type == 'c2i', "FSDP only supports c2i currently."
    
    # =======================================
    #    Initialize Distributed Training
    # =======================================
    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # =======================================
    #    Initialize logger and wandb
    # =======================================
    timestamp = None
    if rank == 0:
        timestamp = time.localtime()
        timestamp = int(time.strftime("%Y%m%d%H%M%S", timestamp))
    # Convert timestamp to a tensor for broadcasting
    timestamp_tensor = torch.tensor([timestamp] if timestamp is not None else [0.0], dtype=torch.double).to(device)
    # Broadcast the timestamp to all processes
    dist.broadcast(timestamp_tensor, src=0)
    # All processes receive the timestamp
    timestamp = int(timestamp_tensor.item())

    model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
    experiment_dir = f"{args.save_path}/{args.sub_exp_dir}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    optim_checkpoint_dir = f"{experiment_dir}/optim_checkpoints"  # Stores saved optimizer checkpoints

    with open(args.tok_config, "r") as f:
        tok_config = yaml.safe_load(f)

    causal_type = tok_config["model"]["causal_settings"]["causal_type"]

    total_steps = args.iterations
    # Setup an experiment folder:
    # configs for constant + cooldown training
    if args.lr_scheduler == "wsd":
        use_wsd = True
        # Temporally fixed fraction
        fract_decay = args.fract_decay
        cd_sub_dir = experiment_dir + "/cd_records" + f"/cd_fract_{fract_decay}_to_{total_steps:07d}"
        cd_checkpoint_dir = f"{cd_sub_dir}/checkpoints"
        cd_optim_checkpoint_dir = f"{cd_sub_dir}/optim_checkpoints"

        _, const_end_flag = wsd_find_newest_ckpt(
                                                config={"trainer": {}}, 
                                                const_ckpt_dir=checkpoint_dir, 
                                                cd_sub_dir=cd_sub_dir, 
                                                total_steps=total_steps,
                                                fract_decay=fract_decay,
                                                )

        # add extra wsd setting for trial_name
        trial_name = args.sub_exp_dir + f"_ar_cd_fract_{fract_decay}_to_{total_steps}"
    else:
        use_wsd = False
        trial_name = args.sub_exp_dir

    exp_dir = cd_sub_dir if (use_wsd and const_end_flag) else experiment_dir

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # os.makedirs(cd_sub_dir, exist_ok=True)
        os.makedirs(cd_checkpoint_dir, exist_ok=True)
        os.makedirs(cd_optim_checkpoint_dir, exist_ok=True) # Create optimizer checkpoint folder
        os.makedirs(optim_checkpoint_dir, exist_ok=True)

        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")

        if not args.no_wandb:
            try:
                init_wandb(
                    project_name=wandb_project_name,
                    config={"dataset":"imagenet"},
                    exp_dir=exp_dir,
                    name=trial_name
                    )
            except Exception as e:
                print(e)
                logger.info("wandb init failed, please check your wandb config")
                logger.info("Running without wandb")
                args.no_wandb = True
    else:
        logger = create_logger(None)
    # training args
    logger.info(f"{args}")
    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")



    # ======================================================
    #     Initialize Dataloader
    # ======================================================
    # Setup data:
    if args.dataset == "imagenet":
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        dataset = build_dataset(args, transform=transform)
    elif args.dataset == "imagenet_code":
        dataset = build_dataset(args)
    else:
        raise NotImplementedError

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


    # tokenizers will not be sharded in FSDP
    if args.dataset == 'imagenet_code':
        # flip_info = 'with' if dataset.flip else 'without'
        aug_info = 10 if 'ten_crop' in dataset.feature_dir else 1
        aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
        logger.info(f"Dataset contains {len(dataset):,} images ({args.code_path}) "
                    f"{aug_info} crop augmentation")
    else:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

        # load the tokenizer
        tokenizer = create_local_tokenizer(args=args, device=device)
        # save the config of the tokenizer
        if args.quant_way == 'vq':
            # copy the tok_config to the experiment folder
            shutil.copyfile(args.tok_config, f"{exp_dir}/{os.path.basename(args.tok_config)}")
            # create a txt file recording the vq ckpt path
            with open(f"{exp_dir}/vq_ckpt.txt", "w") as f:
                f.write(args.vq_ckpt)

        elif args.quant_way == 'fsq':
            raise NotImplementedError
            tokenizer_dict = {
                'quant_way': args.quant_way,
                'fsq_model': args.fsq_model,
                'codebook_size_fsq': args.codebook_size_fsq,
                'fsq_ckpt': args.fsq_ckpt, 
            }

  
    # ======================================================
    #     Initialize model and resume
    # ======================================================
    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    if args.grad_ckpt:
        raise NotImplementedError("grad_ckpt not implemented yet for FSDP")
    else:
        model = GPT_models[args.gpt_model](
            vocab_size=tok_config["model"]["init_args"]["codebook_size"],
            block_size=tok_config["model"]["init_args"]["num_latent_tokens"],
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
            resid_dropout_p=dropout_p,
            ffn_dropout_p=dropout_p,
            drop_path_rate=args.drop_path_rate,
            token_dropout_p=args.token_dropout_p,
            rope=args.rope,
            use_adaLN=args.adaLN,
            use_simple_adaLN=args.simple_adaLN,
            use_qk_norm=args.qk_norm,
        ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    torch.cuda.synchronize()
    dist.barrier()


    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

        
    # Prepare models for training:
    # We do auto resume for model training, by checking the latest checkpoint.
    # find the latest checkpoint
    if len(glob(f"{checkpoint_dir}/*.pt")) != 0:
        if args.lr_scheduler == "wsd":
            # Is the constant training already ended?
            resume_checkpoint, const_end_flag = wsd_find_newest_ckpt(
                                                    config={"trainer": {}}, 
                                                    const_ckpt_dir=checkpoint_dir, 
                                                    cd_sub_dir=cd_sub_dir, 
                                                    total_steps=total_steps,
                                                    fract_decay=fract_decay,
                                                    )

            latest_checkpoint = resume_checkpoint
            checkpoint = torch.load(latest_checkpoint, map_location="cpu")
        else:
            latest_checkpoint = max(glob(f"{checkpoint_dir}/*.pt"), key=get_int_prefix_value)
            checkpoint = torch.load(latest_checkpoint, map_location="cpu")

        model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
        start_epoch = train_steps // (len(dataset) // args.global_batch_size)
        del checkpoint
        logger.info(f"Resume training from checkpoint: {latest_checkpoint}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            with torch.no_grad():
                update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    if args.compile:
        raise NotImplementedError("compile not implemented yet for FSDP")
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        

    model = setup_fsdp_sync(model, args, device)
    torch.cuda.synchronize()
    dist.barrier()

    model.train()

    if args.ema:
        ema = setup_fsdp_sync(ema, args, device)
        ema.eval()
    else:
        ema = None


    # ======================================================
    #     Initialize optimizer and resume
    # ======================================================
    # a major different part for FSDP is that optimizer will be saved seperately for each rank
    optimizer = create_optimizer_by_name(
                    model, args.weight_decay, 
                    args.lr, (args.beta1, args.beta2), rank, logger
                )

    # auto resume for optimizer
    exp_dir = cd_sub_dir if (use_wsd and const_end_flag) else experiment_dir
    if len(glob(f"{optim_checkpoint_dir}/*")) != 0:

        if args.lr_scheduler == "wsd":
            # Is the constant training already ended?
            if const_end_flag:
                latest_optim_checkpoint_dir = f"{cd_optim_checkpoint_dir}/{train_steps:07d}"
                latest_optim_checkpoint = os.path.join(
                                            latest_optim_checkpoint_dir,
                                            f"optimizer.{rank:05d}-of-"
                                            f"{dist.get_world_size():05d}.pt",
                                          )
            else:
                latest_optim_checkpoint_dir = f"{optim_checkpoint_dir}/{train_steps:07d}"
                latest_optim_checkpoint = os.path.join(
                                            latest_optim_checkpoint_dir,
                                            f"optimizer.{rank:05d}-of-"
                                            f"{dist.get_world_size():05d}.pt",
                                          )
        else:
            latest_optim_checkpoint_dir = f"{optim_checkpoint_dir}/{train_steps:07d}"
            latest_optim_checkpoint = os.path.join(
                                        latest_optim_checkpoint_dir,
                                        f"optimizer.{rank:05d}-of-"
                                        f"{dist.get_world_size():05d}.pt",
                                        )

        opt_state_world_size = len([
            x for x in os.listdir(latest_optim_checkpoint_dir)
            if x.startswith("optimizer.") and x.endswith(".pt")
        ])
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )

        logger.info(f"Resuming optimizer states from: {latest_optim_checkpoint}")
        optimizer.load_state_dict(torch.load(latest_optim_checkpoint, map_location="cpu"))


    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0


    if args.lr_scheduler == "const":
        scheduler = const_lr(
                        optimizer, 
                        float(args.lr), 
                        int(args.warmup) if args.warmup is None else args.warmup, 
                        total_steps)
    elif args.lr_scheduler == "cosine":
        scheduler = cosine_schedule_with_warmup_v2(
                        optimizer,
                        float(args.lr),
                        int(args.warmup) if args.warmup is None else args.warmup,
                        total_steps,
                        end_lr=float(args.end_lr)
                    )
    elif args.lr_scheduler == "wsd":
        scheduler = wsd_lr(
                        optimizer,
                        base_lr=float(args.lr),
                        warmup_length=int(args.warmup) if args.warmup is None else args.warmup,
                        steps=total_steps,
                        fract_decay=float(fract_decay),
                    )

    else:
        logger.error(
            f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
        exit(1)


    start_time = time.time()
    start_step_one_epoch = train_steps % (len(dataset) // args.global_batch_size)
    wandb_updates = []

    epochs = ceil(total_steps / (len(dataset) // args.global_batch_size))

    logger.info(f"Training for {epochs} epochs...")
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

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            # print("x shape", x.shape)
            # print("y shape", y.shape)
            if args.dataset != 'imagenet_code':
                with torch.no_grad():
                    if args.quant_way == 'vq':
                        _, _, [_, _, indices] = tokenizer.encode(
                                                    x,
                                                    causal_type=causal_type,
                                                    )
                        # print(indices.shape)
                    else:
                        raise NotImplementedError
                    # elif args.quant_way == 'fsq':
                    #     _, indices, _ = tokenizer.encode(x)
            else:
                indices = x

            y = y.to(device, non_blocking=True)
            z_indices = indices.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0], f"z:{z_indices.shape}, c:{c_indices.shape}"

            ptdtype = {
                'none': torch.float32, 
                'bf16': torch.bfloat16, 
                'fp16': torch.float16

            }[args.mixed_precision]
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)
            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            if args.lr_scheduler != "const":
                current_lr = scheduler(train_steps)
            else:
                current_lr = args.lr

           # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            if args.ema:
                with torch.no_grad():
                    update_ema(ema, model.module)

            if use_wsd:
                # update the wsd state for constant stage
                const_end_flag = (train_steps >= hold_iterations)
                # also update the exp_dir
                exp_dir = cd_sub_dir if (use_wsd and const_end_flag) else experiment_dir

            # Log loss values:
            running_loss += loss.item()
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
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"\
                            f", lr: {current_lr:.4e}"
                    )
                wandb_updates.append(
                    {
                        "train_loss": avg_loss,
                        "train_steps_per_sec": steps_per_sec,
                        "iteration": train_steps,
                        "epoch": epoch,
                        "lr": current_lr,
                    }
                )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if (train_steps % args.ckpt_every == 0 and train_steps > 0) or train_steps == total_steps:
                if args.compile:
                    raise NotImplementedError
                    # model_weight = model.module._orig_mod.state_dict()

                ### saving model parameters
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    consolidated_model_state_dict = model.state_dict()
                    if args.ema:
                        # Save EMA state as well
                        with FSDP.state_dict_type(
                            ema,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                        ):
                            ema_state_dict = ema.state_dict()
                        checkpoint = {
                            "model": consolidated_model_state_dict,
                            "ema": ema_state_dict,
                            "steps": train_steps,
                            "args": args
                        }
                    else:
                        checkpoint = {
                            "model": consolidated_model_state_dict,
                            "steps": train_steps,
                            "args": args
                        }


                    if rank == 0:
                        save_ckpt_dir = cd_checkpoint_dir if (use_wsd and const_end_flag) else checkpoint_dir
                        checkpoint_path = f"{save_ckpt_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved consolidated to {checkpoint_path}")
    
                dist.barrier()
                del consolidated_model_state_dict
                if args.ema:
                    del ema_state_dict

                ### saving optimizer
                opt_state_fn = (
                    f"optimizer.{dist.get_rank():05d}-of-"
                    f"{dist.get_world_size():05d}.pt"
                )
                optim_save_ckpt_dir = cd_optim_checkpoint_dir if (use_wsd and const_end_flag) else optim_checkpoint_dir
                optimizer_save_path = os.path.join(optim_save_ckpt_dir, f"{train_steps:07d}", opt_state_fn)

                if not os.path.exists(os.path.dirname(optimizer_save_path)):
                    os.makedirs(os.path.dirname(optimizer_save_path), exist_ok=True)

                torch.save(
                    optimizer.state_dict(), 
                    optimizer_save_path
                )
                dist.barrier()
                logger.info(f"Saved optimizer to {optimizer_save_path}")


                if rank == 0:
                    if not args.no_wandb:
                        wandb_cache_file_append(wandb_updates, exp_dir=exp_dir)
                        upload_wandb_cache(exp_dir=exp_dir)
                        wandb.finish()
                        init_wandb(
                            project_name=wandb_project_name,
                            config={"dataset":"imagenet"},
                            exp_dir=exp_dir,
                            name=trial_name,
                            )
                        
                    manage_ckpt_num(
                        checkpoint_dir,
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
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    # depracated
    parser.add_argument("--grad-ckpt", action='store_true', help="whether using gradient checkpointing")

    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--save-path", type=str, default="results/gpt")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--iterations", type=int, default=1500_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--lr-scheduler", type=str, default="const", choices=["cosine", "const", "wsd"])
    parser.add_argument("--fract-decay", type=float, default=0.2, help="fraction of the total iterations to decay the learning rate, \
                        used when lr-scheduler=wsd")
    parser.add_argument("--end-lr", type=float, default=1e-5)
    parser.add_argument("--ema", action='store_true', help="whether using ema training")

    parser.add_argument("--sub-exp-dir", type=str, required=True, help="experiment directory for a single run")
    parser.add_argument("--milestone-step", type=int, default=50_000, help="milestone step for checkpoint saving")
    parser.add_argument("--milestone-start", type=int, default=50_000, help="milestone start for checkpoint saving")
    parser.add_argument("--max-milestone-num", type=int, default=40, help="max milestone num for checkpoint saving")
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    parser.add_argument("--no-wandb", action='store_true', help="whether not using wandb")


    parser.add_argument("--quant-way", type=str, choices=['vq', 'fsq'], default='vq')

    # model specific arguments for vq
    # parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--tok-config", type=str, default=None, help="config path for vq model")

    # model specific arguments for fsq, Not implemented yet
    parser.add_argument("--fsq-model", type=str, choices=list(FSQ_models.keys()), default="FSQ-16")
    parser.add_argument("--fsq-ckpt", type=str, default=None, help="ckpt path for fsq model")
    parser.add_argument("--codebook-size-fsq", type=str, default="8^8")
    to_int_or_none = lambda x: None if x == "None" else int(x)
    parser.add_argument("--early-stop-iter", type=to_int_or_none, default=None)
    parser.add_argument("--rope", action='store_true', help="whether using rotary embedding")
    parser.add_argument("--adaLN", action='store_true', help="whether using adaptive layer normalization")
    parser.add_argument("--simple-adaLN", action='store_true', help="whether using simple adaptive layer normalization")
    parser.add_argument("--qk-norm", action='store_true', help="whether using query and key normalization")


    # fsdp args
    parser.add_argument("--data-parallel", type=str, choices=["sdp", "fsdp", "hsdp"], default="sdp")
    parser.add_argument("--grad-precision", type=str, choices=["fp32", "fp16", "bf16"])

    args = parser.parse_args()
    wandb_project_name = args.wandb_project if args.wandb_project is not None else wandb_project_name

    main(args)
