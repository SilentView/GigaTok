"""
Calculate the valiation cross entropy loss for gpt models on ImageNet val set.
Plan to also implement the perplexity calculation.
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from glob import glob
import json
from copy import deepcopy
import os
import shutil
import time
import inspect
import argparse
from itertools import islice
import itertools
from tqdm import tqdm

from utils.logger import create_logger
# from utils.resume_log import (
#     init_wandb, upload_wandb_cache, wandb_cache_file_append,
#     manage_ckpt_num, get_int_prefix_value, wsd_find_newest_ckpt
# )
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from dataset.augmentation import center_crop_arr
# from autoregressive.models.gpt import GPT_models
from autoregressive.models.gpt_1d import GPT_models
from autoregressive.models.gpt import GPT_models_2d
# from autoregressive.models.gpt_ic import GPT_IC_models

from tokenizer.tokenizer_image.vq.vq_train import wsd_find_newest_ckpt

from autoregressive.train.train_c2i import create_local_tokenizer

import yaml




def print_master(msg):
    if dist.get_rank() == 0:
        print(msg)


#################################################################################
#                                  Evaluation
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    init_distributed_mode(args)
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    args.global_batch_size = args.per_proc_batch_size * dist.get_world_size()
    rank = dist.get_rank()
    node_rank = int(os.environ.get('NODE_RANK', 0))
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)


    model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)

    with open(args.tok_config, "r") as f:
        tok_config = yaml.safe_load(f)

    causal_type = tok_config["model"]["causal_settings"]["causal_type"]


    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
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
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print_master(f"Dataset contains {len(dataset):,} images ({args.data_path})")


    latent_size = args.image_size // args.downsample_size
    if args.gpt_2d:
        model = GPT_models_2d[args.gpt_model](
            vocab_size=tok_config["model"]["init_args"]["codebook_size"],
            block_size=tok_config["model"]["init_args"]["num_latent_tokens"],
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
            resid_dropout_p=0,
            ffn_dropout_p=0,
            drop_path_rate=0,
            token_dropout_p=0,
        ).to(device)
        pass
    else:
        model = GPT_models[args.gpt_model](
            vocab_size=tok_config["model"]["init_args"]["codebook_size"],
            block_size=tok_config["model"]["init_args"]["num_latent_tokens"],
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
            resid_dropout_p=0,
            ffn_dropout_p=0,
            drop_path_rate=0,
            token_dropout_p=0,
            rope=args.rope,
            use_adaLN=args.adaLN,
            use_simple_adaLN=args.simple_adaLN,
            use_qk_norm=args.qk_norm,
            use_flash_attn=args.flash_attn,
        ).to(device)
    print_master(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")


    # load the tokenizer
    if args.quant_way == 'fsq':
        raise NotImplementedError
 
    tokenizer = create_local_tokenizer(args=args, device=device)
      
    # load the gpt model
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp: # fsdp
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    model.load_state_dict(model_weight, strict=False)
    model.eval()
    del checkpoint

    if args.compile:
        print(f"compiling the model...")
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile") 


    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]


    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision =='fp16'))
    # Variables for monitoring/logging purposes:
    running_loss_list = []
    loader = tqdm(loader) if rank == 0 else loader
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
        with torch.cuda.amp.autocast(dtype=ptdtype):  
            with torch.no_grad():
                logits, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)
        running_loss_list.append(loss.item())

    dist.barrier()
    world_size = dist.get_world_size()
    gather_loss_val = [None for _ in range(world_size)]
    dist.all_gather_object(gather_loss_val, running_loss_list)

    gpt_iter = args.gpt_ckpt.split('/')[-1].split('.')[0]
    gpt_folder = os.path.dirname(os.path.dirname(args.gpt_ckpt))
    if node_rank == 0 and rank == 0:
        gather_loss_val = list(itertools.chain(*gather_loss_val))
        loss_val = sum(gather_loss_val) / len(gather_loss_val)
        print("val_loss: %f " % loss_val)
        result_file = f"{gpt_folder}/val_results.txt"
        print("writing results to {}".format(result_file))
        with open(result_file, 'a') as f:
                print("gpt_iter: %s " % gpt_iter, file=f)
                print("val_loss: %f /n" % loss_val, file=f)
        print("Done!")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-fsdp", action='store_true')


    parser.add_argument("--code-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--gpt-2d", action="store_true", help="use 2D GPT model")

    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--save-path", type=str, default="results/gpt")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")

    parser.add_argument("--quant-way", type=str, choices=['vq', 'fsq'], default='vq')

    # model specific arguments for vq
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--tok-config", type=str, default=None, help="config path for vq model")

    # model specific arguments for fsq
    parser.add_argument("--early-stop-iter", type=int, default=None)
    parser.add_argument("--rope", action='store_true', help="whether using rotary embedding")
    parser.add_argument("--adaLN", action='store_true', help="whether using adaptive layer normalization")
    parser.add_argument("--simple-adaLN", action='store_true', help="whether using simple adaptive layer normalization")
    parser.add_argument("--qk-norm", action='store_true', help="whether using query and key normalization")
    parser.add_argument("--flash-attn", action='store_true', help="whether using flash attention")

    args = parser.parse_args()
    # default_wandb_name = None
    # wandb_project_name = args.wandb_project if args.wandb_project is not None else default_wandb_name
    main(args)


