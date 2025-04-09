import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
import os
import json
from PIL import Image
import numpy as np
import math
import argparse
import shutil
import subprocess
import itertools
import yaml

from torchvision.utils import make_grid

from autoregressive.models.gpt_1d import GPT_models
from autoregressive.models.generate import generate

from utils.model_init import load_model_from_config
from utils.imgnet_idx2label import idx2firstlabel

def qualitative_eval(args):
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    with open(args.tok_config, "r") as f:
        tok_config = yaml.safe_load(f)

    num_token = tok_config["model"]["init_args"]["num_latent_tokens"] if args.num_token is None else args.num_token

    # Create and load tokenizer model
    if args.quant_way == 'vq':
        tokenizer_model = load_model_from_config(args.tok_config)
    elif args.quant_way == 'fsq':
        raise NotImplementedError
        tokenizer_model = FSQ_models[args.fsq_model](
            levels=codebook_size_to_levels(args.codebook_size_fsq),
        )
    else:
        raise ValueError("please check quant way")

    tokenizer_model.to(device)
    tokenizer_model.eval()
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

    codebook_size = tok_config["model"]["init_args"]["codebook_size"]
    tok_model_cls = tok_config["model"]["model_cls"]
    codebook_embed_dim = tok_config["model"]["init_args"]["codebook_embed_dim"]

    # Create folder to save samples:
    # Define folder name based on parameters
    model_string_name = args.gpt_model.replace("/", "-")
    folder_name = f"{model_string_name}-size-{args.image_size}-size-{args.image_size_eval}-{tok_model_cls}-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-{args.cfg_schedule}-seed-{args.global_seed}"

    sample_folder_dir = f"{args.sample_dir}/{folder_name}/"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving samples at {sample_folder_dir}")
    dist.barrier()

    # Load GPT model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=codebook_size,
        block_size=num_token,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        rope=args.rope,
        use_adaLN=args.adaLN,
        use_simple_adaLN=args.simple_adaLN,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:  # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint

    if args.compile:
        print(f"Compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        )  # requires PyTorch 2.0 (optional)
    else:
        print(f"No model compile")

    dist.barrier()

    # Prepare class indices
    if args.class_idx is not None:
        class_indices = args.class_idx
    else:
        # If no specific class is provided, sample randomly
        class_indices = list(range(args.num_classes))
    
    # Determine the number of classes to process per GPU
    total_classes = len(class_indices)
    classes_per_gpu = math.ceil(total_classes / dist.get_world_size())
    start_idx = rank * classes_per_gpu
    end_idx = min(start_idx + classes_per_gpu, total_classes)
    local_class_indices = class_indices[start_idx:end_idx]

    if rank == 0:
        print(f"Total classes to generate: {total_classes}")
        print(f"Classes assigned to this GPU: {local_class_indices}")

    # Prepare sampling parameters
    samples_per_class = math.ceil(args.qual_num / total_classes)
    pbar = tqdm(local_class_indices, desc=f"Rank {rank} Sampling Classes") if rank == 0 else local_class_indices

    # Dictionary to hold images for visualization
    images_single_class = {}
    images_multi_class = []

    total_generated = 0
    for cls_idx in pbar:
        # Generate samples for the current class
        n = args.per_proc_batch_size
        # Adjust batch size if necessary
        num_batches = math.ceil(samples_per_class / n)
        class_images = []

        for _ in range(num_batches):
            current_batch_size = min(n, samples_per_class - len(class_images))
            if current_batch_size <= 0:
                break
            # Sample inputs with the specified class index
            c_indices = torch.full((current_batch_size,), cls_idx, device=device)
            
            qzshape = [len(c_indices), codebook_embed_dim, 1, num_token]

            if args.cfg_schedule == "step":
                cfg_schedule_kwargs = {
                    "window_start": 0.18,
                    "window_end": 1.0,
                    }
            else:
                cfg_schedule_kwargs = {}
            
            with torch.no_grad():
                index_sample = generate(
                    gpt_model, c_indices, num_token,
                    cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
                    temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, sample_logits=True,
                    cfg_schedule=args.cfg_schedule,
                    cfg_schedule_kwargs=cfg_schedule_kwargs,
                )

                samples = tokenizer_model.decode_code(index_sample, qzshape)  # output value is between [-1, 1]

            if args.image_size_eval != args.image_size:
                samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')

            # size: [batch_size, height, width, channel]
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Collect samples for the current class
            for sample in samples:
                class_images.append(sample)
                images_multi_class.append((cls_idx, sample))
                total_generated += 1
                if total_generated >= args.qual_num:
                    break
            if total_generated >= args.qual_num:
                break

        images_single_class[cls_idx] = class_images[:samples_per_class]

    dist.barrier()

    # Only rank 0 will handle visualization and saving
    # Gather `images_single_class` from all ranks
    gather_single_class = [None for _ in range(dist.get_world_size())]
    gather_multi_class = [None for _ in range(dist.get_world_size())]

    dist.all_gather_object(gather_single_class, images_single_class)
    dist.all_gather_object(gather_multi_class, images_multi_class)

    if rank == 0:
        # Merge gathered data
        # gather_single_class = list(itertools.chain(*gather_single_class))
        # gather_multi_class = list(itertools.chain(*gather_multi_class))
        all_images_single_class = {}
        all_images_multi_class = []
        if gather_single_class is not None:
            for gpu_data in gather_single_class:
                for cls, imgs in gpu_data.items():
                    if cls not in all_images_single_class:
                        all_images_single_class[cls] = []
                    all_images_single_class[cls].extend(imgs)

        if gather_multi_class is not None:
            for gpu_data in gather_multi_class:
                all_images_multi_class.extend(gpu_data)

        # Create visualization grids
        # 1. Image Grid for Each Single Class
        for cls_idx, imgs in all_images_single_class.items():
            # make sure the number of images is a multiple of 6
            to_show_imgs = torch.tensor(imgs).permute(0, 3, 1, 2)   # B, C, H, W
            to_show_num = to_show_imgs.shape[0] - to_show_imgs.shape[0] % 6
            to_show_imgs = to_show_imgs[:to_show_num]
            print(to_show_imgs.shape)
            grid = make_grid(to_show_imgs, nrow=6, padding=2)
            print(grid.shape)
            # 2. Permute the dimensions from [C, H, W] to [H, W, C]
            grid = grid.permute(1, 2, 0)
            grid_image = Image.fromarray(grid.numpy().astype(np.uint8))
            grid_image.save(os.path.join(sample_folder_dir, f"class_{cls_idx}_grid.png"))
            print(f"Saved grid for class {cls_idx} with {len(imgs)} images.")

        # 2. Image Grid for Different Classes
        # Assume you want to arrange one image per class in a grid
        if len(all_images_multi_class) > 0:
            # Select one image per class for the grid
            unique_classes = {}
            for cls_idx, img in all_images_multi_class:
                if cls_idx not in unique_classes:
                    unique_classes[cls_idx] = img
                if len(unique_classes) >= len(class_indices):
                    break
            grid_imgs = [unique_classes[cls] for cls in sorted(unique_classes.keys())]
            to_show_imgs = torch.tensor(grid_imgs).permute(0, 3, 1, 2)
            to_show_num = to_show_imgs.shape[0] - to_show_imgs.shape[0] % 6
            to_show_imgs = to_show_imgs[:to_show_num]
            grid = make_grid(to_show_imgs, nrow=6, padding=2)
            grid = grid.permute(1, 2, 0)    # [C, H, W] -> [H, W, C]
            grid_image = Image.fromarray(grid.numpy().astype(np.uint8))
            grid_image.save(os.path.join(sample_folder_dir, f"multi_class_grid.png"))
            print(f"Saved multi-class grid with {len(grid_imgs)} images.")
            # save the actual cls indices as a 2d array
            cls_indices = [cls for cls in sorted(unique_classes.keys())]
            cls_indices = cls_indices[:to_show_num]
            cls_array = np.array(cls_indices).reshape(len(cls_indices) // 6, 6)
            np.savetxt(os.path.join(sample_folder_dir, f"multi_class_grid_cls_indices.txt"), cls_array, fmt="%d")

    dist.barrier()
    dist.destroy_process_group()


def sample_visualization(args):
    """
    1. Can control the number of images to generate
    2. Can control the number of classes to generate (and which)
       Images from the same class will be saved in the same folder
    """
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    with open(args.tok_config, "r") as f:
        tok_config = yaml.safe_load(f)

    num_token = tok_config["model"]["init_args"]["num_latent_tokens"] if args.num_token is None else args.num_token

    # Create and load tokenizer model
    if args.quant_way == 'vq':
        tokenizer_model = load_model_from_config(args.tok_config)
    elif args.quant_way == 'fsq':
        raise NotImplementedError
        tokenizer_model = FSQ_models[args.fsq_model](
            levels=codebook_size_to_levels(args.codebook_size_fsq),
        )
    else:
        raise ValueError("please check quant way")

    tokenizer_model.to(device)
    tokenizer_model.eval()
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

    codebook_size = tok_config["model"]["init_args"]["codebook_size"]
    tok_model_cls = tok_config["model"]["model_cls"]
    codebook_embed_dim = tok_config["model"]["init_args"]["codebook_embed_dim"]

    # Define folder name based on parameters
    model_string_name = args.gpt_model.replace("/", "-")
    folder_name = f"{model_string_name}-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temp-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-{args.cfg_schedule}-seed-{args.global_seed}"

    sample_folder_dir = f"{args.sample_dir}/{folder_name}/"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving samples at {sample_folder_dir}")
    dist.barrier()

    # Load GPT model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=codebook_size,
        block_size=num_token,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        rope=args.rope,
        use_adaLN=args.adaLN,
        use_simple_adaLN=args.simple_adaLN,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:  # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint

    if args.compile:
        print(f"Compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        )  # requires PyTorch 2.0 (optional)
    else:
        print(f"No model compile")

    dist.barrier()

    # Prepare class indices
    if args.class_idx is not None:
        class_indices = args.class_idx
    else:
        # If no specific class is provided, sample randomly
        class_indices = list(range(args.num_classes))
    
    # Determine the number of classes to process per GPU
    total_classes = len(class_indices)
    classes_per_gpu = math.ceil(total_classes / dist.get_world_size())
    start_idx = rank * classes_per_gpu
    end_idx = min(start_idx + classes_per_gpu, total_classes)
    local_class_indices = class_indices[start_idx:end_idx]

    if rank == 0:
        print(f"Total classes to generate: {total_classes}")
        print(f"Classes assigned to this GPU: {local_class_indices}")

    # Prepare sampling parameters
    samples_per_class = math.ceil(args.qual_num / total_classes)
    pbar = tqdm(local_class_indices, desc=f"Rank {rank} Sampling Classes") if rank == 0 else local_class_indices

    # Dictionary to hold images for visualization
    images_single_class = {}

    total_generated = 0
    rank = int(os.environ.get('RANK', 0))

    for cls_idx in pbar:
        # Generate samples for the current class
        n = args.per_proc_batch_size
        # Adjust batch size if necessary
        num_batches = math.ceil(samples_per_class / n)
        cls_saved_imgs_num = 0

        for b_i in range(num_batches):
            current_batch_size = min(n, samples_per_class - cls_saved_imgs_num)
            if current_batch_size <= 0:
                break
            # Sample inputs with the specified class index
            c_indices = torch.full((current_batch_size,), cls_idx, device=device)
            
            qzshape = [len(c_indices), codebook_embed_dim, 1, num_token]

            if args.cfg_schedule == "step":
                cfg_schedule_kwargs = {
                    "window_start": 0.18,
                    "window_end": 1.0,
                    }
            else:
                cfg_schedule_kwargs = {}
            
            with torch.no_grad():
                index_sample = generate(
                    gpt_model, c_indices, num_token,
                    cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
                    temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, sample_logits=True,
                    cfg_schedule=args.cfg_schedule,
                    cfg_schedule_kwargs=cfg_schedule_kwargs,
                )

                samples = tokenizer_model.decode_code(index_sample, qzshape)  # output value is between [-1, 1]

            if args.image_size_eval != args.image_size:
                samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')

            # size: [batch_size, height, width, channel]
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            save_path = os.path.join(sample_folder_dir, f"class_{cls_idx}_{idx2firstlabel(cls_idx)}")
            os.makedirs(save_path, exist_ok=True)
            # Collect samples for the current class
            for i, sample in enumerate(samples):
                index = cls_saved_imgs_num
                s_f_path = os.path.join(save_path, f"{idx2firstlabel(cls_idx)}_{index:06d}.png")
                Image.fromarray(sample).save(s_f_path)
                cls_saved_imgs_num += 1
                total_generated += 1
                if total_generated >= args.qual_num:
                    break
            if total_generated >= args.qual_num:
                break
            # directly save the samples

        # images_single_class[cls_idx] = class_images[:samples_per_class]

    dist.barrier()

    # Only rank 0 will handle visualization and saving
    # Gather `images_single_class` from all ranks
    # gather_single_class = [None for _ in range(dist.get_world_size())]
    # dist.all_gather_object(gather_single_class, images_single_class)

    # if rank == 0:
    #     # Merge gathered data
    #     # gather_single_class = list(itertools.chain(*gather_single_class))
    #     # gather_multi_class = list(itertools.chain(*gather_multi_class))
    #     all_images_single_class = {}
    #     if gather_single_class is not None:
    #         for gpu_data in gather_single_class:
    #             for cls_idx, imgs in gpu_data.items():
    #                 if cls_idx not in all_images_single_class:
    #                     all_images_single_class[cls_idx] = []
    #                 all_images_single_class[cls_idx].extend(imgs)

        # save the samples per class
        # for cls_idx, imgs in all_images_single_class.items():
        #     save_path = os.path.join(sample_folder_dir, f"class_{cls_idx}_{idx2firstlabel(cls_idx)}")
        #     os.makedirs(save_path, exist_ok=True)
        #     for i, img in enumerate(imgs):
        #         img = Image.fromarray(img.astype(np.uint8))
        #         img.save(os.path.join(save_path, f"{idx2firstlabel(cls_idx)}_{i}.png"))
        #     print(f"Saved {len(imgs)} images for class {cls_idx} ({idx2firstlabel(cls_idx)})")

    dist.barrier()
    dist.destroy_process_group()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")

    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)

    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--tok-config", type=str, default=None, help="config path for vq model")

    parser.add_argument("--quant-way", type=str, choices=['vq', 'fsq'], default='vq')

    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=2.0)
    parser.add_argument("--cfg-schedule", type=str, default="constant")
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    parser.add_argument("--num-token", type=int, default=None, help="number of tokens to sample with")

    parser.add_argument("--qual-num", type=int, default=100, help="Total number of images to generate")
    parser.add_argument("--rope", action='store_true', help="whether using rotary embedding")
    parser.add_argument("--adaLN", action='store_true', help="whether using adaptive layer normalization")
    parser.add_argument("--simple-adaLN", action='store_true', help="whether using simple adaptive layer normalization")

    # New argument for specifying class indices
    to_list = lambda x: list(map(int, x.split(',')))
    parser.add_argument("--class-idx", type=to_list, default=[207, 360, 387, 974, 88, 979, 417, 279], help="class indices to sample with")
    args = parser.parse_args()

    # qualitative_eval(args)
    sample_visualization(args)

