import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import os
from datetime import timedelta
import shutil
import json
from glob import glob
from PIL import Image


import numpy as np
import argparse
import itertools
import subprocess

from tokenizer.tokenizer_image.lpips import LPIPS

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset
# from tokenizer.tokenizer_image.vq.vq_model import VQ_models
from tokenizer.tokenizer_image.fsq.fsq_model import FSQ_models
# from tokenizer.tokenizer_image.lq.lq_model import LQ_models

from utils.model_init import load_model_from_config, custom_load

import yaml



def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path



def quantitative_eval(args, cur_en_q_level=None):
    if args.merge:
        if torch.cuda.is_available():
            torch.set_grad_enabled(False)
            # Setup DDP:
            dist.init_process_group("nccl")
            rank = dist.get_rank()
        else:
            rank = int(os.environ['LOCAL_RANK'])
        if rank == 0:
            sample_dir = args.sample_dir[:-1] if args.sample_dir.endswith("/") else args.sample_dir
            name = args.model_config.split("/")[-1].split(".")[0] + "_eval"

            # merge all the .json files 
            results = []
            check_dir = os.path.join(os.path.dirname(sample_dir),
                                     "eval_results")
            for file in os.listdir(check_dir):
                if file.endswith(".json"):
                    with open(os.path.join(check_dir, file), "r") as f:
                        results.append(json.load(f))
            # upload the results
            return
        else:
            return

    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)

    causal_type = config["model"]["causal_settings"]["causal_type"]
    dynamic_length_train = config["model"]["causal_settings"]["dynamic_length_train"]
    min_level = config["model"]["causal_settings"]["min_level"]

    # store the results to args.sample_dir/
    sample_dir = args.sample_dir[:-1] if args.sample_dir.endswith("/") else args.sample_dir

    save_path = os.path.dirname(sample_dir) + "/eval_results/" + sample_dir.split("/")[-1] + ".json"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = json.load(f)
        # print the results
        node_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        if node_rank == 0 and rank == 0:
            print("Eval results:")
            for key, value in results.items():
                print(f"{key}: {value}")
        return


    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group(
        "nccl",
        timeout=timedelta(hours=1)  # 1 hour
        )
 
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    if args.quant_way == 'vq':
        tokenizer_model = load_model_from_config(config)
    elif args.quant_way == 'fsq':
        tokenizer_model = FSQ_models[args.fsq_model](
            levels=args.levels
        )
    elif args.quant_way == 'lq':
        tokenizer_model = LQ_models[args.lq_model](
            levels=args.levels
        )
    else:
        raise ValueError("please check quant way")

    tokenizer_model.to(device)
    tokenizer_model.eval()
    print(f"VQ Model Parameters(inference): {sum(p.numel() for p in tokenizer_model.parameters()):,}")
    if args.quant_way == 'vq':
        ckpt_path = args.vq_ckpt
    elif args.quant_way == 'fsq':
        ckpt_path = args.fsq_ckpt
    elif args.quant_way == 'lq':
        ckpt_path = args.lq_ckpt
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

    # tokenizer_model.load_state_dict(model_weight)
    custom_load(tokenizer_model, model_weight)
    del checkpoint

    # Create folder to save samples:
    if args.quant_way == 'vq':
        folder_name = (f"{config['model']['model_cls']}-{args.dataset}-size-{args.image_size}-size-{args.image_size_eval}"
                    f"-codebook-size-{config['model']['init_args']['codebook_size']}-dim-{config['model']['init_args']['codebook_embed_dim']}-seed-{args.global_seed}" + \
                    ("" if cur_en_q_level is None else f"-c-type-{causal_type}-cur-level-{cur_en_q_level}")
                    )
    elif args.quant_way == 'lq':
        folder_name = (f"{args.lq_model}-{args.dataset}-size-{args.image_size}-size-{args.image_size_eval}")
    elif args.quant_way == 'fsq':
        folder_name = (f"{args.fsq_model}-{args.dataset}-size-{args.image_size}-size-{args.image_size_eval}")
    else:
        raise ValueError("please check quant way")

    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()


    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    if args.dataset == 'imagenet':
        dataset = build_dataset(args, transform=transform)
        num_fid_samples = 50000
    elif args.dataset == 'coco':
        dataset = build_dataset(args, transform=transform)
        num_fid_samples = 5000
    else:
        raise Exception("please check dataset")
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
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

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()

    # loss_fn_alex = LPIPS(net='alex').to(device)  # best forward scores
    loss_fn_vgg = LPIPS().eval().to(device)   # closer to "traditional" perceptual loss, when used for optimization

    
    psnr_val_rgb = []
    ssim_val_rgb = []
    lpips_val = []
    loader = tqdm(loader) if rank == 0 else loader
    total = 0
    used_indices = []
    cnt = 0
    for x, _ in loader:
        if args.image_size_eval != args.image_size:
            rgb_gts = F.interpolate(x, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        else:
            rgb_gts = x

        ori_x = x.clone().to(device)

        rgb_gts = (rgb_gts.permute(0, 2, 3, 1).to("cpu").numpy() + 1.0) / 2.0 # rgb_gt value is between [0, 1]
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            if args.quant_way == 'vq':
                latent, _, [_, _, indices] = tokenizer_model.encode(
                                                x, 
                                                num_en_q_level=cur_en_q_level, 
                                                causal_type=causal_type)
                samples = tokenizer_model.decode_code(indices, latent.shape) # output value is between [-1, 1]
            elif args.quant_way == 'fsq':
                latent, indices, _ = tokenizer_model.encode(x)
                samples = tokenizer_model.decode_code(indices)
            elif args.quant_way == 'lq':
                latent, _, indices = tokenizer_model.encode(x)
                samples = tokenizer_model.decode_code(indices)
            if args.image_size_eval != args.image_size:
                samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')

            rec_x = samples.clone()
            
            used_indices.extend(indices.cpu().view(-1).numpy().tolist())
            cnt += 1
            if cnt % 1000 == 0:
                # deduplicate
                used_indices = list(set(used_indices)) 
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # lpips_alex = loss_fn_alex(ori_x, rec_x)
        lpips_vgg = loss_fn_vgg(ori_x, rec_x)
        lpips_val.append(lpips_vgg.mean().item())
        # Save samples to disk as individual .png files
        for i, (sample, rgb_gt) in enumerate(zip(samples, rgb_gts)):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            # metric
            rgb_restored = sample.astype(np.float32) / 255. # rgb_restored value is between [0, 1]
            psnr = psnr_loss(rgb_restored, rgb_gt)
            ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=2.0, channel_axis=-1)
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)
            
        total += global_batch_size

    # ------------------------------------
    #       Summary
    # ------------------------------------
    # Make sure all processes have finished saving their samples
    dist.barrier()
    world_size = dist.get_world_size()
    gather_psnr_val = [None for _ in range(world_size)]
    gather_ssim_val = [None for _ in range(world_size)]
    gather_lpips_val = [None for _ in range(world_size)]
    gather_used_indices = [None for _ in range(world_size)]

    dist.all_gather_object(gather_psnr_val, psnr_val_rgb)
    dist.all_gather_object(gather_ssim_val, ssim_val_rgb)
    dist.all_gather_object(gather_lpips_val, lpips_val)
    dist.all_gather_object(gather_used_indices, used_indices)


    if rank == 0:
        gather_psnr_val = list(itertools.chain(*gather_psnr_val))
        gather_ssim_val = list(itertools.chain(*gather_ssim_val))        
        gather_lpips_val = list(itertools.chain(*gather_lpips_val))
        gather_used_indices = list(itertools.chain(*gather_used_indices))
        psnr_val_rgb = sum(gather_psnr_val) / len(gather_psnr_val)
        ssim_val_rgb = sum(gather_ssim_val) / len(gather_ssim_val)
        lpips_val = sum(gather_lpips_val) / len(gather_lpips_val)
        try:
            if args.quant_way == 'vq':
                codebook_usage = len(set(gather_used_indices)) / tokenizer_model.quantize.n_e
            else:
                codebook_usage = len(set(gather_used_indices)) / tokenizer_model.quantize.codebook_size
        except TypeError as e:
            print(e)
            print(type(gather_used_indices[0][:10]))

        print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))
        print("LPIPS_vgg: %f " % lpips_val)
        print("codebook usage: %f" % (codebook_usage))

        result_file = f"{sample_folder_dir}_results.txt"
        print("writing results to {}".format(result_file))
        with open(result_file, 'w') as f:
            print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb), file=f)
            print("LPIPS_vgg: %f " % lpips_val, file=f)
            print("codebook usage: %f" % (codebook_usage), file=f)

        create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
        print("Done.")
    
    if rank == 0:
        """
        Further measure rFID
        columns = ["iteration", "PSNR", "SSIM", "rFID"]
        """
        def _wrap_key(key):
            if cur_en_q_level is None:
                return key
            else:
                return f"{key}_{cur_en_q_level}_{causal_type}"
        

        try:
            iteration = int(ckpt_path.split("/")[-1].split(".")[0])
        except:
            iteration = ""

        result_dict = {}
        result_dict.update(
            {
                _wrap_key("PSNR"): psnr_val_rgb,
                _wrap_key("SSIM"): ssim_val_rgb,
                _wrap_key("codebook_usage"): codebook_usage,
                _wrap_key("lpips_vgg"): lpips_val,
                "iteration": iteration
            }
        )
        # find the .npz file
        npz_file_path = f"{sample_folder_dir}.npz"
        # run the evaluator script
        evaluate_script = eval_script_template.replace("<test_npz_file>", npz_file_path)
        ret = subprocess.run(evaluate_script, shell=True)
        assert ret.returncode == 0, "evaluate script failed"

        txt_path = npz_file_path.replace(".npz", ".txt")
        with open(txt_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("FID"):
                result_dict.update({_wrap_key("rFID"): float(line.split(":")[-1].strip())})
            if line.startswith("sFID"):
                result_dict.update({_wrap_key("sFID"): float(line.split(":")[-1].strip())})

        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                results = json.load(f)
            results.update(result_dict)

            with open(save_path, "w") as f:
                json.dump(results, f, indent=4)
        else:
            with open(save_path, "w") as f:
                json.dump(result_dict, f, indent=4)

        if args.clear_cache:
            for file in os.listdir(args.sample_dir):
                if not file.endswith(".txt"):
                    if os.path.isdir(os.path.join(args.sample_dir, file)):
                        shutil.rmtree(os.path.join(args.sample_dir, file), ignore_errors=True)
                    else:
                        try:
                            os.remove(os.path.join(args.sample_dir, file))
                        except FileNotFoundError:
                            pass
    dist.barrier()
    dist.destroy_process_group()


    


def qualitative_eval(args, cur_en_q_level=None):
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

    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)

    causal_type = config["model"]["causal_settings"]["causal_type"]
    dynamic_length_train = config["model"]["causal_settings"]["dynamic_length_train"]
    min_level = config["model"]["causal_settings"]["min_level"]

    # TODO: check if the inference setting match the causal training setting

    # create and load model
    if args.quant_way == 'vq':
        tokenizer_model = load_model_from_config(config)

    elif args.quant_way == 'fsq':
        tokenizer_model = FSQ_models[args.fsq_model](
            levels=args.levels
        )
    elif args.quant_way == 'lq':
        tokenizer_model = LQ_models[args.lq_model](
            levels=args.levels
        )
    else:
        raise ValueError("please check quant way")

    tokenizer_model.to(device)
    tokenizer_model.eval()

    if args.lpips:
        # loss_fn_alex = LPIPS(net='alex').to(device)  # best forward scores
        loss_fn_vgg = LPIPS().eval().to(device)   # closer to "traditional" perceptual loss, when used for optimization


    if args.quant_way == 'vq':
        ckpt_path = args.vq_ckpt
    elif args.quant_way == 'fsq':
        ckpt_path = args.fsq_ckpt
    elif args.quant_way == 'lq':
        ckpt_path = args.lq_ckpt
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

    # Create folder to save samples:
    try:
        iteration = int(ckpt_path.split("/")[-1].split(".")[0])
    except:
        iteration = ""
        iteration = ""
    if args.quant_way == 'vq':
        save_folder_name = (f"{config['model']['model_cls']}-{args.dataset}-size-{args.image_size}-size-{args.image_size_eval}"
                    f"-codebook-size-{config['model']['init_args']['codebook_size']}-dim-{config['model']['init_args']['codebook_embed_dim']}-seed-{args.global_seed}-{iteration}" + \
                    "" if cur_en_q_level is None else f"-c-type-{causal_type}-cur-level-{cur_en_q_level}"
                    )
    elif args.quant_way == 'fsq':
        save_folder_name = (f"{args.fsq_model}"
                    f"-codebook-size-{args.codebook_size}-dim-{args.codebook_embed_dim}-seed-{args.global_seed}-{iteration}")
    elif args.quant_way == 'lq':
        save_folder_name = (f"{args.lq_model}"
                    f"-codebook-size-{args.codebook_size}-dim-{args.codebook_embed_dim}-seed-{args.global_seed}-{iteration}")
    else:
        raise ValueError("please check quant way")

    sample_folder_dir = f"{args.sample_dir}/{save_folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

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
        shuffle=False,
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


    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    
    loader = tqdm(loader) if rank == 0 else loader
    total = 0
    lpips_val = []
    psnr_val = []
    ssim_val = []
    for x, _ in loader:
        if args.image_size_eval != args.image_size:
            rgb_gts = F.interpolate(x, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        else:
            rgb_gts = x
        if args.lpips:
            ori_x = x.clone().to(device)
        rgb_gts = (rgb_gts.permute(0, 2, 3, 1).to("cpu").numpy() + 1.0) / 2.0 # rgb_gt value is between [0, 1]
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            if args.quant_way == 'vq':
                latent, _, [_, _, indices] = tokenizer_model.encode(
                                                x, 
                                                num_en_q_level=cur_en_q_level, 
                                                causal_type=causal_type)
                samples = tokenizer_model.decode_code(indices, latent.shape) # output value is between [-1, 1]
            elif args.quant_way == 'fsq':
                latent, indices, _ = tokenizer_model.encode(x)
                samples = tokenizer_model.decode_code(indices)
            elif args.quant_way == 'lq':
                latent, _, indices = tokenizer_model.encode(x)
                samples = tokenizer_model.decode_code(indices)
            if args.image_size_eval != args.image_size:
                samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        if args.lpips:
            rec_x = samples.clone()
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # lpips takes [-1, 1] input
        if args.lpips:
            # lpips_alex = loss_fn_alex(ori_x, rec_x)
            lpips_vgg = loss_fn_vgg(ori_x, rec_x)
            lpips_val.append(lpips_vgg)

            # rgb_restored = sample.astype(np.float32) / 255. # rgb_restored value is between [0, 1]
            # psnr = psnr_loss(rgb_restored, rgb_gt)
            # ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=2.0, channel_axis=-1)
            # psnr_val.append(psnr)
            # ssim_val.append(ssim)

        # Save samples to disk as individual .png files
        for i, (sample, rgb_gt) in enumerate(zip(samples, rgb_gts)):
            index = i * dist.get_world_size() + rank + total
            if args.lpips:
                lpips_value = lpips_vgg[i].item()
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}_{lpips_value:.4f}.png")
            else:
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            # float to uint8
            rgb_gt = (rgb_gt * 255).astype(np.uint8)
            Image.fromarray(rgb_gt).save(f"{sample_folder_dir}/{index:06d}_gt.png")
            
        total += global_batch_size

    # ------------------------------------
    #       Summary
    # ------------------------------------
    # Make sure all processes have finished saving their samples
    dist.barrier()
    dist.destroy_process_group()

    if rank == 0 and args.lpips:
        lpips_val = [val.cpu() for val in lpips_val]
        # print(f"lpips_val: {lpips_val}")
        print(f"lpips_val_mean: {np.mean(lpips_val)}")
        # print(f"lpips_val_std: {np.std(lpips_val)}")
        # save as txt file
        with open(os.path.join(sample_folder_dir, "lpips_val.txt"), "w") as f:
            f.write(f"lpips_val_mean: {np.mean(lpips_val)}")

    if args.plot:
        # deprecated
        plot_trending(args.sample_dir)
    

def plot_trending(exp_dir):
    """
    Assuming different subfolders of levels 
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    items = []
    for sub_folder in os.listdir(exp_dir):
        if os.path.isdir(os.path.join(exp_dir, sub_folder)):
            # read the lpips value from the file name
            cur_level = int(sub_folder.split("-")[-1])
            for file in os.listdir(os.path.join(exp_dir, sub_folder)):
                if file.endswith(".png") and "gt" not in file:
                    # calculate lpips
                    try:
                        lpips_val = float(file.replace(".png", "").split("_")[1])
                    except IndexError:
                        continue
                    img_idx = file.split("_")[0]
                    items.append({
                        "lpips_val": lpips_val,
                        "img_path": os.path.join(exp_dir, sub_folder, file),
                        "img_idx": img_idx,
                        "cur_level": cur_level
                    })
    
    # plot the number of unique img_idx rows of imgs and the columns are 
    # the images with lpips value above, and from left
    # Sort items by image index and LPIPS value
    items.sort(key=lambda x: (x['img_idx'], x['cur_level']))
    
    # Group items by img_idx
    grouped_items = {}
    for item in items:
        img_idx = item['img_idx']
        if img_idx not in grouped_items:
            grouped_items[img_idx] = []
        grouped_items[img_idx].append(item)

    # Plot each row for each img_idx
    for img_idx, img_group in grouped_items.items():
        # Sort the images by cur_level within each img_idx group
        img_group.sort(key=lambda x: x['cur_level'])
        
        # Number of columns corresponds to the number of cur_levels
        num_cols = len(img_group)
        fig, axs = plt.subplots(1, num_cols, figsize=(4 * num_cols, 6))

        if num_cols == 1:
            axs = [axs]  # Ensure axs is iterable even for a single column
        
        for i, item in enumerate(img_group):
            img = mpimg.imread(item['img_path'])
            axs[i].imshow(img)
            axs[i].axis('off')
            
            # Add title with cur_level and lpips_val
            axs[i].set_title(f"Level: {item['cur_level']}\nLPIPS: {item['lpips_val']:.4f}", fontsize=12)
        
        # Save the figure for this img_idx
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"trending_{img_idx}.png"))
        plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'coco', 'imagenet_reconstruct'], default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--sample-dir", type=str, default="results/reconstructions/vq")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--quant-way", type=str, choices=['vq', 'fsq', 'lq'], default='vq')

    # model specific arguments for vq
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")

    # inference configs for multi-level reconstruction
    str2list = lambda x: None if x == "None" else [int(item) for item in x.split(',')]
    parser.add_argument("--mul-rec-levels", type=str2list, default=None)

    # model specific arguments for fsq
    parser.add_argument("--fsq-model", type=str, choices=list(FSQ_models.keys()), default="FSQ-16")
    parser.add_argument("--fsq-ckpt", type=str, default=None, help="ckpt path for fsq model")

    str2list = lambda x: [int(item) for item in x.split(',')]
    parser.add_argument("--levels", type=str2list, default=str2list("8, 8, 8, 6, 5"), help="levels for fsq model")

    parser.add_argument("--clear-cache", action="store_true", help="whether to clear all the images and .npz files")
    parser.add_argument("--merge", action="store_true", help="whether to merge all the results")

    parser.add_argument("--lpips", action="store_true", help="whether to merge all the results")
    parser.add_argument("--plot", action="store_true", help="whether to plot an increasing token_num trending plot")

    parser.add_argument("--eval-python-path", type=str, default="python", help="python path for the specific environment used in gFID etc. evaluation")
    parser.add_argument("--gt-npz-path", type=str, default="VIRTUAL_imagenet256_labeled.npz", help="path to the ground truth npz file for gFID etc. evaluation")

    parser.add_argument("--qualitative", action="store_true", help="whether to evaluate the model quantitatively")

    args = parser.parse_args()

    EVAL_PYTHON_PATH = args.eval_python_path
    GT_NPZ_PATH = args.gt_npz_path

    eval_script_template = \
    f"""
    cd evaluations/c2i

    {EVAL_PYTHON_PATH} \
    evaluator.py \
    {GT_NPZ_PATH} \
    <test_npz_file>
    """

    if not args.qualitative:
        if args.mul_rec_levels is not None and not args.merge:
            for cur_en_level in args.mul_rec_levels:
                quantitative_eval(args, cur_en_q_level=cur_en_level)
        else:
            quantitative_eval(args)
    else:
        # reconstruct all the images in one folder, and send them to 
        if args.mul_rec_levels is not None:
            for cur_en_level in args.mul_rec_levels:
                qualitative_eval(args, cur_en_q_level=cur_en_level)

        else:
            qualitative_eval(args)