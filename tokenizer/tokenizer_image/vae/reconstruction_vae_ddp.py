import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
from datetime import timedelta
import shutil
import json
from glob import glob
from PIL import Image
import os
import itertools
from PIL import Image
import numpy as np
import argparse
import random
import itertools
import subprocess

from tokenizer.tokenizer_image.lpips import LPIPS
import wandb


from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
# from diffusers.models import AutoencoderKL

from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset

from utils.resume_log import init_wandb, update_wandb_log, wandb_cache_file_append
from utils.google_drive_util import create_folder, upload_file, PARENT_FOLDER_ID
from utils.model_init import load_model_from_config, custom_load

import yaml



class SingleFolderDataset(Dataset):
    def __init__(self, directory, transform=None):
        super().__init__()
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, file_name) for file_name in os.listdir(directory)
                            if os.path.isfile(os.path.join(directory, file_name))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    
    random.shuffle(samples) # This is very important for IS(Inception Score) !!!
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


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
            init_wandb(
                project_name=args.wandb_project,
                config={"dataset": args.dataset},
                exp_dir=args.eval_wandb_dir,
                name=name,
                eval_run=True
            )
            # merge all the .json files 
            results = []
            check_dir = os.path.join(os.path.dirname(sample_dir),
                                     "eval_results")
            for file in os.listdir(check_dir):
                if file.endswith(".json"):
                    with open(os.path.join(check_dir, file), "r") as f:
                        results.append(json.load(f))
            # upload the results
            for result in results:
                wandb.log(result)
            wandb.finish()
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
    tokenizer_model = load_model_from_config(config)
    tokenizer_model.to(device)
    tokenizer_model.eval()
    print(f"VAE Model Parameters(inference): {sum(p.numel() for p in tokenizer_model.parameters()):,}")

    ckpt_path = args.ckpt
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
    folder_name = (f"{config['model']['model_cls']}-{args.dataset}-size-{args.image_size}-size-{args.image_size_eval}"
                f"-dim-{config['model']['init_args']['latent_embed_dim']}-seed-{args.global_seed}" + \
                ("" if cur_en_q_level is None else f"-c-type-{causal_type}-cur-level-{cur_en_q_level}")
                )

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
            posteriors, _ = tokenizer_model.encode(
                                x, 
                                return_latent_only=True, 
                                num_en_q_level=cur_en_q_level, 
                                causal_type=causal_type,
                                )
            z = posteriors.sample()
            samples = tokenizer_model.decode(z)
            rec_x = samples.clone()
            
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

        print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))
        print("LPIPS_vgg: %f " % lpips_val)

        result_file = f"{sample_folder_dir}_results.txt"
        print("writing results to {}".format(result_file))
        with open(result_file, 'w') as f:
            print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb), file=f)
            print("LPIPS_vgg: %f " % lpips_val, file=f)

        create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
        print("Done.")
    
    if rank == 0:
        """
        Further measure rFID, and upload the info to the specific wandb project
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
                _wrap_key("lpips_vgg"): lpips_val,
                "iteration": iteration
            }
        )
        # find the .npz file
        npz_file_path = f"{sample_folder_dir}.npz"
        # run the evaluator script
        evaluate_script = eval_script_template.replace("<test_npz_file>", npz_file_path)
        # upload the results to wandb
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
                if not file.endswith(".txt") or (not file.endswith(".json")):
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
    tokenizer_model = load_model_from_config(config)
    tokenizer_model.to(device)
    tokenizer_model.eval()

    if args.lpips:
        # loss_fn_alex = LPIPS(net='alex').to(device)  # best forward scores
        loss_fn_vgg = LPIPS().eval().to(device)   # closer to "traditional" perceptual loss, when used for optimization


    ckpt_path = args.ckpt
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
    google_drive_foldre_name = (f"{config['model']['model_cls']}-{args.dataset}-size-{args.image_size}-size-{args.image_size_eval}"
                f"-dim-{config['model']['init_args']['latent_embed_dim']}-seed-{args.global_seed}-{iteration}" + \
                "" if cur_en_q_level is None else f"-c-type-{causal_type}-cur-level-{cur_en_q_level}"
                )

    sample_folder_dir = f"{args.sample_dir}/{google_drive_foldre_name}"
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
            posteriors, _ = tokenizer_model.encode(
                                x, 
                                return_latent_only=True, 
                                num_en_q_level=cur_en_q_level, 
                                causal_type=causal_type,
                                )
            z = posteriors.sample()
            samples = tokenizer_model.decode(z)

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
    if rank == 0 and args.google_up:
        # upload to google drive
        folder_id = create_folder(google_drive_foldre_name, parent_folder_id=PARENT_FOLDER_ID)
        for file in os.listdir(sample_folder_dir):
            if file.endswith(".png"):
                # upload to google drive
                upload_file(os.path.join(sample_folder_dir, file), parent_folder_id=folder_id)
        print("Done.")

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

    parser.add_argument("--google-up", action="store_true")

    # model specific arguments for vq
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help="ckpt path for the vae model")

    # inference configs for multi-level reconstruction
    str2list = lambda x: None if x == "None" else [int(item) for item in x.split(',')]
    parser.add_argument("--mul-rec-levels", type=str2list, default=None)

    parser.add_argument("--clear-cache", action="store_true", help="whether to clear all the images and .npz files")
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb for logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    parser.add_argument("--eval-wandb-dir", type=str, default=None, help="directory restoring the run id for a single run")
    parser.add_argument("--merge", action="store_true", help="whether to merge all the results")

    parser.add_argument("--lpips", action="store_true", help="whether to merge all the results")
    parser.add_argument("--plot", action="store_true", help="whether to plot an increasing token_num trending plot")
    parser.add_argument("--root", type=str, default="/mnt/bn/data-aigc-video/tianwei")

    parser.add_argument("--qualitative", action="store_true", help="whether to evaluate the model quantitatively")

    args = parser.parse_args()

    ROOT = args.root
    EVAL_PYTHON_PATH = f"{ROOT}/miniconda3/envs/tok_eval/bin/python"
    GT_NPZ_PATH = f"{ROOT}/code/Tokenizer1D/results/reconstructions/val_imagenet.npz"

    eval_script_template = \
    f"""
    cd evaluations/c2i

    {EVAL_PYTHON_PATH} \
    evaluator.py \
    {GT_NPZ_PATH} \
    <test_npz_file>
    """

    if not args.qualitative:
        if args.wandb:
            assert args.wandb_project is not None, "Please specify a wandb project name"
            assert args.eval_wandb_dir is not None, "Please specify a experiment directory name"
        
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