# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
#   llamagen

import torch  
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True  
import torch.distributed as dist  
from torch.utils.data import DataLoader, Dataset, Subset  
from torch.utils.data.distributed import DistributedSampler  
from torchvision import transforms  
import numpy as np  
import argparse  
import os  
import yaml  
from tqdm import tqdm  

from utils.distributed import init_distributed_mode  
from dataset.augmentation import center_crop_arr  
from dataset.build import build_dataset  
from utils.model_init import load_model_from_config, custom_load  
from utils.imgnet_idx2label import idx2firstlabel

class IndexedDataset(Dataset):  
    def __init__(self, dataset):  
        self.dataset = dataset  
    def __getitem__(self, index):  
        data = self.dataset[index]  
        return data + (index,)  # Append the original dataset index  
    def __len__(self):  
        return len(self.dataset)  

def filter_dataset(dataset, args):  
    filtered_indices = []  
    codes_dir = os.path.join(args.code_path, f'{args.dataset}{args.image_size}_codes')  
    labels_dir = os.path.join(args.code_path, f'{args.dataset}{args.image_size}_labels')  
    rank = dist.get_rank()  
    loader = tqdm(range(len(dataset))) if rank == 0 else range(len(dataset))
    for i in loader:
        file_idx = i  
        code_file = os.path.join(codes_dir, f'{file_idx}.npy')  
        label_file = os.path.join(labels_dir, f'{file_idx}.npy')  
        if rank == 0 and i < 100:
            print(f"Checking: code_file: {code_file}, label_file: {label_file}")
            print(f"checking existing result: {os.path.exists(code_file) and os.path.exists(label_file)}")
        if not (os.path.exists(code_file) and os.path.exists(label_file)):  
            filtered_indices.append(i)  
    print(f"Resuming: {len(filtered_indices)} samples remaining out of {len(dataset)}.")  
    return Subset(dataset, filtered_indices)  

def main(args):  
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."  
    if not args.debug:  
        init_distributed_mode(args)  
        rank = dist.get_rank()  
        device = rank % torch.cuda.device_count()  
        seed = args.global_seed * dist.get_world_size() + rank  
        torch.manual_seed(seed)  
        torch.cuda.set_device(device)  
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")  
    else:  
        device = 'cuda'  
        rank = 0  

    if args.debug or rank == 0:  
        os.makedirs(args.code_path, exist_ok=True)  
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_codes'), exist_ok=True)  
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_labels'), exist_ok=True)  

    with open(args.model_config, "r") as f:  
        config = yaml.safe_load(f)  
    tokenizer_model = load_model_from_config(config)  
    tokenizer_model.to(device)  
    tokenizer_model.eval()  
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")  
    tokenizer_model.load_state_dict(checkpoint["model"])  
    del checkpoint  
    if not args.debug:
        del tokenizer_model.decoder  
        if hasattr(tokenizer_model, "s1to2decoder"):  
            del tokenizer_model.s1to2decoder  
        torch.cuda.empty_cache()  

    if args.ten_crop:  
        crop_size = int(args.image_size * args.crop_range)  
        transform = transforms.Compose([  
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),  
            transforms.TenCrop(args.image_size),  
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)  
        ])  
    else:  
        crop_size = args.image_size  
        transform = transforms.Compose([  
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)  
        ])  
    dataset = build_dataset(args, transform=transform)  
    dataset = IndexedDataset(dataset)  # Wrap the dataset to include sample indices  
    if args.resume:  
        dataset = filter_dataset(dataset, args)  
        dist.barrier()
    if not args.debug:  
        sampler = DistributedSampler(  
            dataset,  
            num_replicas=dist.get_world_size(),  
            rank=rank,  
            shuffle=False,  
            seed=args.global_seed  
        )  
    else:  
        sampler = None  
    loader = DataLoader(  
        dataset,  
        batch_size=args.batch_size,  
        shuffle=False,  
        sampler=sampler,  
        num_workers=args.num_workers,  
        pin_memory=True,  
        drop_last=False  
    )  

    loader = tqdm(loader) if rank == 0 else loader  
    for batch in loader:  
        x, y, idx = batch  # x: features, y: labels, idx: original dataset indices  
        x = x.to(device)  
        y = y.to(device)  
        if args.ten_crop:  
            B = x.shape[0]  
            x_all = x.flatten(0, 1)  # Reshape from [B, num_crops, C, H, W] to [B*num_crops, C, H, W]  
            num_aug = 10  
        else:  
            B = x.shape[0]  
            x_flip = torch.flip(x, dims=[-1])  
            x_all = torch.cat([x, x_flip])  
            num_aug = 2  
        with torch.no_grad():  
            if args.quant_way == 'vq':  
                _, _, [_, _, indices] = tokenizer_model.encode(x_all)  
            elif args.quant_way == 'fsq':  
                _, indices, _ = tokenizer_model.encode(x)  
        codes = indices.reshape(B, num_aug, -1)  
        codes_np = codes.detach().cpu().numpy()  
        labels_np = y.detach().cpu().numpy()  

        for i in range(B):  
            file_idx = int(idx[i].item())  # Use the dataset-provided index for consistent numbering  
            np.save(f'{args.code_path}/{args.dataset}{args.image_size}_codes/{file_idx}.npy', codes_np[i])  
            np.save(f'{args.code_path}/{args.dataset}{args.image_size}_labels/{file_idx}.npy', labels_np[i])
        
        if args.debug:
            # also save the reconstructed images for checking
            to_save_img_indices = indices.reshape(B, num_aug, -1)
            to_save_img_indices = to_save_img_indices[:, 0, :]
            qzshape = [B, 8, 1, 256]
            # decode the codes
            with torch.no_grad():
                sample = tokenizer_model.decode_code(to_save_img_indices, qzshape)
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            if not os.path.exists(f'{args.code_path}/{args.dataset}{args.image_size}_samples'):
                os.makedirs(f'{args.code_path}/{args.dataset}{args.image_size}_samples')
            for i in range(B):
                file_idx = int(idx[i].item())  # Use the dataset-provided index for consistent numbering
                img_cls = idx2firstlabel(labels_np[i])
                Image.fromarray(samples[i]).save(f'{args.code_path}/{args.dataset}{args.image_size}_samples/{img_cls}_{labels_np[i]}_{file_idx}.png')



    if not args.debug:  
        dist.barrier()  
        dist.destroy_process_group()  

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--data-path", type=str, required=True)  
    parser.add_argument("--code-path", type=str, required=True)  
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)  
    parser.add_argument("--dataset", type=str, default='imagenet')  
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)  
    parser.add_argument("--ten-crop", action='store_true', help="whether using random crop")  
    parser.add_argument("--crop-range", type=float, default=1.1, help="expanding range of center crop")  
    parser.add_argument("--global-seed", type=int, default=0)  
    parser.add_argument("--num-workers", type=int, default=24)  
    parser.add_argument("--batch-size", type=int, default=32)  
    parser.add_argument("--model-config", type=str, required=True)  
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")  
    parser.add_argument("--quant-way", type=str, choices=['vq', 'fsq', 'lq'], default='vq')  
    parser.add_argument("--debug", action='store_true')  
    parser.add_argument("--resume", action='store_true', help="resume and filter out already processed samples")  
    args = parser.parse_args()  
    main(args)
