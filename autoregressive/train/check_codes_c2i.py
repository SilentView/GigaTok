import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq_model import VQ_models
from tokenizer.tokenizer_image.fsq.fsq_model import FSQ_models
import shutil
from dataset.augmentation import random_crop_arr
from utils.model_init import load_model_from_config
import yaml
from PIL import Image
import numpy as np
from utils.imgnet_idx2label import idx2firstlabel


def create_local_tokenizer(args, device):
    """
    Create a tokenizer that is local to each rank.
    """
    tokenizer_model = load_model_from_config(args.tok_config)
    ckpt_path = args.vq_ckpt
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "ema" in checkpoint:
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    else:
        raise Exception("Please check model weight")

    tokenizer_model.load_state_dict(model_weight)
    tokenizer_model.eval()
    return tokenizer_model


def decode_and_save_images(loader, tokenizer, device, save_path):
    """
    Decode the indices and save images with labels as filenames.
    """
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        print(f"Processing batch {idx}")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        
        with torch.no_grad():
            B = x.shape[0]
            qzshape = [B, 8, 1, 256]
            samples = tokenizer.decode_code(x, qzshape)
            decoded_images = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for i in range(x.shape[0]):  # For each image in the batch
            image = decoded_images[i]
            label = y[i].item()
            image_pil = Image.fromarray(image)

            # Save the image with its label as the filename
            img_cls = idx2firstlabel(label)
            label_str = img_cls
            file_i = idx * i + i

            image_pil.save(os.path.join(save_path, f"{label_str}_{file_i}.png"))
            print(f"Saved image for label {label} at {label_str}_{file_i}.png")


def main(args):
    assert torch.cuda.is_available(), "Training requires at least one GPU."
    
    # Set up device
    device = "cuda:0"
    torch.manual_seed(args.global_seed)
    torch.cuda.set_device(device)

    # Load tokenizer
    tokenizer = create_local_tokenizer(args, device)
    # del tokenizer.encoder
    # del tokenizer.s2to1encoder
    tokenizer.to(device)

    # Set up dataset and dataloader
    if args.dataset == "imagenet_code":
        dataset = build_dataset(args)
    else:
        raise NotImplementedError
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Decode and save images
    decode_and_save_images(loader, tokenizer, device, args.save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--vq-ckpt", type=str, default=None, help="Path to the VQ model checkpoint")
    parser.add_argument("--tok-config", type=str, default=None, help="Path to the tokenizer config")
    parser.add_argument("--save-path", type=str, default="decoded_images", help="Directory to save the decoded images")
    parser.add_argument("--dataset", type=str, default="imagenet_code", help="Dataset name")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--quant-way", type=str, default="vq")

    args = parser.parse_args()
    main(args)
