# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
"""
Searching the best cfg value and schedules for a generator

There will be 2 searching patterns:
- All strategies
- Single specified strategy.
    - For single specified strategy, all hyperparameters will be searched, trying to find the minimum of FID
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
import pandas as pd
import os
import json
from PIL import Image
import numpy as np
import math
import argparse
import shutil
import subprocess
import yaml
from datetime import timedelta
import multiprocessing
import time

from einops import rearrange
import itertools

from autoregressive.models.gpt_1d import GPT_models
from autoregressive.models.gpt import GPT_models_2d

from autoregressive.models.generate import generate

from utils.model_init import load_model_from_config





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


def create_npz_from_sample_folder(sample_dir, num=50_000):
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

def create_npz_from_sample_folder_mp(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    import os
    from multiprocessing import Pool, cpu_count
    from PIL import Image
    import numpy as np
    from tqdm import tqdm

    # Read the first image to get dimensions
    sample_pil = Image.open(os.path.join(sample_dir, f"{0:06d}.png"))
    sample_np = np.asarray(sample_pil).astype(np.uint8)
    H, W, C = sample_np.shape

    # Pre-allocate the array for efficiency
    samples = np.empty((num, H, W, C), dtype=np.uint8)

    # Define a helper function that reads and processes a single image
    def read_image(i):
        try:
            sample_pil = Image.open(os.path.join(sample_dir, f"{i:06d}.png"))
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            return i, sample_np
        except Exception as e:
            print(f"Error processing image {i:06d}.png: {e}")
            return i, None

    # Use a multiprocessing pool to read images in parallel
    with Pool(processes=cpu_count()) as pool:
        for i, sample_np in tqdm(pool.imap_unordered(read_image, range(num)), total=num, desc="Reading images"):
            if sample_np is not None:
                if sample_np.shape != (H, W, C):
                    raise ValueError(f"Image {i:06d}.png has incorrect shape {sample_np.shape}")
                samples[i] = sample_np
            else:
                raise ValueError(f"Failed to process image {i:06d}.png")

    assert samples.shape == (num, H, W, C)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path



def quantitative_eval(args, independent=False):

    if independent:
        # Setup PyTorch:
        assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
        torch.set_grad_enabled(False)

        # Setup DDP:
        # dist.init_process_group("nccl")
        dist.init_process_group(
            "nccl",
            timeout=timedelta(hours=1)  # 1 hour
            )

    rank = dist.get_rank()
    node_rank = int(os.environ.get('NODE_RANK', 0))
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    if args.quant_way == 'vq':
        tokenizer_model = load_model_from_config(args.tok_config)
    elif args.quant_way == 'fsq':
        raise NotImplementedError("fsq not implemented yet")
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


    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]

    with open(args.tok_config, "r") as f:
        tok_config = yaml.safe_load(f)

    num_token = tok_config["model"]["init_args"]["num_latent_tokens"] if args.num_token is None else args.num_token
    codebook_size = tok_config["model"]["init_args"]["codebook_size"]
    tok_model_cls = tok_config["model"]["model_cls"]
    codebook_embed_dim = tok_config["model"]["init_args"]["codebook_embed_dim"]

    if args.gpt_2d:
        gpt_model = GPT_models_2d[args.gpt_model](
            vocab_size=codebook_size,
            block_size=num_token,
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
        ).to(device=device, dtype=precision)
    else:
        gpt_model = GPT_models[args.gpt_model](
            vocab_size=codebook_size,
            block_size=num_token,
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
            rope=args.rope,
            use_adaLN=args.adaLN,
            use_simple_adaLN=args.simple_adaLN,
            use_qk_norm=args.qk_norm,
            use_flash_attn=args.flash_attn,
        ).to(device=device, dtype=precision)

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
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile") 

    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    # if args.from_fsdp:
    #     ckpt_string_name = args.gpt_ckpt.split('/')[-2]
    # else:
    #     ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    folder_name = f"{model_string_name}-size-{args.image_size}-size-{args.image_size_eval}-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-{args.cfg_schedule}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if node_rank == 0 and rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    
    # check if the results already exist, if exists, skip the sampling
    # if specifically counting time consumption there will be no skipping
    txt_path = f"{sample_folder_dir}.txt"
    result_dict = {}
    if os.path.exists(txt_path) and not args.time_cnt_only:
        with open(txt_path, "r") as f:
            lines = f.readlines()
        check_metrics = ["FID", "sFID", "Inception Score", "Precision", "Recall"]
        for line in lines:
            for metric in check_metrics:
                if line.startswith(metric):
                    result_dict.update({metric: float(line.split(":")[-1].strip())})
        if len(result_dict) == len(check_metrics):
            dist.barrier()
            if independent:
                dist.destroy_process_group()
            print(f"Results already exist, skip the sampling")
            print(line)
            return

    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if node_rank == 0 and rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if node_rank == 0 and rank == 0 else pbar
    total = 0
    generation_costs = []
    decode_costs = []
    for _ in pbar:
        # Sample inputs:
        t1 = time.time()
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)
        qzshape = [len(c_indices), codebook_embed_dim, 1, num_token]

        index_sample = generate(
            gpt_model, c_indices, num_token,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True,
            cfg_schedule=args.cfg_schedule,
            cfg_schedule_kwargs=args.cfg_schedule_kwargs,
            )
        
        t2 = time.time()
        generation_costs.append(t2 - t1)
        t1 = time.time()
        if args.quant_way == 'vq':
            samples = tokenizer_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
            t2 = time.time()
            decode_costs.append(t2 - t1)
        elif args.quant_way == 'fsq':
            raise NotImplementedError("fsq not implemented yet")
        if args.image_size_eval != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')

        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        
        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
    
    dist.barrier()
    world_size = dist.get_world_size()
    gather_generation_costs = [None for _ in range(world_size)]
    gather_decode_costs = [None for _ in range(world_size)]

    dist.all_gather_object(gather_generation_costs, generation_costs)
    dist.all_gather_object(gather_decode_costs, decode_costs)

    if rank == 0:
        # average accross all ranks
        gather_generation_costs = list(itertools.chain(*gather_generation_costs))
        gather_decode_costs = list(itertools.chain(*gather_decode_costs))        
        generaiont_t_cost = sum(gather_generation_costs) / len(gather_generation_costs)
        decode_t_cost = sum(gather_decode_costs) / len(gather_decode_costs)
        print(f"generation time cost: {generaiont_t_cost}")
        print(f"decode time cost: {decode_t_cost}")
    
    dist.barrier()

    if args.time_cnt_only:
        if args.clear_cache and rank == 0:
            # start the cache clearing process asynchronously and continue with other computations.
            shutil.rmtree(sample_folder_dir)
            # os.remove(sample_folder_dir + ".npz")
        if independent:
            dist.barrier()
            dist.destroy_process_group()
        return

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if node_rank == 0 and rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")

    if node_rank == 0 and rank == 0:
        """
        Further measure FID and sFID
        columns = ["iteration", "FID", "sFID"]
        """

        try:
            iteration = int(ckpt_path.split("/")[-1].split(".")[0])
        except:
            iteration = ""

        result_dict = {}
        result_dict.update(
            {
                "iteration": iteration
            }
        )
        # find the .npz file
        npz_file_path = f"{sample_folder_dir}.npz"
        # run the evaluator script
        evaluate_script = eval_script_template.replace("<test_npz_file>", npz_file_path)
        print("running evaluate_script: ", evaluate_script)
        subprocess.run(evaluate_script, shell=True)
        txt_path = npz_file_path.replace(".npz", ".txt")
        with open(txt_path, "r") as f:
            lines = f.readlines()
        
        check_metrics = ["FID", "sFID", "Inception Score", "Precision", "Recall"]
        for line in lines:
            for metric in check_metrics:
                if line.startswith(metric):
                    result_dict.update({metric: float(line.split(":")[-1].strip())})

        # store the results to args.sample_dir/
        sample_dir = args.sample_dir[:-1] if args.sample_dir.endswith("/") else args.sample_dir

        # save_path = os.path.dirname(sample_dir) + "/eval_results/" + sample_dir.split("/")[-1] + ".json"
        # print("save_path: ", save_path)
        # if not os.path.exists(os.path.dirname(save_path)):
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # with open(save_path, "w") as f:
        #     json.dump(result_dict, f, indent=4)
       
        # record the searching results in a csv file
        # csv file columns: ["annotation", "cfg_scale", "cfg_schedule", "FID", "Inception Score", "Precision", "Recall", "sFID"]

        csv_file_path = os.path.dirname(sample_dir) + "/search_results/" + sample_dir.split("/")[-1] + ".csv"
        print("csv_file_path: ", csv_file_path)
        if not os.path.exists(os.path.dirname(csv_file_path)):
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        
        if not os.path.exists(csv_file_path):
            df = pd.DataFrame(columns=[
                                "annotation", 
                                "cfg_scale", 
                                "cfg_schedule", 
                                "FID", 
                                "Inception Score", 
                                "Precision", 
                                "Recall", 
                                "sFID"]
                            )
            annotation = ";".join([f"{key}={value:.2f}" for key, value in args.cfg_schedule_kwargs.items()])
            row = pd.DataFrame(
                {"annotation": [annotation], 
                "cfg_scale": [args.cfg_scale], 
                "cfg_schedule": [args.cfg_schedule],
                "FID": [result_dict["FID"]],
                "Inception Score": [result_dict["Inception Score"]],
                "Precision": [result_dict["Precision"]],
                "Recall": [result_dict["Recall"]],
                "sFID": [result_dict["sFID"]]
                })
            
            df = pd.concat([df, row], ignore_index=True)
            df.to_csv(csv_file_path, index=False)
        else:
            df = pd.read_csv(csv_file_path)
            annotation = ";".join([f"{key}={value:.2f}" for key, value in args.cfg_schedule_kwargs.items()])
            row = pd.DataFrame(
                {"annotation": [annotation],
                "cfg_scale": [args.cfg_scale], 
                "cfg_schedule": [args.cfg_schedule],
                "FID": [result_dict["FID"]],
                "Inception Score": [result_dict["Inception Score"]],
                "Precision": [result_dict["Precision"]],
                "Recall": [result_dict["Recall"]],
                "sFID": [result_dict["sFID"]]
                })
            
            df = pd.concat([df, row], ignore_index=True)
            df.to_csv(csv_file_path, index=False)

        # if args.clear_cache:
        #     cache_clear_process = multiprocessing.Process(
        #         target=clear_cache, 
        #         args=(sample_folder_dir,)
        #     )
        #     cache_clear_process.start()
        #     cache_clear_process.daemon = True

        if args.clear_cache:
            try:
                shutil.rmtree(sample_folder_dir)
            except:
                # try again
                try:
                    shutil.rmtree(sample_folder_dir)
                except:
                    pass
            os.remove(sample_folder_dir + ".npz")

    dist.barrier()

    if independent:
        dist.destroy_process_group()

    if not args.search and rank == 0:
        print(lines)

 
        
def clear_cache(sample_folder_dir):
    try:
        shutil.rmtree(sample_folder_dir)
        os.remove(f"{sample_folder_dir}.npz")
        print("Cache successfully cleared.")
    except Exception as e:
        print(f"Error clearing cache: {e}")


def empirical_cfg_start(args):
    if hasattr(args, "cfg_scale_start"):
        cfg_scale_start = args.cfg_scale_start
        return cfg_scale_start

    if args.cfg_schedule == 'step':
        # using empirically best cfgs as the starting point
        if args.gpt_model == "GPT-B":
            if args.simple_adaLN or args.adaLN or "v4b" in args.gpt_ckpt:
                if "0250000.pt" in args.gpt_ckpt:
                    cfg_scale_start = 3.0
                else:
                    cfg_scale_start = 2.75
            else:
                cfg_scale_start = 2.5

        elif args.gpt_model == "GPT-L":
            if args.simple_adaLN or args.adaLN or "v4b" in args.gpt_ckpt:
                if "0250000.pt" in args.gpt_ckpt:
                    cfg_scale_start = 2.5
                else:
                    cfg_scale_start = 1.75
            else:
                cfg_scale_start = 1.75
        else:
            cfg_scale_start = 1.25

    elif args.cfg_schedule =='constant':
        # constant cfg_scale searc
        if args.gpt_model == "GPT-B":
            if "XXL" in args.tok_config:
                cfg_scale_start = 1.75
            else:
                cfg_scale_start = 2.0
        elif args.gpt_model == "GPT-L":
            cfg_scale_start = 1.75
        else:
            cfg_scale_start = 1.75

    elif args.cfg_schedule == 'rectangular':
        if args.gpt_model == "GPT-B":
            cfg_scale_start = 3.0
        elif args.gpt_model == "GPT-L":
            cfg_scale_start = 2.5
        else:
            cfg_scale_start = 1.75
    else:
        raise NotImplementedError(f"cfg_schedule {args.cfg_schedule} not implemented yet")
 
    return cfg_scale_start


def search_cfg_only(args, annotation, increase_first=True):
    """
    search with the given setting for only different cfg.
    """
    cfg_scale_start = empirical_cfg_start(args)
    # cfg_scale_start = empirical_cfg_start(args)
    cfg_scale_step = 0.25

    sample_dir = args.sample_dir[:-1] if args.sample_dir.endswith("/") else args.sample_dir
    csv_file_path = os.path.dirname(sample_dir) + "/search_results/" + sample_dir.split("/")[-1] + ".csv"

    args.cfg_schedule_kwargs = {} if not hasattr(args, "cfg_schedule_kwargs") else args.cfg_schedule_kwargs

    while True:
        next_cfg_scale, cur_best_fid, end_flag = get_next_search_state_cfg(
            csv_file_path=csv_file_path,
            cfg_schedule=args.cfg_schedule,
            initial_cfg_scale=cfg_scale_start,
            step_size=cfg_scale_step,
            annotation=annotation,
            increase_first=increase_first,
        )

        rank = int(os.environ.get('RANK', 0))
        node_rank = int(os.environ.get('NODE_RANK', 0))
        # if node_rank == 0 and rank == 0:
        #     print(f"cur best cfg_scale={args.cfg_scale}")
        #     print(f"cur best fid={cur_best_fid}")
        if end_flag:
            # if node_rank == 0 and rank == 0:
            #     print(f"searching for cfg_scale is done for annotation={annotation},\
            #           the current best cfg_scale={args.cfg_scale}, the corresponding fid={cur_best_fid}")
            break
        args.cfg_scale = next_cfg_scale
        # the results will be written into a csv file
        quantitative_eval(args)
    
    return args.cfg_scale



def announce_best_setting(csv_file_path, cfg_schedule):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # raise FileNotFoundError(f"The file {csv_file_path} does not exist.")
        # return the initial_cfg_scale if the file does not exist
        print(f"The file {csv_file_path} does not exist.")
        return
    except pd.errors.EmptyDataError:
        # If the CSV is empty, start with the initial cfg_scale
        print(f"The file {csv_file_path} is empty.")
        return

    # Filter the DataFrame based on cfg_schedule
    df_filtered = df[df["cfg_schedule"] == cfg_schedule]
    # change the nan to ""
    df_filtered = df_filtered.fillna("")
    if df_filtered.empty:
        # No prior searches for this cfg_schedule; start with the initial_cfg_scale
        print(f"No prior searches for this cfg_schedule.")

    # Find the row with the minimal fid
    min_fid_row = df_filtered.loc[df_filtered["FID"].idxmin()]
    current_cfg_scale = min_fid_row["cfg_scale"]
    current_annotation = min_fid_row["annotation"]
    best_IS = min_fid_row["Inception Score"]

    print(f"The best setting found for cfg_schedule={cfg_schedule} is \
          annotation={current_annotation} cfg={current_cfg_scale} with FID={min_fid_row['FID']}, IS={best_IS}, \
        Precision={min_fid_row['Precision']}, Recall={min_fid_row['Recall']}, sFID={min_fid_row['sFID']}")




def search(args):
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    # dist.init_process_group("nccl")
    dist.init_process_group(
        "nccl",
        timeout=timedelta(hours=1)  # 1 hour
        )

    if args.cfg_schedule == 'step':
        cfg_start_ratio = args.step_start_ratio

        sample_dir = args.sample_dir[:-1] if args.sample_dir.endswith("/") else args.sample_dir
        csv_file_path = os.path.dirname(sample_dir) + "/search_results/" + sample_dir.split("/")[-1] + ".csv"

        args.cfg_schedule_kwargs = {
            "window_start": cfg_start_ratio
        }
        annotation = ";".join([f"{key}={value:.2f}" for key, value in args.cfg_schedule_kwargs.items()])

        best_cfg = search_cfg_only(args, annotation=annotation)

        # announce the best setting
        rank = int(os.environ.get('RANK', 0))
        node_rank = int(os.environ.get('NODE_RANK', 0))
        if node_rank == 0 and rank == 0:
            announce_best_setting(csv_file_path, args.cfg_schedule)

   
    if args.cfg_schedule == 'rectangular':
        # grid search for window_start, window_end
        # and for each combination, there will be a cfg_scale searched
        # for next step, search, the cfg_scale_start will be the best cfg_scale found in the previous step
        window_start_step = 0.05
        window_start_init = 0.1
        window_start_end = 0.8
        window_start_choices = np.arange(window_start_init, window_start_end + 0.01, window_start_step)
        
        window_end_step = 0.05
        window_end_init = 0.5  # You can adjust this initial value
        window_end_choices = np.arange(window_end_init, 1.01, window_end_step)  # 1.01 to include 1.0

        # compose the iteration objects in a 1d sequence so that we can use tqdm
        combs = [
            (window_start, window_end)
            for window_start in window_start_choices
            for window_end in window_end_choices
            if window_start < window_end  # Ensure window_start is always less than window_end
        ]

        for window_start, window_end in tqdm(combs, total=len(combs), desc="seaching progress"):
            cfg_schedule_kwargs = {
                "window_start": window_start,
                "window_end": window_end,
            }
            args.cfg_schedule_kwargs = cfg_schedule_kwargs
            # check for existing search results
            annotation = ";".join([f"{k}={v:.2f}" for k, v in cfg_schedule_kwargs.items()])

            best_cfg_scale = search_cfg_only(args, annotation=annotation, increase_first=False)
            args.cfg_scale_start = best_cfg_scale
        
        # announce the best setting
        rank = int(os.environ.get('RANK', 0))
        node_rank = int(os.environ.get('NODE_RANK', 0))
        if node_rank == 0 and rank == 0:
            announce_best_setting(csv_file_path, args.cfg_schedule)

    
    if args.cfg_schedule =='constant':
       
        sample_dir = args.sample_dir[:-1] if args.sample_dir.endswith("/") else args.sample_dir
        csv_file_path = os.path.dirname(sample_dir) + "/search_results/" + sample_dir.split("/")[-1] + ".csv"

        args.cfg_schedule_kwargs = {}
        annotation = ";".join([f"{key}={value:.2f}" for key, value in args.cfg_schedule_kwargs.items()])

        search_cfg_only(args, annotation=annotation)
        # announce the best setting
        rank = int(os.environ.get('RANK', 0))
        node_rank = int(os.environ.get('NODE_RANK', 0))
        if node_rank == 0 and rank == 0:
            announce_best_setting(csv_file_path, args.cfg_schedule)


    dist.destroy_process_group()


def get_next_search_state_cfg(
        csv_file_path, 
        cfg_schedule, 
        initial_cfg_scale, 
        annotation,
        step_size=0.25, 
        max_cfg_scale=None, 
        min_cfg_scale=1.0,
        increase_first=False
    ):
    """
    Determine the next cfg_scale to search based on existing results.

    Note that this only searches for cfg with given annotation (super parameter) and cfg_scale
    
    Parameters:
    - csv_file_path (str): Path to the CSV file containing search results.
    - cfg_schedule (str or int): The specific configuration schedule to filter.
    - initial_cfg_scale (int or float): The starting cfg_scale if no prior searches exist.
    - step_size (int or float): The increment/decrement step for searching neighboring cfg_scales.
    - max_cfg_scale (int or float, optional): The maximum allowable cfg_scale.
    - min_cfg_scale (int or float, optional): The minimum allowable cfg_scale.
    
    Returns:
    - next_cfg_scale (int or float or None): The next cfg_scale to search, or None if a local minimum is reached.
    - cur_best_fid
    - end_flag (bool): Whether the search has reached a local minimum.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # raise FileNotFoundError(f"The file {csv_file_path} does not exist.")
        # return the initial_cfg_scale if the file does not exist
        return initial_cfg_scale, None, False
    except pd.errors.EmptyDataError:
        # If the CSV is empty, start with the initial cfg_scale
        return initial_cfg_scale, None,  False

    # Filter the DataFrame based on cfg_schedule
    df_filtered = df[df["cfg_schedule"] == cfg_schedule]
    # change the nan to ""
    df_filtered = df_filtered.fillna("")
    df_filtered = df_filtered[df_filtered["annotation"] == annotation]

    if df_filtered.empty:
        # No prior searches for this cfg_schedule; start with the initial_cfg_scale
        return initial_cfg_scale, None, False

    # Find the row with the minimal fid
    min_fid_row = df_filtered.loc[df_filtered["FID"].idxmin()]
    current_cfg_scale = min_fid_row["cfg_scale"]

    # Define potential next cfg_scales
    potential_next_scales = [
        current_cfg_scale + step_size,
        current_cfg_scale - step_size
    ] if increase_first else [
        current_cfg_scale - step_size,
        current_cfg_scale + step_size
    ]

    # Apply constraints if any
    if max_cfg_scale is not None:
        potential_next_scales = [scale for scale in potential_next_scales if scale <= max_cfg_scale]
    if min_cfg_scale is not None:
        potential_next_scales = [scale for scale in potential_next_scales if scale >= min_cfg_scale]

    # Check which potential scales have not been tried yet
    tried_scales = set(df_filtered["cfg_scale"])
    for scale in potential_next_scales:
        if scale not in tried_scales:
            return scale, min_fid_row["FID"], False

    # If all neighboring scales have been tried, check if they resulted in higher or equal fid
    # This implies a local minimum
    return None, min_fid_row["FID"], True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--gpt-2d", action='store_true', help="whether to use 2D GPT(2d rope)")

    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)

    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--tok-config", type=str, default=None, help="config path for vq model")

    parser.add_argument("--quant-way", type=str, choices=['vq', 'fsq'], default='vq')

    parser.add_argument("--clear-cache", action="store_true", help="whether to clear all the images and .npz files")
    parser.add_argument("--merge", action="store_true", help="whether to merge all the results")


    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=2.0)
    parser.add_argument("--cfg-schedule", type=str, default="constant")
    parser.add_argument("--search", action="store_true", help="whether to search the best cfg scale")
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    parser.add_argument("--num-token", type=int, default=None, help="number of tokens to sample with")

    parser.add_argument("--qualitative", action="store_true", help="whether to evaluate the model quantitatively")
    parser.add_argument("--qual-num", type=int, default=100)
    parser.add_argument("--rope", action='store_true', help="whether using rotary embedding")
    parser.add_argument("--adaLN", action='store_true', help="whether using adaptive layer normalization")
    parser.add_argument("--simple-adaLN", action='store_true', help="whether using simple adaptive layer normalization")
    parser.add_argument("--qk-norm", action='store_true', help="whether using query and key normalization")
    parser.add_argument("--flash-attn", action='store_true', help="whether using flash attention")
    parser.add_argument("--step-start-ratio", type=float, default=0.18, help="start ratio of the step")
    parser.add_argument("--time-cnt-only", action='store_true', help="whether only count the time")
    parser.add_argument("--eval-python-path", type=str, default="python", help="python path for the specific environment used in gFID etc. evaluation")
    parser.add_argument("--gt-npz-path", type=str, default="VIRTUAL_imagenet256_labeled.npz", help="path to the ground truth npz file for gFID etc. evaluation")
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


    if args.search:
        if args.cfg_schedule not in ["rectangular", "constant", "step"]:
            raise NotImplementedError(f"Only rectangular and constant schedules are supported, currently {args.cfg_schedule} is not supported")
        search(args)
    else:
        if args.cfg_schedule == "step":
            args.cfg_schedule_kwargs = {
                "window_start": args.step_start_ratio
            }
        else:
            args.cfg_schedule_kwargs = {}
        quantitative_eval(args, independent=True)