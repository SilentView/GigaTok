"""
This script is used to manage the checkpoints and the running log.
The file structure for results files are:

- results
    - tokenizers (This level will share a project name)
        - vqgan_imgnet_{exp_idx}
            - config.json
            - checkpoints ({num_step}.pth)
            - log (Local log only for timestamps and warning messages)
    - generators
        - ....
"""
import os
# import glob
from glob import glob
import time
import shutil
import json

import wandb
import torch

from typing import List, Dict


###################
# Resume related functions
###################

def init_wandb(
    project_name:str,
    config:dict,
    exp_dir,
    # for the same project, training and eval will have different run id
    eval_run=False,
    name=None
    ):
    """
    This function will create a cache file to store the logs in 
    {exp_dir}/wandb_cache.json
    """

    # first check for id
    confg_file = os.path.join(exp_dir, "config.json")
    if os.path.exists(confg_file):
        with open(confg_file, "r") as f:
            data = json.load(f)
        if eval_run:
            id = data.get("wandb_run_id_eval", None)
        else:
            id = data.get("wandb_run_id", None)
    else:
        id = None
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        # track hyperparameters and run metadata
        config=config,
        resume="allow",
        name=name,
        id=id
    )
    save_wandb_run_id(exp_dir, eval_run=eval_run)
    save_wandb_project(exp_dir)

    if eval_run:
        # wanbd cache will only be needed for eval runs, because 
        # eval run may launch parallel evaluations for a single run.
        with open(os.path.join(exp_dir, "wandb_cache.json"), "w") as f:
            json.dump([], f, indent=4)


def wandb_cache_file_append(
        data:List[Dict], 
        exp_dir, 
    ):
    for d in data:
        assert "iteration" in d, "data must contain 'iteration'"
        # use item for Tensor
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.item()
    cache_file_path = os.path.join(exp_dir, "wandb_cache.json")
    if not os.path.exists(cache_file_path):
        with open(cache_file_path, "w") as f:
            json.dump([], f, indent=4)
        cache_data = []
    else:
        with open(cache_file_path, "r") as f:
            cache_data = json.load(f)
    cache_data.extend(data)
    with open(cache_file_path, "w") as f:
        json.dump(cache_data, f, indent=4)


def upload_wandb_cache(exp_dir):
    cache_file_path = os.path.join(exp_dir, "wandb_cache.json")
    with open(cache_file_path, "r") as f:
        cache_data = json.load(f)
    for data in cache_data:
        wandb.log(data)
    
    with open(cache_file_path, "w") as f:
        json.dump([], f, indent=4)


def update_wandb_log(
    update_dict:dict
):
    assert "iteration" in update_dict, "update_dict must contain 'iteration'" 
    # assert "epoch" in update_dict, "update_dict must contain 'epoch'"
    wandb.log(update_dict)


def save_wandb_run_id(exp_dir, eval_run=False):
    confg_file = os.path.join(exp_dir, "config.json")
    if not os.path.exists(confg_file):
        data = {}
    else:
        with open(confg_file, "r") as f:
            data = json.load(f)
    run_id_key = "wandb_run_id_eval" if eval_run else "wandb_run_id"
    if run_id_key in data:
        assert data[run_id_key] == wandb.run.id, f"{run_id_key} in config.json is not the same as the current run"
    data[run_id_key] = wandb.run.id
    if not os.path.exists(confg_file):
        os.makedirs(os.path.dirname(confg_file), exist_ok=True)
    with open(confg_file, "w") as f:
        json.dump(data, f, indent=4)

def save_wandb_project(exp_dir):
    confg_file = os.path.join(exp_dir, "config.json")
    if not os.path.exists(confg_file):
        data = {}
    else:
        with open(confg_file, "r") as f:
            data = json.load(f)
    if "wandb_project" in data:
        try:
            assert data["wandb_project"] == wandb.run.project, "wandb_project in config.json is not the same as the current run"
        except AssertionError as e:
            print(confg_file)
            raise e
    data["wandb_project"] = wandb.run.project
    if not os.path.exists(confg_file):
        os.makedirs(os.path.dirname(confg_file), exist_ok=True)
    with open(confg_file, "w") as f:
        json.dump(data, f, indent=4)


    

def get_int_prefix_value(f_name):
    f_name = f_name.split("/")[-1]
    f_name = f_name.split(".")[0]
    f_value = int(f_name)
    return f_value


def manage_ckpt_num(
        ckpt_dir, 
        milestone_step=100_000,
        milestone_start=100_000,
        max_milestone_num=5,
        max_none_milestone_num=10
        ):
    """
    Checkpoints will be saved at a very frequent interval. But each time a milestone is reached,
    the checkpoints before the milestone will be deleted (except those saved at a milestone tick).
    We keep only max_milestone_num milestones.
    """
    ckpt_files = glob(os.path.join(ckpt_dir, "*.pt"))
    ckpt_file_values = [[f, get_int_prefix_value(f)] for f in ckpt_files]
    ckpt_file_values = sorted(ckpt_file_values, key=lambda x: x[1])

    milestones = []
    for f, value in ckpt_file_values:
        if value >= milestone_start and (value - milestone_start) % milestone_step == 0:
            milestones.append([f, value])
    
    if len(milestones) > max_milestone_num:
        milestones_to_keep = milestones[-max_milestone_num:]
        milestones_to_delete = milestones[:-max_milestone_num]
        # remove the milestones that are not to be kept 
        for f, _ in milestones_to_delete:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
    
    # remove the non-milestone checkpoints before the last milestone

    non_milestone_items = [item for item in ckpt_file_values if item not in milestones]

    for f, v in non_milestone_items[:-max_none_milestone_num]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    if len(milestones) == 0:
        return

    last_milestone = milestones[-1]
    for f, v in non_milestone_items:
        if v < last_milestone[1]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass


def manage_fsdp_ckpt_num(
        ckpt_dir, 
        optim_ckpt_dir,
        milestone_step=100_000,
        milestone_start=100_000,
        max_milestone_num=5,
        max_none_milestone_num=10
        ):
    """
    Checkpoints will be saved at a very frequent interval. But each time a milestone is reached,
    the checkpoints before the milestone will be deleted (except those saved at a milestone tick).
    We keep only max_milestone_num milestones.
    This function is specifically for mananging the checkpoints saved by fsdp. The difference is that
    it also manages the optimizer state ckpts.
    """
    ckpt_files = glob(os.path.join(ckpt_dir, "*.pt"))
    ckpt_file_values = [[f, get_int_prefix_value(f)] for f in ckpt_files]
    ckpt_file_values = sorted(ckpt_file_values, key=lambda x: x[1])

    optim_ckpt_dirs = glob(os.path.join(optim_ckpt_dir, "*"))
    optim_ckpt_values = [[f, get_int_prefix_value(f)] for f in optim_ckpt_dirs]
    optim_ckpt_values = sorted(optim_ckpt_values, key=lambda x: x[1])

    milestones = []
    for f, value in ckpt_file_values:
        if value >= milestone_start and (value - milestone_start) % milestone_step == 0:
            milestones.append([f, value])
    
    optim_milestones = []
    for f, value in optim_ckpt_values:
        if value >= milestone_start and (value - milestone_start) % milestone_step == 0:
            optim_milestones.append([f, value])
    
    if len(milestones) > max_milestone_num:
        milestones_to_keep = milestones[-max_milestone_num:]
        milestones_to_delete = milestones[:-max_milestone_num]
        # remove the milestones that are not to be kept 
        for f, _ in milestones_to_delete:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
    
    if len(optim_milestones) > max_milestone_num:
        optim_milestones_to_keep = optim_milestones[-max_milestone_num:]
        optim_milestones_to_delete = optim_milestones[:-max_milestone_num]
        # remove the milestones that are not to be kept
        for f, _ in optim_milestones_to_delete:
            # remove the directory
            try:
                shutil.rmtree(f)
            except FileNotFoundError:
                pass
    
    # remove the non-milestone checkpoints before the last milestone
    non_milestone_items = [item for item in ckpt_file_values if item not in milestones]
    for f, v in non_milestone_items[:-max_none_milestone_num]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    
    optim_non_milestone_items = [item for item in optim_ckpt_values if item not in optim_milestones]
    for f, v in optim_non_milestone_items[:-max_none_milestone_num]:
        # remove the directory
        try:
            shutil.rmtree(f)
        except FileNotFoundError:
            pass


    if len(milestones) == 0 and len(optim_milestones) == 0:
        return

    last_milestone = milestones[-1]
    for f, v in non_milestone_items:
        if v < last_milestone[1]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
    
    optim_last_milestone = optim_milestones[-1]
    for f, v in optim_non_milestone_items:
        if v < optim_last_milestone[1]:
            # remove the directory
            try:
                shutil.rmtree(f)
            except FileNotFoundError:
                pass


def wsd_find_newest_ckpt(
        config, 
        const_ckpt_dir, 
        cd_sub_dir, 
        total_steps,
        fract_decay,
    ):
    """
    Find the newest checkpoint for constant + cooldown training
    Search both where the constant stage checkpoints are stored and where the cooldown stage checkpoints are stored
    All the training progresses are expressed by iteration steps

    Illustration of file structure:
    - exp_name 
        - checkpoints (for constant lr part)
        config.json   (wandb_config for constant part)
        log.txt
        - cd_records
            - cd_fract_0.2_from_N
                - checkpoints
                    {CUR_n}.pt
                config.json   (wandb_config for cool down part)
                log.txt
    """
    const_end_flag = False
    fract_decay = fract_decay if fract_decay is not None else config["trainer"].get("fract_decay", 0.2)
    # constant_epochs = int(args.epochs * fract_decay)
    constant_steps = int(total_steps * (1 - fract_decay))
    if len(glob(f"{const_ckpt_dir}/*.pt"))!= 0:
        latest_checkpoint_const = max(glob(f"{const_ckpt_dir}/*.pt"), key=get_int_prefix_value)
    else:
        const_end_flag = False
        return None, const_end_flag

    largest_step_const = int(latest_checkpoint_const.split('/')[-1].split('.')[0])

    # find the needed epochs
    if largest_step_const >= constant_steps:
        const_end_flag = True
        # check the sub folder for cooldown stage latest ckpt
        if len(glob(f"{cd_sub_dir}/checkpoints/*.pt")) != 0:
            latest_checkpoint_cd = max(glob(f"{cd_sub_dir}/checkpoints/*.pt"), 
                                       key=get_int_prefix_value)
            resume_checkpoint = latest_checkpoint_cd
        else:
            # find the biggest const step that is smaller than the constant steps
            const_ckpts = glob(f"{const_ckpt_dir}/*.pt")
            const_ckpts = [ckpt for ckpt in const_ckpts if get_int_prefix_value(ckpt) <= constant_steps]
            resume_checkpoint = max(const_ckpts, key=get_int_prefix_value)
            const_end_flag = False
   
        assert os.path.exists(resume_checkpoint), \
            f"cannot find the latest checkpoint for constant + cooldown training to resume"
    else:
        const_end_flag = False
        resume_checkpoint = latest_checkpoint_const
    
    return resume_checkpoint, const_end_flag




import unittest
from unittest.mock import patch, MagicMock, call
import os

# class TestManageCkptNum(unittest.TestCase):
#     @patch('os.remove')
#     @patch('glob')
#     def test_manage_ckpt_num(self, mock_glob, mock_remove):
#         # Mock the glob function to return a list of fake checkpoint files
#         mock_glob.return_value = [
#             'ckpt/010000.pth', 'ckpt/020000.pth', 'ckpt/030000.pth',
#             'ckpt/100000.pth', 'ckpt/200000.pth', 'ckpt/300000.pth',
#             'ckpt/400000.pth', 'ckpt/500000.pth', 'ckpt/600000.pth',
#             'ckpt/610000.pth', 'ckpt/620000.pth', 'ckpt/630000.pth',
#         ]
        
#         # Call the function with the test parameters
#         manage_ckpt_num(
#             ckpt_dir='ckpt',
#             milestone_step=100_000,
#             milestone_start=100_000,
#             max_milestone_num=2
#         )
        
#         # Check that os.remove was called with the correct files
#         mock_remove.assert_any_call('ckpt/100000.pth')
#         mock_remove.assert_any_call('ckpt/200000.pth')
#         mock_remove.assert_any_call('ckpt/300000.pth')
#         mock_remove.assert_any_call('ckpt/010000.pth')
#         mock_remove.assert_any_call('ckpt/020000.pth')
#         mock_remove.assert_any_call('ckpt/030000.pth')
#         mock_remove.assert_any_call('ckpt/400000.pth')
        
#         # Check that os.remove was not called with the last 2 milestone files
#         self.assertNotIn(call('ckpt/500000.pth'), mock_remove.call_args_list)
#         self.assertNotIn(call('ckpt/600000.pth'), mock_remove.call_args_list)
        
#         # Check that os.remove was called 6 times in total
#         self.assertEqual(mock_remove.call_count, 7)

if __name__ == "__main__":
    unittest.main()
