import numpy as np
import torch


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        if "lr_mult" in param_group:
            # print(f"Assigning learning rate {new_lr * param_group['lr_mult']} to group")
            param_group["lr"] = new_lr * param_group["lr_mult"]
        else:
            param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step, init_div_factor=100):
    ratio = (step / warmup_length) + (1 - step / warmup_length) / init_div_factor
    return base_lr * ratio
    # return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def const_lr_cooldown(
        optimizer, 
        base_lr, 
        warmup_length, 
        steps, 
        cooldown_steps, 
        cooldown_power=1.0, 
        cooldown_end_lr=0.
    ):

    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e/es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def wsd_lr(
        optimizer, 
        base_lr, 
        warmup_length, 
        steps, 
        final_lr_factor=0.0,
        init_div_factor=100,
        fract_decay=0.2,
        decay_type="sqrt",
    ):
    """
    Adapted from https://github.com/epfml/schedules-and-scaling/src/optim/utils.py
    This is a function that returns a function that adjusts the learning rate of the optimizer.
    Args:
        steps: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end
        warmup_length: length of iterations used for warmup
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
    """
    n_anneal_steps = int(fract_decay * steps)
    n_hold = steps - n_anneal_steps

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step, init_div_factor=init_div_factor)
        elif step < n_hold:
            lr = base_lr
        else:
            if decay_type == "linear":
                lr_factor = final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
                lr = base_lr * lr_factor

            elif decay_type == "exp":
                lr = final_lr_factor ** ((step - n_hold) / n_anneal_steps)
            elif decay_type == "cosine":
                lr = base_lr * (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + np.cos(np.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
            elif decay_type == "square":
                lr_factor = final_lr_factor + (1 - final_lr_factor) * max(
                    1 - ((step - n_hold) / n_anneal_steps) ** 2, 0
                )

                lr = base_lr * lr_factor

            elif decay_type == "sqrt":
                lr_factor = final_lr_factor + (1 - final_lr_factor) * max(
                    1 - np.sqrt((step - n_hold) / n_anneal_steps), 0
                )

                lr = base_lr * lr_factor

            else:
                raise ValueError(
                    f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp']"
                )
        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster



def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        elif step >= steps:
            lr = 0.0
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def cosine_schedule_with_warmup_v2(
        optimizer, 
        base_lr, 
        warmup_length, 
        steps, 
        num_cycles=0.5, 
        end_lr=0.0,
        init_div_factor=100,
    ):
    """
    Args:
        optimizer: A torch.optim.Optimizer, the optimizer for which to schedule the learning rate.
        num_warmup_steps: An integer, the number of steps for the warmup phase.
        num_training_steps: An integer, the total number of training steps.
        num_cycles : A float, the number of periods of the cosine function in a schedule (the default is to 
            just decrease from the max value to 0 following a half-cosine).
        last_epoch: An integer, the index of the last epoch when resuming training.
        base_lr: A float, the base learning rate.
        end_lr: A float, the final learning rate.
    """

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step, init_div_factor=init_div_factor)
        elif step >= steps:
            lr = end_lr
        else:
            progress = (step - warmup_length) / (steps - warmup_length)
            lr = end_lr + 0.5 * (base_lr - end_lr) * (1 + np.cos(np.pi * num_cycles * 2.0 * progress))
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster
