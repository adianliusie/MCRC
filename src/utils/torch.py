import random
import torch
import numpy as np

#== Reproducibility ===============================================================================#
def set_rand_seed(seed_num=None):
    if seed_num:
        print('setting seed ', seed_num)
        random.seed(seed_num)
        torch.manual_seed(seed_num)
        np.random.seed(seed_num)

#== Optimisation ==================================================================================#
def build_triangular_scheduler(optimizer, num_warmup_steps:str, num_steps:str):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / num_warmup_steps
        return (num_steps - step) / (num_steps - num_warmup_steps)

    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )
    return scheduler