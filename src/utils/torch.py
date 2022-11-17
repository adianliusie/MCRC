import torch

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