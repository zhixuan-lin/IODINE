import torch
from torch import optim


def make_optimizer(cfg, model: torch.nn.Module):
    lr = cfg.TRAIN.BASE_LR
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    
    return optimizer


def make_scheduler(cfg, optimizer):
    return None
