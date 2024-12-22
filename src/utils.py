
import torch
import math
from torch import nn
from torch.optim import Adam, AdamW, SGD
from typing import Any, Dict, List, Tuple

class WarmupCosineSchedule:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        t_total: int,
        lr_start: float,
        lr_end: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.step = 0

    def step(self):
        self.step += 1
        if self.step <= self.warmup_steps:
            lr = self.lr_start + (self.lr_end - self.lr_start) * self.step / self.warmup_steps
        else:
            lr = self.lr_end + 0.5 * (self.lr_start - self.lr_end) * (
                1 + math.cos(math.pi * (self.step - self.warmup_steps) / (self.t_total - self.warmup_steps))
            )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
    

def get_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str,
    start_lr: float,
    weight_decay: float,
    **kwargs: Any,
):
    """Get optimizer for training.
    Args:
        model (nn.Module): Model to optimize
        optimizer_name (str): Name of optimizer
        start_lr (float): Initial learning rate
        max_lr (float): Maximum learning rate
        min_lr (float): Minimum learning rate
        weight_decay (float): Weight decay
        grad_clip (float): Gradient clipping value
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    if optimizer_name == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=start_lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=start_lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=start_lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")
    
    return optimizer

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str,
    max_lr: float,
    min_lr: float,
    warmup_epochs: int,
    num_epochs: int,
    **kwargs: Any,
):
    """Get learning rate scheduler.
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler_type (str): Type of scheduler
        max_lr (float): Maximum learning rate
        min_lr (float): Minimum learning rate
        warmup_epochs (int): Number of warmup epochs
        num_epochs (int): Number of epochs
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
    """
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs,
            eta_min=min_lr,
        )
    elif scheduler_type == "warmup_cosine":
        return WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=warmup_epochs,
            t_total=num_epochs,
            lr_start=max_lr,
            lr_end=min_lr,
        )
    else:
        raise ValueError(f"Invalid scheduler: {scheduler_type}")