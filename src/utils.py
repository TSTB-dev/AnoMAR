
import torch
import math
from torch import nn
from torch.optim import Adam, AdamW, SGD
from typing import Any, Dict, List, Tuple

def patchify(imgs, p):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def pidx_to_ppos(patch_idx, h, w):
    """
    Convert patch index to patch position
    Args:
        patch_idx: (N, L), patch index in [0, patch_size**2)
        h: height of image
        w: width of image
    Return:
        patch_pos: (N, L, 2), patch position in (x, y)
    """
    patch_pos = torch.zeros((patch_idx.shape[0], patch_idx.shape[1], 2), dtype=torch.int)
    for i, idx in enumerate(patch_idx):
        patch_pos[i] = torch.stack([idx % w, idx // w], dim=1)
    return patch_pos

def ppos_to_pidx(patch_pos, h, w):
    """
    Convert patch position to patch index
    Args:
        patch_pos: (N, L, 2), patch position in (x, y)
        h: height of image
        w: width of image
    Return:
        patch_idx: (N, L), patch index in [0, patch_size**2)
    """
    patch_idx = patch_pos[:, :, 1] * w + patch_pos[:, :, 0]
    return patch_idx

def ppos_to_pmask(patch_pos, h, w):
    """
    Convert patch position to binary mask of patches
    Args:
        patch_pos: (N, L, 2), patch position in (x, y)
        h: height of image
        w: width of image
    Return:
        mask: (N, H, W), binary mask
    """
    mask = torch.zeros((patch_pos.shape[0], 1, h, w), dtype=torch.float32)
    for i, pos in enumerate(patch_pos):
        for x, y in pos:
            mask[i, 0, y, x] = 1
    return mask

def pmask_to_ppos(mask):
    """
    Convert binary mask of patches to patch position
    Args:
        mask: (N, H, W), binary mask
    Return:
        patch_pos: (N, L, 2), patch position in (x, y)
    """
    patch_pos = torch.stack(mask.nonzero(), dim=1)
    return patch_pos

def pidx_to_pmask(patch_idx, h, w):
    """
    Convert patch index to binary mask of patches
    Args:
        patch_idx: (N, L), patch index in [0, patch_size**2)
        h: height of image
        w: width of image
    Return:
        mask: (N, H, W), binary mask
    """
    patch_pos = pidx_to_ppos(patch_idx, h, w)
    mask = ppos_to_pmask(patch_pos, h, w)
    return mask

def pmask_to_pidx(mask, h, w):
    """
    Convert binary mask of patches to patch index
    Args:
        mask: (N, H, W), binary mask
        h: height of image
        w: width of image
    Return:
        patch_idx: (N, L), patch index in [0, patch_size**2)
    """
    patch_pos = pmask_to_ppos(mask)
    patch_idx = ppos_to_pidx(patch_pos, h, w)
    return patch_idx

def ppos_to_imask(patch_pos, h, w, patch_size):
    """Convert patch position to binary mask of image
    Args:
        patch_pos: (N, L, 2), patch position in (x, y)
        h: height of image
        w: width of image
        patch_size: size of patch
    Return:
        mask: (N, H, W), binary mask (float32)
    """
    H, W = h * patch_size, w * patch_size
    mask = torch.ones((patch_pos.shape[0], 1, H, W), dtype=torch.float32)
    for i, pos in enumerate(patch_pos):
        for x, y in pos:
            mask[i, 0, y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size] = 0
    return mask

def imask_to_ppos(mask, patch_size):
    """Convert binary mask of image to patch position
    Args:
        mask: (N, H, W), binary mask
        patch_size: size of patch
    Return:
        patch_pos: (N, L, 2), patch position in (x, y)
    """
    patch_pos = torch.stack(mask.nonzero(), dim=1)
    patch_pos = patch_pos[:, :, [1, 2]]
    patch_pos = patch_pos - patch_pos % patch_size
    return patch_pos

def pidx_to_imask(patch_idx, h, w, patch_size):
    """Convert patch index to binary mask of image
    Args:
        patch_idx: (N, L), patch index in [0, patch_size**2)
        h: height of image
        w: width of image
        patch_size: size of patch
    Return:
        mask: (N, H, W), binary mask
    """
    patch_pos = pidx_to_ppos(patch_idx, h, w)
    mask = ppos_to_imask(patch_pos, h, w, patch_size)
    return mask

def imask_to_pidx(mask, h, w, patch_size):
    """Convert binary mask of image to patch index
    Args:
        mask: (N, H, W), binary mask
        h: height of image
        w: width of image
        patch_size: size of patch
    Return:
        patch_idx: (N, L), patch index in [0, patch_size**2)
    """
    patch_pos = imask_to_ppos(mask, patch_size)
    patch_idx = ppos_to_pidx(patch_pos, h, w)
    return patch_idx


def calculate_mask_coverage(mask_batch, h, w):
    """Calculate mask coverage. 

    Args:
        mask_batch (tensor): Indices of masked patches. Shape: (B, L)
        h (int): Height of the feature map
        w (int): Width of the feature map
    Returns:
        mask_coverage (float): Mask coverage
    """
    mask = pidx_to_pmask(mask_batch, h, w)  # (B, h, w)
    mask_or = torch.any(mask, dim=0).float()  # (h, w)
    mask_coverage = torch.mean(mask_or)  # scalar
    return mask_coverage  

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def get_avg(self):
        return self.avg

class WarmupCosineAnnealingScheduler:
    def __init__(self, optimizer, warmup_steps: int, t_total: int, init_lr: float, peak_lr: float):
        """
        Args:
            optimizer (Optimizer): The optimizer to update.
            warmup_steps (int): Number of steps to warm up.
            t_total (int): Total number of steps for the schedule.
            init_lr (float): Initial learning rate.
            peak_lr (float): Peak learning rate after warmup.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.min_lr = 1e-6
        self.step_num = 0
        
        # Initialize the optimizer's learning rate.
        self._set_lr(init_lr)

    def _set_lr(self, lr: float):
        """Sets the learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.last_lr = lr

    def step(self):
        """Updates the learning rate based on the current step."""
        self.step_num += 1
        
        if self.step_num < self.warmup_steps:
            # Linear warmup: increase lr from init_lr to peak_lr.
            lr = self.init_lr + (self.peak_lr - self.init_lr) * (self.step_num / self.warmup_steps)
        else:
            # Cosine annealing: decay lr from peak_lr to min_lr.
            progress = (self.step_num - self.warmup_steps) / max(1, (self.t_total - self.warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
        
        self._set_lr(lr)

    def get_last_lr(self):
        """Returns the last set learning rate."""
        return self.last_lr
    

def get_optimizer(
    models: List[nn.Module],
    *,
    optimizer_name: str,
    init_lr: float,
    weight_decay: float,
    **kwargs: Any,
):
    """Get optimizer for training.
    Args:
        model (nn.Module): Model to optimize
        optimizer_name (str): Name of optimizer
        init_lr (float): Initial learning rate
        max_lr (float): Maximum learning rate
        min_lr (float): Minimum learning rate
        weight_decay (float): Weight decay
        grad_clip (float): Gradient clipping value
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    params = []
    for model in models:
        params += list(model.parameters())
    
    if isinstance(init_lr, str):
        init_lr = float(init_lr)
    
    if optimizer_name == "adam":
        optimizer = Adam(
            params,
            lr=init_lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = AdamW(
            params,
            lr=init_lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )
    elif optimizer_name == "sgd":
        optimizer = SGD(
            params,
            lr=init_lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}")
    
    return optimizer

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str,
    init_lr: float,
    peak_lr: float,
    warmup_epochs: int,
    num_epochs: int,
    iter_per_epoch: int,
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
            T_max=iter_per_epoch * num_epochs,
            eta_min=init_lr,
        )
    elif scheduler_type == "warmup_cosine":
        return WarmupCosineAnnealingScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_epochs * iter_per_epoch,
            t_total=num_epochs * iter_per_epoch,
            init_lr=init_lr,
            peak_lr=peak_lr,
        )
    else:
        raise ValueError(f"Invalid scheduler: {scheduler_type}")