import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from PIL import Image
from tqdm import tqdm
import math

import random
import numpy as np
from einops import rearrange
from pathlib import Path
import matplotlib.pyplot as plt

import argparse
import yaml
from sklearn.metrics import roc_auc_score
import json
from pprint import pprint
from collections import defaultdict
from torchvision import transforms

from utils import AverageMeter, calculate_mask_coverage
from datasets import build_dataset, EvalDataLoader
from models import create_vae, AutoencoderKL, create_mar_model, EncoderDecoerMAR
from backbones import get_backbone
from denoiser import get_denoiser, Denoiser
from mask import RandomMaskCollator, BlockRandomMaskCollator, CheckerBoardMaskCollator, ConstantMaskCollator, SlidingWindowMaskCollator, \
    indices_to_mask, mask_to_indices, PatchRandomMaskCollator
    
def parser_args():
    parser = argparse.ArgumentParser(description="AnoMAR Training")
    
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--diffusion_config', type=str, help='Path to the diffusion config file', required=True)
    parser.add_argument('--diffusion_ckpt', type=str, default='diffusion_ckpt', help='Path to the diffusion checkpoint', required=True)
    parser.add_argument('--backbone_name', type=str, default='enet', help='Backbone name')
    parser.add_argument('--mask_strategy', type=str, default='checkerboard', help='Mask strategy')
    parser.add_argument('--num_inference_steps', type=int, default=100, help='Number of inference steps')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('--save_dir', type=str, default='backbone_proj', help='Path to save the backbone projection')
    
    return parser.parse_args()

def main(args):
    
    def load_config(config_path):
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return config

    config = load_config(args.diffusion_config)
    pprint(config)
    
    # set seed
    seed = config['meta']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    device = "cuda"
    
    config["diffusion"]["num_sampling_steps"] = str(args.num_inference_steps)
    img_size = config['data']['img_size']
    num_samples = args.num_samples
    category = config['data']['category']
    
    train_dataset = build_dataset(**config['data'])
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, \
        pin_memory=config['data']['pin_memory'], num_workers=config['data']['num_workers'])
    
    in_sh = (3, img_size, img_size)
    
    # Load diffusion model
    denoiser: Denoiser = get_denoiser(**config['diffusion'], input_shape=in_sh)
    model_ckpt = torch.load(args.diffusion_ckpt, map_location='cpu', weights_only=True)
    keys_to_replace = []
    # Identify the keys to be renamed
    for k in model_ckpt.keys():
        if 'module' in k:
            keys_to_replace.append(k)

    # Modify the keys outside the loop
    for k in keys_to_replace:
        model_ckpt[k.replace('module.', 'net.')] = model_ckpt.pop(k)
        
    results = denoiser.load_state_dict(model_ckpt, strict=False)
    print(results)
    denoiser.to(device).eval()
    
    # Load backbone model
    model_kwargs = {
        'model_type': 'efficientnet-b4',
        'outblocks': (1, 5, 9, 21),
        'outstrides': (2, 4, 8, 16),
        'pretrained': True,
        'stride': 16
    }
    # print(f"Using feature space reconstruction with {model_kwargs['model_type']} backbone")
    # model_kwargs = {"model_type": "pdn_medium"}
    # model_kwargs = {
    #     'model_type': 'wide_resnet',
    #     'ckpt_path': f'./{category}_fe'
    # }
    backbone = get_backbone(**model_kwargs)
    backbone.to(device)
    
    backbone_proj = torch.nn.Conv2d(272, 272, kernel_size=1, bias=True)
    
    backbone_proj.to(device)
    
    mask_strategy = args.mask_strategy
    if mask_strategy == "random":
        mask_collator = RandomMaskCollator(
            ratio=mask_ratio, input_size=in_sh[1], patch_size=1
        )
    elif mask_strategy == "block":
        kwargs = {
            "aspect_min": 0.75,
            "aspect_max": 1.5,
            "scale_min": 0.4,
            "scale_max": 0.7,
            "num_blocks": 2,
        }
        mask_collator = BlockRandomMaskCollator(
            input_size=in_sh[1], patch_size=1, mask_ratio=mask_ratio, **kwargs
        )
    elif mask_strategy == "checkerboard":
        mask_collator = CheckerBoardMaskCollator(
            input_size=in_sh[1], patch_size=1, min_divisor=2, max_divisor=2
        )
    elif mask_strategy == "constant":
        mask_ratio = config['data']['mask']['ratio']
        mask_collator = ConstantMaskCollator(
            ratio=mask_ratio, input_size=in_sh[1], patch_size=1
        )
    elif mask_strategy == "sliding_window":
        config['data']['mask']['order'] = 'raster'
        mask_collator = SlidingWindowMaskCollator(
            input_size=in_sh[1], patch_size=1, **config['data']['mask']
        )
        args.num_masks = len(mask_collator)
        print(f"For sliding window mask, num_masks is overrided to {args.num_masks}")
    elif mask_strategy == "prandom":
        mask_collator = PatchRandomMaskCollator(
            input_size=in_sh[1], patch_size=16, mask_ratio=mask_ratio
        )
    else:
        raise ValueError(f"Invalid mask strategy: {mask_strategy}")
    
    masks = mask_collator.collate_all_masks()  # (N, M)
    
    # Build optimizer (only for backbone proj)
    params = list(backbone_proj.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    loss_meter = AverageMeter()
    for epoch in tqdm(range(args.num_epochs)):
        train_one_epoch(
            epoch,
            train_loader,
            denoiser,
            backbone,
            backbone_proj,
            masks,
            optimizer,
            loss_meter,
            in_sh,
            num_samples,
            device
        )
    
    # Save the model
    save_dir = os.path.join(args.save_dir, 'backbone_proj.pth')
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(backbone_proj.state_dict(), save_dir)

def ddrm_sample(
    denoiser,
    masks,
    images, 
    in_sh,
    num_samples,    
    device,
):
    m = rearrange(masks, 'b (h w) -> b h w', h=in_sh[1], w=in_sh[2])
    m = m.repeat_interleave(num_samples, dim=0)
    images = images.repeat_interleave(num_samples, dim=0)
    
    x_T = torch.randn((len(images), *in_sh), device=device)
    model_kwargs = dict(c=None, z=None, mask_indices=None, z_vis=None)
    sample_fn = denoiser.net.forward
    x_org = images
    inv_m = 1 - m.float()
    m, inv_m = m.unsqueeze(1), inv_m.unsqueeze(1)
    
    indices = list(range(denoiser.sample_diffusion.num_timesteps))[::-1]
    alphas = 1.0 - denoiser.sample_diffusion.betas
    sigmas = np.sqrt(1/alphas - 1)

    eta = 0.85
    etaB = 1.0
    x = x_T
    for i in indices:
        t = torch.tensor([i] * len(x_org)).to(device)
        
        # For first denoising step
        if i == denoiser.sample_diffusion.num_timesteps - 1:
            noise = torch.randn(x_org.shape, device=device)
            x_m = sigmas[i] * noise * m   # unconditional for masked pixels
            x_v = x_org * inv_m + sigmas[i] * noise * inv_m # conditional for unmasked pixels
            x = x_m + x_v
            # scale x_t to VP domain
            x = x * math.sqrt(alphas[i])
            x.to(device)
            continue
        
        with torch.no_grad():
            out = denoiser.sample_diffusion.p_sample(
                sample_fn,
                x,
                t,
                clip_denoised=False,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=model_kwargs,
                temperature=1.0,
            )
            # pred_x0 = out["pred_xstart"]
            eps = out["eps"]
            pred_x0 = x - sigmas[i] * eps 
        
        # Predict x_{t-1}
        sigma_t = sigmas[i]
        prev_sigma_t = sigmas[i+1]
        noise = torch.randn(x_org.shape, device=device)
        x_m = m * (pred_x0 + math.sqrt(1 - eta**2) * sigma_t * (x - pred_x0) / prev_sigma_t + eta * noise * sigma_t)
        x_v = inv_m * ((1 - etaB) * pred_x0 + etaB * x_org + noise * sigma_t)    
        
        x = x_m + x_v
    
    return x, x_org  # (B*K, C, H, W)

def train_one_epoch(
    epoch,
    train_loader,
    denoiser,
    backbone,
    backbone_proj,
    masks,
    optimizer,
    loss_meter,
    in_sh,
    num_samples,
    device
):
    # Set models to train mode
    denoiser.eval()
    backbone.train()
    backbone_proj.train()
    
    for i, data in enumerate(train_loader):
        images = data['samples'].to(device)
        mask_idx = random.choice(range(len(masks)))
        mask = masks[mask_idx].to(device).unsqueeze(0).repeat(images.shape[0], 1)

        with torch.no_grad():
            # get samples
            samples, samples_org = ddrm_sample(
                denoiser,
                mask,
                images,
                in_sh,
                num_samples,
                device,
            )
            
        # Forward pass
        features = backbone(samples)
        # Residual connection
        features_proj = backbone_proj(features) + features
        features_org = backbone(samples_org).detach()
        
        # compute loss
        loss = F.mse_loss(features_proj, features_org)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Update loss meter
        loss_meter.update(loss.item())
        
        # Print loss
        if i % 5 == 0:
            print(f"Epoch {epoch}, Iter {i}, Loss {loss_meter.avg}")
            loss_meter.reset()
        
if __name__ == "__main__":
    args = parser_args()
    main(args)
    
    
