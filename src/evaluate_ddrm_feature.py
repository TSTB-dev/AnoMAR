
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
    parser = argparse.ArgumentParser(description='AnoMAR Sampling')
    
    parser.add_argument('--num_samples', type=int, help='Number of samples to generate', default=1)
    parser.add_argument('--num_inference_steps', type=int, help='Number of inference steps')
    parser.add_argument('--num_iterations', type=int, help='Number of iterations', default=1)
    parser.add_argument('--quant_thresh', type=float, help='Quantization threshold', default=0.8)
    parser.add_argument('--sampler', type=str, help='Sampler to use', default='org')  # ddim or org
    parser.add_argument('--temperature', type=float, help='Temperature for sampling', default=1.0)
    parser.add_argument('--num_masks', type=int, help='Number of masks to generate', default=1)
    parser.add_argument('--recon_space', type=str, default='latent', help='Reconstruction space')  # ['latent', 'pixel', 'feature']
    parser.add_argument('--aggregation', type=str, default='mean', help='Aggregation method')  # ['mean', 'max']
    parser.add_argument('--eta', type=float, help='Eta for sampling', default=0.85)
    parser.add_argument('--etaB', type=float, help='EtaB for sampling', default=1.0)
    parser.add_argument('--save_images', action='store_true', help='Save images')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--model_ckpt', type=str, help='Path to the model checkpoint')
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--sample_indices', type=int, nargs='+', help='Sample indices to visualize', default=[])
    
    return parser.parse_args()
    
@torch.no_grad()
def main(args):
    
    def load_config(config_path):
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return config

    config = load_config(args.config_path)
    pprint(config)
    
    # set seed
    seed = config['meta']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    config["diffusion"]["num_sampling_steps"] = str(args.num_inference_steps)
    img_size = config['data']['img_size']
    device = args.device
    num_samples = args.num_samples
    category = config['data']['category']
    
    dataset_config = config['data']
    dataset_config['transform_type'] = 'ddad'
    train_dataset = build_dataset(**dataset_config)
    dataset_config['train'] = False
    dataset_config['anom_only'] = True
    anom_dataset = build_dataset(**dataset_config)
    dataset_config['anom_only'] = False
    dataset_config['normal_only'] = True
    normal_dataset = build_dataset(**dataset_config)

    img_size = config['data']['img_size']
    patch_size = 16
    in_sh = (272, img_size // 16, img_size // 16)
    
    # build mim model
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=in_sh)
    model_ckpt = torch.load(args.model_ckpt, map_location='cpu', weights_only=True)
    keys_to_replace = []
    # Identify the keys to be renamed
    for k in model_ckpt.keys():
        if 'module' in k:
            keys_to_replace.append(k)

    # Modify the keys outside the loop
    for k in keys_to_replace:
        model_ckpt[k.replace('module.', 'net.')] = model_ckpt.pop(k)
        
    results = model.load_state_dict(model_ckpt, strict=False)
    print(results)
    model.to(device).eval()
    
    if args.recon_space == 'feature':
        model_kwargs = {
            'model_type': 'efficientnet-b4',
            'outblocks': (1, 5, 9, 21),
            'outstrides': (2, 4, 8, 16),
            'pretrained': True,
            'stride': 16
        }
        feature_extractor = get_backbone(**model_kwargs)
        feature_extractor.to(device).eval()
        
    # build mask collator
    mask_strategy = "random"
    mask_ratio = 0.75

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
            ratio=mask_ratio, input_size=in_sh[1], patch_size=patch_size
        )
    elif mask_strategy == "sliding_window":
        config['data']['mask']['order'] = 'raster'
        mask_collator = SlidingWindowMaskCollator(
            input_size=in_sh[1], patch_size=patch_size, **config['data']['mask']
        )
        args.num_masks = len(mask_collator)
        print(f"For sliding window mask, num_masks is overrided to {args.num_masks}")
    elif mask_strategy == "prandom":
        mask_collator = PatchRandomMaskCollator(
            input_size=in_sh[1], patch_size=16, mask_ratio=mask_ratio
        )
    else:
        raise ValueError(f"Invalid mask strategy: {mask_strategy}")
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=False)
    normal_loader = DataLoader(normal_dataset, args.batch_size, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    # masks = mask_collator.generate_random_mask(args.num_masks)  # (N, M)
    masks = mask_collator.collate_all_masks(args.num_masks)  # (N, M)
    args.num_masks = len(masks)

    def reshape_mask(m):  
        """ (B, hw) -> (B, hp, wp) """
        m = rearrange(m, 'b (h w) -> b h w', h=in_sh[1], w=in_sh[2])
        m = torch.repeat_interleave(m, repeats=patch_size, dim=1)
        m = torch.repeat_interleave(m, repeats=patch_size, dim=2)
        m = m.float()
        return m

    print(f"Compute global statistics...")
    features = []
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        img = data["samples"].to(device)
        with torch.no_grad():
            x, _ = feature_extractor(img)
            features.append(x)
    features = torch.cat(features, dim=0)   # (N, c, h, w)
    avg_glo = features.mean(dim=(0, 2, 3))  # (c, )
    std_glo = features.std(dim=(0, 2, 3))  # (c, )
    print(avg_glo[0], std_glo[0])
    
    # Evaluation on normal samples
    print("Evaluating on normal samples")
    normal_scores = []
    mask_coverage_meter = AverageMeter()
    sample_dict = defaultdict(dict)
    for step_idx, batch in tqdm(enumerate(normal_loader), total=len(normal_loader)):
        bs = len(batch["samples"])
        m = masks.repeat(bs, 1).to(device)  # (B, M)
        
        images = batch["samples"].to(device).repeat_interleave(args.num_masks, dim=0)  # (B*nm, c, h, w)
        labels = batch["clslabels"].to(device).repeat_interleave(args.num_masks, dim=0)  # (B*nm)
        
        m = rearrange(m, 'b (h w) -> b h w', h=in_sh[1], w=in_sh[2])
        m = m.repeat_interleave(num_samples, dim=0)  # (B*K, h, w)
        
        # Feature extraction
        features, _ = feature_extractor(images)  # (B*K, c, h, w)
        features = (features - avg_glo.view(1, -1, 1, 1)) / std_glo.view(1, -1, 1, 1)
        
        # Initialize x_T and etc.
        x_T = torch.randn((len(images), *in_sh), device=device)  # (B*K, c, h, w)
        x_org = features
        cls_embed = model.cls_embed(labels)  # (B, Z)
        cls_embed = torch.repeat_interleave(cls_embed, num_samples, dim=0)  # (B*K, Z)
        model_kwargs = dict(c=cls_embed, z=None, mask_indices=None, z_vis=None)
        sample_fn = model.net.forward
        inv_m = 1 - m.float() # (B*K, h, w)
        m, inv_m = m.unsqueeze(1), inv_m.unsqueeze(1)  # (B*K, 1, h, w)
        
        indices = list(range(model.sample_diffusion.num_timesteps))[::-1]
        alphas = 1.0 - model.sample_diffusion.betas
        sigmas = np.sqrt(1/alphas - 1)
        
        def ddrm_sample(model, x, x_org, indices, m, inv_m, sigmas, alphas, eta, etaB):
            for i in indices:
                if i % 20 == 0:
                    tqdm.write(f"Step [{i}/{model.sample_diffusion.num_timesteps}]")
                t = torch.tensor([i] * len(x_org)).to(device)
                
                # For first denoising step
                if i == model.sample_diffusion.num_timesteps - 1:
                    noise = torch.randn(x_org.shape, device=device)
                    x_m = sigmas[i] * noise * m   # unconditional for masked pixels
                    x_v = x_org * inv_m + sigmas[i] * noise * inv_m # conditional for unmasked pixels
                    x = x_m + x_v
                    # scale x_t to VP domain
                    x = x * math.sqrt(alphas[i])
                    x.to(device)
                    continue
                
                with torch.no_grad():
                    out = model.sample_diffusion.p_sample(
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
            return x
        
        def update_mask(anom_map, m, quant_thresh):
            thresh = torch.sort(anom_map.view(-1), descending=False)[0][int(quant_thresh * anom_map.numel())]
            anom_map = anom_map > thresh
            anom_map = anom_map.to(device).float()  # (B, 1, h, w)
            anom_map = anom_map.repeat_interleave(num_samples, dim=0)  # (B*K, 1, h, w)
            m = m * anom_map
            inv_m = 1 - m 
            return m, inv_m

        # DDRM Sampling
        for mi in range(args.num_iterations):
            
            tqdm.write(f"Iteration [{mi + 1}] - Mask Coverage: {m.float().mean().item():.4f}")
            x = ddrm_sample(model, x_T, x_org, indices, m, inv_m, sigmas, alphas, args.eta, args.etaB)
            pred_features = inv_m * x_org + m * x  # (B*K, c, h, w)
            anom_map = torch.sum((pred_features - x_org)**2, dim=1)  # (B*K, h, w)
            
            anom_map = rearrange(anom_map, '(b nm ns) h w -> b nm ns h w', b=bs, nm=args.num_masks, ns=args.num_samples)
            anom_map = torch.min(anom_map, dim=2).values  # (B, nm, h, w)
            m, inv_m = update_mask(anom_map, m, args.quant_thresh)
        
        # x = ddrm_sample(model, x_T, x_org, indices, m, inv_m, sigmas, alphas, args.eta, args.etaB)
        # pred_latents_map = inv_m * x_org + m * x
        
        if args.recon_space == 'feature':
            anom_score = torch.mean(anom_map, dim=(2,3))  # (B, nm)
            anom_score = torch.mean(anom_score, dim=1)  # (B)
            normal_scores += anom_score.tolist()
        else:
            raise ValueError(f"Invalid reconstruction space: {args.recon_space}")
        
    normal_scores = torch.tensor(normal_scores)
    
    # Evaluation on anomalous samples
    print("Evaluating on anomalous samples")
    anom_loader = DataLoader(anom_dataset, args.batch_size, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    anom_scores = []            
    anom_types = []
    for step_idx, batch in tqdm(enumerate(anom_loader), total=len(anom_loader)):
        bs = len(batch["samples"])
        m = masks.repeat(bs, 1).to(device)  # (B, M)
        
        images = batch["samples"].to(device).repeat_interleave(args.num_masks, dim=0)
        labels = batch["clslabels"].to(device).repeat_interleave(args.num_masks, dim=0)
        anom_types += batch["anom_type"]
        
        m = rearrange(m, 'b (h w) -> b h w', h=in_sh[1], w=in_sh[2])
        m = m.repeat_interleave(num_samples, dim=0)  # (B*K, h, w)
        
        # Feature extraction
        features, _ = feature_extractor(images)  # (B*K, c, h, w)
        features = (features - avg_glo.view(1, -1, 1, 1)) / std_glo.view(1, -1, 1, 1)
        
        # Initialize x_T and etc.
        x_T = torch.randn((len(images), *in_sh), device=device)  # (B*K, c, h, w)

        cls_embed = model.cls_embed(labels)  # (B, Z)
        cls_embed = torch.repeat_interleave(cls_embed, num_samples, dim=0)  # (B*K, Z)
        model_kwargs = dict(c=cls_embed, z=None, mask_indices=None, z_vis=None)
        sample_fn = model.net.forward
        x_org = features
        inv_m = 1 - m.float() # (B*K, h, w)
        m, inv_m = m.unsqueeze(1), inv_m.unsqueeze(1)  # (B*K, 1, h, w)
        
        indices = list(range(model.sample_diffusion.num_timesteps))[::-1]
        
        # DDRM
        for mi in range(args.num_iterations):
            tqdm.write(f"Iteration [{mi + 1}] - Mask Coverage: {m.float().mean().item():.4f}")
            x = ddrm_sample(model, x_T, x_org, indices, m, inv_m, sigmas, alphas, args.eta, args.etaB)
            pred_features = inv_m * x_org + m * x
            anom_map = torch.sum((pred_features - x_org)**2, dim=1)  # (B*K, h, w)
            anom_map = rearrange(anom_map, '(b nm ns) h w -> b nm ns h w', b=bs, nm=args.num_masks, ns=args.num_samples)
            anom_map = torch.min(anom_map, dim=2).values  # (B, nm, h, w)
            m, inv_m = update_mask(anom_map, m, args.quant_thresh)
        # x = ddrm_sample(model, x_T, x_org, indices, m, inv_m, sigmas, alphas, args.eta, args.etaB)
        
        if args.recon_space == 'feature':
            anom_score = torch.mean(anom_map, dim=(2,3))  # (B, nm)
            anom_score = torch.mean(anom_score, dim=1)  # (B)
            anom_scores += anom_score.tolist()
        else:
            raise ValueError(f"Invalid reconstruction space: {args.recon_space}")
            
    anom_scores = torch.tensor(anom_scores)

    print("Calculating AUC...🧑‍⚕️")
    print(f"===========================================")
    y_true = torch.cat([torch.zeros_like(normal_scores), torch.ones_like(anom_scores)])
    y_score = torch.cat([normal_scores, anom_scores])
    auc_all = roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy())
    print(f"Image-level AUC 👓: {auc_all:.4f} [{category}]")
    print(f"===========================================")
    
    print(f"Calculating AUC for each anomaly type...🧑‍⚕️")
    print(f"===========================================")
    unique_anom_types = list(sorted(set(anom_types)))
    auc_dict = {}
    for anom_type in unique_anom_types:
        y_true = torch.cat([torch.zeros_like(normal_scores), torch.ones_like(anom_scores[np.array(anom_types) == anom_type])])
        y_score = torch.cat([normal_scores, anom_scores[np.array(anom_types) == anom_type]])
        auc = roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy())
        auc_dict[anom_type] = auc
        print(f"AUC [{anom_type}]: {auc:.4f}")
    print(f"===========================================")
    
    print("Saving results...📁")
    results = {
        "normal_scores": normal_scores.cpu().numpy().tolist(),
        "anom_scores": anom_scores.cpu().numpy().tolist(),
        "auc": auc_all, 
        "num_masks": args.num_masks,
        "recon_space": args.recon_space,
        "mask_coverage": mask_coverage_meter.avg,
        "mask_strategy": mask_strategy,
        "mask_ratio": mask_ratio if mask_strategy in ['random', 'block', 'constant'] else None,
    }
    results.update(auc_dict)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved at {output_dir / 'eval_results.json'}")

if __name__ == '__main__':
    args = parser_args()
    main(args)
        
        
    
    
    
    