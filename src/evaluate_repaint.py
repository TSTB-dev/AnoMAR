import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from PIL import Image
from tqdm import tqdm

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

from utils import AverageMeter, calculate_mask_coverage
from datasets import build_dataset, EvalDataLoader
from models import create_vae, AutoencoderKL, create_mar_model, EncoderDecoerMAR
from backbones import get_backbone
from denoiser import get_denoiser, Denoiser
from mask import RandomMaskCollator, BlockRandomMaskCollator, CheckerBoardMaskCollator, ConstantMaskCollator, SlidingWindowMaskCollator, \
    indices_to_mask, mask_to_indices

def parser_args():
    parser = argparse.ArgumentParser(description='AnoMAR Sampling')
    
    parser.add_argument('--num_samples', type=int, help='Number of samples to generate', default=1)
    parser.add_argument('--num_inference_steps', type=int, help='Number of inference steps')
    parser.add_argument('--start_step', type=int, default=10, help='timestep to start denoising')
    parser.add_argument('--sampler', type=str, help='Sampler to use', default='org')  # ddim or org
    parser.add_argument('--temperature', type=float, help='Temperature for sampling', default=1.0)
    parser.add_argument('--eta', type=float, help='Eta for sampling', default=1.0)
    parser.add_argument('--num_masks', type=int, help='Number of masks to generate', default=1)
    parser.add_argument('--recon_space', type=str, default='latent', help='Reconstruction space')  # ['latent', 'pixel', 'feature']
    parser.add_argument('--aggregation', type=str, default='mean', help='Aggregation method')  # ['mean', 'max']
    parser.add_argument('--num_resamples', type=int, default=1, help='Number of resamples')
    parser.add_argument('--num_jumps', type=int, default=1, help='Number of jumps')
    parser.add_argument('--save_images', action='store_true', help='Save images')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--model_ckpt', type=str, help='Path to the model checkpoint')
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--sample_indices', type=int, nargs='+', help='Sample indices to visualize', default=[])
    
    return parser.parse_args()

def postprocess(x):
    x = x / 2 + 0.5
    return x.clamp(0, 1)

def convert2image(x):
    if x.dim() == 3:
        return x.permute(1, 2, 0).cpu().numpy()
    elif x.dim() == 4:
        return x.permute(0, 2, 3, 1).cpu().numpy()
    else:
        return x.cpu().numpy()
    
def gt_mask_to_mask(gt_mask, in_sh, expand_size):
    # gt_mask: (B, 1, H, W)
    mask = F.interpolate(gt_mask, size=(in_sh[1], in_sh[2]), mode='nearest')
    mask = expand_bool_mask(mask, expand_size)
    return mask 

def expand_bool_mask(m, p: int):
    # m: (B, h, w) 
    if p <= 0:
        return m
    m_float = m.float()  # (B, 1, h, w)
    dilated = F.max_pool2d(m_float, kernel_size=2*p+1, stride=1, padding=p)
    return dilated.squeeze(1) > 0
    
    
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
    dataset_config['transform_type'] = 'default'
    dataset_config['train'] = False
    dataset_config['anom_only'] = True
    anom_dataset = build_dataset(**dataset_config)
    dataset_config['anom_only'] = False
    dataset_config['normal_only'] = True
    normal_dataset = build_dataset(**dataset_config)
    
    # build backbone model
    if not 'backbone' in config:
        config['backbone'] = config['vae']
    if 'vae' in config['backbone']['model_type']:
        backbone: AutoencoderKL = create_vae(**config['backbone'])
    else:
        backbone = get_backbone(**config['backbone'])
    backbone.to(device).eval()
    
    backbone_embed_dim = config['backbone']['embed_dim']
    backbone_stride = config['backbone']['stride']
    img_size = config['data']['img_size']
    patch_size = 16
    in_sh = (backbone_embed_dim, img_size // backbone_stride, img_size // backbone_stride)
    in_shb = (args.num_samples * args.num_masks, *in_sh)
    
    # build mim model
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=in_sh)
    model_ckpt = torch.load(args.model_ckpt, map_location='cpu', weights_only=True)
    results = model.load_state_dict(model_ckpt, strict=True)
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
        print(f"Using feature space reconstruction with {model_kwargs['model_type']} backbone")
        feature_extractor = get_backbone(**model_kwargs)
        feature_extractor.to(device).eval()
        
    # build mask collator
    mask_strategy = "block"
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
            input_size=in_sh[1], patch_size=patch_size, **config['data']['mask']
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
    else:
        raise ValueError(f"Invalid mask strategy: {mask_strategy}")
    
    normal_loader = EvalDataLoader(normal_dataset, args.num_masks, collate_fn=mask_collator)

    # For visualization
    if args.save_images:    
        sample_indices = random.sample(range(min(len(anom_dataset), len(normal_dataset))), 2) if not args.sample_indices else args.sample_indices
        sample_dict = {}
    else:
        sample_indices = []

    def encode_images(x):
        if 'vae' in config['backbone']['model_type']:
            post = backbone.encode(x)
            return post.sample().mul_(0.2325)
        else:
            return backbone(x)
        
    def decode_images(x):
        if 'vae' in config['backbone']['model_type']:
            return backbone.decode(x / 0.2325)
        else:
            raise ValueError(f"Backbone {config['backbone']['model_type']} does not support decoding")

    def reshape_mask(m):  
        """ (B, hw) -> (B, hp, wp) """
        m = rearrange(m, 'b (h w) -> b h w', h=in_sh[1], w=in_sh[2])
        m = torch.repeat_interleave(m, repeats=patch_size, dim=1)
        m = torch.repeat_interleave(m, repeats=patch_size, dim=2)
        m = m.float()
        return m

    def generate_indices_with_jump(total_steps, r, jump):
        jumps = {}
        for j in range(0, total_steps - jump, jump):
            jumps[j] = r - 1
        
        t = total_steps - 1
        indices = []
        while t >= 1:
            t -= 1
            indices.append(t)
            
            if jumps.get(t, 0) > 0:
                jumps[t] -= 1
                for _ in range(r):
                    t += 1
                    indices.append(t)
        indices.append(0)
        return indices
        
    def anomaly_score(x, y):
        anom_map = torch.mean((x - y)**2, dim=1)  # (B*K, h, w)
        anom_map = anom_map.view(-1, num_samples, in_sh[1], in_sh[2])  # (B, K, h, w)
        anom_map = torch.min(anom_map, dim=1).values  # (B, h, w)
        if args.aggregation == 'mean':
            anom_score = torch.mean(anom_map)
        elif args.aggregation == 'max':
            anom_score = torch.max(torch.mean(anom_map, dim=1))
        return anom_score
    
    def anomaly_map(x, y):
        anom_map = torch.mean((x - y)**2, dim=1)  # (B*K, h, w)
        anom_map = anom_map.view(-1, num_samples, in_sh[1], in_sh[2])  # (B, K, h, w)
        anom_map = torch.min(anom_map, dim=1).values # (B, h, w)
        return anom_map

    num_patches = in_sh[1] * in_sh[2]
    
    # Evaluation on normal samples
    print("Evaluating on normal samples")
    normal_scores = []
    mask_coverage_meter = AverageMeter()
    loss_meter = AverageMeter()
    for i, (batch, mask_indices) in tqdm(enumerate(normal_loader), total=len(normal_loader)):
        # TODO: replace with random anomaly masks
        # mask_indices = torch.stack([anom_masks[b] for b in range(len(mask_indices))]) # (B, M)
        
        mask = indices_to_mask(mask_indices, num_patches)
        mask = mask.to(device)
        mask_indices = mask_indices.to(device)
        
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        gt_masks = batch["masks"]  # (B, 1, H, W) 
        
        mask_cov = calculate_mask_coverage(mask_indices, in_sh[1], in_sh[2])
        mask_coverage_meter.update(mask_cov.item(), images.size(0))
        
        latents = encode_images(images)
        m = rearrange(mask, 'b (h w) -> b h w', h=in_sh[1], w=in_sh[2])
        m = m.repeat_interleave(num_samples, dim=0)  # (B*K, h, w)
        
        # Initialize x_T and etc.
        x_T = torch.randn(in_shb, device=device)  # (B*K, c, h, w)
        indices = generate_indices_with_jump(args.num_inference_steps, args.num_resamples, args.num_jumps)
        cls_embed = model.cls_embed(labels)  # (B, Z)
        cls_embed = torch.repeat_interleave(cls_embed, num_samples, dim=0)  # (B*K, Z)
        model_kwargs = dict(c=cls_embed, z=None, mask_indices=None, z_vis=None)
        sample_fn = model.net.forward
        x_org = latents.repeat_interleave(num_samples, dim=0)  # (B*K, c, h, w)
        inv_m = 1 - m.float() # (B*K, h, w)
        m, inv_m = m.unsqueeze(1), inv_m.unsqueeze(1)  # (B*K, 1, h, w)
        
        # Repaint loop
        prev_j = 0
        x = x_T 
        for j in indices:
            t = torch.tensor([j] * len(x_org)).to(device)
            
            # If prev_i < i, then we diffuse x_{t}
            if prev_j < j:
                x = model.sample_diffusion.q_sample(x, t[0], noise=None)
                prev_j = j
                continue
        
            # Predict x_{t-1}^{unknown} from x_t
            with torch.no_grad():
                pred_x = model.sample_diffusion.p_sample(
                    sample_fn,
                    x,
                    t,
                    clip_denoised=False,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=model_kwargs,
                    temperature=1.0,
                )["sample"]
            
            # sample x_{t-1}^{known} from x_vis
            vis_x = model.sample_diffusion.q_sample(
                x_org, t[0], noise=None
            )
            
            # composite x_{t-1}^{unknown} and x_{t-1}^{known}
            x = inv_m * vis_x + m * pred_x
            prev_j = j
        
        pred_latents_map = inv_m * x_org + m * x  # (B*K, c, h, w)
        
        if args.recon_space == 'latent':
            anom_score = anomaly_score(pred_latents_map, x_org)
            normal_scores.append(anom_score)
        elif args.recon_space == 'pixel':
            decoded_images = decode_images(pred_latents_map)
            decoded_images_org = decode_images(x_org)
            anom_score = anomaly_score(decoded_images, decoded_images_org)
            normal_scores.append(anom_score)
        elif args.recon_space == 'feature':
            decoded_images = decode_images(pred_latents_map)  # (B*K, 3, H, W)
            decoded_images_org = decode_images(x_org)  # (B*K, 3, H, W)
            features = feature_extractor(decoded_images)  # (B*K, c, h, w)
            features_org = feature_extractor(decoded_images_org)  # (B*K, c, h, w)
            anom_score = anomaly_score(features, features_org)
            normal_scores.append(anom_score)
        else:
            raise ValueError(f"Invalid reconstruction space: {args.recon_space}")
        
        if i in sample_indices:
            b = 0
            pred_imgs = decode_images(pred_latents_map)  # (B*K, 3, H, W)
            mask = reshape_mask(mask).unsqueeze(1)  # (B, H, W)
            org_imgs = convert2image(postprocess(images))  # (B, H, W, C)
            mask = F.interpolate(mask, size=(img_size, img_size), mode='nearest')  # (B, 1, H, W)
            masked_imgs = convert2image(postprocess(images) * (1 - mask))  # (B, H, W, C)
            pred_imgs = convert2image(postprocess(pred_imgs)) # (B*K, H, W, C)
            pred_imgs = pred_imgs.reshape(-1, num_samples, *pred_imgs.shape[1:])[b]  # (K, H, W, C)  
            anom_map = convert2image(anomaly_map(pred_latents_map, x_org))
            
            sample_dict[f'normal_{i}'] = {
                'images': org_imgs[b], # (H, W, C)
                'masked_images': masked_imgs[b],
                'pred_images': pred_imgs,
                'gt_masks': convert2image(gt_masks)[0],  # (H, W)
                'labels': 'normal',
                'anomaly_maps': anom_map[..., b], # (H, W)
                'mask_coverage': mask_cov.item()
            }
        
    normal_scores = torch.tensor(normal_scores)
    print(f"Mask Coverage: {mask_coverage_meter.avg:.4f}")
    print(f"Reconstruction Loss: {loss_meter.avg:.4f}")
    
    # Evaluation on anomalous samples
    print("Evaluating on anomalous samples")
    anom_loader = EvalDataLoader(anom_dataset, args.num_masks, collate_fn=mask_collator, shared_masks=normal_loader.shared_masks)
    anom_scores = []            
    anom_types = []
    anom_masks = []
    mask_coverage_meter = AverageMeter()
    loss_meter = AverageMeter()
    for i, (batch, mask_indices) in tqdm(enumerate(anom_loader), total=len(anom_loader)):
        mask = indices_to_mask(mask_indices, num_patches)
        mask = mask.to(device)
        mask_indices = mask_indices.to(device)
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        anom_types.append(batch["anom_type"][0])
        gt_masks = batch["masks"]
        
        # TODO: 
        # gt_masks_ = gt_masks.clone()
        # gt_masks_ = gt_mask_to_mask(gt_masks_, in_sh, expand_size=1) # (B, h, w)
        # mask = gt_masks_.to(device)
        # mask = rearrange(mask, 'b h w -> b (h w)')
        # mask_indices = mask_to_indices(mask)
        # anom_masks.append(mask_indices.squeeze(0))
        
        mask_cov = calculate_mask_coverage(mask_indices, in_sh[1], in_sh[2])
        mask_coverage_meter.update(mask_cov.item(), images.size(0))
        
        latents = encode_images(images)
        m = rearrange(mask, 'b (h w) -> b h w', h=in_sh[1], w=in_sh[2])
        m = m.repeat_interleave(num_samples, dim=0)  # (B*K, h, w)
        
        # Initialize x_T and etc.
        x_T = torch.randn(in_shb, device=device)  # (B*K, c, h, w)
        indices = generate_indices_with_jump(args.num_inference_steps, args.num_resamples, args.num_jumps)
        cls_embed = model.cls_embed(labels)  # (B, Z)
        cls_embed = torch.repeat_interleave(cls_embed, num_samples, dim=0)  # (B*K, Z)
        model_kwargs = dict(c=cls_embed, z=None, mask_indices=None, z_vis=None)
        sample_fn = model.net.forward
        x_org = latents.repeat_interleave(num_samples, dim=0)  # (B*K, c, h, w)
        inv_m = 1 - m.float() # (B*K, h, w)
        m, inv_m = m.unsqueeze(1), inv_m.unsqueeze(1)  # (B*K, 1, h, w)
        
        # Repaint loop
        prev_j = 0
        x = x_T
        for j in indices:
            t = torch.tensor([j] * len(x_org)).to(device)
            
            # If prev_i < i, then we diffuse x_{t}
            if prev_j < j:
                x = model.sample_diffusion.q_sample(x, t[0], noise=None)
                prev_j = j
                continue
        
            # Predict x_{t-1}^{unknown} from x_t
            with torch.no_grad():
                pred_x = model.sample_diffusion.p_sample(
                    sample_fn,
                    x,
                    t,
                    clip_denoised=False,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=model_kwargs,
                    temperature=1.0,
                )["sample"]
            
            # sample x_{t-1}^{known} from x_vis
            vis_x = model.sample_diffusion.q_sample(
                x_org, t[0], noise=None
            )
            
            # composite x_{t-1}^{unknown} and x_{t-1}^{known}
            x = inv_m * vis_x + m * pred_x
            
            prev_j = j
            
        pred_latents_map = inv_m * x_org + m * x  # (B*K, c, h, w)
        
        if args.recon_space == 'latent':
            anom_score = anomaly_score(pred_latents_map, x_org)
            anom_scores.append(anom_score)
        elif args.recon_space == 'pixel':
            decoded_images = decode_images(pred_latents_map)
            decoded_images_org = decode_images(x_org)
            anom_score = anomaly_score(decoded_images, decoded_images_org)
            anom_scores.append(anom_score)
        elif args.recon_space == 'feature':
            decoded_images = decode_images(pred_latents_map)
            decoded_images_org = decode_images(x_org)
            features = feature_extractor(decoded_images)
            features_org = feature_extractor(decoded_images_org)
            anom_score = anomaly_score(features, features_org)
            anom_scores.append(anom_score)
        else:
            raise ValueError(f"Invalid reconstruction space: {args.recon_space}")
        
        if i in sample_indices:
            b = 0
            pred_imgs = decode_images(pred_latents_map)  # (B*K, 3, H, W)
            mask = reshape_mask(mask).unsqueeze(1)  # (B, H, W)
            org_imgs = convert2image(postprocess(images))  # (B, H, W, C)
            mask = F.interpolate(mask, size=(img_size, img_size), mode='nearest')  # (B, 1, H, W)
            masked_imgs = convert2image(postprocess(images) * (1 - mask))  # (B, H, W, C)
            pred_imgs = convert2image(postprocess(pred_imgs)) # (B*K, H, W, C)
            pred_imgs = pred_imgs.reshape(-1, num_samples, *pred_imgs.shape[1:])[b]
            anom_map = convert2image(anomaly_map(pred_latents_map, x_org))
            sample_dict[f'anom_{i}'] = {
                'images': org_imgs[b], # (H, W, C)
                'masked_images': masked_imgs[b],  
                'pred_images': pred_imgs,
                'gt_masks': convert2image(gt_masks)[b],  # (H, W)
                'labels': 'anomalous',
                'anomaly_maps': anom_map[..., b], # (H, W)
                'mask_coverage': mask_cov.item()
            }
    
    anom_scores = torch.tensor(anom_scores)
    print(f"Mask Coverage: {mask_coverage_meter.avg:.4f}")
    print(f"Reconstruction Loss: {loss_meter.avg:.4f}")
    
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
    
    if args.save_images:
        save_images(sample_dict, anom_scores, args.output_dir, num_samples=args.num_samples)
        print(f"Images saved at {output_dir / 'results.png'}")
            
def save_images(sample_dict, anom_scores, output_dir, num_samples):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    min_score = min(anom_scores)
    max_score = max(anom_scores)
    
    fig, ax = plt.subplots(4, 4 + min(num_samples, 3), figsize=(25, 10))
    font_color = 'blue'
    for i, (key, value) in enumerate(sample_dict.items()):
        if i == 2:
            font_color = 'orange'
        ax[i, 0].imshow(value['images'])
        ax[i, 0].set_title(f"Original {value['labels']}", color=font_color)
        ax[i, 0].axis('off')
        
        ax[i, 1].imshow(value['masked_images'])
        ax[i, 1].set_title(f"Masked {value['labels']} image", color=font_color)
        ax[i, 1].axis('off')
        
        ax[i, 2].imshow(value['pred_images'][0])
        ax[i, 2].set_title(f"Pred images 1", color=font_color)
        ax[i, 2].axis('off')
        
        for j in range(1, min(num_samples, 3)):
            ax[i, 2 + j].imshow(value['pred_images'][j])
            ax[i, 2 + j].set_title(f"Pred images {j + 1}", color=font_color)
            ax[i, 2 + j].axis('off')
        
        # ax[i, 2 + min(num_samples, 3)].imshow(value['anomaly_maps'], vmin=min_score, vmax=max_score)
        ax[i, 2 + min(num_samples, 3)].imshow(value['anomaly_maps'])
        ax[i, 2 + min(num_samples, 3)].set_title(f"Anomaly Map {value['labels']}", color=font_color)
        ax[i, 2 + min(num_samples, 3)].axis('off')
        
        ax[i, 3 + min(num_samples, 3)].imshow(value['gt_masks'], cmap='gray', vmin=min_score, vmax=max_score)
        ax[i, 3 + min(num_samples, 3)].set_title(f"GT Mask {value['labels']}", color=font_color)
        ax[i, 3 + min(num_samples, 3)].axis('off')
            
    plt.tight_layout()
    plt.savefig(output_dir / 'results.png')
    plt.close()

if __name__ == '__main__':
    args = parser_args()
    main(args)
        
        
    
    
    
    