
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
    indices_to_mask, mask_to_indices

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

def feature_distance(x, y, fe, mode='cosine'):
    transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / (2)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    x = transform(x)
    y = transform(y)
    x_feats = fe(x)
    y_feats = fe(y)
    b, c, h, w = x_feats[0].shape
    dist = torch.zeros(b, h, w, device=x.device)
    for i in range(len(x_feats)):
        if mode == 'l2':
            dist_i = (x_feats[i] - y_feats[i]).pow(2).mean(dim=1)  # (B, H, W)
        elif mode == 'l1':
            dist_i = (x_feats[i] - y_feats[i]).abs().mean(dim=1)
        elif mode == 'cosine':
            c_i, h_i, w_i= x_feats[i].shape[-3:]
            dist_i = F.cosine_similarity(x_feats[i].view(b, c_i, -1), y_feats[i].view(b, c_i, -1), dim=1)
            dist_i = 1 - dist_i.view(b, h_i, w_i)
        dist_i = F.interpolate(dist_i.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
        dist += dist_i
    return dist
    
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
    # in_sh = (backbone_embed_dim, img_size // backbone_stride, img_size // backbone_stride)
    in_sh = (3, img_size, img_size)
    in_shb = (args.num_samples * args.num_masks, *in_sh)
    
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
        # print(f"Using feature space reconstruction with {model_kwargs['model_type']} backbone")
        # model_kwargs = {"model_type": "pdn_medium"}
        # model_kwargs = {
        #     'model_type': 'wide_resnet',
        #     'ckpt_path': f'./{category}_fe'
        # }
        feature_extractor = get_backbone(**model_kwargs)
        feature_extractor.to(device).eval()
        
    # build mask collator
    mask_strategy = "checkerboard"
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
    else:
        raise ValueError(f"Invalid mask strategy: {mask_strategy}")
    
    normal_loader = DataLoader(normal_dataset, args.batch_size, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    # masks = mask_collator.generate_random_mask(args.num_masks)  # (N, M)
    masks = mask_collator.collate_all_masks()  # (N, M)
    args.num_masks = len(masks)
    
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
        # anom_map = anom_map.view(-1, num_samples, 16, 16)  # (B, K, h, w)
        # anom_map = torch.min(anom_map, dim=1).values # (B, h, w)
        return anom_map

    num_patches = in_sh[1] * in_sh[2]
    
    # Evaluation on normal samples
    print("Evaluating on normal samples")
    normal_scores = []
    mask_coverage_meter = AverageMeter()
    loss_meter = AverageMeter()
    sample_dict = defaultdict(dict)
    img_saved = False
    for step_idx, batch in tqdm(enumerate(normal_loader), total=len(normal_loader)):
        bs = len(batch["samples"])
        # TODO: replace with random anomaly masks
        # mask_indices = torch.stack([anom_masks[b] for b in range(len(mask_indices))]) # (B, M)
        m = masks.repeat(bs, 1).to(device)  # (B, M)
        # midx = mask_to_indices(m).to(device)  # (B, M)
        
        images = batch["samples"].to(device).repeat_interleave(args.num_masks, dim=0)  # (B*nm, c, h, w)
        labels = batch["clslabels"].to(device).repeat_interleave(args.num_masks, dim=0)  # (B*nm)
        gt_masks = batch["masks"]  # (B, 1, H, W)
        
        # mask_cov = calculate_mask_coverage(midx, in_sh[1], in_sh[2])
        # mask_coverage_meter.update(mask_cov.item(), len(images))
        
        # latents = encode_images(images)
        m = rearrange(m, 'b (h w) -> b h w', h=in_sh[1], w=in_sh[2])
        m = m.repeat_interleave(num_samples, dim=0)  # (B*K, h, w)
        
        # Initialize x_T and etc.
        x_T = torch.randn((len(images), *in_sh), device=device)  # (B*K, c, h, w)
        cls_embed = model.cls_embed(labels)  # (B, Z)
        cls_embed = torch.repeat_interleave(cls_embed, num_samples, dim=0)  # (B*K, Z)
        # model_kwargs = dict(c=cls_embed, z=None, mask_indices=None, z_vis=None)
        model_kwargs = dict(c=None, z=None, mask_indices=None, z_vis=None)
        sample_fn = model.net.forward
        # x_org = latents.repeat_interleave(num_samples, dim=0)  # (B*K, c, h, w)
        x_org = images.repeat_interleave(num_samples, dim=0)  # (B*K, c, h, w)
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
            anom_map = anom_map.to(device).unsqueeze(1).float()  # (B, 1, h, w)
            anom_map = anom_map.repeat_interleave(num_samples, dim=0)  # (B*K, 1, h, w)
            m = m * anom_map
            inv_m = 1 - m 
            return m, inv_m

        # DDRM Sampling
        for mi in range(args.num_iterations):
            
            # tqdm.write(f"Iteration [{mi + 1}] - Mask Coverage: {m.float().mean().item():.4f}")
            x = ddrm_sample(model, x_T, x_org, indices, m, inv_m, sigmas, alphas, args.eta, args.etaB)
            pred_latents_map = inv_m * x_org + m * x  # (B*K, c, h, w)
            # decoded_images = decode_images(pred_latents_map)  # (B*K, 3, H, W)
            # decoded_images_org = decode_images(x_org)  # (B*K, 3, H, W)
            decoded_images = pred_latents_map
            decoded_images_org = x_org
            features = feature_extractor(decoded_images)  # (B*K, c, h, w)
            features_org = feature_extractor(decoded_images_org)  # (B*K, c, h, w)
            anom_map = anomaly_map(features, features_org)
            # anom_map = feature_distance(decoded_images, decoded_images_org, feature_extractor, mode='cosine')
            
            anom_map = rearrange(anom_map, '(b nm ns) h w -> b nm ns h w', b=bs, nm=args.num_masks, ns=args.num_samples)
            anom_map = torch.min(anom_map, dim=2).values  # (B, nm, h, w)
            # m, inv_m = update_mask(anom_map, m, args.quant_thresh)
        
        # x = ddrm_sample(model, x_T, x_org, indices, m, inv_m, sigmas, alphas, args.eta, args.etaB)
        # pred_latents_map = inv_m * x_org + m * x
        
        if args.recon_space == 'latent':
            anom_score = anomaly_score(pred_latents_map, x_org)
            anom_map = anomaly_map(pred_latents_map, x_org)
            normal_scores.append(anom_score)
        elif args.recon_space == 'pixel':
            decoded_images = decode_images(pred_latents_map)
            decoded_images_org = decode_images(x_org)
            anom_score = anomaly_score(decoded_images, decoded_images_org)
            anom_map = anomaly_map(decoded_images, decoded_images_org)
            normal_scores.append(anom_score)
        elif args.recon_space == 'feature':
            # decoded_images = decode_images(pred_latents_map)  # (B*K, 3, H, W)
            # decoded_images_org = decode_images(x_org)  # (B*K, 3, H, W)
            # features = feature_extractor(decoded_images)  # (B*K, c, h, w)
            # features_org = feature_extractor(decoded_images_org)  # (B*K, c, h, w)
            # anom_map = anomaly_map(features, features_org)
            # anom_score = anomaly_score(features, features_org)
            anom_score = torch.mean(anom_map, dim=(2,3))  # (B, nm)
            anom_score = torch.mean(anom_score, dim=1)  # (B)
            normal_scores += anom_score.tolist()
        else:
            raise ValueError(f"Invalid reconstruction space: {args.recon_space}")
        
        if args.save_images and not img_saved:
            img_saved = True
            idxs = (0,1)
            # B, nm, ns, 
            c, h, w = in_sh[0], in_sh[1], in_sh[2]
            preds = pred_latents_map.view(bs, args.num_masks, args.num_samples, *pred_latents_map.shape[1:])
            reshaped_masks = rearrange(masks, 'b (h w) -> b h w', h=h, w=w).unsqueeze(1)  # (nm, 1, H, W)
            reshaped_masks = reshaped_masks.unsqueeze(0).repeat(bs, 1, 1, 1, 1)  # (B, nm, 1, H, W)
            reshaped_masks = reshaped_masks.unsqueeze(2).repeat(1, 1, args.num_samples, 1, 1, 1)  # (B, nm, ns, 1, H, W)
            imgs = images.view(bs, args.num_masks, args.num_samples, *images.shape[1:])  # (B, nm, ns, C, H, W)
            
            for b in idxs:
                pred_imgs_b = preds[b, 0].view(-1, c, h, w)  # (ns, C, H, W)
                masks_b = reshaped_masks[b, 0].view(-1, 1, h, w)  # (ns, 1, H, W)
                org_imgs_b = imgs[b, 0].view(-1, c, h, w)  # (ns, C, H, W)
                org_imgs = convert2image(postprocess(org_imgs_b))[0]  # (H, W, C)
                pred_imgs = convert2image(postprocess(pred_imgs_b))  # (ns, H, W, C)
                masked_imgs = convert2image(postprocess(org_imgs_b.cpu()[0]) * (1 - masks_b.cpu()))[0]  # (ns, H, W, C)
                maps = convert2image(anom_map[b, 0])  # (H, W)
                
                sample_dict[f'normal_{b}'] = {
                    'images': org_imgs,
                    'masked_images': masked_imgs,
                    'pred_images': pred_imgs,
                    'gt_masks': convert2image(gt_masks)[b],  # (H, W)
                    'labels': 'normal',
                    'anomaly_maps': maps,
                }
        
        # pred_imgs = decode_images(pred_latents_map)  # (B*K, 3, H, W)
        # pred_imgs = pred_latents_map
        # masks = reshape_mask(masks).unsqueeze(1)  # (B, 1, H, W)
        # org_imgs = convert2image(postprocess(images))  # (B, H, W, C)
        # mask = F.interpolate(mask, size=(img_size, img_size), mode='nearest')  # (B, 1, H, W)
        # masked_imgs = convert2image(postprocess(images) * (1 - mask))  # (B, H, W, C)
        # pred_imgs = convert2image(postprocess(pred_imgs)) # (B*K, H, W, C)
        # pred_imgs = pred_imgs.reshape(-1, num_samples, *pred_imgs.shape[1:])[b]  # (K, H, W, C)  
        # anom_map = convert2image(anom_map)
        
        # sample_dict[f'normal_{1}'] = {
        #     'images': org_imgs[b], # (H, W, C)
        #     'masked_images': masked_imgs[b],
        #     'pred_images': pred_imgs,
        #     'gt_masks': convert2image(gt_masks)[0],  # (H, W)
        #     'labels': 'normal',
        #     'anomaly_maps': anom_map[..., b], # (H, W)
        # }
        
    normal_scores = torch.tensor(normal_scores)
    # print(f"Mask Coverage: {mask_coverage_meter.avg:.4f}")
    # print(f"Reconstruction Loss: {loss_meter.avg:.4f}")
    
    # Evaluation on anomalous samples
    print("Evaluating on anomalous samples")
    anom_loader = DataLoader(anom_dataset, args.batch_size, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    anom_scores = []            
    anom_types = []
    anom_masks = []
    # mask_coverage_meter = AverageMeter()
    loss_meter = AverageMeter()
    img_saved = False
    for step_idx, batch in tqdm(enumerate(anom_loader), total=len(anom_loader)):
        # mask = indices_to_mask(mask_indices, num_patches)
        # mask = mask.to(device)
        # mask_indices = mask_indices.to(device)
        bs = len(batch["samples"])
        m = masks.repeat(bs, 1).to(device)  # (B, M)
        # midx = mask_to_indices(m).to(device)  # (B, M)
        
        images = batch["samples"].to(device).repeat_interleave(args.num_masks, dim=0)
        labels = batch["clslabels"].to(device).repeat_interleave(args.num_masks, dim=0)
        anom_types += batch["anom_type"]
        gt_masks = batch["masks"]
        
        # TODO: 
        # gt_masks_ = gt_masks.clone()
        # gt_masks_ = gt_mask_to_mask(gt_masks_, in_sh, expand_size=1) # (B, h, w)
        # mask = gt_masks_.to(device)
        # mask = rearrange(mask, 'b h w -> b (h w)')
        # mask_indices = mask_to_indices(mask)
        # anom_masks.append(mask_indices.squeeze(0))
        
        # mask_cov = calculate_mask_coverage(midx, in_sh[1], in_sh[2])
        # mask_coverage_meter.update(mask_cov.item(), images.size(0))
        
        # latents = encode_images(images)
        m = rearrange(m, 'b (h w) -> b h w', h=in_sh[1], w=in_sh[2])
        m = m.repeat_interleave(num_samples, dim=0)  # (B*K, h, w)
        
        # Initialize x_T and etc.
        x_T = torch.randn((len(images), *in_sh), device=device)  # (B*K, c, h, w)

        cls_embed = model.cls_embed(labels)  # (B, Z)
        cls_embed = torch.repeat_interleave(cls_embed, num_samples, dim=0)  # (B*K, Z)
        # model_kwargs = dict(c=cls_embed, z=None, mask_indices=None, z_vis=None)
        model_kwargs = dict(c=None, z=None, mask_indices=None, z_vis=None)
        sample_fn = model.net.forward
        # x_org = latents.repeat_interleave(num_samples, dim=0)  # (B*K, c, h, w)
        x_org = images.repeat_interleave(num_samples, dim=0)  # (B*K, c, h, w)
        inv_m = 1 - m.float() # (B*K, h, w)
        m, inv_m = m.unsqueeze(1), inv_m.unsqueeze(1)  # (B*K, 1, h, w)
        
        indices = list(range(model.sample_diffusion.num_timesteps))[::-1]
        
        # DDRM
        for mi in range(args.num_iterations):
            # tqdm.write(f"Iteration [{mi + 1}] - Mask Coverage: {m.float().mean().item():.4f}")
            x = ddrm_sample(model, x_T, x_org, indices, m, inv_m, sigmas, alphas, args.eta, args.etaB)
            pred_latents_map = inv_m * x_org + m * x
            # decoded_images = decode_images(pred_latents_map)
            # decoded_images_org = decode_images(x_org)
            decoded_images = pred_latents_map
            decoded_images_org = x_org
            # anom_map = feature_distance(decoded_images, decoded_images_org, feature_extractor, mode='cosine')
            
            # anom_map = rearrange(anom_map, '(b nm ns) h w -> b nm ns h w', b=bs, nm=args.num_masks, ns=args.num_samples)
            # anom_map = torch.min(anom_map, dim=2).values  # (B, nm, h, w)
            features = feature_extractor(decoded_images)
            features_org = feature_extractor(decoded_images_org)
            anom_map = anomaly_map(features, features_org)
            anom_map = rearrange(anom_map, '(b nm ns) h w -> b nm ns h w', b=bs, nm=args.num_masks, ns=args.num_samples)
            anom_map = torch.min(anom_map, dim=2).values  # (B, nm, h, w)
            # m, inv_m = update_mask(anom_map, m, args.quant_thresh)
        # x = ddrm_sample(model, x_T, x_org, indices, m, inv_m, sigmas, alphas, args.eta, args.etaB)
        # pred_latents_map = inv_m * x_org + m * x
        
        if args.recon_space == 'latent':
            anom_score = anomaly_score(pred_latents_map, x_org)
            anom_map = anomaly_map(pred_latents_map, x_org)
            anom_scores.append(anom_score)
        elif args.recon_space == 'pixel':
            decoded_images = decode_images(pred_latents_map)
            decoded_images_org = decode_images(x_org)
            anom_map = anomaly_map(decoded_images, decoded_images_org)
            anom_score = anomaly_score(decoded_images, decoded_images_org)
            anom_scores.append(anom_score)
        elif args.recon_space == 'feature':
            # decoded_images = decode_images(pred_latents_map)
            # decoded_images_org = decode_images(x_org)
            # features = feature_extractor(decoded_images)
            # features_org = feature_extractor(decoded_images_org)
            # anom_map = anomaly_map(features, features_org)
            # anom_score = anomaly_score(features, features_org)
            anom_score = torch.mean(anom_map, dim=(2,3))  # (B, nm)
            anom_score = torch.mean(anom_score, dim=1)  # (B)
            anom_scores += anom_score.tolist()
        else:
            raise ValueError(f"Invalid reconstruction space: {args.recon_space}")
        
        if args.save_images and not img_saved:
            img_saved = True
            idxs = (0,1)
            # B, nm, ns, 
            preds = pred_latents_map.view(bs, args.num_masks, args.num_samples, *pred_latents_map.shape[1:])
            reshaped_masks = rearrange(masks, 'b (h w) -> b h w', h=h, w=w).unsqueeze(1)
            
            reshaped_masks = reshaped_masks.unsqueeze(0).repeat(bs, 1, 1, 1, 1)
            reshaped_masks = reshaped_masks.unsqueeze(2).repeat(1, 1, args.num_samples, 1, 1, 1)
            imgs = images.view(bs, args.num_masks, args.num_samples, *images.shape[1:])
            
            c, h, w = in_sh[0], in_sh[1], in_sh[2]
            for b in idxs:
                pred_imgs_b = preds[b, 0].view(-1, c, h, w)
                masks_b = reshaped_masks[b, 0].view(-1, 1, h, w)
                org_imgs_b = imgs[b, 0].view(-1, c, h, w)
                org_imgs = convert2image(postprocess(org_imgs_b))[0]
                pred_imgs = convert2image(postprocess(pred_imgs_b))
                masked_imgs = convert2image(postprocess(org_imgs_b.cpu()[0]) * (1 - masks_b.cpu()))[0]
                maps = convert2image(anom_map[b, 0])
                
                sample_dict[f'anom_{b}'] = {
                    'images': org_imgs,
                    'masked_images': masked_imgs,
                    'pred_images': pred_imgs,
                    'gt_masks': convert2image(gt_masks)[b],
                    'labels': anom_types[b],
                    'anomaly_maps': maps,
                }
            
    anom_scores = torch.tensor(anom_scores)
    # print(f"Mask Coverage: {mask_coverage_meter.avg:.4f}")
    # print(f"Reconstruction Loss: {loss_meter.avg:.4f}")
    
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
        
        
    
    
    
    