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
from models import create_vae, AutoencoderKL, create_mim_model, MaskedImageModelingModel
from backbones import get_backbone
from mask import RandomMaskCollator, BlockRandomMaskCollator, CheckerBoardMaskCollator, indices_to_mask, mask_to_indices

def parser_args():
    parser = argparse.ArgumentParser(description='AnoMAR Sampling')
    
    parser.add_argument('--num_masks', type=int, help='Number of masks to generate', default=1)
    parser.add_argument('--recon_space', type=str, default='latent', help='Reconstruction space')  # ['latent', 'pixel', 'feature']
    parser.add_argument('--save_images', action='store_true', help='Save images')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--model_ckpt', type=str, help='Path to the model checkpoint')
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    
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
    
    img_size = config['data']['img_size']
    device = args.device
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
    if 'vae' in config['backbone']['model_type']:
        backbone: AutoencoderKL = create_vae(**config['backbone'])
    else:
        backbone = get_backbone(config['backbone']['model_type'], **config['backbone'])
    backbone.to(device).eval()
    
    backbone_embed_dim = config['backbone']['embed_dim']
    backbone_stride = config['backbone']['stride']
    img_size = config['data']['img_size']
    patch_size = config['mim']['patch_size']
    mim_in_sh = (backbone_embed_dim, img_size // backbone_stride, img_size // backbone_stride)
    
    # build mim model
    model: MaskedImageModelingModel = create_mim_model(**config['mim'])
    model_ckpt = torch.load(args.model_ckpt, map_location='cpu', weights_only=True)
    results = model.load_state_dict(model_ckpt, strict=True)
    print(results)
    model.to(device).eval()
    
    if args.recon_space == 'feature':
        model_name = 'efficientnet-b4'
        print(f"Using feature space reconstruction with {model_name} backbone")
        feature_extractor = get_backbone(model_name=model_name)
        feature_extractor.to(device).eval()
        
    # build mask collator
    mask_strategy = config['data']['mask']['strategy']
    mask_ratio = config['data']['mask']['ratio']

    if mask_strategy == "random":
        mask_collator = RandomMaskCollator(
            ratio=mask_ratio, input_size=mim_in_sh[1], patch_size=patch_size
        )
    elif mask_strategy == "block":
        mask_collator = BlockRandomMaskCollator(
            input_size=mim_in_sh[1], patch_size=patch_size, mask_ratio=mask_ratio, **config['data']['mask']
        )
    elif mask_strategy == "checkerboard":
        mask_collator = CheckerBoardMaskCollator(
            input_size=mim_in_sh[1], patch_size=patch_size, **config['data']['mask']
        )
    else:
        raise ValueError(f"Invalid mask strategy: {mask_strategy}")
    
    normal_loader = EvalDataLoader(normal_dataset, args.num_masks, collate_fn=mask_collator)
    # For visualization
    if args.save_images:    
        sample_indices = random.sample(range(min(len(normal_dataset), len(anom_dataset))), 2)
        sample_dict = {}
    else:
        sample_indices = []
    
    # Evaluation on normal samples
    print("Evaluating on normal samples")
    normal_scores = []
    mask_coverage_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    for i, (batch, mask_indices) in tqdm(enumerate(normal_loader), total=len(normal_loader)):
        mask = indices_to_mask(mask_indices, model.num_patches)
        mask = mask.to(device)
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        gt_masks = batch["masks"]  # (B, 1, H, W) 
        
        mask_cov = calculate_mask_coverage(mask_indices, mim_in_sh[1], mim_in_sh[2])
        mask_coverage_meter.update(mask_cov.item(), images.size(0))
        
        def encode_images(x):
            if 'vae' in config['backbone']['model_type']:
                post = backbone.encode(x)
                return post.sample().mul_(0.2325)
            else:
                return backbone(x)
    
        def reshape_mask(m):  
            """ (B, hw) -> (B, hp, wp) """
            m = rearrange(m, 'b (h w) -> b h w', h=mim_in_sh[1], w=mim_in_sh[2])
            m = torch.repeat_interleave(m, repeats=patch_size, dim=1)
            m = torch.repeat_interleave(m, repeats=patch_size, dim=2)
            m = m.float()
            return m
            
        def decode_images(x):
            if 'vae' in config['backbone']['model_type']:
                return backbone.decode(x / 0.2325)
            else:
                raise ValueError(f"Backbone {config['backbone']['model_type']} does not support decoding")
        
        outputs = model(encode_images(images), mask)
        target_features = outputs["targets"]  # (B, c, h, w)
        pred_features = outputs["preds"]  # (B, c, h, w)
        loss_meter.update(outputs["loss"].item(), images.size(0))

        mask = reshape_mask(mask).unsqueeze(1)  # (B, 1, h, w)
        pred_features = pred_features * mask + target_features * (1 - mask)
        pred_images = decode_images(pred_features)
        
        def anomaly_score(x, y, m):
            anom_map = m.squeeze(1) * torch.mean((x - y)**2, dim=1)  # (B, h, w)
            anom_score = torch.mean(anom_map[anom_map != 0])
            anom_map = torch.mean(anom_map, dim=0)  # (h, w)
            return anom_score, anom_map
        
        if args.recon_space == 'latent':
            anom_score, anom_map = anomaly_score(target_features, pred_features, mask)
            normal_scores.append(anom_score)
        elif args.recon_space == 'pixel':
            raise NotImplementedError
        elif args.recon_space == 'feature':
            raise NotImplementedError   
        else:
            raise ValueError(f"Invalid reconstruction space: {args.recon_space}")
        
        if i in sample_indices:
            org_imgs = convert2image(postprocess(images))  # (B, H, W, C)
            mask = F.interpolate(mask, size=(img_size, img_size), mode='nearest')  # (B, 1, H, W)
            masked_imgs = convert2image(postprocess(images) * (1 - mask))  # (B, H, W, C)
            pred_imgs = convert2image(postprocess(pred_images))
            
            sample_dict[f'normal_{i}'] = {
                'images': org_imgs[0], # (H, W, C)
                'masked_images': masked_imgs[0],
                'pred_images': pred_imgs,
                'gt_masks': convert2image(gt_masks)[0],  # (H, W)
                'labels': 'normal',
                'anomaly_maps': convert2image(anom_map), # (H, W)
                'mask_coverage': mask_cov.item()
            }
        
    normal_scores = torch.tensor(normal_scores)
    print(f"Normal Loss: {loss_meter.avg:.4f}")
    
    # Evaluation on anomalous samples
    print("Evaluating on anomalous samples")
    anom_scores = []            
    anom_types = []
    # Ensure that the masks are shared between normal and anomalous samples
    anom_loader = EvalDataLoader(anom_dataset, args.num_masks, collate_fn=mask_collator, shared_masks=normal_loader.shared_masks)
    for i, (batch, mask_indices) in tqdm(enumerate(anom_loader), total=len(anom_loader)):
        mask = indices_to_mask(mask_indices, model.num_patches)
        mask = mask.to(device)
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        anom_types.append(batch["anom_type"][0])
        gt_masks = batch["masks"]
        
        mask_cov = calculate_mask_coverage(mask_indices, mim_in_sh[1], mim_in_sh[2])
        mask_coverage_meter.update(mask_cov.item(), images.size(0))
        
        outputs = model(encode_images(images), mask)
        target_features = outputs["targets"]  # (B, c, h, w)
        pred_features = outputs["preds"]  # (B, c, h, w)
        loss_meter.update(outputs["loss"].item(), images.size(0))

        mask = reshape_mask(mask).unsqueeze(1)  # (B, 1, h, w)
        pred_features = pred_features * mask + target_features * (1 - mask)
        pred_images = decode_images(pred_features)
        
        if args.recon_space == 'latent':
            anom_score, anom_map = anomaly_score(target_features, pred_features, mask)
            anom_scores.append(anom_score)
        elif args.recon_space == 'pixel':
            raise NotImplementedError
        elif args.recon_space == 'feature':
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid reconstruction space: {args.recon_space}")
        
        if i in sample_indices:
            org_imgs = convert2image(postprocess(images))  # (B, H, W, C)
            mask = F.interpolate(mask, size=(img_size, img_size), mode='nearest')  # (B, 1, H, W)
            masked_imgs = convert2image(postprocess(images) * (1 - mask))  # (B, H, W, C)
            pred_imgs = convert2image(postprocess(pred_images))
            
            sample_dict[f'anom_{i}'] = {
                'images': org_imgs[0], # (H, W, C)
                'masked_images': masked_imgs[0],  
                'pred_images': pred_imgs,
                'gt_masks': convert2image(gt_masks)[0],  # (H, W)
                'labels': 'anomalous',
                'anomaly_maps': convert2image(anom_map), # (H, W)
                'mask_coverage': mask_cov.item()
            }
                
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
        "mask_ratio": mask_ratio,
    }
    results.update(auc_dict)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved at {output_dir / 'eval_results.json'}")
    
    if args.save_images:
        save_images(sample_dict, anom_scores, args.output_dir, num_masks=args.num_masks)
        print(f"Images saved at {output_dir / 'results.png'}")
            
def save_images(sample_dict, anom_scores, output_dir, num_masks):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    min_score = min(anom_scores)
    max_score = max(anom_scores)
    
    fig, ax = plt.subplots(4, 4 + min(num_masks, 3), figsize=(25, 10))
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
        
        for j in range(1, min(num_masks, 3)):
            ax[i, 2 + j].imshow(value['pred_images'][j])
            ax[i, 2 + j].set_title(f"Pred images {j + 1}", color=font_color)
            ax[i, 2 + j].axis('off')
        
        ax[i, 2 + min(num_masks, 3)].imshow(value['anomaly_maps'], vmin=min_score, vmax=max_score)
        ax[i, 2 + min(num_masks, 3)].set_title(f"Anomaly Map {value['labels']}", color=font_color)
        ax[i, 2 + min(num_masks, 3)].axis('off')
        
        ax[i, 3 + min(num_masks, 3)].imshow(value['gt_masks'], cmap='gray', vmin=0, vmax=1)
        ax[i, 3 + min(num_masks, 3)].set_title(f"GT Mask {value['labels']}", color=font_color)
        ax[i, 3 + min(num_masks, 3)].axis('off')
            
    plt.tight_layout()
    plt.savefig(output_dir / 'results.png')
    plt.close()

if __name__ == '__main__':
    args = parser_args()
    main(args)
        
        
    
    
    
    