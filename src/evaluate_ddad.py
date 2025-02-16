import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from PIL import Image
from tqdm import tqdm

import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import argparse
import yaml
from sklearn.metrics import roc_auc_score
import json
from pprint import pprint

from datasets import build_dataset
from denoiser import get_denoiser, Denoiser
from models import create_vae, AutoencoderKL
from backbones import get_backbone

from torch.nn import functional as F
from torchvision import transforms

def parser_args():
    parser = argparse.ArgumentParser(description='AnoMAR Sampling')
    
    parser.add_argument('--num_samples', type=int, help='Number of samples to generate', default=1)
    parser.add_argument('--num_inference_steps', type=int, help='Number of inference steps')
    parser.add_argument('--recon_space', type=str, default='latent', help='Reconstruction space')  # ['latent', 'pixel', 'feature']
    parser.add_argument('--start_step', type=int, default=10, help='timestep to start denoising')
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

def gaussian_blur(x, kernel_size=3, sigma=1.0):
    import cv2
    x = x.permute(0, 2, 3, 1).cpu().numpy()
    x = cv2.GaussianBlur(x, (kernel_size, kernel_size), sigma)
    x = torch.tensor(x).permute(0, 3, 1, 2)
    return x

def convert2image(x):
    if x.dim() == 3:
        return x.permute(1, 2, 0).cpu().numpy()
    elif x.dim() == 4:
        return x.permute(0, 2, 3, 1).cpu().numpy()
    else:
        return x.cpu().numpy()

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
    batch_size = args.batch_size
    num_samples = args.num_samples
    category = config['data']['category']

    dataset_config = config['data']
    dataset_config['batch_size'] = batch_size
    dataset_config['transform_type'] = 'ddad'
    dataset_config['train'] = False
    dataset_config['anom_only'] = True
    anom_dataset = build_dataset(**dataset_config)
    dataset_config['anom_only'] = False
    dataset_config['normal_only'] = True
    normal_dataset = build_dataset(**dataset_config)
    anom_loader = DataLoader(anom_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

    img_size = config['data']['img_size']
    diff_in_sh = (3, img_size, img_size)
    
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    denoiser_ckpt = torch.load(args.model_ckpt, map_location="cpu", weights_only=True)
    keys_to_replace = []
    # Identify the keys to be renamed
    for k in denoiser_ckpt.keys():
        if 'module' in k:
            keys_to_replace.append(k)

    # Modify the keys outside the loop
    for k in keys_to_replace:
        denoiser_ckpt[k.replace('module.', 'net.')] = denoiser_ckpt.pop(k)
        
    results = model.load_state_dict(denoiser_ckpt, strict=False)
    print(results)
    model.to(device)
    model.eval()
    
    if args.recon_space == 'feature':
        # model_kwargs = {
        #     'model_type': 'efficientnet-b4',
        #     'outblocks': (1, 5, 9, 21),
        #     'outstrides': (2, 4, 8, 16),
        #     'pretrained': True,
        #     'stride': 16
        # }
        # print(f"Using feature space reconstruction with {model_kwargs['model_type']} backbone")
        # model_kwargs = {"model_type": "pdn_medium"}
        model_kwargs = {
            'model_type': 'wide_resnet',
            'ckpt_path': f'./{category}_fe'
        }
        feature_extractor = get_backbone(**model_kwargs)
        feature_extractor.to(device).eval()
    
    # For visualization
    sample_indices = random.sample(range(min(len(anom_loader), len(normal_loader))), 2)
    sample_dict = {}
    
    # Evaluation on normal samples
    print("Evaluating on normal samples✔️")
    normal_scores = []
    bidx = list(range(batch_size, len(normal_loader), batch_size))
    for i, batch in tqdm(enumerate(normal_loader), total=len(normal_loader)):

        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        masks = batch["masks"]  # (B, 1, H, W)
        
        # Prepare timesteps
        t = torch.tensor([args.start_step] * num_samples * len(images)).to(device)  # (B * K, )
        
        def perturb(x, t):
            x = x.repeat_interleave(num_samples, dim=0)  # (B*K, c, h, w)
            
            noised_x = model.q_sample(x, t)  # (B*K, c, h, w)
            
            return x, noised_x
        
        org_x, noised_x = perturb(images, t)  # (B*K, c, h, w)
        
        # decode
        def denoising(noised_x, t, labels):
            labels = labels.repeat_interleave(num_samples, dim=0)
            denoized_x = model.denoise_from_intermediate(noised_x, t, None)  
            return denoized_x
        
        x_rec = denoising(noised_x, t, labels)
        
        # calculate scores
        def anomaly_score(x, x_rec):
            # import pdb; pdb.set_trace()
            diff = torch.mean((x - x_rec).pow(2), dim=1)
            mse = diff.view(-1, num_samples, 16, 16)
            mse = torch.min(mse, dim=1).values  # (B, H, W)
            mse = torch.mean(mse, dim=(1,2))
            
            anom_map = torch.mean((x - x_rec).pow(2), dim=1)  # (B*K, H, W)
            anom_map = anom_map.view(-1, num_samples, *anom_map.shape[1:])
            anom_map = torch.min(anom_map, dim=1).values  # (B, H, W)
            return mse, anom_map

        if args.recon_space == 'latent':
            anom_score, anom_map = anomaly_score(org_latents, denoized_latents)
            normal_scores.append(anom_score)
        elif args.recon_space == 'pixel':
            anom_score, anom_map = anomaly_score(images.repeat_interleave(num_samples, dim=0), x_rec)
            normal_scores.append(anom_score)
        elif args.recon_space == 'feature':
            decoded_images = x_rec
            decoded_images_org = org_x  # (B*K, 3, H, W)
            # features = feature_extractor(decoded_images)  # (B*K, c, h, w)
            # features_org = feature_extractor(decoded_images_org)  # (B*K, c, h, w)
            # anom_score, anom_map = anomaly_score(features, features_org)
            anom_map = feature_distance(decoded_images, decoded_images_org, feature_extractor)
            anom_score = torch.mean(anom_map, dim=(1, 2))
            normal_scores.append(anom_score)
        else:
            raise ValueError("Invalid reconstruction space")        

        if i in sample_indices:
            sample_dict[f"normal_{i}"] = {
                "images": convert2image(postprocess(images))[0],  # (H, W, C)
                "noised_images": convert2image(postprocess(noised_x).reshape(-1, num_samples, *noised_x.shape[1:])[0][0]),  # (H, W, C)
                "reconstructed_images": convert2image(postprocess(x_rec).reshape(-1, num_samples, *x_rec.shape[1:])[0]),  # (K, H, W, C)
                "anomaly_maps": convert2image(anom_map[0]),  # (H, W)
                "gt_masks": convert2image(masks)[0],  # (H, W)
                "labels": "normal", 
            }
    
    normal_scores = torch.cat(normal_scores)
    
    # Evaluation on anomalous samples
    print("Evaluating on anomalous samples✖️")
    anom_scores = []
    anom_types = []
    for i, batch in tqdm(enumerate(anom_loader), total=len(anom_loader)):
        
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        anom_type = batch["anom_type"]
        anom_types += anom_type
        masks = batch["masks"]  # (B, 1, H, W)
        
        # Prepare timesteps
        t = torch.tensor([args.start_step] * num_samples * len(images)).to(device)  # (B * K, )
        
        org_x, noised_x = perturb(images, t)  # (B*K, c, h, w)
        x_rec = model.denoise_from_intermediate(noised_x, t, labels)  # (B*K, c, h, w)
        
        if args.recon_space == 'latent':
            anom_score, anom_map = anomaly_score(x_rec, org_x)
            anom_scores.append(anom_score)
        elif args.recon_space == 'pixel':
            anom_score, anom_map = anomaly_score(images.repeat_interleave(num_samples, dim=0), x_rec)
            anom_scores.append(anom_score)
        elif args.recon_space == 'feature':
            decoded_images = x_rec
            decoded_images_org = org_x  # (B*K, 3, H, W)
            # features = feature_extractor(decoded_images)  # (B*K, c, h, w)
            # features_org = feature_extractor(decoded_images_org)  # (B*K, c, h, w)
            # anom_score, anom_map = anomaly_score(features, features_org)
            anom_map = feature_distance(decoded_images, decoded_images_org, feature_extractor)
            anom_score = torch.mean(anom_map, dim=(1, 2))
            anom_scores.append(anom_score)
        else:
            raise ValueError("Invalid reconstruction space")
        
        if i in sample_indices:
            sample_dict[f"anom_{i}"] = {
                "images": convert2image(postprocess(images))[0],  
                "noised_images": convert2image(postprocess(noised_x).reshape(-1, num_samples, *noised_x.shape[1:])[0][0]),
                "reconstructed_images": convert2image(postprocess(x_rec).reshape(-1, num_samples, *x_rec.shape[1:])[0]),
                "anomaly_maps": convert2image(anom_map[0]),
                "gt_masks": convert2image(masks)[0],
                "labels": "anomalous"
            }
    anom_scores = torch.cat(anom_scores)
    
    print("Calculating AUC...🧑‍⚕️")
    print(f"===========================================")
    y_true = torch.cat([torch.zeros_like(normal_scores), torch.ones_like(anom_scores)])
    y_score = torch.cat([normal_scores, anom_scores])
    auc = roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy())
    print(f"Image-level AUC 👓: {auc:.4f} [{category}]")
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
        "auc": auc, 
        "num_samples": num_samples,
        "recon_space": args.recon_space,
        "start_step": args.start_step,
        "num_inference_steps": args.num_inference_steps,
    }
    results.update(auc_dict)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved at {output_dir / 'eval_results.json'}")
    
    if args.save_images:
        save_images(sample_dict, anom_scores, args.output_dir, num_samples)
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
        
        ax[i, 1].imshow(value['noised_images'])
        ax[i, 1].set_title(f"Noised {value['labels']}", color=font_color)
        ax[i, 1].axis('off')
        
        ax[i, 2].imshow(value['reconstructed_images'][0])
        ax[i, 2].set_title(f"Reconstructed {1}", color=font_color)
        ax[i, 2].axis('off')
        
        for j in range(1, min(num_samples, 3)):
            ax[i, 2 + j].imshow(value['reconstructed_images'][j])
            ax[i, 2 + j].set_title(f"Reconstructed {j + 1}", color=font_color)
            ax[i, 2 + j].axis('off')
        
        ax[i, 2 + min(num_samples, 3)].imshow(value['anomaly_maps'], vmin=min_score, vmax=max_score)
        ax[i, 2 + min(num_samples, 3)].set_title(f"Anomaly Map {value['labels']}", color=font_color)
        ax[i, 2 + min(num_samples, 3)].axis('off')
        
        ax[i, 3 + min(num_samples, 3)].imshow(value['gt_masks'], cmap='gray', vmin=0, vmax=1)
        ax[i, 3 + min(num_samples, 3)].set_title(f"GT Mask {value['labels']}", color=font_color)
        ax[i, 3 + min(num_samples, 3)].axis('off')
            
    plt.tight_layout()
    plt.savefig(output_dir / 'results.png')
    plt.close()

    

if __name__ == '__main__':
    args = parser_args()
    main(args)
            
            
            
            
            
            
            
            
                
                
            
    
    