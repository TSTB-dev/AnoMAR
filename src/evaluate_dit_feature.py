import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from PIL import Image
from tqdm import tqdm
import lpips

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
    parser.add_argument('--save_all_images', action='store_true', help='Save all images')
    parser.add_argument('--eval_dataset', type=str, help='Dataset to evaluate', default=None)
    parser.add_argument('--eval_category', type=str, help='Category to evaluate', default=None)
    
    
    return parser.parse_args()

def postprocess(x):
    x = x / 2 + 0.5
    return x.clamp(0, 1)

def postprocess_lpips(x):
    # -> [-1, 1]
    x = x * 2 - 1  # Assume x is in [0, 1]

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
    
    config["diffusion"]["num_sampling_steps"] = str(args.num_inference_steps)
    img_size = config['data']['img_size']
    device = args.device
    batch_size = args.batch_size
    num_samples = args.num_samples
    dataset = config['data']['dataset_name'] if args.eval_dataset is None else args.eval_dataset
    category = config['data']['category'] if args.eval_category is None else args.eval_category
    print(f"Evaluating on category: {category}")
    
    dataset_config = config['data']
    dataset_config['batch_size'] = batch_size
    dataset_config['category'] = category
    dataset_config['dataset_name'] = dataset
    dataset_config['transform_type'] = 'default'
    dataset_config['train'] = True
    dataset_config['normal_only'] = False
    train_dataset = build_dataset(**dataset_config)
    dataset_config['train'] = False
    dataset_config['anom_only'] = True
    anom_dataset = build_dataset(**dataset_config)
    dataset_config['anom_only'] = False
    dataset_config['normal_only'] = True
    normal_dataset = build_dataset(**dataset_config)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    anom_loader = DataLoader(anom_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    
    vae: AutoencoderKL = create_vae(**config['vae'])
    vae.to(device).eval()
    vae_embed_dim = config['vae']['embed_dim']
    vae_stride = config['vae']['stride']
    img_size = config['data']['img_size']
    diff_in_sh = (272, 16, 16)
    
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    denoiser_ckpt = torch.load(args.model_ckpt, map_location="cpu", weights_only=True)
    results = model.load_state_dict(denoiser_ckpt, strict=True)
    print(results)
    model.to(device)
    model.eval()
    
    if args.recon_space == 'feature':
        model_kwargs = {
            'model_type': 'efficientnet-b4',
            'outblocks': (1, 5, 9, 21),
            'outstrides': (2, 4, 8, 16),
            'pretrained': True,
            'stride': 16
        }
        # model_kwargs = {'model_type': 'wide_resnet50_2'}
        print(f"Using feature space reconstruction with {model_kwargs['model_type']} backbone")
        feature_extractor = get_backbone(**model_kwargs)
        feature_extractor.to(device).eval()
    
    # For visualization
    if args.save_images:
        sample_indices = random.sample(range(min(len(anom_loader), len(normal_loader))), 2) if not args.save_all_images else range(min(len(anom_loader), len(normal_loader)))
    else:
        sample_indices = []
    sample_dict = {}
    
    feature_extractor.eval()
    print(f"Computing global feature statistics for {len(train_dataset)} samples")
    features = []
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        img = data["samples"].to(device)
        with torch.no_grad():
            x, _ = feature_extractor(img)
            features.append(x)
    features = torch.cat(features, dim=0)   # (N, c, h, w)
    avg_glo = features.mean(dim=(0, 2, 3))  # (c, )
    std_glo = features.std(dim=(0, 2, 3))  # (c, )
    
    # Evaluation on normal samples
    print("Evaluating on normal samples✔️")
    normal_scores = []
    for i, batch in tqdm(enumerate(normal_loader), total=len(normal_loader)):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
#         masks = batch["masks"]  # (B, 1, H, W)
        # Prepare timesteps
        t = torch.tensor([args.start_step] * num_samples * len(images)).to(device)  # (B * K, )
        
        def perturb(x, t):
            z, _ = feature_extractor(x)
            # Normalize x
            z = (z - avg_glo.view(1, -1, 1, 1)) / (std_glo.view(1, -1, 1, 1) + 1e-6)
            z = z.repeat_interleave(num_samples, dim=0)  # (B*K, c, h, w)
            
            noised_z = model.q_sample(z, t)  # (B*K, c, h, w)
            return noised_z, z
        
        noised_latents, org_latents = perturb(images, t)  # (B*K, c, h, w)
        
        # decode
        def denoising(noised_z, t, labels):
            labels = labels.repeat_interleave(num_samples, dim=0)
            denoized_z = model.denoise_from_intermediate(noised_z, t, labels)  
            return denoized_z
        
        denoized_latents = denoising(noised_latents, t, labels)
        
        # calculate scores
        def anomaly_score(x, x_rec):
            diff = torch.mean((x - x_rec).pow(2), dim=1)
            mse = diff.view(-1, num_samples, diff_in_sh[1], diff_in_sh[2])
            mse = torch.min(mse, dim=1).values  # (B, H, W)
            mse = torch.mean(mse, dim=(1, 2))  # (B, )
            # mse = mse.max(dim=1).values  # (B, W)
            # mse = mse.max(dim=1).values  # (B, )
            
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
            # decoded_images = x_rec
            # decoded_images_org = vae.decode(org_latents / 0.2325)  # (B*K, 3, H, W)
            # features = feature_extractor(decoded_images)  # (B*K, c, h, w)
            # features_org = feature_extractor(decoded_images_org)  # (B*K, c, h, w)
            features = denoized_latents
            features_org = org_latents
            # pix_map = torch.mean((decoded_images - decoded_images_org).pow(2), dim=1)  # (B*K, H, W)
            # pix_map = pix_map.view(-1, num_samples, *pix_map.shape[1:]) 
            # pix_map = torch.min(pix_map, dim=1).values  # (B, H, W)
            # pix_score = torch.mean(pix_map, dim=(1, 2))  # (B, )
            anom_score, anom_map = anomaly_score(features, features_org)
            # normalize and combine
            # anom_score = 0.5 * anom_score + 0.5 * pix_score
            # anom_map = pix_map
            # loss_fn_alex = lpips.LPIPS(net='alex', spatial=True).to(device)
            # loss_fn_vgg_sp = lpips.LPIPS(net='vgg', spatial=True).to(device)
            # loss_fn_vgg = lpips.LPIPS(net='vgg', spatial=False).to(device)
            # d_alex = loss_fn_alex(decoded_images, decoded_images_org)
            # anom_map = loss_fn_vgg_sp(decoded_images, decoded_images_org)
            anom_score = anom_map.mean()
            normal_scores.append(anom_score.item())
        else:
            raise ValueError("Invalid reconstruction space")        

        if i in sample_indices:
            sample_dict[f"normal_{i}"] = {
                "images": convert2image(postprocess(decoded_images_org))[0],  # (H, W, C)
                "noised_images": convert2image(postprocess(noised_x).reshape(-1, num_samples, *noised_x.shape[1:])[0][0]),  # (H, W, C)
                "reconstructed_images": convert2image(postprocess(x_rec).reshape(-1, num_samples, *x_rec.shape[1:])[0]),  # (K, H, W, C)
                "anomaly_maps": convert2image(anom_map[0]),  # (H, W)
                "gt_masks": convert2image(masks)[0],  # (H, W)
                "labels": "normal", 
            }
    
    normal_scores = torch.Tensor(normal_scores)
    
    # Evaluation on anomalous samples
    print("Evaluating on anomalous samples✖️")
    anom_scores = []
    anom_types = []
    for i, batch in tqdm(enumerate(anom_loader), total=len(anom_loader)):
        
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        anom_type = batch["anom_type"]
        anom_types += anom_type
       #  masks = batch["masks"]  # (B, 1, H, W)
        
        # Prepare timesteps
        t = torch.tensor([args.start_step] * num_samples * len(images)).to(device)  # (B * K, )
        
        noised_latents, org_latents = perturb(images, t)  # (B*K, c, h, w)
        denoized_latents = denoising(noised_latents, t, labels)
        
        if args.recon_space == 'latent':
            anom_score, anom_map = anomaly_score(org_latents, denoized_latents)
            anom_scores.append(anom_score)
        elif args.recon_space == 'pixel':
            anom_score, anom_map = anomaly_score(images.repeat_interleave(num_samples, dim=0), x_rec)
            anom_scores.append(anom_score)
        elif args.recon_space == 'feature':
            features = denoized_latents
            features_org = org_latents
            anom_score, anom_map = anomaly_score(features, features_org)
            # normalize and combine
            # anom_score = (anom_score - anom_score.min()) / (anom_score.max() - anom_score.min() + 1e-6)
            # pix_score = (pix_score - pix_score.min()) / (pix_score.max() - pix_score.min() + 1e-6)
            # anom_score = 0.5 * anom_score + 0.5 * pix_score
            # anom_map = pix_map
            # anom_score, anom_map = anomaly_score(features, features_org)
            
            # loss_fn_vgg_sp = lpips.LPIPS(net='vgg', spatial=True).to(device)
            # loss_fn_vgg = lpips.LPIPS(net='vgg', spatial=False).to(device)
            # anom_map = loss_fn_vgg_sp(decoded_images, decoded_images_org)
            # anom_score = anom_map.mean()
            
            anom_scores.append(anom_score.item())
        else:
            raise ValueError("Invalid reconstruction space")
        
        if i in sample_indices:
            sample_dict[f"anom_{i}"] = {
                "images": convert2image(postprocess(decoded_images_org))[0],  
                "noised_images": convert2image(postprocess(noised_x).reshape(-1, num_samples, *noised_x.shape[1:])[0][0]),
                "reconstructed_images": convert2image(postprocess(x_rec).reshape(-1, num_samples, *x_rec.shape[1:])[0]),
                "anomaly_maps": convert2image(anom_map[0]),
                "gt_masks": convert2image(masks)[0],
                "labels": "anomalous"
            }
    anom_scores = torch.Tensor(anom_scores)
    
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
    scores_dict = {}
    scores_dict["good"] = normal_scores.cpu().numpy().tolist()
    for anom_type in unique_anom_types:
        y_true = torch.cat([torch.zeros_like(normal_scores), torch.ones_like(anom_scores[np.array(anom_types) == anom_type])])
        y_score = torch.cat([normal_scores, anom_scores[np.array(anom_types) == anom_type]])
        scores_dict[anom_type] = y_score.cpu().numpy().tolist()
        auc_cat = roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy())
        auc_dict[anom_type] = auc_cat
        print(f"AUC [{anom_type}]: {auc_cat:.4f}")
    print(f"===========================================")
    
    print("Saving results...📁")
    results = {
        "normal_scores": normal_scores.cpu().numpy().tolist(),
        "anom_scores": anom_scores.cpu().numpy().tolist(),
        "scores_dict": scores_dict,
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
        if args.save_all_images:
            # import pdb; pdb.set_trace()
            save_all_images(sample_dict, anom_scores, args.output_dir, num_samples)
        else:
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

def save_all_images(
    sample_dict, 
    anom_scores, 
    output_dir, 
    num_samples
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    min_score = min(anom_scores)
    max_score = max(anom_scores)
    
    items = list(sample_dict.items())
    items = tqdm(items)
    for key, value in items:
        fig, ax = plt.subplots(1, 5 + min(num_samples, 3), figsize=(25, 5))
        font_color = 'blue'
        ax[0].imshow(value['images'])
        ax[0].set_title(f"Original {value['labels']}", color=font_color)
        ax[0].axis('off')
        
        ax[1].imshow(value['noised_images'])
        ax[1].set_title(f"Noised {value['labels']}", color=font_color)
        ax[1].axis('off')
        
        ax[2].imshow(value['reconstructed_images'][0])
        ax[2].set_title(f"Reconstructed {1}", color=font_color)
        ax[2].axis('off')
        
        for j in range(1, min(num_samples, 3)):
            ax[2 + j].imshow(value['reconstructed_images'][j])
            ax[2 + j].set_title(f"Reconstructed {j + 1}", color=font_color)
            ax[2 + j].axis('off')
        
        ax[2 + min(num_samples, 3)].imshow(value['anomaly_maps'], vmin=min_score, vmax=max_score)
        ax[2 + min(num_samples, 3)].set_title(f"Anomaly Map {value['labels']}", color=font_color)
        ax[2 + min(num_samples, 3)].axis('off')
        
        ax[3 + min(num_samples, 3)].imshow(value['gt_masks'], cmap='gray', vmin=0, vmax=1)
        ax[3 + min(num_samples, 3)].set_title(f"GT Mask {value['labels']}", color=font_color)
        ax[3 + min(num_samples, 3)].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{key}.png')
        plt.close()
    
    print(f"Images saved at {output_dir}")
    
    
    

if __name__ == '__main__':
    args = parser_args()
    main(args)
            
            
            
            
            
            
            
            
                
                
            
    
    