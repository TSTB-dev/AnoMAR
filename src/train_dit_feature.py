
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
from tqdm import tqdm
from pathlib import Path

import copy
import argparse
import yaml
from pprint import pprint

from datasets import build_dataset
from utils import get_optimizer, get_lr_scheduler
from denoiser import get_denoiser, Denoiser
from backbones import get_backbone

from einops import rearrange
from sklearn.metrics import roc_curve, roc_auc_score

import wandb
from dotenv import load_dotenv
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

def parse_args():
    parser = argparse.ArgumentParser(description="AnoMAR Training")
    
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the current process')
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')

    args = parser.parse_args()
    return args

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
    # create wandb project
    wandb.init(project="AnoMAR", entity="shuns71", config=config)
    
    # set seed
    seed = config['meta']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    dataset_config = config['data']
    device = config['meta']['device']
    batch_size = config['data']['batch_size']
    train_dataset = build_dataset(**config['data'])
    dataset_config['train'] = False
    dataset_config['anom_only'] = True
    anom_dataset = build_dataset(**dataset_config)
    dataset_config['anom_only'] = False
    dataset_config['normal_only'] = True
    normal_dataset = build_dataset(**dataset_config)
    anom_loader = DataLoader(anom_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    normal_loader = DataLoader(normal_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, \
        pin_memory=config['data']['pin_memory'], num_workers=config['data']['num_workers'], drop_last=True)

    diff_in_sh = (272, 16, 16)  # For EfficientNet-b4
    # diff_in_sh = (1792, 16, 16)  # For WideResNet50-2
    # diff_in_sh = (384, 28, 28)  # For PDN-medium
    # diff_in_sh = (272, 32, 32)
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    ema_decay = config['diffusion']['ema_decay']
    model_ema = copy.deepcopy(model)
    model.to(device)
    model_ema.to(device)
    
    # feature_ln = torch.nn.LayerNorm(diff_in_sh[0]).to(device)

    model_kwargs = {
        'model_type': 'efficientnet-b4',
        'outblocks': (1, 5, 9, 21),
        'outstrides': (2, 4, 8, 16),
        'pretrained': True,
        'stride': 16
    }
    # model_kwargs = {
    #     "model_type": "wide_resnet50_2"}
    # model_kwargs = {
    #     "model_type": "pdn_medium",
    # }
    print(f"Using feature space reconstruction with {model_kwargs['model_type']} backbone")
    
    feature_extractor = get_backbone(**model_kwargs)
    feature_extractor.to(device).eval()

    # optimizer = get_optimizer([model, feature_ln], **config['optimizer'])
    optimizer = get_optimizer([model], **config['optimizer'])
    if config['optimizer']['scheduler_type'] == 'none':
        pass
    else:
        scheduler = get_lr_scheduler(optimizer, **config['optimizer'], iter_per_epoch=len(train_loader))

    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(save_dir))

    # save config
    save_path = save_dir / "config.yaml"
    with open(save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config is saved at {save_path}")

    feature_extractor.eval()
    print(f"Computing global feature statistics for {len(train_dataset)} samples")
    features = []
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        img = data["samples"].to(device)
        with torch.no_grad():
            x, _ = feature_extractor(img)
            # x = feature_extractor(img)
            features.append(x)
    features = torch.cat(features, dim=0)   # (N, c, h, w)
    avg_glo = features.mean(dim=(0, 2, 3))  # (c, )
    std_glo = features.std(dim=(0, 2, 3))  # (c, )
    
    model.train()
    print(f"Steps per epoch: {len(train_loader)}")
    
    es_count = 0
    best_auc = 0
    for epoch in range(config['optimizer']['num_epochs']):
        for i, data in enumerate(train_loader):
            img, labels = data["samples"], data["clslabels"]    # (B, C, H, W), (B,)
            img = img.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                x, _ = feature_extractor(img)  # (B, c, h, w)
                # x = feature_extractor(img)  # (B, c, h, w)
                
                # Normalize x
                x = (x - avg_glo.view(1, -1, 1, 1)) / (std_glo.view(1, -1, 1, 1) + 1e-6)
                
            loss = model(x, labels)  
            
            # backward
            optimizer.zero_grad()
            loss.backward()
                    
            if config['optimizer']['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['grad_clip'])
            optimizer.step()
            
            scheduler.step()
            
            # update ema
            for ema_param, model_param in zip(model_ema.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1.0 - ema_decay)
            
            if i % config["logging"]["log_interval"] == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss {loss.item()}")      
                tb_writer.add_scalar("Loss", loss.item(), epoch * len(train_loader) + i)  
                wandb.log({"Loss": loss.item(), "LR": scheduler.get_last_lr()})
                

        if (epoch + 1) % config["logging"]["save_interval"] == 0:
            save_path = save_dir / f"model_latest.pth"
            torch.save(model.state_dict(), save_path)
            save_path = save_dir / f"model_ema_latest.pth"
            torch.save(model_ema.state_dict(), save_path)
            print(f"Model is saved at {save_dir}")
        
        if (epoch + 1) % config["evaluation"]["eval_interval"] == 0:
            current_auc = evaluate(
                model,
                feature_extractor,
                anom_loader,
                normal_loader,
                config, 
                diff_in_sh,
                epoch + 1,
                config["evaluation"]["eval_step"],
                device,
                (avg_glo, std_glo)
            )
            
            if current_auc > best_auc:
                best_auc = current_auc
                save_path = save_dir / f"model_best.pth"
                torch.save(model.state_dict(), save_path)
                save_path = save_dir / f"model_ema_best.pth"
                torch.save(model_ema.state_dict(), save_path)
                print(f"Model is saved at {save_dir}")
                
                es_count = 0
            else:
                es_count += config["evaluation"]["eval_interval"]

            wandb.log({"AUC": current_auc})
            print(f"AUC: {current_auc} at epoch {epoch}")

            if es_count >= config["evaluation"]["early_stop"]:
                print(f"Early stopping at epoch {epoch}")
                break
            
            
    print("Training is done!")
    tb_writer.close()
    
    # save model
    save_path = save_dir / "model_latest.pth"
    torch.save(model.state_dict(), save_path)
    save_path = save_dir / "model_ema_latest.pth"
    torch.save(model_ema.state_dict(), save_path)
    print(f"Model is saved at {save_dir}")


def init_denoiser(num_inference_steps, device, config, in_sh, inherit_model=None):
    config["diffusion"]["num_sampling_steps"] = str(num_inference_steps)
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=in_sh)
    
    if inherit_model is not None:
        for p, p_inherit in zip(model.parameters(), inherit_model.parameters()):
            p.data.copy_(p_inherit.data)
    model.to(device).eval()
    return model

@torch.no_grad()
def evaluate(denoiser, feature_extractor, anom_loader, normal_loader, config, in_sh, epoch, eval_step, device, \
    global_stats):
    avg_glo, std_glo = global_stats
    denoiser.eval()
    feature_extractor.eval()
    
    eval_denoiser = init_denoiser(eval_step, device, config, in_sh, inherit_model=denoiser)
    
    print(f"Evaluating on {len(anom_loader)} anomalous samples and {len(normal_loader)} normal samples")
    
    start_t = torch.tensor([0] * 8, device=device, dtype=torch.long)
    normal_scores = []
    for i, batch in enumerate(normal_loader):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        
        features, _ = feature_extractor(images)
        # features = feature_extractor(images)
        features = (features - avg_glo.view(1, -1, 1, 1)) / (std_glo.view(1, -1, 1, 1) + 1e-6)
        latents_last = eval_denoiser.ddim_reverse_sample(
            features, start_t, labels, eta=0.0
        )
        latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
        ats = torch.abs(latents_last_l2 - torch.sqrt(torch.tensor([0], device=device, dtype=torch.float32))) 
        min_ats_spatial = ats.view(ats.shape[0], -1).min(dim=1)[0]  # (bs, )
        max_ats_spatial = ats.view(ats.shape[0], -1).max(dim=1)[0]  # (bs, )
        ats = torch.abs(min_ats_spatial - max_ats_spatial)  # (bs, )
        
        normal_scores.extend(ats.cpu().numpy())
        
    anomaly_scores = []
    for i, batch in enumerate(anom_loader):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        
        features, _ = feature_extractor(images)
        # features = feature_extractor(images)
        features = (features - avg_glo.view(1, -1, 1, 1)) / (std_glo.view(1, -1, 1, 1) + 1e-6)
        latents_last = eval_denoiser.ddim_reverse_sample(
            features, start_t, labels, eta=0.0
        )
        latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
        ats = torch.abs(latents_last_l2 - torch.sqrt(torch.tensor([0], device=device, dtype=torch.float32))) 
        min_ats_spatial = ats.view(ats.shape[0], -1).min(dim=1)[0]
        max_ats_spatial = ats.view(ats.shape[0], -1).max(dim=1)[0]
        ats = torch.abs(min_ats_spatial - max_ats_spatial)
        
        anomaly_scores.extend(ats.cpu().numpy())
    
    normal_scores = np.array(normal_scores)
    anomaly_scores = np.array(anomaly_scores)

    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    y_score = np.concatenate([normal_scores, anomaly_scores])
    
    roc_auc = roc_auc_score(y_true, y_score)
    return roc_auc
        
        
        
        

# @torch.no_grad()
# def evaluate(denoiser, feature_extractor, anom_loader, normal_loader, in_sh, epoch, start_step, device, \
#     global_stats):
#     avg_glo, std_glo = global_stats
#     denoiser.eval()
#     feature_extractor.eval()
    
#     print(f"Evaluating on {len(anom_loader)} anomalous samples and {len(normal_loader)} normal samples")
#     normal_scores = []
#     for i, batch in enumerate(normal_loader):
#         images = batch["samples"].to(device)
#         labels = batch["clslabels"].to(device)
#         # Prepare timesteps
#         t = torch.tensor([start_step] * len(images)).to(device)  # (B, )
        
#         def perturb(x, t):
#             z, _ = feature_extractor(x)  # (B, c, h, w)
            
#             # Normalize z
#             z = (z - avg_glo.view(1, -1, 1, 1)) / (std_glo.view(1, -1, 1, 1) + 1e-6)
#             noised_z = denoiser.q_sample(z, t)  # (B, c, h, w)
            
#             return noised_z, z, 
        
#         noised_latents, org_latents = perturb(images, t)  # (B, c, h, w)
        
#         # decode
#         def denoising(noised_z, t, labels):
#             denoized_z = denoiser.denoise_from_intermediate(noised_z, t, labels)  
#             return denoized_z
#         denoized_latents = denoising(noised_latents, t, labels)
        
#         # calculate scores
#         def anomaly_score(x, x_rec):
#             diff = torch.mean((x - x_rec).pow(2), dim=1)
#             mse = diff.view(-1, in_sh[1], in_sh[2])
#             mse = torch.mean(mse, dim=(1, 2))  # (B, )
#             # mse = mse.max(dim=1).values  # (B, W)
#             # mse = mse.max(dim=1).values  # (B, )
            
#             anom_map = torch.mean((x - x_rec).pow(2), dim=1)  # (B*K, H, W)
#             anom_map = anom_map.view(-1, *anom_map.shape[1:])
#             # anom_map = torch.min(anom_map, dim=1).values  # (B, H, W)
#             return mse, anom_map
        
#         anom_score, anom_map = anomaly_score(org_latents, denoized_latents)
#         normal_scores.append(anom_score)
    
#     normal_scores = torch.cat(normal_scores, dim=0)
    
#     anom_scores = []
#     for i, batch in enumerate(anom_loader):
#         images = batch["samples"].to(device)
#         labels = batch["clslabels"].to(device)
#         # Prepare timesteps
#         t = torch.tensor([start_step] * len(images)).to(device)
        
#         noised_latents, org_latents = perturb(images, t)  # (B, c, h, w)
#         denoized_latents = denoising(noised_latents, t, labels)
        
#         anom_score, anom_map = anomaly_score(org_latents, denoized_latents)
        
#         anom_scores.append(anom_score)
#     anom_scores = torch.cat(anom_scores, dim=0)

#     y_true = torch.cat([torch.zeros_like(normal_scores), torch.ones_like(anom_scores)])
#     y_score = torch.cat([normal_scores, anom_scores])
#     from sklearn.metrics import roc_auc_score
#     auc = roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy())
#     print(f"AUC: {auc} at epoch {epoch}")
#     wandb.log({"AUC": auc})
#     return auc

if __name__ == "__main__":
    args = parse_args()
    main(args)


    
    
    
        
      
            
    
        
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
