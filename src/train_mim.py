
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
from pathlib import Path

import copy
import argparse
import yaml
from pprint import pprint

from datasets import build_dataset
from utils import get_optimizer, get_lr_scheduler
from models import create_vae, AutoencoderKL, create_mim_model, MaskedImageModelingModel
from backbones import get_backbone
from mask import RandomMaskCollator, BlockRandomMaskCollator, CheckerBoardMaskCollator, indices_to_mask, mask_to_indices

def parse_args():
    parser = argparse.ArgumentParser(description="AnoMAR Training")
    
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the current process')
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')

    args = parser.parse_args()
    return args

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
    
    device = config['meta']['device']
    batch_size = config['data']['batch_size']
    
    train_dataset = build_dataset(**config['data'])
    
    # build backbone model
    if 'vae' in config['backbone']['model_type']:
        backbone: AutoencoderKL = create_vae(**config['backbone'])
    else:
        backbone = get_backbone(**config['backbone'])
    backbone.to(device).eval()
    
    for param in backbone.parameters():
        param.requires_grad = False
    
    backbone_embed_dim = config['backbone']['embed_dim']
    backbone_stride = config['backbone']['stride']
    img_size = config['data']['img_size']
    mim_in_sh = (backbone_embed_dim, img_size // backbone_stride, img_size // backbone_stride)
    
    # build mim model
    model: MaskedImageModelingModel = create_mim_model(**config['mim'])
    model.to(device)
    if config['mim']['ckpt_path'] is not None:
        model.load_state_dict(torch.load(config['mim']['ckpt_path']))
        print(f"Loaded model from {config['mim']['ckpt_path']}")
    
    ema_decay = config['mim']['ema_decay']
    model_ema = copy.deepcopy(model)
    model_ema.to(device)
    
    # build mask collator
    mask_strategy = config['data']['mask']['strategy']
    mask_ratio = config['data']['mask']['ratio']
    patch_size = config['mim']['patch_size']
    
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
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=mask_collator, \
        pin_memory=config['data']['pin_memory'], num_workers=config['data']['num_workers'])

    optimizer = get_optimizer(model, **config['optimizer'])
    scheduler = get_lr_scheduler(optimizer, **config['optimizer'])

    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(save_dir))

    # save config
    save_path = save_dir / "config.yaml"
    with open(save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config is saved at {save_path}")
    
    model.train()
    for epoch in range(config['optimizer']['num_epochs']):
        for i, (data, mask_indices) in enumerate(train_loader):
            img, labels = data["samples"], data["clslabels"]    # (B, C, H, W), (B,)
            img = img.to(device)
            labels = labels.to(device)
            
            mask = indices_to_mask(mask_indices.to(device), model.num_patches)
            
            # forward
            with torch.no_grad():
                if 'vae' in config['backbone']['model_type']:
                    posterior = backbone.encode(img)  # (B, c, h, w)
                    x = posterior.sample().mul_(0.2325)  # (B, c, h, w)
                else:
                    x = backbone(img)  # (B, c, h, w)
            
            outputs = model(x, mask)
            loss = outputs['loss']

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update ema
            for ema_param, model_param in zip(model_ema.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1.0 - ema_decay)
            
            if i % config["logging"]["log_interval"] == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss {loss.item()}")      
                tb_writer.add_scalar("Loss", loss.item(), epoch * len(train_loader) + i)  
                
                if config["logging"]["save_images"]:
                    pass
                    # save_images(model_ema, vae, tb_writer, epoch, i, device)
        
        if (epoch + 1) % config["logging"]["save_interval"] == 0:
            save_path = save_dir / f"model_latest.pth"
            torch.save(model.state_dict(), save_path)
            save_path = save_dir / f"model_ema_latest.pth"
            torch.save(model_ema.state_dict(), save_path)
            print(f"Model is saved at {save_dir}")
        
        scheduler.step()

    print("Training is done!")
    tb_writer.close()
    
    # save model
    save_path = save_dir / "model_latest.pth"
    torch.save(model.state_dict(), save_path)
    save_path = save_dir / "model_ema_latest.pth"
    torch.save(model_ema.state_dict(), save_path)
    print(f"Model is saved at {save_dir}")
    
def save_images(model, vae, tb_writer, epoch, i, device):
    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)
                
            