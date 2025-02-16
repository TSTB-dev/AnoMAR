
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
from denoiser import get_denoiser, Denoiser
from models import create_vae, AutoencoderKL

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
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, \
        pin_memory=config['data']['pin_memory'], num_workers=config['data']['num_workers'])
    
    vae: AutoencoderKL = create_vae(**config['vae'])
    vae.to(device).eval()
    vae_embed_dim = config['vae']['embed_dim']
    vae_stride = config['vae']['stride']
    img_size = config['data']['img_size']
    # diff_in_sh = (vae_embed_dim, img_size // vae_stride, img_size // vae_stride)
    diff_in_sh = (3, img_size, img_size)
    
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    ema_decay = config['diffusion']['ema_decay']
    model_ema = copy.deepcopy(model)
    model.to(device)
    model_ema.to(device)
    
    optimizer = get_optimizer([model], **config['optimizer'])
    if config['optimizer']['scheduler_type'] == 'none':
        pass
    else:
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
        for i, data in enumerate(train_loader):
            img, labels = data["samples"], data["clslabels"]    # (B, C, H, W), (B,)
            img = img.to(device)
            labels = labels.to(device)
            
            # # vae encode
            # with torch.no_grad():
            #     posterior = vae.encode(img)  # (B, c, h, w)
            #     x = posterior.sample().mul_(0.2325)  # (B, c, h, w)
            
            x = img
            # forward
            # img = _process_inputs(img)
            loss = model(x, labels)  
            
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
                    save_images(model_ema, vae, tb_writer, epoch, i, device)

        if (epoch + 1) % config["logging"]["save_interval"] == 0:
            save_path = save_dir / f"model_latest.pth"
            torch.save(model.state_dict(), save_path)
            save_path = save_dir / f"model_ema_latest.pth"
            torch.save(model_ema.state_dict(), save_path)
            print(f"Model is saved at {save_dir}")
        
        if config['optimizer']['scheduler_type'] == 'none':
            pass
        else:
            print(f"Epoch {epoch}: LR {scheduler.get_last_lr()}, Loss {loss.item()}")
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


    
    
    
        
      
            
    
        
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
