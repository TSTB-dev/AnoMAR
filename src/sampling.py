import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from PIL import Image
from tqdm import tqdm

import numpy as np
from pathlib import Path

import argparse
import yaml
from pprint import pprint

from datasets import build_dataset
from denoiser import get_denoiser, Denoiser
from models import create_vae, AutoencoderKL

def parser_args():
    parser = argparse.ArgumentParser(description='AnoMAR Sampling')
    
    parser.add_argument('--num_samples', type=int, help='Number of samples to generate')
    parser.add_argument('--num_inference_steps', type=int, help='Number of inference steps')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--model_ckpt', type=str, help='Path to the model checkpoint')
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    
    return parser.parse_args()

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
    
    
    train_dataset = build_dataset(**config['data'])
    num_classes = train_dataset.num_classes
    class_labels = train_dataset[0]['clslabels']

    vae: AutoencoderKL = create_vae(**config['vae'])
    vae.to(device).eval()
    vae_embed_dim = config['vae']['embed_dim']
    vae_stride = config['vae']['stride']
    img_size = config['data']['img_size']
    diff_in_sh = (vae_embed_dim, img_size // vae_stride, img_size // vae_stride)
    
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    results = model.load_state_dict(torch.load(args.model_ckpt, weights_only=True, map_location=device), strict=False)
    print(results)
    model.to(device)
    model.eval()
    
    for i in tqdm(range(args.num_samples)):
        import pdb; pdb.set_trace()
        labels = torch.tensor([class_labels] * batch_size).to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            sample = model.sample(diff_in_sh, labels)  # (B, c, h, w)
            # decode
            images = vae.decode(sample / 0.2325)  # (B, C, H, W)
        
        def postprocess(x):
            # [-1, 1] -> [0, 1]
            return x / 2 + 0.5
        images = postprocess(images).cpu()  # (B, C, H, W)
        images = images.permute(0, 2, 3, 1).numpy()  # (B, H, W, C)
        images = (images * 255).astype(np.uint8)  

        for j in range(batch_size):
            img = images[j]  # (H, W, C)
            output_path = Path(args.output_dir) / f"{batch_size * i + j}.png"
            Image.fromarray(img).save(output_path)
        
if __name__ == '__main__':
    args = parser_args()
    main(args)