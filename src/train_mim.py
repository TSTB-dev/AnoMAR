
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
    
    
