
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
from pathlib import Path

import signal
import copy
from einops import rearrange
import argparse
import yaml
from pprint import pprint
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import logging

from datasets import build_dataset, AD_CLASSES, LOCO_CLASSES, ICLDataLoader
from utils import get_optimizer, get_lr_scheduler
from models import create_vae, AutoencoderKL, create_emar_model, EncoderMAR, get_unmasked_indices
from denoiser import get_denoiser, Denoiser
from backbones import get_backbone
from mask import RandomMaskCollator, BlockRandomMaskCollator, CheckerBoardMaskCollator, indices_to_mask, mask_to_indices

def parse_args():
    parser = argparse.ArgumentParser(description="ICL Training")
    
    parser.add_argument('--devices', nargs='+', type=int, default=[0], help='GPU ids to use')
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')

    args = parser.parse_args()
    return args

def init_distributed(port=12345, rank_and_world_size=(None, None)):
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    
    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'
    
    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logging.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank
    try:
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
    except Exception as e:
        world_size, rank = 1, 0
        logging.info(f'distributed training not available {e}')
    
    return world_size, rank

def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def shutdown(signal, frame):
    logging.info("Shutting down...")
    dist.destroy_process_group()
    sys.exit(0)
    

def main(rank, config, num_gpus, devices, ds_list, ds_indices):

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank])
    
    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    
    world_size, rank = init_distributed(rank_and_world_size=(rank, num_gpus))
    signal.signal(signal.SIGINT, shutdown)
    
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logging.info(f"Config: {config}")
    
    # set seed
    seed = config['meta']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['data']['batch_size']
    
    # build backbone model
    if 'vae' in config['backbone']['model_type']:
        backbone: AutoencoderKL = create_vae(**config['backbone'])
    else:
        backbone = get_backbone(**config['backbone'])
    backbone.to(device).eval()
    
    for param in backbone.parameters():
        param.requires_grad = False
        
    mask_strategy = config['data']['mask']['strategy']
    mask_ratio = config['data']['mask']['ratio']
    
    backbone_embed_dim = config['backbone']['embed_dim']
    backbone_stride = config['backbone']['stride']
    img_size = config['data']['img_size']
    in_sh = (backbone_embed_dim, img_size // backbone_stride, img_size // backbone_stride)
    
    # build mim model
    model = get_denoiser(**config['diffusion'], input_shape=in_sh).to(device)
    ddp_model = DDP(model, static_graph=True)
    
    if mask_strategy == "random":
        mask_collator = RandomMaskCollator(
            input_size=in_sh[1], patch_size=1, **config['data']['mask']
        )
    elif mask_strategy == "block":
        mask_collator = BlockRandomMaskCollator(
            input_size=in_sh[1], patch_size=1, mask_ratio=mask_ratio, **config['data']['mask']
        )
    elif mask_strategy == "checkerboard":
        mask_collator = CheckerBoardMaskCollator(
            input_size=in_sh[1], patch_size=1, **config['data']['mask']
        )
    else:
        raise ValueError(f"Invalid mask strategy: {mask_strategy}")
    
    # train_loader = ICLDataLoader(ds_list, mask_collator, batch_size)
    ds_list = [ds_list[i] for i in ds_indices]
    samplers = [DistributedSampler(ds, num_replicas=num_gpus, rank=rank, shuffle=True) for ds in ds_list]
    loaders = [DataLoader(ds, batch_size=batch_size, sampler=sampler, collate_fn=mask_collator) for ds, sampler in zip(ds_list, samplers)]

    optimizer = get_optimizer(ddp_model, **config['optimizer'])
    scheduler = get_lr_scheduler(optimizer, **config['optimizer'])

    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(save_dir))

    # save config
    save_path = save_dir / "config.yaml"
    with open(save_path, 'w') as f:
        yaml.dump(config, f)
    logging.info(f"Config is saved at {save_path}")
    
    ddp_model.train()
    try:
        for epoch in range(config['optimizer']['num_epochs']):
            
            # Set random seed for dataloaders
            for loader in loaders:
                loader.sampler.set_epoch(epoch)
            iterators = [iter(loader) for loader in loaders]
            num_loaders = len(loaders)
            total_steps = sum([len(loader) for loader in loaders])
            
            for i in range(total_steps):
                loader_idx = i % num_loaders
                try:
                    data, mask_indices = next(iterators[loader_idx])
                except StopIteration:
                    continue
                
                img, labels = data["samples"], data["clslabels"]    # (B, C, H, W), (B,)
                mask_indices = mask_indices.to(device)
                img = img.to(device)
                labels = labels.to(device)
                
                # forward
                with torch.no_grad():
                    if 'vae' in config['backbone']['model_type']:
                        posterior = backbone.encode(img)  # (B, c, h, w)
                        x = posterior.sample().mul_(0.2325)  # (B, c, h, w)
                    else:
                        x = backbone(img)  # (B, c, h, w)
                
                # Rollout -> (B, B, c, h, w)
                # We assume the first element is target to masked image prediction
                # and the rest are for the context images
                # [[x_0, x_1, ..., x_B], [x_1, x_2, ..., x_B, x_0], ..., [x_B, x_0, ..., x_B-1]]
                rolled_x = torch.stack([torch.roll(x, -i, dims=1) for i in range(len(img))])
                rolled_x = rearrange(rolled_x, "b1 b2 c h w -> b1 b2 (h w) c")
                target_x = rearrange(x.clone().detach(), "b c h w -> b (h w) c")
                unmasked_indices = get_unmasked_indices(mask_indices, target_x.size(1))
                
                # Get the masked target and unmasked target
                # (B, M, C), (B, V, C)
                # TODO: we don't use unmasked region of target images. 
                masked_target_x = torch.gather(target_x, 1, mask_indices.unsqueeze(-1).expand(-1, -1, target_x.size(-1)))
                unmasked_target_x = torch.gather(target_x, 1, unmasked_indices.unsqueeze(-1).expand(-1, -1, target_x.size(-1)))
                
                # Get model inputs
                # condition: (B, B-1, N, C)
                # target: (B, M, C)
                cond = rolled_x[:, 1:, ...]
                loss = ddp_model(masked_target_x, z=cond, mask_indices=mask_indices)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if torch.isnan(loss):
                    logging.info("Loss is NaN")
                    exit()
                
                if i % config["logging"]["log_interval"] == 0:
                    logging.info(f"Epoch {epoch}, Iter {i}, Loss {loss.item():.5f}")      
                    tb_writer.add_scalar("Loss", loss.item(), epoch * total_steps + i)  
                    
                    if config["logging"]["save_images"]:
                        pass
                        # save_images(model_ema, vae, tb_writer, epoch, i, device)
            
            if (epoch + 1) % config["logging"]["save_interval"] == 0:
                save_path = save_dir / f"model_latest.pth"
                torch.save(model.state_dict(), save_path)
                logging.info(f"Model is saved at {save_dir}")
            
            scheduler.step()
    except KeyboardInterrupt:
        logging.info("Training is interrupted!")
    finally:
        dist.destroy_process_group()

    logging.info("Training is done!")
    tb_writer.close()
    
    # save model
    save_path = save_dir / "model_latest.pth"
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model is saved at {save_dir}")
    
def save_images(model, vae, tb_writer, epoch, i, device):
    pass

if __name__ == "__main__":
    import multiprocessing as mp
    args = parse_args()
    num_gpus = len(args.devices)
    config = load_config(args.config_path)
    
    # assign datasets in a way that each process has a different datasets
    ## Note: we exclude one class for evaluation
    exclude_class = config['data']['exclude_class']
    ds_list = []
    class_names = []
    if config['data']['dataset_name'] == 'mvtec_ad':
        num_classes = len(AD_CLASSES) 
        class_names = AD_CLASSES
    elif config['data']['dataset_name'] == 'mvtec_loco':
        num_classes = len(LOCO_CLASSES) 
        class_names = LOCO_CLASSES 
    else:
        raise ValueError(f"Invalid dataset: {config['data']['dataset_name']}")
    
    for i in range(num_classes):
        if class_names[i] != exclude_class:
            ds = build_dataset(**config['data'], category=class_names[i])
            ds_list.append(ds)

    mp.set_start_method('spawn', True)
    
    processes = []
    for rank in args.devices:
        ds_indices = [i for i in range(len(ds_list)) if i % num_gpus == rank]
        p = mp.Process(target=main, args=(rank, config, num_gpus, args.devices, ds_list, ds_indices))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
            