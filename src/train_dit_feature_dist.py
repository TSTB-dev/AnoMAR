
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
from models import create_vae, AutoencoderKL
from backbones import get_backbone

from einops import rearrange

import torch.distributed as dist
import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import wandb
os.environ["WANDB_SILENT"] = "true"

from dotenv import load_dotenv
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

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
            logger.info('SLURM vars not set (distributed training not available)')
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
        logger.info(f'distributed training not available {e}')
    
    return world_size, rank


def parse_args():
    parser = argparse.ArgumentParser(description="AnoMAR Training")
    
    parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'], help='GPU devices')
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

def main(rank, args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.devices[rank].split(":")[-1])
    
    def load_config(config_path):
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.info(exc)
        return config

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    assert os.path.exists(args.config_path), f"Config file not found: {args.config_path}"

    world_size, rank = init_distributed(rank_and_world_size=(rank, len(args.devices)))
    logger.info(f"Running... (rank: {rank}/{world_size})")
    
    config = load_config(args.config_path)
    pprint(config)
    # create wandb project
    if rank == 0:
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
    anom_datasets = anom_dataset.datasets
    normal_datasets = normal_dataset.datasets

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, \
        pin_memory=config['data']['pin_memory'], num_workers=config['data']['num_workers'], drop_last=False)
    train_loader_ddp = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )
    
    vae: AutoencoderKL = create_vae(**config['vae'])
    vae.to(device).eval()
    vae_embed_dim = config['vae']['embed_dim']
    vae_stride = config['vae']['stride']
    img_size = config['data']['img_size']
    # diff_in_sh = (vae_embed_dim, img_size // vae_stride, img_size // vae_stride)
    # diff_in_sh = (3, img_size, img_size)
    diff_in_sh = (272, 16, 16)
    
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    model_ema = copy.deepcopy(model)
    model_without_ddp = model
    model_without_ddp.to(device)
    model_ema.to(device)
    model.to(device)
    model = DistributedDataParallel(model, static_graph=True)
    ema_decay = config['diffusion']['ema_decay']
    # feature_ln = torch.nn.LayerNorm(diff_in_sh[0]).to(device)

    model_kwargs = {
        'model_type': 'efficientnet-b4',
        'outblocks': (1, 5, 9, 21),
        'outstrides': (2, 4, 8, 16),
        'pretrained': True,
        'stride': 16
    }
    logger.info(f"Using feature space reconstruction with {model_kwargs['model_type']} backbone")
    
    feature_extractor = get_backbone(**model_kwargs)
    feature_extractor.to(device).eval()

    # optimizer = get_optimizer([model, feature_ln], **config['optimizer'])
    optimizer = get_optimizer([model], **config['optimizer'])
    if config['optimizer']['scheduler_type'] == 'none':
        pass
    else:
        scheduler = get_lr_scheduler(optimizer, **config['optimizer'], iter_per_epoch=len(train_loader_ddp))

    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(save_dir))

    # save config
    save_path = save_dir / "config.yaml"
    with open(save_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Config is saved at {save_path}")

    feature_extractor.eval()
    logger.info(f"Computing global feature statistics for {len(train_dataset)} samples")
    features = []
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        img = data["samples"].to(device)
        with torch.no_grad():
            x, _ = feature_extractor(img)
            features.append(x)
    features = torch.cat(features, dim=0)   # (N, c, h, w)
    avg_glo = features.mean(dim=(0, 2, 3))  # (c, )
    std_glo = features.std(dim=(0, 2, 3))  # (c, )
    
    model.train()
    logger.info(f"Steps per epoch: {len(train_loader_ddp)}")
    for epoch in range(config['optimizer']['num_epochs']):
        for i, data in enumerate(train_loader_ddp):
            img, labels = data["samples"], data["clslabels"]    # (B, C, H, W), (B,)
            img = img.to(device)
            labels = labels.to(device)
            
            # # vae encode
            with torch.no_grad():
                # posterior = vae.encode(img)  # (B, c, h, w)
                # x = posterior.sample().mul_(0.2325)  # (B, c, h, w)
                x, _ = feature_extractor(img)  # (B, c, h, w)
                
                # Normalize x
                x = (x - avg_glo.view(1, -1, 1, 1)) / std_glo.view(1, -1, 1, 1)
                
                # Logging feature distribution stats
                avg_x = x.mean()
                std_x = x.std()
                min_x = x.min()
                max_x = x.max()
                # tb_writer.add_scalar("Feature/mean", avg_x.mean())
                # tb_writer.add_scalar("Feature/std", std_x.mean())
                # tb_writer.add_scalar("Feature/min", min_x.mean())
                # tb_writer.add_scalar("Feature/max", max_x.mean())
                # wandb.log({"Feature/mean": avg_x.mean(), "Feature/std": std_x.mean()})
                # wandb.log({"Feature/min": min_x.mean(), "Feature/max": max_x.mean()})
                # logger.info(f"Feature stats: mean {avg_x}, std {std_x}, min {min_x}, max {max_x}")
                
                # Normalize x
                # x = x / x.norm(dim=1, keepdim=True)
            # x = rearrange(x, 'b c h w -> b (h w) c')
            # x = feature_ln(x)  # LN
            # x = rearrange(x, 'b (h w) c -> b c h w', h=16, w=16)
            # x = img
            # forward
            # img = _process_inputs(img)
            loss = model(x, labels)  
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # Log gradients stats
            for name, param in model.named_parameters():
                if param.grad is not None and rank == 0:
                    # logger.info(name, param.grad.data.norm(2).item())
                    wandb.log({f"Grad/{name}": param.grad.data.norm(2).item()})
                    
            if config['optimizer']['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['grad_clip'])
            optimizer.step()
            scheduler.step()
            
            # update ema
            if rank == 0:
                for ema_param, model_param in zip(model_ema.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1.0 - ema_decay)
            
            if i % config["logging"]["log_interval"] == 0 and rank == 0:
                logger.info(f"Epoch {epoch}, Iter {i}, Loss {loss.item()}")      
                tb_writer.add_scalar("Loss", loss.item(), epoch * len(train_loader_ddp) + i)  
                wandb.log({"Loss": loss.item(), "LR": scheduler.get_last_lr()})
                
                if config["logging"]["save_images"]:
                    save_images(model_ema, vae, tb_writer, epoch, i, device)
            
            dist.barrier()

        if (epoch + 1) % config["logging"]["save_interval"] == 0 and rank == 0:
            
            save_path = save_dir / f"model_latest.pth"
            torch.save(model_without_ddp.state_dict(), save_path)
            save_path = save_dir / f"model_ema_latest.pth"
            torch.save(model_ema.state_dict(), save_path)
            logger.info(f"Model is saved at {save_dir}")
        
        dist.barrier()
        if (epoch + 1) % config["evaluation"]["eval_interval"] == 0:
            proc_idxs = [i for i in range(len(anom_datasets)) if i % world_size == rank]
            eval_results = {}
            for i in proc_idxs:
                results = evaluate(
                    model_without_ddp,
                    feature_extractor,
                    vae,
                    anom_datasets[i],
                    normal_datasets[i],
                    diff_in_sh,
                    epoch + 1,
                    config["evaluation"]["start_step"],
                    device,
                    (avg_glo, std_glo)
                )
                eval_results.update(results)
            
            if rank == 0:
                gathered_data = [None for _ in range(world_size)]
            else:
                gathered_data = None
            # collect stats
            dist.gather_object(eval_results, gathered_data, dst=0)
            if rank == 0:
                eval_results = {}
                for data in gathered_data:
                    eval_results.update(data)
                logger.info(f"Eval results: {eval_results}")
                wandb.log(eval_results)
                tb_writer.add_scalars("AUC", eval_results, epoch)
                
        dist.barrier()
        
    logger.info("Training is done!")
    tb_writer.close()
    
    # save model
    save_path = save_dir / "model_latest.pth"
    torch.save(model_without_ddp.state_dict(), save_path)
    save_path = save_dir / "model_ema_latest.pth"
    torch.save(model_ema.state_dict(), save_path)
    logger.info(f"Model is saved at {save_dir}")
    
    dist.destroy_process_group()
    
def save_images(model, vae, tb_writer, epoch, i, device):
    pass

@torch.no_grad()
def evaluate(denoiser, feature_extractor, vae, anom_dataset, normal_dataset, in_sh, epoch, start_step, device, \
    global_stats):
    anom_loader = DataLoader(anom_dataset, batch_size=1, shuffle=False, num_workers=1)
    normal_loader = DataLoader(normal_dataset, batch_size=1, shuffle=False, num_workers=1)
    category = normal_dataset.category
    
    avg_glo, std_glo = global_stats
    denoiser.eval()
    feature_extractor.eval()
    
    logger.info(f"Evaluating on {len(anom_loader)} anomalous samples and {len(normal_loader)} normal samples [{category}]")
    normal_scores = []
    for i, batch in enumerate(normal_loader):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        # Prepare timesteps
        t = torch.tensor([start_step] * len(images)).to(device)  # (B, )
        
        def perturb(x, t):
            z, _ = feature_extractor(x)  # (B, c, h, w)
            
            # Normalize z
            z = (z - avg_glo.view(1, -1, 1, 1)) / std_glo.view(1, -1, 1, 1)
            noised_z = denoiser.q_sample(z, t)  # (B, c, h, w)
            
            return noised_z, z, 
        
        noised_latents, org_latents = perturb(images, t)  # (B, c, h, w)
        
        # decode
        def denoising(noised_z, t, labels):
            denoized_z = denoiser.denoise_from_intermediate(noised_z, t, labels)  
            return denoized_z
        denoized_latents = denoising(noised_latents, t, labels)
        
        # calculate scores
        def anomaly_score(x, x_rec):
            diff = torch.mean((x - x_rec).pow(2), dim=1)
            mse = diff.view(-1, in_sh[1], in_sh[2])
            mse = torch.mean(mse, dim=(1, 2))  # (B, )
            # mse = mse.max(dim=1).values  # (B, W)
            # mse = mse.max(dim=1).values  # (B, )
            
            anom_map = torch.mean((x - x_rec).pow(2), dim=1)  # (B*K, H, W)
            anom_map = anom_map.view(-1, *anom_map.shape[1:])
            # anom_map = torch.min(anom_map, dim=1).values  # (B, H, W)
            return mse, anom_map
        
        anom_score, anom_map = anomaly_score(org_latents, denoized_latents)
        normal_scores.append(anom_score)
    
    normal_scores = torch.cat(normal_scores, dim=0)
    
    anom_scores = []
    for i, batch in enumerate(anom_loader):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        # Prepare timesteps
        t = torch.tensor([start_step] * len(images)).to(device)
        
        noised_latents, org_latents = perturb(images, t)  # (B, c, h, w)
        denoized_latents = denoising(noised_latents, t, labels)
        
        anom_score, anom_map = anomaly_score(org_latents, denoized_latents)
        
        anom_scores.append(anom_score)
    anom_scores = torch.cat(anom_scores, dim=0)

    y_true = torch.cat([torch.zeros_like(normal_scores), torch.ones_like(anom_scores)])
    y_score = torch.cat([normal_scores, anom_scores])
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy())
    logger.info(f"AUC: {auc} at epoch {epoch}")
    return {
        f"{category}": auc
    }

if __name__ == "__main__":
    args = parse_args()
    num_gps = len(args.devices)
    mp.set_start_method("spawn", True)
    
    processes = []
    for rank in range(num_gps):
        p = mp.Process(
            target=main,
            args=(rank, args)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


    
    
    
        
      
            
    
        
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
