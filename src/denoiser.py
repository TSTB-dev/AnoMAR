import math

import numpy as np
import torch
from torch import nn
from torch import Tensor
import enum

from typing import Tuple

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from diffusion import create_diffusion, SpacedDiffusion
from models import create_denising_model
from models.vision_transformer import PosEmbedding

class Denoiser(nn.Module):
    def __init__(
        self, 
        *,
        model_type: str,
        num_classes: int,
        input_shape: Tuple[int, int, int],
        z_channels: int, 
        depth: int, 
        width: int, 
        num_sampling_steps : int, 
        num_repeat: int = 1,
        grad_checkpoint=False, 
        conditioning_scheme="none",
        pos_embed: PosEmbedding=None,
        patch_size: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        class_dropout_prob: float = 0.,
        learn_sigma: bool = False,
        **kwargs
    ):
        super(Denoiser, self).__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.in_channels = input_shape[0]
        self.in_res = input_shape[1]
        self.z_channels = z_channels
        self.depth = depth
        self.width = width
        self.num_sampling_steps = num_sampling_steps
        self.pos_embed = pos_embed
        self.conditioning_scheme = conditioning_scheme
        self.num_repeat = num_repeat
        
        self.cls_embed = nn.Embedding(num_classes, width)
        
        self.net = create_denising_model(
            model_type=model_type,
            in_channels=self.in_channels,
            in_res=self.in_res,
            model_channels=width,
            out_channels=self.in_channels if not learn_sigma else self.in_channels * 2,
            z_channels=z_channels,
            num_blocks=depth,
            grad_checkpoint=grad_checkpoint,
            conditioning_scheme=conditioning_scheme,
            pos_embed=pos_embed,
            patch_size=patch_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
            **kwargs
        )
        
        self.train_diffusion: SpacedDiffusion = create_diffusion(timestep_respacing="", noise_schedule='linear', learn_sigma=learn_sigma, rescale_learned_sigmas=False)
        self.sample_diffusion: SpacedDiffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule='linear')
        
    def forward(self, target, cls_label=None, z=None, mask_indices=None, z_vis=None):
        """Denoising step for training.
        Args:
            target (Tensor): the target image to denoise (B, c, h, w) or (B, N, C)
            cls_label (Tensor): the class label (B, )
            z (Tensor): the conditioning variable (B, Z)
            mask_indices (Tensor): the mask indices (B, )
        Returns:
            Tensor: the loss
        """
        if z is not None:
            assert mask_indices is not None, "mask_indices must be specified if z is provided"
            assert self.conditioning_scheme != "none", "conditioning_scheme must be specified if z is provided"
            z = torch.repeat_interleave(z, self.num_repeat, dim=0)  # (B*N, ...)
        
        if z_vis is not None:
            z_vis = torch.repeat_interleave(z_vis, self.num_repeat, dim=0)  # (B*N, ...)
            
        # repeat target and mask_indices
        target = torch.repeat_interleave(target, self.num_repeat, dim=0)  # (B*N, C, H, W)
        if mask_indices is not None:
            mask_indices = torch.repeat_interleave(mask_indices, self.num_repeat, dim=0)  # (B*N, M)
        
        # class embedding
        cls_embed = None
        if cls_label is not None:
            cls_embed = self.cls_embed(cls_label)  # (B, Z)
            cls_embed = torch.repeat_interleave(cls_embed, self.num_repeat, dim=0)  # (B*N, Z)

        # sample timestep
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0], ), device=target.device)  # (B*N, )
        
        # denoising
        model_kwargs = dict(c=cls_label, z=z, mask_indices=mask_indices, z_vis=z_vis)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict['loss']
        return loss.mean()  # mean over the batch
    
    def sample(self, input_shape, cls_label=None, z=None, mask_indices=None, temperature=1.0, cfg=1.0, z_vis=None, device="cuda", strategy="org"):
        """Denoising step for sampling.
        Args:
            input_shape (Tuple[int, int, int, int]): the input shape (B, C, H, W)
            cls_label (Tensor): the class label (B, )
            z (Tensor): the conditioning variable (B, Z)
            temperature (float): the temperature for sampling
            cfg (float): the cfg scale
        Returns:
            Tensor: the denoised image
        """
        cls_embed = None
        if cls_label is not None:
            cls_embed = self.cls_embed(cls_label)  # (B, Z)
        
        if not cfg == 1.0:
            # do classifer free guidance
            noise = torch.randn(*input_shape).to(device)  # (B//2, C)
            noise = torch.cat([noise, noise], dim=0)  # (B, C)
            model_kwargs = dict(c=cls_embed, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(*input_shape).to(device)  # (B, C, H, W)
            model_kwargs = dict(c=cls_label, z=z, mask_indices=mask_indices, z_vis=z_vis)
            sample_fn = self.net.forward
        
        # sampling loop
        if strategy == "org":
            sample = self.sample_diffusion.p_sample_loop(
                sample_fn,
                noise.shape,
                noise, 
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                temperature=temperature,
                device=device
            )
        elif strategy == "ddim":
            sample = self.sample_diffusion.ddim_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
                eta=0.0
            )
        return sample
    
    def q_sample(self, x_start: Tensor, t: Tensor, noise=None) -> Tensor:
        """Sample from q(x_t | x_{t-1})
        Args:
            x_start (Tensor): the starting image (B, c, h, w)
            t (Tensor): the timestep (B, )
            noise (Tensor): the noise tensor (B, c, h, w)
        Returns:
            Tensor: the sampled image (B, c, h, w) representing x_t
        """
        return self.sample_diffusion.q_sample(x_start, t, noise=noise)
    
    def denoise_from_intermediate(self,  x_t: Tensor, t: Tensor, cls_label=None, z = None, mask_indices=None, cfg=1.0, \
        sampler: str="org", eta: float = 0.0, temperature: float = 1.0, z_vis=None) -> Tensor:
        """Denoise from intermediate state x_t to x_0
        Args:
            x_t (Tensor): the intermediate diffusion latent variables (B, c, h, w)
            t (Tensor): the timestep (B, )
            cls_label (Tensor): the class label (B, )
            z (Tensor): the conditioning variable (B, Z)
            mask_indices (Tensor): the mask indices (B, M)
            cfg (float): the cfg scale
            sampler (str): the sampler type, "org", "ddim"
            eta (float): the eta value for ddim sampling
            temperature (float): the temperature for sampling
        Returns:
            Tensor: the denoised image (B, c, h, w)
        """
        assert torch.where(t == t[0], 1, 0).sum() == t.shape[0], "All timesteps must be the same"

        cls_embed = None
        if cls_label is not None:
            cls_embed = self.cls_embed(cls_label)  # (B, Z)
            
        if not cfg == 1.0:
            # do classifer free guidance
            model_kwargs = dict(c=cls_embed, cfg_scale=cfg, z=z, mask_indices=mask_indices)
            sample_fn = self.net.forward_with_cfg
        else:
            model_kwargs = dict(c=cls_label, z=z, mask_indices=mask_indices, z_vis=z_vis)
            sample_fn = self.net.forward
            
        indices = list(range(t[0].item()))[::-1]
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_t.device)  # (B, )
            if sampler == "org":
                out = self.sample_diffusion.p_sample(
                    sample_fn,
                    x_t,
                    t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    temperature=temperature,
                )
                x_t = out["sample"]
            elif sampler == "ddim":
                out = self.sample_diffusion.ddim_sample(
                    sample_fn,
                    x_t,
                    t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                x_t = out["sample"]
            else:
                raise ValueError(f"Invalid sampler type: {sampler}")
        return x_t
    
    def ddim_reverse_sample(self, x_t: Tensor, t: Tensor, cls_label=None, z=None, mask_indices=None, cfg=1.0, eta=0.0, z_vis=None) -> Tensor:
        assert torch.where(t == t[0], 1, 0).sum() == t.shape[0], "All timesteps must be the same"
        
        cls_embed = None
        if cls_label is not None:
            cls_embed = self.cls_embed(cls_label)
        
        if not cfg == 1.0:
            # do classifer free guidance
            model_kwargs = dict(c=cls_embed, cfg_scale=cfg, z=z, mask_indices=mask_indices)
            sample_fn = self.net.forward_with_cfg
        else:
            model_kwargs = dict(c=cls_label, z=z, mask_indices=mask_indices, z_vis=z_vis)
            sample_fn = self.net.forward
        
        indices = list(range(t[0].item(), int(self.num_sampling_steps)))
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_t.device)
            out = self.sample_diffusion.ddim_reverse_sample(
                sample_fn,
                x_t,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                eta=eta
            )
            x_t = out["sample"]
        return x_t
    
    def p_sample(self, x_t: Tensor, t: Tensor, cls_label=None, z=None, mask_indices=None, cfg=1.0, z_vis=None, return_noise=False) -> Tensor:
        cls_embed = None
        if cls_label is not None:
            cls_embed = self.cls_embed(cls_label)
        
        if not cfg == 1.0:
            # do classifer free guidance
            model_kwargs = dict(c=cls_embed, cfg_scale=cfg, z=z, mask_indices=mask_indices)
            sample_fn = self.net.forward_with_cfg
        else:
            model_kwargs = dict(c=cls_label, z=z, mask_indices=mask_indices, z_vis=z_vis)
            sample_fn = self.net.forward
        
        out = self.sample_diffusion.p_sample(
            sample_fn,
            x_t,
            t,
            clip_denoised=False,
            model_kwargs=model_kwargs
        )
        if not return_noise:
            return out["eps"]
        return out["sample"]
            
    
def get_denoiser(**kwargs) -> Denoiser:
    return Denoiser(**kwargs)