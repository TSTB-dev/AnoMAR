import math

import numpy as np
import torch
from torch import Tensor, nn
import enum

from typing import Tuple

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
    
    def forward(self, x: Tensor) -> Tensor:
        """Channel shuffle operation, x : (B, C, H, W)"""
        B, C, H, W = x.shape
        assert C % self.groups == 0, "Number of channels must be divisible by number of groups"
        x = x.view(B, self.groups, C // self.groups, H, W)  # grouping
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)  # shuffling
        return x

class ConvBnSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)
    
class ResBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        half_dim_in = in_channels // 2
        half_dim_out = out_channels // 2
        self.branch1 = nn.Sequential(
            nn.Conv2d(half_dim_in, half_dim_in, 3, 1, 1, groups=half_dim_in),
            nn.BatchNorm2d(half_dim_in),
            ConvBnSiLU(half_dim_in, half_dim_out, 1, 1, 0),
        )
    
        self.branch2 = nn.Sequential(
            ConvBnSiLU(half_dim_in, half_dim_in, 1, 1, 0),
            nn.Conv2d(half_dim_in, half_dim_in, 3, 1, 1, groups=half_dim_in),
            nn.BatchNorm2d(half_dim_in),
            ConvBnSiLU(half_dim_in, half_dim_out, 1, 1, 0),
        )
        self.ch_shuffle = ChannelShuffle(groups=2)
        
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        return self.ch_shuffle(torch.cat([x1, x2], dim=1))
    
class ResDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        half_dim_out = out_channels // 2
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            ConvBnSiLU(in_channels, half_dim_out, 1, 1, 0),
        )
        self.branch2 = nn.Sequential(
            ConvBnSiLU(in_channels, half_dim_out, 1, 1, 0),
            nn.Conv2d(half_dim_out, half_dim_out, 3, 2, 1, groups=half_dim_out),
            nn.BatchNorm2d(half_dim_out),
            ConvBnSiLU(half_dim_out, half_dim_out, 1, 1, 0),
        ) 
        self.ch_shuffle = ChannelShuffle(groups=2)
    
    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return self.ch_shuffle(torch.cat([x1, x2], dim=1))

class TimeStepEmbedder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, out_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.act = nn.SiLU()
    
    def forward(self, x, t):
        t_emb = self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb
        return self.act(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.conv0 = nn.Sequential(
            *[
                ResBottleneck(in_channels, in_channels)
                for i in range(3)],
            ResBottleneck(in_channels, out_channels//2)
        )
        self.time_mlp = TimeStepEmbedder(embed_dim=time_embed_dim, hidden_dim=out_channels, out_dim=out_channels//2)
        self.conv1 = ResDownsample(out_channels//2, out_channels)
        
    def forward(self, x, t=None):
        x_shortcut = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x_shortcut, t)
        x = self.conv1(x_shortcut)
        return [x, x_shortcut]
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv0 = nn.Sequential(
            *[
                ResBottleneck(in_channels, in_channels)
                for i in range(3)],
            ResBottleneck(in_channels, in_channels//2)
        )
        self.time_mlp = TimeStepEmbedder(embed_dim=time_embed_dim, hidden_dim=in_channels, out_dim=in_channels//2)
        self.conv1 = ResBottleneck(in_channels//2, out_channels//2)
    
    def forward(self, x, x_shortcut, t=None):
        x = self.upsample(x)
        x = torch.cat([x, x_shortcut], dim=1)
        x = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x, t)
        x = self.conv1(x)
        return x

class Unet(nn.Module):
    def __init__(self, num_timesteps, time_embed_dim, in_channels=3, out_channels=2, \
        base_dim=32, dim_mults=[2, 4, 8, 16]):
        super().__init__()
        assert isinstance(dim_mults, (list, tuple)), "dim_mults must be a list or tuple"
        assert base_dim % 2 == 0, "Base dimension must be divisible by 2"
        
        channels = self._cal_channels(base_dim, dim_mults)
        
        self.init_conv = ConvBnSiLU(in_channels, base_dim, 3, 1, 1)
        self.time_embed = nn.Embedding(num_timesteps, time_embed_dim)
        
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(c[0], c[1], time_embed_dim) for c in channels
        ])    
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(c[1], c[0], time_embed_dim) for c in channels[::-1]
        ])
        
        self.mid_block = nn.Sequential(
            *[ResBottleneck(channels[-1][1], channels[-1][1]) for _ in range(2)],
            ResBottleneck(channels[-1][1], channels[-1][1]//2)
        )
        
        self.final_conv = nn.Conv2d(in_channels=channels[0][0]//2, out_channels=out_channels, kernel_size=1)
    
    def _cal_channels(self, base_dim, dim_mults):
        dims=[base_dim * x for x in dim_mults]
        dims.insert(0, base_dim)
        channels = []
        for i in range(len(dims) - 1):
            channels.append((dims[i], dims[i+1]))
        return channels
    
    def forward(self, x, t=None, **kwargs):
        x = self.init_conv(x)
        if t is not None:
            t = self.time_embed(t)
        enc_shortcuts = []
        for block in self.encoder_blocks:
            x, x_shortcut = block(x, t)
            enc_shortcuts.append(x_shortcut)
        x = self.mid_block(x)
        enc_shortcuts = enc_shortcuts[::-1]
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, enc_shortcuts[i], t)
        return self.final_conv(x)
    
    def forward_with_cfg(self, x, t=None, c=None, cfg_scale=1.0):
        half = x[:len(x)//2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t=t, c=c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]  
        cond_eps, uncond_eps = torch.split(eps, len(eps)//2, dim=0)  # (B/2, C)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)