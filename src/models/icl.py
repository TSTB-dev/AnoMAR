from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from .vision_transformer import VisionTransformerEncoder, VisionTransformerPredictor
# from .backbone import BackboneModel

import sys
sys.path.append("src/models")

from vision_transformer import mask_to_indices, get_unmasked_indices
from vision_transformer import MultiHeadAttentionBlock, FeedForwardBlock, PosEmbedding


class ICLContextEncoder(nn.Module):
    def __init__(
        self,
        num_patches: int,
        in_channels: int,
        num_blocks: int,
        embed_dim: int,
        num_heads: int,
        num_context_tokens: int,
        out_channels: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        layer_norm = nn.LayerNorm,
    ):
        super(ICLContextEncoder, self).__init__()
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_context_tokens = num_context_tokens
        self.out_channels = out_channels
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate  
        self.attn_drop_rate = attn_drop_rate
        self.layer_norm = layer_norm
        
        num_tokens = num_patches + num_context_tokens
        
        self.in_proj = nn.Linear(in_channels, embed_dim)
        self.pos_embed = PosEmbedding(embed_dim, num_patches)
        self.instance_embed = nn.Parameter(torch.randn(1000, embed_dim))
        self.context_tokens = nn.Parameter(torch.randn(num_context_tokens, embed_dim))
        
        self.blocks = nn.ModuleList([
            nn.ModuleList([MultiHeadAttentionBlock(
                embed_dim,
                num_heads,
            ), FeedForwardBlock(
                embed_dim,
                int(embed_dim * mlp_ratio),
                embed_dim,
            )])
            for _ in range(num_blocks)
        ])
        self.out_proj = nn.Linear(embed_dim, out_channels)
        
        self.layer_norm = layer_norm(embed_dim)
        
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        
    def forward(self, x):
        """
        x: (B, K, N, C)
        """
        B, K, N, C = x.shape
        
        # Pos embedding
        x = rearrange(x, "B K N C -> (B K) N C")
        x = self.in_proj(x)
        pos_embed = self.pos_embed(x)
        x = rearrange(x + pos_embed, "(B K) N C -> B K N C", B=B, K=K)
        # Instance embedding
        ins_embed = self.instance_embed.unsqueeze(0).expand(B, -1, -1).unsqueeze(2).expand(-1, -1, N, -1)
        x = x + ins_embed[:, 1:K+1]  
        
        x = rearrange(x, "B K N C -> B (K N) C")
        
        # Concat Context tokens
        cts = self.context_tokens.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cts, x], dim=1)  # (B, L + KN, C)
        
        attn_ws = []
        for attn, ffn in self.blocks:
            res = x
            x, attn_w = attn(self.layer_norm(x))
            attn_ws.append(attn_w)
            x = x + res
            
            res = x
            x = ffn(self.layer_norm(x))
            x = x + res
        
        # Extract context tokens
        out = x[:, :self.num_context_tokens]
        
        # out proj
        out = self.out_proj(out)
        
        return out, attn_ws
            
        