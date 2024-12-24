"""
We modify Diffusion Transformer which is introduced in the paper "Scalable Diffusion Models with Transformers", William Peebles, Saining Xe. 
The original code is available at https://github.com/facebookresearch/DiT/blob/main/models.py.
"""

import torch
from torch import Tensor, nn
import numpy as np
import math
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from models.vision_transformer import PosEmbedding

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embed_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embed_size = frequency_embed_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embed_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create a sinusodial timestep embedding.
        Args:
            t (Tensor): tensor of shape (B, )
            dim (int): embedding dimension
            max_period (int): maximum period
        Returns:
            Tensor: tensor of shape (B, D)
        """
        half = dim // 2
        freqs = torch.exp(
            -1 * math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)  # (B, 2/D)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, D)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # (B, D)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embed_size)  # (B, D)
        t_emb = self.mlp(t_freq)
        return t_emb
    
class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        
        use_cfg_embedding = dropout_prob > 0.0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)  # we add cfg embedding which represent "no class information"
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops label information to enable CFG. 
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob  # (B, )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_drop_out = train and self.dropout_prob > 0.0
        if use_drop_out or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
class ConditionEmbedder(nn.Module):
    """Embeds other conditional variables into vector representations."""
    def __init__(self, input_channels, hidden_size):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    
    def forward(self, x):
        return self.mlp(x)
    

class CrossAttention(nn.Module):
    """
    Cross-attention module expanded from timm's Attention module.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, y):
        """Compute cross-attention.
        Args: 
            x (Tensor): input tensor (B, N, C), used as query
            y (Tensor): input tensor (B, M, C), used as key and value
        Returns:
            Tensor: output tensor (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], self.k_norm(qkv[1]), self.q_norm(qkv[2])  # q, k, v: (B, num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, num_heads, N, M)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4., **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    
    def forward(self, x, c, z=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # each: (B, C)
        if z is not None:
            assert z.shape[2] == x.shape[2], "Input and cross-attention tensors must have the same channel dimension."
            x = torch.cat([x, z], dim=1)  # (B, N + M, C)
        
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiTBlockWithCrossAttention(nn.Module):
    """
    A DiT block with adaptive layer norm zero conditioning and cross-attention.
    Norm => Self-Attention => Norm => Cross-Attention => Norm => MLP
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4., **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
    
    def forward(self, x, c, y):
        """
        Args:
            x (Tensor): input tensor (B, N, C)
            c (Tensor): conditioning tensor (B, D)
            y (Tensor): other conditioning tensor (B, M, C)
        Returns:
            Tensor: output tensor (B, N, C)
        """
        assert x.shape[2] == y.shape[2], "Input and cross-attention tensors must have the same channel dimension."
        # Compute conditioning 
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        # each: (B, C) x 9
        
        # self-attention
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # cross-attention
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), shift_mca, scale_mca), modulate(self.norm2(y), shift_mca, scale_mca))
        # mlp
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    


class DiT(nn.Module):
    """
    Diffusion Transformer expanded for various conditioning schemes.
    """
    def __init__(
        self, 
        input_size=224,
        patch_size=16, 
        in_channels=3, 
        hidden_size=384,
        depth=4,
        num_heads=8,
        mlp_ratio=4.,
        conditioning_scheme='cross_attention',  # one of ['cross_attention', 'self_attention', 'none', 'adaLN]
        class_dropout_prob=0.0,
        num_classes=15,
        learn_sigma=False,
        pos_embed: PosEmbedding = None
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.conditioning_scheme = conditioning_scheme
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma
        
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.x_embedder_linear = nn.Linear(in_channels, hidden_size, bias=True)
        self.z_embedder = ConditionEmbedder(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        if pos_embed is None:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            import pdb; pdb.set_trace()
            assert pos_embed.shape == (1, num_patches, hidden_size), "pos_embed shape must be (1, N, C)"
            self.pos_embed = nn.Parameter(pos_embed)
        
        self.blocks = nn.ModuleList([
            DiTBlockWithCrossAttention(hidden_size, num_heads, mlp_ratio=mlp_ratio) if conditioning_scheme == 'cross_attention' else DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.inherit_pos_embed = None
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def unpatchify(self, x):
        """Convert sequence of tokens into image-like tensor.
        Args:
            x (Tensor): tensor of shape (B, N, patch_size*patch_size*in_channels)
        Returns:
            Tensor: tensor of shape (B, C, H, W)
        """
        C = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=p, p2=p, c=C)
        return x

    def forward(self, x, t, c, z = None, mask_indices=None):
        """Apply model to an input batch.
        Args:
            x (Tensor): input tensor (B, C, H, W)
            t (Tensor): timestep tensor (B, )
            c (Tensor): label tensor (B, )
            z (Tensor): other conditioning variables, e.g., MIM predictor's outptus (B, V, D)
            mask (Tensor): mask tensor (B, N), specifying which tokens are masked
        Returns:
            Tensor: output tensor (B, C, H, W) or (B, M, C)
        """
        if len(x.shape) == 4:
            # x: (B, C, H, W) -> (B, N, C)  
            # Apply patch embedding and add positional embedding 
            # we assume this type of input is VAE latents
            x = self.x_embedder(x) + self.pos_embed
        elif len(x.shape) == 3:
            # x: (B, N, C)
            # We assume this type of input is already patchified image, i.e., MIM model's outputs. 
            x = self.x_embedder_linear(x)
            x = self.pos_embed(x, apply_indices=mask_indices) if self.inherit_pos_embed is None else self.inherit_pos_embed(x, apply_indices=mask_indices)
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")
        
        _, N, _ = x.shape
        t = self.t_embedder(t)  # (B, D)

        cond = t + c
        if self.conditioning_scheme == 'adaLN':
            assert z.shape[1] == t.shape[1], "conditioning tensor must have the same channel dimension with timeembedding"
            cond = cond + z
        
        if z is not None:
            z = self.z_embedder(z)
        
        for block in self.blocks:
            x = block(x, cond, z)  # (B, N, C)
            
        x = self.final_layer(x, cond)  # (B, N, C)
        if self.conditioning_scheme == 'self_attention':
            x = x[:, :N]  # First N tokens are the original tokens, see DiTBlock. 
            return x  # (B, M, C)
        elif self.conditioning_scheme == 'cross_attention' or self.conditioning_scheme == 'adaLN':
            return x  # (B, M, C)
        else:
            return self.unpatchify(x)  # (B, C, H, W)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)