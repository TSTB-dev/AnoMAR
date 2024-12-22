from .mlp import SimpleMLPAdaLN
from .unet import Unet
from .dit import DiT
from .vae import AutoencoderKL

def create_vae(
    model_type: str,
    embed_dim = 16,
    ch_mult = (1, 1, 2, 2, 4),
    ckpt_path = None,
    **kwargs
):
    assert ckpt_path is not None, "Checkpoint path must be provided"
    if model_type == "vae_kl":
        return AutoencoderKL(
            embed_dim=embed_dim,
            ch_mult=ch_mult,
            ckpt_path=ckpt_path,
        )

def create_denising_model(
    model_type: str,
    in_channels: int,
    in_res: int,
    model_channels: int,
    out_channels: int,
    z_channels: int,
    num_blocks: int,
    grad_checkpoint: bool = False
):
    if model_type == "mlp":
        return SimpleMLPAdaLN(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            z_channels=z_channels,
            num_blocks=num_blocks,
            grad_checkpoint=grad_checkpoint
        )
    elif model_type == "unet":
        return Unet(
            num_timesteps=1000,
            time_embed_dim=128,
            in_channels=in_channels,
            out_channels=out_channels,
            dim_mults=[2, 4, 8],
        )
    elif model_type == "dit":
        return DiT(
            input_size=in_res,
            patch_size=2,
            in_channels=in_channels,
            hidden_size=model_channels,
            depth=num_blocks,
            num_heads=8,
            mlp_ratio=4,
            class_dropout_prob=0.,
            num_classes=15,
            learn_sigma=False,
        )
    else:
        raise ValueError(f"Model type {model_type} not supported")