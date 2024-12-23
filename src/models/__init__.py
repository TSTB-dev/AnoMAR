from .mlp import SimpleMLPAdaLN
from .unet import Unet
from .dit import DiT
from .vae import AutoencoderKL
from .mim import mim_tiny, mim_small, mim_base, mim_large, mim_huge, mim_gigant, predictor_tiny, \
    predictor_small, predictor_base, predictor_large, predictor_huge, predictor_gigant, PREDICTOR_SUPPORTEED_MODELS, MIM_SUPPORTEED_MODELS, MaskedImageModelingModel
from . import mim

def create_mim_model(model_type, **kwargs):
    assert model_type in MIM_SUPPORTEED_MODELS, f"Model {model_type} not supported"
    if 'tiny' in model_type:
        return mim_tiny(**kwargs)
    elif 'small' in model_type:
        return mim_small(**kwargs)
    elif 'base' in model_type:
        return mim_base(**kwargs)
    elif 'large' in model_type:
        return mim_large(**kwargs)
    elif 'huge' in model_type:
        return mim_huge(**kwargs)
    elif 'gigant' in model_type:
        return mim_gigant(**kwargs)

def create_predictor_model(model_type: str, **kwargs):
    assert model_type in PREDICTOR_SUPPORTEED_MODELS, f"Model {model_type} not supported"
    return getattr(mim, model_type)

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