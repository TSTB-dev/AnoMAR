from .mlp import SimpleMLPAdaLN
from .unet import UNetModel 
from .dit import DiT, ICLDiT
from .vae import AutoencoderKL
from .mim import mim_tiny, mim_small, mim_base, mim_large, mim_huge, mim_gigant, predictor_tiny, \
    predictor_small, predictor_base, predictor_large, predictor_huge, predictor_gigant, PREDICTOR_SUPPORTEED_MODELS, MIM_SUPPORTEED_MODELS, MaskedImageModelingModel, EncoderDecoerMAR, \
        mar_tiny, mar_small, mar_base, mar_large, mar_huge, mar_gigant, emar_tiny, emar_small, emar_base, emar_large, emar_huge, emar_gigant, EMAR_SUPPORTEED_MODELS, EncoderMAR, get_unmasked_indices
from . import mim
from .icl import ICLContextEncoder
from .unet_ddad import UNetModel as UNetModelDDAD
def create_emar_model(denoiser, **kwargs):
    model_type = kwargs['model_type']
    assert model_type in MIM_SUPPORTEED_MODELS, f"Model {model_type} not supported"
    if 'tiny' in model_type:
        return emar_tiny(denoiser, **kwargs)
    elif 'small' in model_type:
        return emar_small(denoiser, **kwargs)
    elif 'base' in model_type:
        return emar_base(denoiser, **kwargs)
    elif 'large' in model_type:
        return emar_large(denoiser, **kwargs)
    elif 'huge' in model_type:
        return emar_huge(denoiser, **kwargs)
    elif 'gigant' in model_type:
        return emar_gigant(denoiser, **kwargs)

def create_mar_model(denoiser, **kwargs):
    model_type = kwargs['model_type']
    assert model_type in MIM_SUPPORTEED_MODELS, f"Model {model_type} not supported"
    if 'tiny' in model_type:
        return mar_tiny(denoiser, **kwargs)
    elif 'small' in model_type:
        return mar_small(denoiser, **kwargs)
    elif 'base' in model_type:
        return mar_base(denoiser, **kwargs)
    elif 'large' in model_type:
        return mar_large(denoiser, **kwargs)
    elif 'huge' in model_type:
        return mar_huge(denoiser, **kwargs)
    elif 'gigant' in model_type:
        return mar_gigant(denoiser, **kwargs)

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
    patch_size: int = 1,
    num_heads: int = 8,
    mlp_ratio: int = 4,
    class_dropout_prob: float = 0.,
    num_classes: int = 15,
    learn_sigma: bool = False,
    grad_checkpoint: bool = False, 
    conditioning_scheme: str = 'none',
    pos_embed = None,
    channel_mult = (1, 1, 2, 2),
    **kwargs
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
        # return UNetModel(
        #     image_size=256,
        #     in_channels=3,
        #     model_channels=256,
        #     out_channels=3,
        #     num_res_blocks=2,
        #     attention_resolutions=[],
        #     channel_mult=(1, 1, 2, 2, 4, 4),
        #     num_heads=16,
        #     num_heads_upsample=-1,
        #     use_fp16=False,
        # )
        return UNetModel(
            image_size=in_res,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_blocks,
            attention_resolutions=[2, 4, 8],
            channel_mult=channel_mult,
            num_heads=16,
            num_heads_upsample=-1,
            use_fp16=False,
        )
        # return UNetModelDDAD(
        #     in_res,
        #     64,
        #     dropout=0.3,
        #     n_heads=2,
        #     in_channels=in_channels,
        # )
    elif model_type == "dit":
        return DiT(
            input_size=in_res,
            patch_size=patch_size,
            in_channels=in_channels,
            cond_channels=z_channels,
            hidden_size=model_channels,
            depth=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
            conditioning_scheme=conditioning_scheme,
            pos_embed=pos_embed
        )
    elif model_type == "icldit":
        return ICLDiT(
            input_size=in_res,
            patch_size=patch_size,
            in_channels=in_channels,
            cond_channels=z_channels,
            hidden_size=model_channels,
            depth=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            learn_sigma=learn_sigma,
            conditioning_scheme=conditioning_scheme,
            pos_embed=pos_embed
        )
    else:
        raise ValueError(f"Model type {model_type} not supported")