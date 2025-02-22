import sys
sys.path.append('./src/backbones')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights, ResNet50_Weights

from .efficientnet import build_efficient
from .resnet import wide_resnet101_2

def get_efficientnet(model_name, **kwargs):
    return build_efficient(model_name, **kwargs)

def get_pdn_small(out_channels=384, padding=False, **kwargs):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )

def get_pdn_medium(out_channels=384, padding=False, **kwargs):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )

def get_backbone_model(model_name):
    if model_name == "vgg19":
        return models.vgg19(weights=VGG19_Weights.DEFAULT)
    elif model_name == "efficientnet-s":
        return models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    elif model_name == "efficientnet-m":
        return models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    elif model_name == "efficientnet-l":
        return models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    elif model_name == "resnet50":
        return models.resnet50(weights=ResNet50_Weights.DEFAULT)
    elif model_name == "identical":
        return nn.Identity()

def get_backbone(**kwargs):
    model_name = kwargs['model_type']
    if 'pdn_small' in model_name:
        return get_pdn_small(**kwargs)
    elif 'pdn_medium' in model_name:
        return get_pdn_medium(**kwargs)
    elif 'efficientnet' in model_name:
        # net = get_efficientnet(model_name, pretrained=True, outblocks=[1, 5, 9, 21], outstrides=[2, 4, 8, 16])
        net =  get_efficientnet(model_name, **kwargs)
        return BackboneWrapper(net, [0.125, 0.25, 0.5, 1.0])
    elif 'vgg' in model_name:
        return BackboneModel(model_name, [3, 8, 17, 26])
    elif 'wide_resnet' in model_name:
        ckpt_path = kwargs.get('ckpt_path', None)
        pretrained = False if ckpt_path is not None else True
        model = wide_resnet101_2(pretrained=pretrained)
        if ckpt_path is not None:
            print(f"Loading checkpoint from {ckpt_path}")
            model_ckpt = torch.load(ckpt_path, weights_only=True)
            for k,v in list(model_ckpt.items()):
                k_new = k.replace("module.","")
                # replace the key name
                model_ckpt[k_new] = model_ckpt.pop(k)
            model.load_state_dict(model_ckpt)
        return model
    else:
        raise ValueError(f"Invalid backbone model: {model_name}")

def get_intermediate_output_hook(layer, input, output):
    BackboneModel.intermediate_cache.append(output)

class BackboneModel(nn.Module):
    intermediate_cache = []
    
    def __init__(self, model_name: str, extract_indices: list, feature_res: int = 64):
        super(BackboneModel, self).__init__()
        self.model_name = model_name
        self.model = get_backbone_model(model_name)
        self.model.eval()
        self.model_name = model_name
        self.extract_indices = extract_indices
        self.feature_res = feature_res
        
        if model_name in ["pdn_small", "pdn_medium"]:
            self.feature_dim = 384
        elif model_name == "identical":
            self.feature_dim = 3
        else:
            self._register_hook()
        
    
    def _register_hook(self):
        self.layer_hooks = []
        feature_dim = 0
        if self.model_name == "vgg19":
            for i, layer_idx in enumerate(self.extract_indices):
                module = self.model.features[layer_idx-1]
                if isinstance(module, nn.Conv2d):
                    feature_dim += module.out_channels
                elif isinstance(module, nn.Sequential):
                    if isinstance(module[-1], nn.SiLU):
                        feature_dim += module[-3].out_channels
                    else:
                        feature_dim += module[-1].out_channels
                layer_to_hook = self.model.features[layer_idx]
                hook = layer_to_hook.register_forward_hook(get_intermediate_output_hook)
                self.layer_hooks.append(hook)
        elif "resnet" in self.model_name:
            for i, layer_idx in enumerate(self.extract_indices):
                module = getattr(self.model, f"layer{layer_idx}")
                feature_dim += module[-1].conv3.out_channels
                layer_to_hook = getattr(self.model, f"layer{layer_idx}")
                hook = layer_to_hook.register_forward_hook(get_intermediate_output_hook)
                self.layer_hooks.append(hook)
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor):
        """Extract features from the backbone model. 
        Args:
            x (torch.Tensor): Input image tensor, shape (B, C, H, W)
            extract_indices (list): List of indices to extract features from the backbone model.
        Returns:
            torch.Tensor: Extracted features, shape (B, C, H', W')
        Examples:
            >>> backbone = get_backbone_model("vgg19", [3, 8, 17, 26])
            >>> features = backbone.extract_features(x)  # x shape (B, 960, 64, 64)
        """
        
        if self.model_name in ["pdn_small", "pdn_medium"]:
            with torch.no_grad():
                features = self.model(x)
                features = nn.functional.interpolate(features, size=(self.feature_res, self.feature_res), mode="bilinear", align_corners=False)
            return features
        
        if self.model_name == "identical":
            return x
            
        with torch.no_grad():
            _ = self.model(x)
        self.intermediate_outputs = BackboneModel.intermediate_cache
        self._reset_cache()
        
        for i, intermediate_output in enumerate(self.intermediate_outputs):
            self.intermediate_outputs[i] = nn.functional.interpolate(intermediate_output, size=(self.feature_res, self.feature_res), mode="bilinear", align_corners=False)
        features = torch.cat(self.intermediate_outputs, dim=1)

        return features

    def _reset_cache(self):
        BackboneModel.intermediate_cache = []

class BackboneWrapper(nn.Module):
    def __init__(self, backbone, scale_factors):
        super(BackboneWrapper, self).__init__()
        self.backbone = backbone
        self.scale_factors = scale_factors
        
        self.downsamples = nn.ModuleList()
        for scale_factor in scale_factors:
            self.downsamples.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear'))
        
    def forward(self, x):
        y = self.backbone(x)["features"]
        concat_y = torch.cat([downsample(y[i]) for i, downsample in enumerate(self.downsamples)], dim=1)
        return concat_y, y
    
if __name__ == "__main__":
    model_kwargs = {
        'model_type': 'efficientnet-b4',
        'outblocks': (1, 5, 9, 21),
        'outstrides': (2, 4, 8, 16),
        'pretrained': True,
        'stride': 16
    }
    model = get_backbone(**model_kwargs).to('cuda')
    x = torch.randn(1, 3, 256, 256).to('cuda')
    with torch.no_grad():
        features, y = model(x)
    print(features[0].shape)
    print(f"Sublevel features: {len(y)}")
    print(len(y))
    for i in range(len(y)):
        print(y[i].shape)
    