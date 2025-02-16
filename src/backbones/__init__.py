import sys
sys.path.append('./src/backbones')

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return concat_y
    
if __name__ == "__main__":
    model = get_backbone('efficientnet-b4')
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)