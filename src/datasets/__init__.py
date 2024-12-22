

from torchvision import transforms

from .mnist import MNISTWrapper
from .mvtec_ad import MVTecAD
from .mvtec_loco import MVTecLOCO

def build_transforms(img_size, transform_type):
    # standarization
    default_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    if transform_type == 'default':
        return default_transform
    elif transform_type == 'imagenet':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if transform_type == 'crop':
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Invalid transform: {transform_type}")

def build_dataset(*, dataset_name: str, data_root: str, train: bool, img_size: int, transform_type: str, **kwargs):
    if dataset_name == 'mnist':
        return MNISTWrapper(root=data_root, train=train, transform=build_transforms(img_size, transform_type))
    elif dataset_name == 'mvtec_ad':
        return MVTecAD(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
            transform=build_transforms(img_size, transform_type), is_mask=True, cls_label=True, **kwargs)
    