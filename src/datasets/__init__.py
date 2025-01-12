

from torchvision import transforms
from torch.utils.data import DataLoader

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
    elif dataset_name == 'mvtec_loco':
        return MVTecLOCO(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
            transform=build_transforms(img_size, transform_type), is_mask=True, cls_label=True, **kwargs)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

class EvalDataLoader:
    def __init__(self, dataset, num_repeat, collate_fn, shared_masks=None):
        self.dataset = dataset
        self.num_repeat = num_repeat
        self.collate_fn = collate_fn
        # We repeat each sample in the dataset for the number of times it is required to be evaluated.
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)  
        self.iterator = iter(self.dataloader)
        
        self.shared_masks = shared_masks
    
    def __iter__(self):
        return self
    
    def __next__(self):
        data = next(self.iterator)  
        
        repeated_data = []
        data["samples"] = data["samples"][0]  # (1, C, H, W) -> (C, H, W)
        data["labels"] = data["labels"]
        data["filenames"] = data["filenames"][0]
        data["clsnames"] = data["clsnames"][0]
        data["anom_type"] = data["anom_type"][0]
        data["clslabels"] = data["clslabels"][0]
        data["masks"] = data["masks"][0]  # gt masks
        for _ in range(self.num_repeat):
            repeated_data.append(data)

        # use shared masks accross all test samples
        collated_batch, collated_masks = self.collate_fn(repeated_data)
        if self.shared_masks is None:
            self.shared_masks = collated_masks
        else:
            collated_masks = self.shared_masks
        
        return collated_batch, collated_masks
    
    def __len__(self):
        return len(self.dataloader)