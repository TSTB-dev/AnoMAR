from .mask_collator import RandomMaskCollator, BlockRandomMaskCollator, CheckerBoardMaskCollator
import torch


def indices_to_mask(mask_indices, L):
    """Convert indices to binary mask.
    Args:
        masks_indices (tensor): The indices of masked patches. shape of (B, M), where M is the number of masked patches.
        L (int): The total number of patches.
    Returns:
        mask (tensor): The binary mask. shape of (B, L), where L is the number of patches.
    """
    B, M = mask_indices.shape
    masks = torch.zeros(B, L, device=mask_indices.device)
    masks.scatter_(dim=1, index=mask_indices, value=True)
    inverse_masks = torch.logical_not(masks).float()
    return masks, inverse_masks

def mask_to_indices(masks):
    """Convert binary mask to indices.
    Args:
        masks (tensor): The binary mask. shape of (B, L), where L is the number of patches.
    Returns:
        mask_indices (tensor): The indices of masked patches. shape of (B, M), where M is the number of masked patches.
    """
    mask_indices_ = torch.nonzero(masks, as_tuple=False)  # (L, 2)
    mask_indices = []
    for i in range(masks.shape[0]):
        mask_idx = mask_indices_[mask_indices_[:, 0] == i, 1]
        mask_indices.append(mask_idx)
    mask_indices = torch.stack(mask_indices, dim=0)
    return mask_indices