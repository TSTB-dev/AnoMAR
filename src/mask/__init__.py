from .mask_collator import RandomMaskCollator, BlockRandomMaskCollator, CheckerBoardMaskCollator
import torch


def indices_to_mask(mask_indices, L):
    """Convert indices to binary mask keeping it's orders.
    Args:
        masks_indices (tensor): The indices of masked patches. shape of (B, M), where M is the number of masked patches.
        L (int): The total number of patches.
    Returns:
        mask (tensor): The binary mask. shape of (B, L), where L is the number of patches.
    """
    B, M = mask_indices.shape
    masks = torch.zeros(B, L, device=mask_indices.device, dtype=torch.long)
    for b in range(B):
        for order, idx in enumerate(mask_indices[b]):
            masks[b, idx] = order + 1 
    inverse_masks = (masks != 0)
    return inverse_masks.bool()

def mask_to_indices(masks):
    """Convert binary mask to indices keeping it's orders.
    Args:
        masks (tensor): The binary mask. shape of (B, L), where L is the number of patches.
    Returns:
        mask_indices (tensor): The indices of masked patches. shape of (B, M), where M is the number of masked patches.
    """
    B, L = masks.shape
    
    masks = masks.long()
    mask_indices_list = []
    for b in range(B):
        row = masks[b] 
        nonzero_positions = torch.nonzero(row, as_tuple=False).squeeze(1)
        order_values = row[nonzero_positions]
        _, sorted_idx = torch.sort(order_values)
        sorted_positions = nonzero_positions[sorted_idx]
        mask_indices_list.append(sorted_positions)

    mask_indices = torch.stack(mask_indices_list, dim=0)
    
    return mask_indices