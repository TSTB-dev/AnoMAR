import torch

class DefaultCollator(object):
    def __call__(self, batch):
        collated_batch = torch.utils.data.default_collate(batch)
        return collated_batch, None, None