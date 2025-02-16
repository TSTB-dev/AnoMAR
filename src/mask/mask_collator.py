from multiprocessing import Value

import torch
import math

class ConstantMaskCollator(object):
    def __init__(
        self, 
        ratio=0.75, # ratio of masked patches
        input_size=(224, 224),
        patch_size=16,
        mask_seed=None,
        mask=None
    ):
        super(ConstantMaskCollator, self).__init__()
        
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.mask = mask
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.ratio = ratio
        self.mask_seed = mask_seed
        
        if mask is not None:
            assert mask.size(0) == self.height * self.width, "Mask size should be equal to number of patches"
        else:
            self.mask = self._generate_random_mask()
        
    def _generate_random_mask(self):
        num_patches = self.height * self.width
        num_keep = int(num_patches * (1. - self.ratio))
        m = torch.randperm(num_patches)[num_keep:]
        m = m.sort().values
        return m
    
    def __call__(self, batch):
        '''
        Create constant masks for each sample in the batch
        
        Ouptut:
            collated_batch_org: original batch
            collated_masks: masks for each sample in the batch, (B, M), M: num of masked patches
        '''
        B = len(batch)
        collated_batch_org = torch.utils.data.default_collate(batch)
        collated_masks = self.mask.unsqueeze(0).expand(B, -1).clone()  # (B, M), M: num of masked patches
        return collated_batch_org, collated_masks    

class RandomMaskCollator(object):
    def __init__(
        self,
        ratio=0.75, # ratio of masked patches
        input_size=(224, 224),
        patch_size=16,
        mask_seed = None,
        min_ratio= None,
        max_ratio= None,
        schedule = "const",
        total_steps = 2000,
        **kwargs
    ):
        super(RandomMaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.ratio = ratio
        self.mask_seed = mask_seed
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        if self.min_ratio is None or self.max_ratio is None:
            self.min_ratio = self.ratio
            self.max_ratio = self.ratio
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes (for distributed training)
        self.scheduler = schedule
        self.step_cout = 0
        self.total_steps = total_steps
        self.current_ratio = self.ratio
        
    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def update(self):
        self.step_cout += 1
        if self.scheduler == "linear":
            self.current_ratio =  self.min_ratio + (self.max_ratio - self.min_ratio) * self.step_cout / self.total_steps
        elif self.scheduler == "cosine":
            self.current_ratio = self.min_ratio + (self.max_ratio - self.min_ratio) * (1 + math.cos(math.pi * self.step_cout / self.total_steps)) / 2
        elif self.scheduler == "dynamic":
            pass
        elif self.scheduler == "const":
            pass
        else:
            raise ValueError("Invalid scheduler")
        
    def generate_random_mask(self, num_masks):
        """Generate random mask with random number of masked patches
        Args:
            num_masks: number of masked patches
        Returns:
            mask: random mask, (B, N)
        """
        ratio = self.current_ratio
        num_patches = self.height * self.width
        num_keep = int(num_patches * (1. - ratio))
        masks = []
        for _ in range(num_masks):
            m = torch.randperm(num_patches)[num_keep:]
            m = m.sort().values
            masks.append(m)
        masks = torch.stack(masks, dim=0)  # (B, N)
        m = torch.zeros(len(masks), num_patches)
        m.scatter_(1, masks, 1)
        
        return m
    
    def __call__(self, batch):
        '''
        Create random masks for each sample in the batch
        
        Ouptut:
            collated_batch_org: original batch
            collated_masks: masks for each sample in the batch, (B, M), M: num of masked patches
        '''
        B = len(batch)
        collated_batch_org = torch.utils.data.default_collate(batch)  # Collates original batch
        
        # For distributed training, each process uses different seed to generate masks
        seed = self.step()  # use the shared counter to generate seed
        g = torch.Generator()
        g.manual_seed(seed)
        if self.mask_seed is not None:
            g.manual_seed(self.mask_seed)
        if self.scheduler in ["linear", "cosine"]:
            ratio = self.current_ratio  
        elif self.scheduler == "dynamic":
            ratio = (self.max_ratio - self.min_ratio) * torch.rand(1, generator=g) + self.min_ratio
        elif self.scheduler == "const":
            ratio = self.ratio
        num_patches = self.height * self.width
        num_keep = int(num_patches * (1. - ratio))
        
        collated_masks = []
        for _ in range(B):
            m = torch.randperm(num_patches)[num_keep:]
            m = m.sort().values
            collated_masks.append(m)
        collated_masks = torch.stack(collated_masks, dim=0)  # (B, M), M: num of masked patches
        return collated_batch_org, collated_masks

class BlockRandomMaskCollator(object):
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        mask_ratio: float = 0.75,
        aspect_min: float = 0.75,
        aspect_max: float = 1.5,
        scale_min: float = 0.1,
        scale_max: float = 0.6,
        mask_seed=None,
        num_block_masks=4,  # ★ 追加した引数: 作成するブロックマスクの数, 
        **kwargs
    ):
        super(BlockRandomMaskCollator, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.mask_seed = mask_seed
        self.num_block_masks = num_block_masks  # ★

        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size

        # マルチプロセス等でカウンターを共有するため
        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def generate_block_mask(self, aspect_ratio, scale):
        """Generate block mask with random aspect ratio and scale

        Args:
            aspect_ratio (float): マスクの縦横比
            scale (float): マスクのスケール (高さ方向の割合)

        Returns:
            mask (LongTensor): マスクされたパッチのインデックス (M,)
        """
        mask = torch.zeros(self.height, self.width)
        h = int(self.height * scale)
        w = int(self.width * scale / aspect_ratio)

        y = torch.randint(0, self.height - h + 1, (1,)).item()
        x = torch.randint(0, self.width - w + 1, (1,)).item()

        mask[y:y+h, x:x+w] = 1

        # 2次元 -> 1次元にflattenして，マスクがかかっている部分のインデックスを返す
        mask = torch.nonzero(mask.view(-1)).squeeze(1)  # (M,)
        return mask

    def restrict_mask_ratio(self, mask, num_remove, num_total):
        """マスクされたパッチ数が最終的に num_remove 個になるように制限する

        Args:
            mask (LongTensor): マスクインデックス (M,)
            num_remove (int): マスクするパッチ数（= total - keep）
            num_total (int): 全パッチ数

        Returns:
            mask (LongTensor): 制限後のマスクインデックス (M'<=num_remove)
        """
        num_masked_patches = mask.size(0)

        if num_remove < num_masked_patches:
            # マスクが多すぎる場合は頭から切り捨て
            mask = mask[:num_remove]
        elif num_remove > num_masked_patches:
            # マスクが少ない場合は未マスク部分から補充
            num_fill = num_remove - num_masked_patches
            m = ~torch.isin(torch.arange(num_total), mask)
            unmasked_indices = torch.arange(num_total)[m]
            new_mask = unmasked_indices[torch.randperm(unmasked_indices.size(0))][:num_fill]
            mask = torch.cat([mask, new_mask], dim=0)

        return mask

    def __call__(self, batch):
        """バッチ分のマスクを生成して返す

        Args:
            batch: データローダなどから渡される1バッチ分の入力

        Returns:
            collated_batch_org: 元のバッチ (collate 後)
            collated_masks: (B, M) 形式で，各サンプルのマスクインデックスをまとめたもの
        """
        B = len(batch)
        collated_batch_org = torch.utils.data.default_collate(batch)

        # 分散学習時など，プロセスごとに異なるシードでマスクを作る
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        if self.mask_seed is not None:
            g.manual_seed(self.mask_seed)

        num_patches = self.height * self.width
        num_keep = int(num_patches * (1. - self.mask_ratio))
        num_remove = num_patches - num_keep

        collated_masks = []

        for _ in range(B):
            # ★ num_block_masks 回だけブロックマスクを生成して結合する
            all_mask_indices = []
            for _block_i in range(self.num_block_masks):
                aspect_ratio = torch.empty(1).uniform_(self.aspect_min, self.aspect_max, generator=g).item()
                scale = torch.empty(1).uniform_(self.scale_min, self.scale_max, generator=g).item()

                mask_block = self.generate_block_mask(aspect_ratio, scale)
                all_mask_indices.append(mask_block)

            # すべてのマスクを結合して重ね合わせ（インデックスのユニーク集合を取る）
            merged_mask = torch.unique(torch.cat(all_mask_indices))

            # 合計のマスク数が最終的に num_remove 個になるように制限
            merged_mask = self.restrict_mask_ratio(merged_mask, num_remove, num_patches)
            # ソートしておく（可視化や順序依存処理を想定）
            merged_mask = merged_mask.sort().values

            collated_masks.append(merged_mask)

        collated_masks = torch.stack(collated_masks, dim=0)  # (B, M)
        return collated_batch_org, collated_masks


# class BlockRandomMaskCollator(object):
#     def __init__(
#         self,
#         input_size=(224, 224),
#         patch_size=16,
#         mask_ratio: float = 0.75,
#         aspect_min: float = 0.75,
#         aspect_max: float = 1.5,
#         scale_min: float = 0.1,
#         scale_max: float = 0.4,
#         mask_seed = None,
#         **kwargs
#     ):
#         super(BlockRandomMaskCollator, self).__init__()
#         self.input_size = input_size
#         self.patch_size = patch_size
#         self.mask_ratio = mask_ratio
#         self.aspect_min = aspect_min
#         self.aspect_max = aspect_max
#         self.scale_min = scale_min
#         self.scale_max = scale_max
#         self.mask_seed = mask_seed
        
#         if not isinstance(input_size, tuple):
#             input_size = (input_size, ) * 2
#         self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
#         self._itr_counter = Value('i', -1)

#     def step(self): 
#         i = self._itr_counter
#         with i.get_lock():
#             i.value += 1
#             v = i.value
#         return v
    
#     def generate_block_mask(self, aspect_ratio, scale):
#         """Generate block mask with random aspect ratio and scale
#         Args:
#             aspect_ratio: aspect ratio of the mask
#             scale: scale of the mask
#         Returns:
#             mask: block mask, (M)
#         """
        
#         mask = torch.zeros(self.height, self.width)
#         h = int(self.height * scale)
#         w = int(self.width * scale / aspect_ratio)
#         y = torch.randint(0, self.height - h + 1, (1,)).item()
#         x = torch.randint(0, self.width - w + 1, (1,)).item()

#         mask[y:y+h, x:x+w] = 1
        
#         # (1, H, W) -> (M)
#         mask = torch.nonzero(mask.view(-1)).squeeze(1)  # (M)
#         return mask
    
#     def restrict_mask_ratio(self, mask, num_remove, num_total):
#         """Restrict the number of masked patches to num_keep
#         Args:
#             mask: block mask, (M)
#             num_remove: number of patches to keep
#             num_total: total number of patches
#         Returns:
#             mask: restricted mask, (M')
#         """
#         num_masked_patches = mask.size(0)
        
#         if num_remove < num_masked_patches:
#             mask = mask[:num_remove]
#         elif num_remove > num_masked_patches:
#             # fill mask indices with random patches
#             num_fill = num_remove - num_masked_patches
#             m = ~torch.isin(torch.arange(num_total), mask)
#             unmasked_indices = torch.arange(num_total)[m]
#             new_mask = unmasked_indices[torch.randperm(unmasked_indices.size(0))][:num_fill]
#             mask = torch.cat([mask, new_mask], dim=0)
            
#         return mask
    
#     def __call__(self, batch):
#         """Create random block mask for each sample in the batch
#         Args:
#             original batch
#         Returns:
#             collated_batch_org: original batch
#             collated_masks: masks for each sample in the batch, (B, M), M: num of masked patches
#         """
#         B = len(batch)
#         collated_batch_org = torch.utils.data.default_collate(batch)
        
#         # For distributed training, each process uses different seed to generate masks
#         seed = self.step()
#         g = torch.Generator()
#         g.manual_seed(seed)
#         if self.mask_seed is not None:
#             g.manual_seed(self.mask_seed)
#         num_patches = self.height * self.width
#         num_keep = int(num_patches * (1. - self.mask_ratio))
#         num_remove = num_patches - num_keep
        
#         collated_masks = []
#         for _ in range(B):
#             aspect_ratio = torch.empty(1).uniform_(self.aspect_min, self.aspect_max, generator=g)
#             scale = torch.empty(1).uniform_(self.scale_min, self.scale_max, generator=g)
#             mask = self.generate_block_mask(aspect_ratio, scale)
#             mask = self.restrict_mask_ratio(mask, num_remove, num_patches)
#             mask = mask.sort().values
#             collated_masks.append(mask)
#         collated_masks = torch.stack(collated_masks, dim=0)
#         return collated_batch_org, collated_masks

class SlidingWindowMaskCollator(object):
    def __init__(
        self, 
        input_size=(224, 224),
        patch_size=16,
        window_size=2,
        stride=2,
        order='raster',
        **kwargs
    ):
        super(SlidingWindowMaskCollator, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.stride = stride
        assert order in ['raster', 'random'], "Order should be either 'raster' or 'random'"
        self.order = order
        
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self._itr_counter = Value('i', -1)
        
        self.masks = self._generate_sliding_window_masks()
        
    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def _generate_sliding_window_masks(self):
        num_windows_h = (self.height - self.window_size) // self.stride + 1
        num_windows_w = (self.width - self.window_size) // self.stride + 1
        
        masks = []
        for i in range(num_windows_h):
            for j in range(num_windows_w):
                mask = torch.zeros(self.height, self.width)
                mask[i*self.stride:i*self.stride+self.window_size, j*self.stride:j*self.stride+self.window_size] = 1
                mask = torch.nonzero(mask.view(-1)).squeeze(1)
                masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks
    
    def __len__(self):
        return self.masks.size(0)
    
    def __call__(self, batch):
        B = len(batch)
        collated_batch_org = torch.utils.data.default_collate(batch)
        
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        if self.order == 'raster':
            # For testing
            assert B == self.masks.size(0), f"Number of masks: {self.masks.size(0)} should be equal to batch size: {B}"
            masks = self.masks
        else:
            # For training
            masks = self.masks[torch.randperm(self.masks.size(0), generator=g)]
        collated_masks = masks[:B].clone()
        return collated_batch_org, collated_masks
    

class CheckerBoardMaskCollator(object):
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        min_divisor=1,
        max_divisor=4,
        mode = "random", # random or fixed
        mask_seed = None,
    ):
        super(CheckerBoardMaskCollator, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.min_divisor = min_divisor
        self.max_divisor = max_divisor
        self.mode = mode
        self.mask_seed = mask_seed
        
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self._itr_counter = Value('i', -1)
        
        self.num_patterns = 2 * (max_divisor - min_divisor + 1)
        
        self.num_masks = self.num_patterns * 2  # 2 masks for each pattern

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def collate_all_masks(self):
        """Create all possible checkerboard masks for evaluation"""
        masks = []
        for div in range(self.min_divisor, self.max_divisor+1):
            div = 2 ** div
            mask = self.checkerboard_mask(div)  
            inv_mask = 1 - mask
            masks.append(mask)
            masks.append(inv_mask)
        masks = torch.stack(masks, dim=0)  # (2 * (max_divisor - min_divisor + 1), N)
        masks = masks.view(-1, self.height * self.width)  # (2 * (max_divisor - min_divisor + 1), N)
        return masks

    def checkerboard_mask(self, divisor):
        tile_h = self.height // divisor
        tile_w = self.width // divisor
        y_indices = torch.arange(divisor).view(-1, 1).expand(divisor, divisor)
        x_indices = torch.arange(divisor).view(1, -1).expand(divisor, divisor)
        checkerboard_pattern = (x_indices + y_indices) % 2
        mask = checkerboard_pattern.repeat_interleave(tile_h, dim=0).repeat_interleave(tile_w, dim=1)
        return mask
    
    def generate_checkerboard_mask(self, divisor):
        """Generate checkerboard mask with random divisor
        Args:
            divisor: divisor of the mask
        Returns:
            mask: checkerboard mask, (M)
        """
        mask = self.checkerboard_mask(divisor)
        inv_mask = 1 - mask
        mask = torch.nonzero(mask.view(-1)).squeeze(1)
        inv_mask = torch.nonzero(inv_mask.view(-1)).squeeze(1)
        return mask, inv_mask
    
    def __call__(self, batch):
        """Create checkerboard mask for each sample in the batch
        Args:
            original batch
        Returns:
            collated_batch_org: original batch
            collated_masks: masks for each sample in the batch, (B, M), M: num of masked patches
        """
        B = len(batch)
        collated_batch_org = torch.utils.data.default_collate(batch)
        
        if self.mode == "fixed":
            collated_masks = self.collate_all_masks()
            assert collated_masks.size(0) == B, "Number of masks should be equal to batch size"
            return collated_batch_org, collated_masks
        
        # For distributed training, each process uses different seed to generate masks
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        if self.mask_seed is not None:
            g.manual_seed(self.mask_seed)
        
        collated_masks = []
        for _ in range(B):
            divisor = 2 ** torch.randint(self.min_divisor, self.max_divisor, (1,)).item() 
            mask, inv_mask = self.generate_checkerboard_mask(divisor)
            if torch.rand(1) > 0.5:
                mask = mask.sort().values
                collated_masks.append(mask)
            else:
                inv_mask = inv_mask.sort().values
                collated_masks.append(inv_mask)
        collated_masks = torch.stack(collated_masks, dim=0)
        return collated_batch_org, collated_masks

if __name__ == '__main__':
    # collator = RandomMaskCollator(ratio=0.75, input_size=(224, 224), patch_size=16)
    # collator = BlockRandomMaskCollator(input_size=(224, 224), patch_size=16, mask_ratio=0.75, aspect_min=0.75, aspect_max=1.5, scale_min=0.1, scale_max=0.4)
    collator = CheckerBoardMaskCollator(input_size=(256,256), patch_size=1, min_divisor=2, max_divisor=2)
    all_masks = collator.collate_all_masks()
    print(all_masks.size())
    # save first mask for visualization
    mask = all_masks[-1]
    mask = mask.view(collator.height, collator.width).float()
    import matplotlib.pyplot as plt
    plt.imsave("checkerboard_mask.png", mask, cmap='gray')
    for i in range(len(all_masks)):
        mask = all_masks[i]
        mask = mask.view(collator.height, collator.width).float()
        plt.imsave(f"checkerboard_mask_{i}.png", mask, cmap='gray')
    