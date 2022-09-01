import torch
import numpy as np


class AddGaussianNoise():
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: torch.Tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


class GridMask():
    def __init__(self, p=0.5, d_range=(96, 224), r=0.6, fill=0):
        self.p = p
        self.d_range = d_range
        self.r = r
        self.fill = fill

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """
        sample: torch.Tensor(3, height, width)
        """
        if np.random.uniform() > self.p:
            return sample
        sample = sample.numpy()
        _, H, W = sample.shape
        d = np.random.randint(*self.d_range, dtype=np.uint8)
        r = int(self.r * d)

        mask = np.ones((H + d, W + d), dtype=np.uint8)
        for i in range(0, H + d, d):
            for j in range(0, W + d, d):
                mask[i: i + (d - r), j: j + (d - r)] = 0
        delta_x, delta_y = np.random.randint(0, d, size=2)
        mask = mask[delta_x: delta_x + H, delta_y: delta_y + W]
        sample *= np.expand_dims(mask, 0)

        # fill
        sample += self.fill * (1 - mask)

        return torch.from_numpy(sample)
