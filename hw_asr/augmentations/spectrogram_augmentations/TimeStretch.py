from torchaudio.transforms import TimeStretch
from torch import Tensor, FloatTensor
from numpy.random import uniform
import random
import torch


from hw_asr.augmentations.base import AugmentationBase


class TimeStretchAug(AugmentationBase):
    def __init__(self, min_s, max_s, p, *args, **kwargs):
        self.min_s = min_s
        self.max_s = max_s

        self.p = p
        self._aug = TimeStretch(n_freq=128)

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            rate = float(uniform(self.min_s, self.max_s, 1))
            return self._aug(data, rate).to(torch.float32)
        return data
