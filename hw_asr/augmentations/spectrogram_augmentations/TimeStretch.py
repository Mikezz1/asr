from torchaudio.transforms import TimeStretch
from torch import Tensor
from numpy.random import uniform


from hw_asr.augmentations.base import AugmentationBase


class TimeStratch(AugmentationBase):
    def __init__(self, min_s, max_s,  *args, **kwargs):
        rate = uniform(min_s, max_s, 1)
        self._aug = TimeStretch(fixed_rate=rate, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
