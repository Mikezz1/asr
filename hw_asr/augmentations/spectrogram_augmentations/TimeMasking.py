from torchaudio.transforms import TimeMasking
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeMaskingAug(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = TimeMasking(
            *args, **kwargs, sample_rate=AugmentationBase.sample_rate)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
