from torch import Tensor
from hw_asr.utils.parse_config import ConfigParser


class AugmentationBase:
    sample_rate = 16000

    def __call__(self, data: Tensor) -> Tensor:
        raise NotImplementedError()
