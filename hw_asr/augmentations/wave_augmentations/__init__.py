from hw_asr.augmentations.wave_augmentations.Gain import GainAug
from hw_asr.augmentations.wave_augmentations.AddColoredNoise import AddColoredNoiseAug
from hw_asr.augmentations.wave_augmentations.BackgroundNoise import AddBackgroundNoiseAug
from hw_asr.augmentations.wave_augmentations.PitchShift import PitchShiftAug

__all__ = [
    "GainAug",
    "AddColoredNoiseAug",
    "AddBackgroundNoiseAug",
    "PitchShiftAug"
]
