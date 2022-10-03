import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    # Should return (batch_size, n_mels, t)
    spectrogram = pad_sequence([item['spectrogram'].squeeze().permute(1, 0)
                               for item in dataset_items], batch_first=True)
    spectrogram = spectrogram.permute(0, 2, 1)

    # Should return (batch_size, text_encoded_length)
    text_encoded = pad_sequence([item['text_encoded'].permute(1, 0)
                                for item in dataset_items], batch_first=True)

    text_encoded = text_encoded.squeeze().permute(0, 1)

    # Custom dataset may not return text_encoded_length
    text_encoded_length = torch.Tensor([item['text_encoded'].size()[1]
                                        for item in dataset_items])
    text = [item['text'] for item in dataset_items]
    audio = [item['audio'] for item in dataset_items]

    result_batch = {
        'audio': audio,
        'spectrogram': spectrogram,
        'text': text,
        'text_encoded': text_encoded,
        'text_encoded_length': text_encoded_length
    }
    return result_batch
