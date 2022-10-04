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
                               for item in dataset_items], padding_value=0)
    spectrogram = spectrogram.permute(1, 2, 0)

    # Should return (batch_size, text_encoded_length)
    text_encoded = pad_sequence([item['text_encoded'].permute(1, 0)
                                for item in dataset_items], padding_value=0)

    text_encoded = text_encoded.permute(1, 0, 2).squeeze()

    # lengths for masking during evalu
    text_encoded_length = torch.LongTensor([item['text_encoded'].size()[1]
                                            for item in dataset_items])

    spectrogram_length = torch.LongTensor([item['spectrogram'].size()[1]
                                           for item in dataset_items])

    text = [item['text'] for item in dataset_items]
    audio = [item['audio'] for item in dataset_items]
    audio_path = [item['audio_path'] for item in dataset_items]

    result_batch = {
        'audio': audio,
        'spectrogram': spectrogram,
        'text': text,
        'text_encoded': text_encoded,
        'text_encoded_length': text_encoded_length,
        'spectrogram_length': spectrogram_length,
        'audio_path': audio_path
    }
    return result_batch
