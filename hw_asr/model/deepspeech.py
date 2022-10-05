from turtle import forward
from unicodedata import bidirectional
from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class DeepSpeech(BaseModel):
    def __init__(
            self, n_feats, gru_hidden, n_gru, conv_channels, n_class=28, *args,
            **kwargs):
        super().__init__(n_feats, n_class=n_class, *args, **kwargs)
        self.conv_block = conv_block(in_channels=1,
                                     out_channels=32,
                                     kernel_size=(41, 11),
                                     stride=(2, 2)
                                     )

        self.gru_block = gru_block(input_size=1408,
                                   hidden_size=gru_hidden,
                                   num_layers=n_gru,
                                   bidirectional=True
                                   )

        self.fc = nn.Linear(in_features=gru_hidden,
                            out_features=n_class)

    def forward(self, spectrogram, *args, **kwargs):
        # out =
        spectrogram = spectrogram.unsqueeze(1)
        out = self.conv_block(spectrogram)
        # out.size() = (seq_len, bs, 2*hidden_size)
        # h.size() = (2*num_layers, bs, hidden_size)
        # print(out.size())
        out = out.view(out.size()[0],  out.size()[-1], -1,)
        # print(out.size())
        out, h = self.gru_block(out)
        # print(out.size())
        # print(h.size())
        # print(out.size())
        out = self.fc(out)
        # print(out.size())
        return {"logits": out}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here


def conv_block(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def gru_block(
        input_size, hidden_size, num_layers, bidirectional, *args, **kwargs):
    return nn.Sequential(
        nn.GRU(input_size, hidden_size, num_layers, bidirectional),
    )


def la_conv_block(*args, **kwargs):
    pass


def fc_block(*args, **kwargs):
    pass
