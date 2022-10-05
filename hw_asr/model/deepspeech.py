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
        self.conv_block1 = conv_block(in_channels=1,
                                      out_channels=conv_channels,
                                      kernel_size=(5, 1),
                                      stride=(1, 1),
                                      padding=(2, 0)
                                      )

        self.conv_block2 = conv_block(in_channels=1,
                                      out_channels=conv_channels,
                                      kernel_size=(5, 1),
                                      stride=(1, 1),
                                      padding=(2, 0)
                                      )
        self.conv_block3 = conv_block(in_channels=1,
                                      out_channels=conv_channels,
                                      kernel_size=(5, 1),
                                      stride=(2, 1),
                                      padding=(2, 0)
                                      )

        self.gru_block = gru_block(input_size=8192,
                                   hidden_size=gru_hidden,
                                   num_layers=n_gru,
                                   bidirectional=True
                                   )

        self.fc = nn.Linear(in_features=gru_hidden,
                            out_features=n_class)

    def forward(self, spectrogram, *args, **kwargs):
        # out =
        spectrogram = spectrogram.unsqueeze(1)
        # print(spectrogram.size())
        out = self.conv_block1(spectrogram)
        # print(out.size())
        out = self.conv_block2(spectrogram)
        # print(out.size())
        out = self.conv_block3(spectrogram)
        # out.size() = (seq_len, bs, 2*hidden_size)
        # h.size() = (2*num_layers, bs, hidden_size)
        # print(out.size())
        out = out.view(out.size()[0],  out.size()[-1], -1,)
        # print(out.size())
        # print(out.size())
        out, h = self.gru_block(out)
        # print(out.size())
        # print(out.size())
        # print(h.size())
        # print(out.size())
        out = self.fc(out)
        # print(out.size())
        # print(out.size())
        return {"logits": out}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here


def conv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
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
