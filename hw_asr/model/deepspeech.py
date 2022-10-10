from unicodedata import bidirectional
from torch import nn
from torch.nn import Sequential
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from hw_asr.base import BaseModel


class DeepSpeech(BaseModel):
    def __init__(
            self,  gru_hidden, n_gru, conv_channels, dropout,  n_class=28, *args,
            **kwargs):
        super().__init__(n_class=n_class, *args, **kwargs)
        self.activation = nn.Hardtanh(min_val=0, max_val=20)

        self.spec_freq = 128
        self.conv_h1 = int((self.spec_freq + 2*20 - 1*(41-1)-1)/2 + 1)
        self.conv_h2 = int((self.conv_h1 + 2*10 - 1*(21-1)-1)/2 + 1)

        self.conv_block = nn.Sequential(
            make_conv_layer(in_channels=1,
                            out_channels=conv_channels,
                            kernel_size=(41, 11),
                            stride=(2, 1),
                            dilation=(1, 1),
                            padding=(20, 5),
                            activation=self.activation,
                            ),
            make_conv_layer(in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=(21, 11),
                            stride=(2, 1),
                            dilation=(1, 1),
                            padding=(10, 5),
                            activation=self.activation,
                            ))

        self.gru_block = make_gru_block(input_size=self.conv_h2*conv_channels,  # 8192,
                                        hidden_size=gru_hidden,
                                        num_layers=n_gru,
                                        bidirectional=True,
                                        dropout=dropout,
                                        )

        self.fc1 = nn.Linear(in_features=gru_hidden,
                             out_features=n_class)

    def forward(self, spectrogram, spectrogram_length, *args, **kwargs):

        out = self.conv_block(spectrogram.unsqueeze(1))
        # print(out.size())
        # out = self.conv_block3(out)
        # print(out.size())
        bs, filters, freqs, seq_length = out.size()
        out = out.permute(0, 3, 2, 1)
        out = out.contiguous()\
            .view(bs, seq_length, filters*freqs)
        # print(out.size())
        # print(out.size())
        out = pack_padded_sequence(
            out, lengths=spectrogram_length, batch_first=True,
            enforce_sorted=False)
        out, _ = self.gru_block(out)
        out, _ = pad_packed_sequence(
            out, batch_first=True, total_length=seq_length)
        # print(out.size())
        # print(out.size())
        # print(h.size())
        # print(out.size())
        out = self.fc1(out)
        # print(out.size())
        # print(out.size())
        # print(out.size())
        return {"logits": out}

    def transform_input_lengths(self, input_lengths):
        return input_lengths


def make_conv_layer(
        in_channels, out_channels, dilation, kernel_size, stride, padding,
        activation):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        activation,)


def make_gru_block(
        input_size, hidden_size, num_layers, bidirectional, dropout, *args, **
        kwargs):
    return nn.Sequential(
        nn.GRU(
            input_size, hidden_size, num_layers, bidirectional,
            dropout=dropout),)
