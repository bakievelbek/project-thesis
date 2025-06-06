# model.py

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class TDNNBlock(nn.Module):
    """A simple TDNN layer (1D Conv with context)"""
    def __init__(self, input_size, output_size, context_size=5, dilation=1):
        super().__init__()
        self.tdnn = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=context_size,
            dilation=dilation,
            padding='same'  # requires torch >=1.10
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T)
        out = self.tdnn(x)
        out = self.activation(out)
        return out.transpose(1, 2)  # (B, F, T) -> (B, T, F)

def xavier_init_layer(in_size, out_size, spec_norm=True, layer_type=nn.Linear, **kwargs):
    layer = layer_type(in_size, out_size, **kwargs)
    if spec_norm:
        layer = spectral_norm(layer)
    nn.init.xavier_uniform_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)
    return layer

class Learnable_sigmoid(nn.Module):
    def __init__(self, in_features=257):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return 1.2 * torch.sigmoid(self.slope * x)

class HybridGenerator(nn.Module):
    def __init__(self, input_size=257, tdnn_channels=200, tdnn_context=5, hidden_size=200, num_layers=2, dropout=0):
        super().__init__()
        self.tdnn = TDNNBlock(input_size, tdnn_channels, context_size=tdnn_context)
        self.activation = nn.LeakyReLU(negative_slope=0.3)
        self.blstm = nn.LSTM(
            input_size=tdnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        for name, param in self.blstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

        self.linear1 = xavier_init_layer(400, 300, spec_norm=False)
        self.linear2 = xavier_init_layer(300, input_size, spec_norm=False)
        self.Learnable_sigmoid = Learnable_sigmoid(in_features=input_size)

    def forward(self, x, lengths=None):
        x = self.tdnn(x)
        out, _ = self.blstm(x)  # (B, T, 2*hidden)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.Learnable_sigmoid(out)
        return out

class MetricDiscriminator(nn.Module):
    def __init__(self, kernel_size=(5, 5), base_channels=15):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.3)
        self.BN = nn.BatchNorm2d(num_features=2, momentum=0.01)
        self.conv1 = xavier_init_layer(2, base_channels, layer_type=nn.Conv2d, kernel_size=kernel_size)
        self.conv2 = xavier_init_layer(base_channels, base_channels, layer_type=nn.Conv2d, kernel_size=kernel_size)
        self.conv3 = xavier_init_layer(base_channels, base_channels, layer_type=nn.Conv2d, kernel_size=kernel_size)
        self.conv4 = xavier_init_layer(base_channels, base_channels, layer_type=nn.Conv2d, kernel_size=kernel_size)
        self.Linear1 = xavier_init_layer(base_channels, 50)
        self.Linear2 = xavier_init_layer(50, 10)
        self.Linear3 = xavier_init_layer(10, 1)

    def forward(self, x):
        out = self.BN(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.activation(out)
        out = self.conv4(out)
        out = self.activation(out)
        out = torch.mean(out, (2, 3))
        out = self.Linear1(out)
        out = self.activation(out)
        out = self.Linear2(out)
        out = self.activation(out)
        out = self.Linear3(out)
        return out
