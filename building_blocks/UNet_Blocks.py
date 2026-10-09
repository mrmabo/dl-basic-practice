import torch
from torch import nn


class UNetDoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class UNetDownConv(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            UNetDoubleConv(input_channels, output_channels),
        )

    def forward(self, x):
        return self.block(x)


class UNetUpConv(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=2, stride=2
        )

        self.conv = UNetDoubleConv(output_channels + skip_channels, output_channels)

    def forward(self, x, skip):
        hidden = self.up(x)

        hidden = torch.cat([skip, hidden], dim=1)

        return self.conv(hidden)
