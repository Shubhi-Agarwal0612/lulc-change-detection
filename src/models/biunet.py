"""Bi-temporal UNet for change detection.

Concatenates T1 and T2 images channel-wise and feeds through a standard
U-Net encoder–decoder. When a BiFormer backbone is available it can be
swapped in, but the default is a clean convolutional U-Net.
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Two 3×3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BiUNet(nn.Module):
    """U-Net that expects concatenated bi-temporal input [T1 | T2].

    Args:
        in_channels: Total input channels (num_bands × 2).
        num_classes: Output channels (2 for binary change detection).
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 2):
        super().__init__()

        # Encoder
        self.enc1 = _ConvBlock(in_channels, 64)
        self.enc2 = _ConvBlock(64, 128)
        self.enc3 = _ConvBlock(128, 256)
        self.enc4 = _ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _ConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = _ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = _ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = _ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = _ConvBlock(128, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, in_channels, H, W) — concatenated [T1 | T2].

        Returns:
            (B, num_classes, H, W) logits.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)
