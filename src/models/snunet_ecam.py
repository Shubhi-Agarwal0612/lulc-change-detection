"""SNUNet-ECAM: Siamese Nested UNet with Ensemble Channel Attention.

Reference:
    Fang et al., "SNUNet-CD: A Densely Connected Siamese Network for Change
    Detection of VHR Images," IEEE GRSL, 2021.

The encoder weights are shared between the T1 and T2 streams. Dense skip
connections and an Ensemble Channel Attention Module (ECAM) fuse multi-scale
features for change prediction.
"""

import torch
import torch.nn as nn


class _ConvBlockNested(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU with residual shortcut."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.activation(self.bn1(x))
        x = self.conv2(x)
        x = self.bn2(x)
        return self.activation(x + identity)


class _UpBlock(nn.Module):
    """Learnable 2× upsampling via transposed convolution."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        return self.up(x)


class _ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        mx = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg + mx)


class SNUNet_ECAM(nn.Module):
    """Siamese Nested UNet with Ensemble Channel Attention.

    Args:
        in_ch: Spectral bands per image (3 for RGB).
        out_ch: Output classes (2 for binary change).
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 2):
        super().__init__()
        n1 = 32
        f = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(2, 2)

        # ── Shared encoder ──
        self.conv0_0 = _ConvBlockNested(in_ch, f[0], f[0])
        self.conv1_0 = _ConvBlockNested(f[0], f[1], f[1])
        self.conv2_0 = _ConvBlockNested(f[1], f[2], f[2])
        self.conv3_0 = _ConvBlockNested(f[2], f[3], f[3])
        self.conv4_0 = _ConvBlockNested(f[3], f[4], f[4])

        self.Up1_0 = _UpBlock(f[1])
        self.Up2_0 = _UpBlock(f[2])
        self.Up3_0 = _UpBlock(f[3])
        self.Up4_0 = _UpBlock(f[4])

        # ── Nested decoder ──
        self.conv0_1 = _ConvBlockNested(f[0] * 2 + f[1], f[0], f[0])
        self.conv1_1 = _ConvBlockNested(f[1] * 2 + f[2], f[1], f[1])
        self.Up1_1 = _UpBlock(f[1])
        self.conv2_1 = _ConvBlockNested(f[2] * 2 + f[3], f[2], f[2])
        self.Up2_1 = _UpBlock(f[2])
        self.conv3_1 = _ConvBlockNested(f[3] * 2 + f[4], f[3], f[3])
        self.Up3_1 = _UpBlock(f[3])

        self.conv0_2 = _ConvBlockNested(f[0] * 3 + f[1], f[0], f[0])
        self.conv1_2 = _ConvBlockNested(f[1] * 3 + f[2], f[1], f[1])
        self.Up1_2 = _UpBlock(f[1])
        self.conv2_2 = _ConvBlockNested(f[2] * 3 + f[3], f[2], f[2])
        self.Up2_2 = _UpBlock(f[2])

        self.conv0_3 = _ConvBlockNested(f[0] * 4 + f[1], f[0], f[0])
        self.conv1_3 = _ConvBlockNested(f[1] * 4 + f[2], f[1], f[1])
        self.Up1_3 = _UpBlock(f[1])

        self.conv0_4 = _ConvBlockNested(f[0] * 5 + f[1], f[0], f[0])

        # ── ECAM ──
        self.ca = _ChannelAttention(f[0] * 4, ratio=16)
        self.ca1 = _ChannelAttention(f[0], ratio=16 // 4)
        self.conv_final = nn.Conv2d(f[0] * 4, out_ch, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xA, xB):
        """Forward pass.

        Args:
            xA: (B, C, H, W) — T1 (pre-change) image.
            xB: (B, C, H, W) — T2 (post-change) image.

        Returns:
            (B, out_ch, H, W) change logits.
        """
        # Encoder — stream A (T1)
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))

        # Encoder — stream B (T2)
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        # Nested decoder with dense skip connections
        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        # ECAM fusion
        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))

        return self.conv_final(out)
