"""SiamUNet-diff: Siamese UNet with Feature Differencing.

Reference:
    Daudt et al., "Fully Convolutional Siamese Networks for Change Detection,"
    IEEE ICIP, 2018.

Uses a shared encoder for both temporal inputs and computes absolute feature
differences at each skip connection level, feeding them into the decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d


class SiamUNet_diff(nn.Module):
    """Siamese UNet with feature differencing at skip connections.

    Args:
        in_ch: Spectral bands per image (3 for RGB).
        out_ch: Output classes (2 for binary change).
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 2):
        super().__init__()
        n1 = 16
        f = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # ── Encoder ──
        self.conv11 = nn.Conv2d(in_ch, f[0], 3, padding=1)
        self.bn11 = nn.BatchNorm2d(f[0])
        self.do11 = nn.Dropout2d(0.2)
        self.conv12 = nn.Conv2d(f[0], f[0], 3, padding=1)
        self.bn12 = nn.BatchNorm2d(f[0])
        self.do12 = nn.Dropout2d(0.2)

        self.conv21 = nn.Conv2d(f[0], f[1], 3, padding=1)
        self.bn21 = nn.BatchNorm2d(f[1])
        self.do21 = nn.Dropout2d(0.2)
        self.conv22 = nn.Conv2d(f[1], f[1], 3, padding=1)
        self.bn22 = nn.BatchNorm2d(f[1])
        self.do22 = nn.Dropout2d(0.2)

        self.conv31 = nn.Conv2d(f[1], f[2], 3, padding=1)
        self.bn31 = nn.BatchNorm2d(f[2])
        self.do31 = nn.Dropout2d(0.2)
        self.conv32 = nn.Conv2d(f[2], f[2], 3, padding=1)
        self.bn32 = nn.BatchNorm2d(f[2])
        self.do32 = nn.Dropout2d(0.2)
        self.conv33 = nn.Conv2d(f[2], f[2], 3, padding=1)
        self.bn33 = nn.BatchNorm2d(f[2])
        self.do33 = nn.Dropout2d(0.2)

        self.conv41 = nn.Conv2d(f[2], f[3], 3, padding=1)
        self.bn41 = nn.BatchNorm2d(f[3])
        self.do41 = nn.Dropout2d(0.2)
        self.conv42 = nn.Conv2d(f[3], f[3], 3, padding=1)
        self.bn42 = nn.BatchNorm2d(f[3])
        self.do42 = nn.Dropout2d(0.2)
        self.conv43 = nn.Conv2d(f[3], f[3], 3, padding=1)
        self.bn43 = nn.BatchNorm2d(f[3])
        self.do43 = nn.Dropout2d(0.2)

        # ── Decoder ──
        self.upconv4 = nn.ConvTranspose2d(f[3], f[3], 3, padding=1, stride=2, output_padding=1)
        self.conv43d = nn.ConvTranspose2d(f[4], f[3], 3, padding=1)
        self.bn43d = nn.BatchNorm2d(f[3])
        self.do43d = nn.Dropout2d(0.2)
        self.conv42d = nn.ConvTranspose2d(f[3], f[3], 3, padding=1)
        self.bn42d = nn.BatchNorm2d(f[3])
        self.do42d = nn.Dropout2d(0.2)
        self.conv41d = nn.ConvTranspose2d(f[3], f[2], 3, padding=1)
        self.bn41d = nn.BatchNorm2d(f[2])
        self.do41d = nn.Dropout2d(0.2)

        self.upconv3 = nn.ConvTranspose2d(f[2], f[2], 3, padding=1, stride=2, output_padding=1)
        self.conv33d = nn.ConvTranspose2d(f[3], f[2], 3, padding=1)
        self.bn33d = nn.BatchNorm2d(f[2])
        self.do33d = nn.Dropout2d(0.2)
        self.conv32d = nn.ConvTranspose2d(f[2], f[2], 3, padding=1)
        self.bn32d = nn.BatchNorm2d(f[2])
        self.do32d = nn.Dropout2d(0.2)
        self.conv31d = nn.ConvTranspose2d(f[2], f[1], 3, padding=1)
        self.bn31d = nn.BatchNorm2d(f[1])
        self.do31d = nn.Dropout2d(0.2)

        self.upconv2 = nn.ConvTranspose2d(f[1], f[1], 3, padding=1, stride=2, output_padding=1)
        self.conv22d = nn.ConvTranspose2d(f[2], f[1], 3, padding=1)
        self.bn22d = nn.BatchNorm2d(f[1])
        self.do22d = nn.Dropout2d(0.2)
        self.conv21d = nn.ConvTranspose2d(f[1], f[0], 3, padding=1)
        self.bn21d = nn.BatchNorm2d(f[0])
        self.do21d = nn.Dropout2d(0.2)

        self.upconv1 = nn.ConvTranspose2d(f[0], f[0], 3, padding=1, stride=2, output_padding=1)
        self.conv12d = nn.ConvTranspose2d(f[1], f[0], 3, padding=1)
        self.bn12d = nn.BatchNorm2d(f[0])
        self.do12d = nn.Dropout2d(0.2)
        self.conv11d = nn.ConvTranspose2d(f[0], out_ch, 3, padding=1)

    def _encode(self, x):
        """Shared encoder for one temporal input."""
        x11 = self.do11(F.relu(self.bn11(self.conv11(x))))
        x12 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12, 2, 2)

        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22, 2, 2)

        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33, 2, 2)

        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43, 2, 2)

        return x12, x22, x33, x43, x4p

    def forward(self, xA, xB):
        """Forward pass.

        Args:
            xA: (B, C, H, W) — T1 (pre-change) image.
            xB: (B, C, H, W) — T2 (post-change) image.

        Returns:
            (B, out_ch, H, W) change logits.
        """
        # Encode both streams with shared weights
        x12_1, x22_1, x33_1, x43_1, x4p = self._encode(xA)
        x12_2, x22_2, x33_2, x43_2, _ = self._encode(xB)

        # Decoder with absolute feature differences at skip connections
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))

        return self.conv11d(x12d)
