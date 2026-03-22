"""CloakUNet — fast amortised adversarial delta predictor.

Architecture: encoder-decoder with skip connections.
Output: tanh(x) * EPS_PGD  →  delta in [-EPS_PGD, +EPS_PGD]
Input/output resolution: 512×512.
"""

import torch
import torch.nn as nn
from config import EPS_PGD


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CloakUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = _conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)       # 512 → 256

        self.enc2 = _conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)       # 256 → 128

        self.enc3 = _conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)       # 128 → 64

        # Bottleneck
        self.bottleneck = _conv_block(256, 512)

        # Decoder
        self.up3    = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 64 → 128
        self.dec3   = _conv_block(512, 256)  # 256 (up) + 256 (skip) = 512

        self.up2    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 128 → 256
        self.dec2   = _conv_block(256, 128)  # 128 + 128 = 256

        self.up1    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 256 → 512
        self.dec1   = _conv_block(128, 64)   # 64 + 64 = 128

        self.out    = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        # Decode with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.tanh(self.out(d1)) * EPS_PGD
