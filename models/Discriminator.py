import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent))

from models.conv_stft import STFT
from einops.layers.torch import Rearrange


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class Discriminator(nn.Module):
    """
    Input:
        x, y: B,T
    Return:
        score
    """

    def __init__(self, nframe, nhop, nfft=None, ndf=16, in_channel=2):
        super().__init__()
        self.stft = STFT(nframe, nhop, nfft=nfft)
        self.layers = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.parametrizations.spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            nn.utils.parametrizations.spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1),
            # nn.Sigmoid(),
        )

    @staticmethod
    def compressed_mag_complex(spec, compress_factor=0.3):
        """
        spec: B,2,T,F
        return: B,3,T,F
        """
        mag2 = torch.maximum(
            torch.sum((spec * spec), dim=1, keepdim=True),
            torch.zeros_like(torch.sum((spec * spec), dim=1, keepdim=True)) + 1e-12,
        )
        spec_cpr = torch.pow(mag2, (compress_factor - 1) / 2) * spec
        spec_mag_cpr = torch.pow(mag2, compress_factor / 2)
        features_mix = torch.cat((spec_mag_cpr, spec_cpr), dim=1)

        return features_mix, spec_mag_cpr

    def forward(self, x, y):
        x = self.stft.transform(x)  # B,2,T,F
        y = self.stft.transform(y)  # B,2,T,F

        _, mag_cpr_x = self.compressed_mag_complex(x)
        _, mag_cpr_y = self.compressed_mag_complex(y)
        mag_cpr = torch.cat([mag_cpr_x, mag_cpr_y], dim=1)

        return self.layers(mag_cpr)


if __name__ == "__main__":
    net = Discriminator(512, 256)
    inp = torch.randn(1, 16000)
    out = net(inp)
    print(out)
