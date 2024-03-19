import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from typing import Tuple
from einops import rearrange
import torch
import torch.nn as nn
from models.conv_stft import STFT

from models.DenseNet import DenseNet


class FuseBLK(nn.Module):
    """
    input
    -----
    B,C,T,F
    """

    def __init__(self, in_channels: int, depth: int = 4, dense_kernel: Tuple = (3, 5)):
        super().__init__()

        self.layers = nn.Sequential(
            DenseNet(depth=depth, in_channels=in_channels, kernel_size=dense_kernel)
        )

        pass

    def forward(self, x):
        pass


class MCSE(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.stft = STFT(512, 256, 512)
        pass

    def forward(self, x):
        """
        x:B,T,M
        """
        m = x.size(-1)
        x = rearrange(x, "b t m ->(b m) t")
        xk = self.stft.transform(x)  # B,2,T,F
        # m c, ((r,i), (r,i), ...)
        xk = rearrange(xk, "(b m) c t f->b (m c) t f", m=m)


if __name__ == "__main__":
    inp = torch.randn(2, 16000, 6)
    net = MCSE()
    out = net(inp)
    pass
