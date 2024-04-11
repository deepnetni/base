from typing import Tuple
from numba import uint
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange


class TCNBLK(nn.Module):
    """TCN along the `dilation_dim` dimension.

    Input: B,C,T,F
    Output: B,C,T,F
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        kernel_size: Tuple = (2, 3),
        dilation: Tuple = (1, 1),
        use_skip_connection=True,
        causal: bool = False,
    ):
        super(TCNBLK, self).__init__()
        self.causal = causal
        self.use_skip_connection = use_skip_connection

        self.up_channel = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.PReLU(),
            nn.GroupNorm(1, mid_channels),  # layernorm
        )

        # dilation kernel size is: 1 + dil * (k-1)
        if not causal:
            pad1 = dilation[0] * (kernel_size[0] - 1) // 2
            pad2 = dilation[1] * (kernel_size[1] - 1) // 2
            padding = (pad2, pad2, pad1, pad1)
        else:
            pad1 = dilation[0] * (kernel_size[0] - 1)
            pad2 = dilation[1] * (kernel_size[1] - 1)
            padding = (pad2, 0, pad1, 0)

        self.depthwise_conv = nn.Sequential(
            nn.ConstantPad2d(padding, 0.0),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                groups=mid_channels,
                dilation=dilation,
                stride=1,
            ),
        )
        self.post = nn.Sequential(
            nn.GroupNorm(1, mid_channels), nn.Conv2d(mid_channels, out_channels, (1, 1))
        )

    def forward(self, x):
        x = self.up_channel(x)
        x = self.depthwise_conv(x)
        x = self.post(x)

        return x


if __name__ == "__main__":
    net = TCNBLK(2, 3, 4, dilation=(2, 5), kernel_size=(3, 5))
    inp = torch.randn(1, 2, 10, 7)
    out = net(inp)
    print(out.shape)
    pass
