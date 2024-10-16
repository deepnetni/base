#!/usr/bin/env python3
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange


class DConvBLK(nn.Module):
    """Dilated Convolution
    Input: B,C,T,F
    """

    def __init__(
        self,
        inp_channel: int,
        feat_size: int,
        kernel_size: Tuple[int, int] = (2, 3),
        dil: int = 1,
    ) -> None:
        super().__init__()

        twidth, fwidth = (*kernel_size,)
        npad_t = twidth + (dil - 1) * (twidth - 1) - 1
        npad_f = (fwidth - 1) // 2

        self.layer = nn.Sequential(
            nn.Conv2d(inp_channel, inp_channel, (1, 1), (1, 1)),
            nn.LayerNorm(feat_size),
            nn.PReLU(inp_channel),
            nn.ConstantPad2d((npad_f, npad_f, npad_t, 0), value=0.0),
            nn.Conv2d(
                inp_channel,
                inp_channel,
                kernel_size,
                (1, 1),
                dilation=(dil, 1),
                groups=inp_channel,
            ),
            nn.LayerNorm(feat_size),
            nn.PReLU(inp_channel),
            nn.Conv2d(inp_channel, inp_channel, (1, 1), (1, 1)),
        )

    def forward(self, x):
        return x + self.layer(x)


if __name__ == "__main__":
    net = DConvBLK(2, 4)
    inp = torch.randn(1, 2, 1, 4)
    out = net(inp)
    print(out.shape)
