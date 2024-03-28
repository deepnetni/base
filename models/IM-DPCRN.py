import sys
from pathlib import Path

from einops import rearrange

sys.path.append(str(Path(__file__).parent.parent))
from typing import List, Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.conv_stft import STFT
from models.ft_lstm import FTLSTM_RESNET, FTLSTM_RESNET_ATT


class DenseBlock(nn.Module):  # dilated dense block
    """input_size the F dimension of b,c,t,f"""

    def __init__(self, depth=4, in_channels=64, input_size: int = 257):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        # self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(self, "norm{}".format(i + 1), nn.LayerNorm(input_size))
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class dense_encoder(nn.Module):
    """
    Input: B,C,T,F

    Arugments:
      - in_chanels: C of input;
      - feature_size: F of input
    """

    def __init__(
        self, in_channels: int, feature_size: int, out_channels, depth: int = 4
    ):
        super(dense_encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pre_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
            ),  # [b, 64, nframes, 512]
            nn.LayerNorm(feature_size),
            nn.PReLU(out_channels),
        )
        self.enc_dense = DenseBlock(
            depth=depth, in_channels=out_channels, input_size=feature_size
        )

        self.post_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # //2
                padding=(0, 1),
            ),
            nn.LayerNorm(feature_size // 2 + 1),
            nn.PReLU(out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # no padding
            ),
            nn.LayerNorm(feature_size // 4),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.enc_dense(x)
        x = self.post_conv(x)

        return x


class SPConvTranspose2d(nn.Module):  # sub-pixel convolution
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        # batch_size, nchannels, H, W = out.shape
        out = rearrange(out, "b (r c) t f-> b c t (f r)", r=self.r)
        # out = out.view((batch_size, self.r, nchannels // self.r, H, W))  # b,r1,r2,h,w
        # out = out.permute(0, 2, 3, 4, 1)  # b,r2,h,w,r1
        # out = out.contiguous().view(
        #     (batch_size, nchannels // self.r, H, -1)
        # )  # b, r2, h, w*r
        return out


class dense_decoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, feature_size: int, depth: int = 4
    ):
        super(dense_decoder, self).__init__()
        self.out_channels = 1
        self.in_channels = in_channels
        self.dec_dense = DenseBlock(
            depth=depth, in_channels=in_channels, input_size=feature_size
        )

        self.dec_conv1 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),  # padding F
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),
            nn.LayerNorm(feature_size * 2),
            nn.PReLU(in_channels),
        )

        feature_size = feature_size * 2

        self.dec_conv2 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),
            nn.ConstantPad2d((1, 0, 0, 0), value=0.0),
            nn.LayerNorm(feature_size * 2 + 1),
            nn.PReLU(in_channels),
        )
        #
        self.out_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

    def forward(self, x):
        out = self.dec_dense(x)
        out = self.dec_conv1(out)

        out = self.dec_conv2(out)
        out = self.out_conv(out)
        return out


class BaseCRN(nn.Module):
    def __init__(self, mid_channels: int = 64):
        super().__init__()
        self.stft = STFT(512, 256)
        self.encode_spec = dense_encoder(
            in_channels=2, feature_size=257, out_channels=mid_channels, depth=4
        )
        self.de1 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )
        self.de2 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )

        pass

    def forward(self, x):
        pass
