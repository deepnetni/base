from re import sub
import sys
from pathlib import Path

from einops import rearrange
import einops
from scipy.signal import dimpulse

sys.path.append(str(Path(__file__).parent.parent))
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.LayerNorm(feature_size // 4 + 1),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.enc_dense(x)
        x = self.post_conv(x)

        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: Tuple = (1, 3), r=1):
        super(SPConvTranspose2d, self).__init__()
        npad = kernel_size[1] // 2
        self.layer = nn.Sequential(
            nn.ConstantPad2d((npad, npad, 0, 0), value=0.0),
            nn.Conv2d(
                in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
            ),
        )
        self.r = r

    def forward(self, x):
        out = self.layer(x)
        out = einops.rearrange(out, "b (r c) t f->b c t (f r)", r=self.r)
        # batch_size, nchannels, H, W = out.shape
        # out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        # out = out.permute(0, 2, 3, 4, 1)  # b,nc/r,h,w,r
        # out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class dense_decoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, feature_size: int, depth: int = 4
    ):
        super().__init__()
        self.out_channels = 1
        self.in_channels = in_channels
        self.dec_dense = DenseBlock(
            depth=depth, in_channels=in_channels, input_size=feature_size
        )

        self.dec_conv1 = nn.Sequential(
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),
            nn.LayerNorm(feature_size * 2),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, in_channels, (1, 2)),
            nn.LayerNorm(feature_size * 2 - 1),
            nn.PReLU(in_channels),
        )

        feature_size = feature_size * 2 - 1

        self.dec_conv2 = nn.Sequential(
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),
            nn.LayerNorm(feature_size * 2),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, in_channels, (1, 2)),
            nn.LayerNorm(feature_size * 2 - 1),
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


class SubBands(nn.Module):
    """
    Input: B,C,T,F
    Output: B, C x sub[0] x sub[1], T,F
    """

    def __init__(self, sub_freqs: Tuple = (2, 3)) -> None:
        super(SubBands, self).__init__()
        self.sub_freqs = sub_freqs
        assert sub_freqs[0] != 0

    def forward(self, x):
        """
        x: b,c,t,f
        """
        nB = x.size(0)
        x = torch.concat(
            [x[..., -self.sub_freqs[0] :], x, x[..., : self.sub_freqs[1]]], dim=-1
        )

        x = einops.rearrange(x, "b c t f->(b t) c f 1")
        # to bt,c,f+n1+n2,1

        # c x (sum(subs)+1)
        x = F.unfold(x, kernel_size=(sum(self.sub_freqs) + 1, 1))
        x = einops.rearrange(x, "(b t) c f->b c t f", b=nB)

        return x


class BaseCRN(nn.Module):
    def __init__(self, mid_channels: int = 128):
        super().__init__()
        self.stft = STFT(512, 256)
        self.encode_spec = dense_encoder(
            in_channels=4, feature_size=257, out_channels=mid_channels, depth=4
        )
        self.de_r = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=65,  # 257//4+1
            depth=4,
        )
        self.de_i = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=65,
            depth=4,
        )

        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(mid_channels, 128),
            FTLSTM_RESNET(mid_channels, 128),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(mid_channels, 128),
            FTLSTM_RESNET(mid_channels, 128),
        )

    def forward(self, mic, ref):
        xk_m = self.stft.transform(mic)  # b,2,t,f
        xk_r = self.stft.transform(ref)

        xk_m_r, xk_m_i = xk_m.chunk(2, dim=1)
        xk_r_r, xk_r_i = xk_r.chunk(2, dim=1)

        xk = torch.concat([xk_m_r, xk_r_r, xk_m_i, xk_r_i], dim=1)
        xk = self.encode_spec(xk)  # b,4,t,f

        # xk_r, xk_i = xk.chunk(2, dim=1)
        xk_r = self.rnns_r(xk)
        xk_i = self.rnns_i(xk)

        mask_r = self.de_r(xk_r)
        mask_i = self.de_i(xk_i)

        xk_r = xk_m_r * mask_r - xk_m_i * mask_i
        xk_i = xk_m_r * mask_i + xk_m_i * mask_r

        xk = torch.concat([xk_r, xk_i], dim=1)

        out = self.stft.inverse(xk)

        return out


class BaseCRNwSubBands(nn.Module):
    def __init__(self, mid_channels: int = 128, sub=(2, 3)):
        super().__init__()
        self.stft = STFT(512, 256)
        self.subf = nn.Sequential(
            SubBands(sub),
            dense_encoder(
                in_channels=4 * (sum(sub) + 1),
                feature_size=257,
                out_channels=mid_channels,
                depth=4,
            ),
        )
        self.encode_spec = dense_encoder(
            in_channels=4, feature_size=257, out_channels=mid_channels, depth=4
        )
        self.de_r = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=65,  # 257//4+1
            depth=4,
        )
        self.de_i = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=65,
            depth=4,
        )

        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(mid_channels, 128),
            FTLSTM_RESNET(mid_channels, 128),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(mid_channels, 128),
            FTLSTM_RESNET(mid_channels, 128),
        )

    def forward(self, mic, ref):
        xk_m = self.stft.transform(mic)  # b,2,t,f
        xk_r = self.stft.transform(ref)

        xk_m_r, xk_m_i = xk_m.chunk(2, dim=1)
        xk_r_r, xk_r_i = xk_r.chunk(2, dim=1)

        xk = torch.concat([xk_m_r, xk_r_r, xk_m_i, xk_r_i], dim=1)

        xk_subs = self.subf(xk)
        xk = self.encode_spec(xk)  # b,4,t,f

        xk = xk + xk_subs

        # xk_r, xk_i = xk.chunk(2, dim=1)
        xk_r = self.rnns_r(xk)
        xk_i = self.rnns_i(xk)

        mask_r = self.de_r(xk_r)
        mask_i = self.de_i(xk_i)

        xk_r = xk_m_r * mask_r - xk_m_i * mask_i
        xk_i = xk_m_r * mask_i + xk_m_i * mask_r

        xk = torch.concat([xk_r, xk_i], dim=1)

        out = self.stft.inverse(xk)

        return out


if __name__ == "__main__":
    from thop import profile

    # net = BaseCRN(64)
    net = BaseCRNwSubBands(64)
    inp = torch.randn(1, 16000)
    out = net(inp, inp)

    flops, param = profile(net, inputs=(inp, inp))
    print(flops / 1e9, param / 1e6)

    # net = SubBands(sub_freqs=(1, 3))
    # inp = torch.randn(1, 2, 10, 257)

    # out = net(inp)
    # print(out.shape)
