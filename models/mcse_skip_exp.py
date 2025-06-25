import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from models.aia_trans import (
    AHAM,
    AIA_Transformer_cau,
    AIA_Transformer,
    DF_dense_decoder,
    DfOutputReshapeMF,
    SPConvTranspose2d,
)
from models.CMGAN.conformer import ConformerBlock
from models.CMGAN.generator import TSCB
from models.SpatialFeatures import SpatialFeats
from models.conv_stft import STFT
from models.multiframe import DF
from models.Fusion.ms_cam import AFF
from models.FTConformer import FConformer, FTConformer, TConformerConv, TConformerQKV
from models.ft_lstm import FTLSTM_RESNET
from models.DConvs import DConvBLK
from models.ft_lstm import FTLSTM_RESNET_Train

from models.FactorizedAttention import FactorizedAttn
from utils.register import tables


class MMEBLK_ALL_ATT(nn.Module):
    """Multi-microphone encoder block
    Input: B,(CM),T,F
    """

    def __init__(self, in_channels, out_channels, ndim, n_mic=6) -> None:
        super().__init__()

        self.en_ch = nn.Sequential(
            Rearrange("b (c m) t f->b (m c) t f", m=n_mic),
            nn.Conv2d(
                in_channels,
                out_channels,
                (1, 1),
                (1, 1),
                groups=n_mic,
            ),
            nn.GroupNorm(n_mic, out_channels),
            nn.PReLU(out_channels),
            Rearrange("b (m c) t f->b (c m) t f", m=n_mic),
        )
        self.en_ri = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), groups=2),
            nn.GroupNorm(2, out_channels),
            nn.PReLU(out_channels),
        )
        self.en_spec = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1)),
            nn.GroupNorm(1, out_channels),
            nn.PReLU(out_channels),
        )

        self.attn = TConformerQKV(out_channels, 4, r=2)
        self.attn2 = TConformerQKV(out_channels, 4, r=2)

    def forward(self, x):
        xch = self.en_ch(x)
        xri = self.en_ri(x)
        xsp = self.en_spec(x)

        xff = self.attn(xch, xri, xri)  # B,C,T,F
        xsp = self.attn2(xff, xsp, xsp)

        return xsp, xch, xri


class MMEBLK_ALL_AFF(nn.Module):
    """Multi-microphone encoder block
    Input: B,(CM),T,F
    """

    def __init__(self, in_channels, out_channels, ndim, n_mic=6) -> None:
        super().__init__()

        self.en_ch = nn.Sequential(
            Rearrange("b (c m) t f->b (m c) t f", m=n_mic),
            nn.Conv2d(
                in_channels,
                out_channels,
                (1, 1),
                (1, 1),
                groups=n_mic,
            ),
            nn.GroupNorm(n_mic, out_channels),
            nn.PReLU(out_channels),
            Rearrange("b (m c) t f->b (c m) t f", m=n_mic),
        )
        self.en_ri = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), groups=2),
            nn.GroupNorm(2, out_channels),
            nn.PReLU(out_channels),
        )
        self.en_spec = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1)),
            nn.GroupNorm(1, out_channels),
            nn.PReLU(out_channels),
        )

        self.aff = AFF(out_channels, ndim, r=1)
        self.aff2 = AFF(out_channels, ndim, r=1)

    def forward(self, x):
        xch = self.en_ch(x)
        xri = self.en_ri(x)
        xsp = self.en_spec(x)

        xff = self.aff(xch, xri)  # B,C,T,F
        xsp = self.aff2(xsp, xff)

        return xsp, xch, xri


class MMEBLK(nn.Module):
    """Multi-microphone encoder block
    Input: B,(CM),T,F
    """

    def __init__(self, in_channels, out_channels, ndim, n_mic=6, cau=True) -> None:
        super().__init__()

        n_mic = in_channels // 2
        self.en_ch = nn.Sequential(
            Rearrange("b (c m) t f->b (m c) t f", m=n_mic),
            nn.Conv2d(
                in_channels,
                out_channels,
                (1, 1),
                (1, 1),
                groups=n_mic,
            ),
            nn.GroupNorm(n_mic, out_channels),
            nn.PReLU(out_channels),
            Rearrange("b (m c) t f->b (c m) t f", m=n_mic),
        )
        self.en_ri = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), groups=2),
            nn.GroupNorm(2, out_channels),
            nn.PReLU(out_channels),
        )
        self.en_spec = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1)),
            nn.GroupNorm(1, out_channels),
            nn.PReLU(out_channels),
        )

        self.aff = AFF(out_channels, ndim, r=1)
        self.attn = TConformerQKV(out_channels, 4, r=2, causal=cau)

    def forward(self, x):
        xch = self.en_ch(x)
        xri = self.en_ri(x)
        xsp = self.en_spec(x)

        xff = self.aff(xch, xri)  # B,C,T,F
        xsp = self.attn(xff, xsp, xsp)

        return xsp, xch, xri


class DenseMPBLKBaseline(nn.Module):
    """Dense Multi-Path Block"""

    def __init__(
        self,
        depth=4,
        in_channels=64,
        input_size: int = 257,
        kernel_size=(2, 3),
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        # self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        twidth = kernel_size[0]
        for i in range(self.depth):
            dil = 2**i
            pad_length = twidth + (dil - 1) * (twidth - 1) - 1

            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),  # lrtb
                    nn.Conv2d(
                        in_channels * (i + 1),
                        in_channels,
                        kernel_size=kernel_size,
                        dilation=(dil, 1),
                    ),
                    nn.LayerNorm(input_size),
                    nn.PReLU(in_channels),
                    # nn.Conv2d(in_channels, in_channels, (1, 3), (1, 1), (0, 1)),
                    # nn.LayerNorm(input_size),
                    # nn.PReLU(in_channels),
                ),
            )
            # setattr(
            #     self,
            #     f"pwc{i+1}",
            #     nn.Sequential(
            #         nn.Conv2d(in_channels * (i + 1), in_channels, (1, 3), (1, 1), (0, 1)),
            #         nn.LayerNorm(input_size),
            #         nn.PReLU(in_channels),
            #     ),
            # )
            # setattr(self, "fu{}".format(i + 1), AFF(in_channels, input_size, r=1))
            # setattr(self, "att{}".format(i + 1), FTConformer(in_channels, 4))

        self.post = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1)))

    def forward(self, x):
        skip = x
        # mid = []
        for i in range(self.depth):
            out = getattr(self, "conv{}".format(i + 1))(skip)
            skip = torch.cat([out, skip], dim=1)

        return out, self.post(out)


class DenseMPBLK(nn.Module):
    """Dense Multi-Path Block"""

    def __init__(
        self,
        depth=4,
        in_channels=64,
        input_size: int = 257,
        kernel_size=(2, 3),
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        # self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        twidth = kernel_size[0]
        for i in range(self.depth):
            dil = 2**i
            pad_length = twidth + (dil - 1) * (twidth - 1) - 1

            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),  # lrtb
                    nn.Conv2d(
                        in_channels * (i + 1),
                        in_channels,
                        kernel_size=kernel_size,
                        dilation=(dil, 1),
                    ),
                    nn.LayerNorm(input_size),
                    nn.PReLU(in_channels),
                    # nn.Conv2d(in_channels, in_channels, (1, 3), (1, 1), (0, 1)),
                    # nn.LayerNorm(input_size),
                    # nn.PReLU(in_channels),
                ),
            )
            setattr(
                self,
                f"pwc{i+1}",
                nn.Sequential(
                    nn.Conv2d(in_channels * (i + 1), in_channels, (1, 3), (1, 1), (0, 1)),
                    nn.LayerNorm(input_size),
                    nn.PReLU(in_channels),
                ),
            )
            setattr(self, "fu{}".format(i + 1), AFF(in_channels, input_size, r=1))
            setattr(self, "att{}".format(i + 1), FTConformer(in_channels, 4))
            # setattr(
            #     self,
            #     "post{}".format(i + 1),
            #     nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
            # )

            # setattr(
            #     self,
            #     "normAct{}".format(i + 1),
            #     nn.Sequential(nn.LayerNorm(input_size), nn.PReLU(in_channels)),
            # )

        self.post = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1)))

    def forward(self, x):
        skip = x
        # mid = []
        for i in range(self.depth):
            out_1 = getattr(self, "conv{}".format(i + 1))(skip)
            out_2 = getattr(self, "pwc{}".format(i + 1))(skip)
            # out = torch.concat([out_1, out_2], dim=1)  # bctf
            out = getattr(self, f"fu{i+1}")(out_1, out_2)
            # out = getattr(self, "normAct{}".format(i + 1))(out)
            out = getattr(self, "att{}".format(i + 1))(out)
            # mid.append(getattr(self, "post{}".format(i + 1))(out))
            skip = torch.cat([out, skip], dim=1)

        # return out, mid[::-1]
        return out, self.post(out)


# class DenseMPBLK(nn.Module):
#     """Dense Multi-Path Block"""

#     def __init__(
#         self,
#         depth=4,
#         in_channels=64,
#         input_size: int = 257,
#         kernel_size=(2, 3),
#     ):
#         super().__init__()
#         self.depth = depth
#         self.in_channels = in_channels
#         # self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
#         twidth = kernel_size[0]
#         for i in range(self.depth):
#             dil = 2**i
#             pad_length = twidth + (dil - 1) * (twidth - 1) - 1

#             setattr(
#                 self,
#                 "dconv{}".format(i + 1),
#                 nn.Sequential(
#                     nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),  # lrtb
#                     nn.Conv2d(
#                         in_channels * (i + 1),
#                         in_channels,
#                         kernel_size=kernel_size,
#                         dilation=(dil, 1),
#                     ),
#                     nn.GroupNorm(1, in_channels),
#                     nn.PReLU(in_channels),
#                 ),
#             )
#             setattr(
#                 self,
#                 "dgconv{}".format(i + 1),
#                 nn.Sequential(
#                     nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),  # lrtb
#                     nn.Conv2d(
#                         in_channels * (i + 1),
#                         in_channels,
#                         kernel_size=kernel_size,
#                         dilation=(dil, 1),
#                         groups=i + 1,
#                     ),
#                     nn.GroupNorm(i + 1, in_channels),
#                     nn.PReLU(in_channels),
#                 ),
#             )
#             setattr(
#                 self,
#                 f"gconv{i+1}",
#                 nn.Sequential(
#                     nn.Conv2d(
#                         in_channels * (i + 1),
#                         in_channels,
#                         (1, 3),
#                         (1, 1),
#                         (0, 1),
#                         groups=i + 1,
#                     ),
#                     nn.GroupNorm(i + 1, in_channels),
#                     nn.PReLU(in_channels),
#                 ),
#             )
#             setattr(
#                 self,
#                 f"midconv{i+1}",
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1)),
#                     nn.LayerNorm(input_size),
#                     nn.PReLU(in_channels),
#                 ),
#             )
#             setattr(self, "fu{}".format(i + 1), AFF(in_channels, input_size, r=1))
#             setattr(
#                 self,
#                 "normAct{}".format(i + 1),
#                 nn.Sequential(nn.LayerNorm(input_size), nn.PReLU(in_channels)),
#             )
#             setattr(self, "fumid{}".format(i + 1), AFF(in_channels, input_size, r=1))
#             # setattr(self, "fatt{}".format(i + 1), FConformer(in_channels, 4))
#             # setattr(self, "tatt{}".format(i + 1), TConformerQKV(in_channels, 4))
#             setattr(self, "att{}".format(i + 1), FTConformer(in_channels, 4))

#             setattr(
#                 self,
#                 "post{}".format(i + 1),
#                 nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
#             )

#     def forward(self, x):
#         skip = x
#         mid = []
#         for i in range(self.depth):
#             out = getattr(self, "dconv{}".format(i + 1))(skip)
#             out_1 = getattr(self, "dgconv{}".format(i + 1))(skip)
#             out_2 = getattr(self, "gconv{}".format(i + 1))(skip)
#             x = getattr(self, f"fu{i+1}")(out_1, out_2)
#             x = getattr(self, "normAct{}".format(i + 1))(x)

#             x = getattr(self, f"midconv{i+1}")(x)
#             x = getattr(self, f"fumid{i+1}")(out, x)
#             # x = getattr(self, "fatt{}".format(i + 1))(out_3)
#             out = getattr(self, "att{}".format(i + 1))(x)
#             mid.append(getattr(self, "post{}".format(i + 1))(out))

#             skip = torch.cat([out, skip], dim=1)

#         # return out, mid[::-1]
#         return out, mid


class EncoderBaseline(nn.Module):
    """input B,(CM),T,F"""

    def __init__(
        self,
        in_channels: int,
        feat_size: int,
        depth: int,
        return_steps=False,
    ) -> None:
        super().__init__()

        self.return_steps = return_steps
        self.enc_dense = DenseMPBLKBaseline(
            in_channels=in_channels,
            input_size=feat_size,
            kernel_size=(2, 3),
            depth=depth,
        )

    def forward(self, x):
        x, x_l = self.enc_dense(x)

        if self.return_steps:
            return x, x_l
        else:
            return x


class EncoderATT(nn.Module):
    """input B,(CM),T,F"""

    def __init__(
        self,
        in_channels: int,
        feat_size: int,
        depth: int,
        return_steps=False,
    ) -> None:
        super().__init__()

        self.return_steps = return_steps
        self.enc_dense = DenseMPBLK(
            in_channels=in_channels,
            input_size=feat_size,
            kernel_size=(2, 3),
            depth=depth,
        )

    def forward(self, x):
        x, x_l = self.enc_dense(x)

        if self.return_steps:
            return x, x_l
        else:
            return x


# class DenseBLK(nn.Module):  # dilated dense block
#     def __init__(
#         self,
#         depth=4,
#         in_channels=64,
#         input_size: int = 257,
#         kernel_size=(2, 3),
#     ):
#         super().__init__()
#         self.depth = depth
#         self.in_channels = in_channels
#         # self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
#         twidth = kernel_size[0]
#         for i in range(self.depth):
#             dil = 2**i
#             pad_length = twidth + (dil - 1) * (twidth - 1) - 1
#             setattr(
#                 self,
#                 "pad{}".format(i + 1),
#                 nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),  # lrtb
#             )

#             setattr(
#                 self,
#                 "conv{}".format(i + 1),
#                 nn.Sequential(
#                     nn.Conv2d(
#                         in_channels * (i + 1),
#                         in_channels,
#                         kernel_size=kernel_size,
#                         dilation=(dil, 1),
#                     ),
#                     nn.LayerNorm(input_size),
#                     nn.PReLU(in_channels),
#                     # nn.Conv2d(in_channels, in_channels, (1, 3), (1, 1), (0, 1)),
#                     # nn.LayerNorm(input_size),
#                     # nn.PReLU(in_channels),
#                 ),
#             )
#             setattr(
#                 self,
#                 f"pwc{i+1}",
#                 nn.Sequential(
#                     nn.Conv2d(
#                         in_channels * (i + 1), in_channels, (1, 3), (1, 1), (0, 1)
#                     ),
#                     nn.LayerNorm(input_size),
#                     nn.PReLU(in_channels),
#                 ),
#             )
#             setattr(self, "fu{}".format(i + 1), AFF(in_channels, input_size, r=1))
#             setattr(
#                 self,
#                 "post{}".format(i + 1),
#                 nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
#             )

#             # setattr(self, "norm{}".format(i + 1), nn.LayerNorm(input_size))
#             # setattr(self, "prelu{}".format(i + 1), nn.PReLU(in_channels))

#     def forward(self, x):
#         skip = x
#         mid = []
#         for i in range(self.depth):
#             out = getattr(self, "pad{}".format(i + 1))(skip)
#             out_1 = getattr(self, "conv{}".format(i + 1))(out)
#             out_2 = getattr(self, "pwc{}".format(i + 1))(skip)
#             # out = torch.concat([out_1, out_2], dim=1)  # bctf
#             out = getattr(self, f"fu{i+1}")(out_1, out_2)
#             # out = getattr(self, "norm{}".format(i + 1))(out)
#             # out = getattr(self, "prelu{}".format(i + 1))(out)
#             mid.append(getattr(self, "post{}".format(i + 1))(out))
#             skip = torch.cat([out, skip], dim=1)

#         # return out, mid[::-1]
#         return out, mid


# class Encoder(nn.Module):
#     """input B,(CM),T,F"""

#     def __init__(
#         self, inp_channels: int, mid_channels: int, feat_size: int, return_steps=False
#     ) -> None:
#         super().__init__()

#         self.return_steps = return_steps
#         self.pre_layer = nn.Sequential(
#             nn.Conv2d(inp_channels, mid_channels, (1, 1), (1, 1), (0, 0)),
#             nn.LayerNorm(feat_size),
#             nn.PReLU(mid_channels),
#         )
#         self.enc_dense = DenseBLK(
#             in_channels=mid_channels,
#             input_size=feat_size,
#             kernel_size=(2, 3),
#             depth=4,
#         )
#         self.post_conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=mid_channels,
#                 out_channels=mid_channels,
#                 kernel_size=(1, 3),
#                 stride=(1, 2),  # //2
#                 padding=(0, 1),
#             ),
#             nn.LayerNorm(feat_size // 2 + 1),
#             nn.PReLU(mid_channels),
#             nn.Conv2d(
#                 in_channels=mid_channels,
#                 out_channels=mid_channels,
#                 kernel_size=(1, 3),
#                 stride=(1, 2),  # no padding
#             ),
#             nn.LayerNorm(feat_size // 4),
#             nn.PReLU(mid_channels),
#         )

#     def forward(self, x):
#         x = self.pre_layer(x)
#         x, x_l = self.enc_dense(x)
#         x = self.post_conv(x)

#         if self.return_steps:
#             return x, x_l
#         else:
#             return x


class DenseSkipBLKBaseline(nn.Module):
    def __init__(
        self,
        depth=4,
        in_channels=64,
        input_size: int = 257,
        kernel_size=(2, 3),
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        # self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        twidth = kernel_size[0]
        self.pref = AFF(in_channels, input_size, r=1)

        for i in range(self.depth):
            dil = 2**i
            pad_length = twidth + (dil - 1) * (twidth - 1) - 1

            setattr(
                self,
                "dconv{}".format(i + 1),
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
                    nn.Conv2d(
                        in_channels * (i + 1),
                        in_channels,
                        kernel_size=kernel_size,
                        dilation=(dil, 1),
                    ),
                    nn.LayerNorm(input_size),
                    nn.PReLU(in_channels),
                    # nn.Conv2d(in_channels, in_channels, (1, 3), (1, 1), (0, 1)),
                    # nn.LayerNorm(input_size),
                    # nn.PReLU(in_channels),
                ),
            )

    def forward(self, x, y):
        """y is the skip conn"""
        # skip = self.pref(x, y)
        skip = x

        for i in range(self.depth):
            out = getattr(self, "dconv{}".format(i + 1))(skip)
            skip = torch.cat([out, skip], dim=1)
        return out


class DenseSkipBLK(nn.Module):
    def __init__(
        self,
        depth=4,
        in_channels=64,
        input_size: int = 257,
        kernel_size=(2, 3),
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        # self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        twidth = kernel_size[0]
        self.pref = AFF(in_channels, input_size, r=1)

        for i in range(self.depth):
            dil = 2**i
            pad_length = twidth + (dil - 1) * (twidth - 1) - 1

            setattr(
                self,
                "dconv{}".format(i + 1),
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
                    nn.Conv2d(
                        in_channels * (i + 1),
                        in_channels,
                        kernel_size=kernel_size,
                        dilation=(dil, 1),
                    ),
                    nn.LayerNorm(input_size),
                    nn.PReLU(in_channels),
                    # nn.Conv2d(in_channels, in_channels, (1, 3), (1, 1), (0, 1)),
                    # nn.LayerNorm(input_size),
                    # nn.PReLU(in_channels),
                ),
            )

            setattr(
                self,
                f"conv{i+1}",
                nn.Sequential(
                    nn.Conv2d(in_channels * (i + 1), in_channels, (1, 3), (1, 1), (0, 1)),
                    nn.LayerNorm(input_size),
                    nn.PReLU(in_channels),
                ),
            )
            # setattr(self, "att{}".format(i + 1), FTConformer(in_channels, 4))
            setattr(self, "fu{}".format(i + 1), AFF(in_channels, input_size, r=1))
            # setattr(
            #     self,
            #     "normAct{}".format(i + 1),
            #     nn.Sequential(nn.LayerNorm(input_size), nn.PReLU(in_channels)),
            # )

    def forward(self, x, y):
        """y is the skip conn"""
        # skip = self.pref(x, y)
        skip = x

        for i in range(self.depth):
            # x = torch.concat([skip, y[i]], dim=1)

            out_1 = getattr(self, "dconv{}".format(i + 1))(skip)
            out_2 = getattr(self, "conv{}".format(i + 1))(skip)
            out = getattr(self, "fu{}".format(i + 1))(out_1, out_2)
            # out = getattr(self, "normAct{}".format(i + 1))(out)
            # out = getattr(self, "att{}".format(i + 1))(out)

            skip = torch.cat([out, skip], dim=1)

        return out


class DecoderBaseline(nn.Module):
    def __init__(self, in_channels, feat_size, depth=4) -> None:
        super().__init__()

        self.dec_dense = DenseSkipBLKBaseline(
            # self.dec_dense = DenseSkipATTBLK(
            depth=depth,
            in_channels=in_channels,
            input_size=feat_size,
            kernel_size=(2, 3),
        )

    def forward(self, x, skip):
        out = self.dec_dense(x, skip)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, feat_size, depth=4) -> None:
        super().__init__()

        self.dec_dense = DenseSkipBLK(
            # self.dec_dense = DenseSkipATTBLK(
            depth=depth,
            in_channels=in_channels,
            input_size=feat_size,
            kernel_size=(2, 3),
        )

    def forward(self, x, skip):
        out = self.dec_dense(x, skip)
        return out


class CHRIDecoder(nn.Module):
    def __init__(self, in_channels, feature_size, depth=4) -> None:
        super().__init__()
        self.ch_de = nn.Sequential(
            *[
                nn.Sequential(
                    FConformer(in_channels, num_head=4, r=2),
                    TConformerConv(in_channels, num_head=4, tconv=3, r=2),
                )
                for _ in range(depth)
            ]
        )
        self.ri_de = nn.Sequential(
            *[
                nn.Sequential(
                    FConformer(in_channels, num_head=4, r=2),
                    TConformerConv(in_channels, num_head=4, tconv=3, r=2),
                )
                for _ in range(depth)
            ]
        )

        self.post = AFF(in_channels, feature_size, r=1)

    def forward(self, ch, ri):
        xch = self.ch_de(ch)
        xri = self.ri_de(ri)
        out = self.post(xch, xri)
        return out


class DownSample(nn.Module):
    def __init__(self, in_channels, feat_size) -> None:
        super().__init__()
        self.ds_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # //2
                padding=(0, 1),
            ),
            nn.LayerNorm(feat_size // 2 + 1),
            nn.PReLU(in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # no padding
            ),
            nn.LayerNorm(feat_size // 4),
            nn.PReLU(in_channels),
        )

    def forward(self, x):
        x = self.ds_conv(x)
        return x


class UpSample_check(nn.Module):
    def __init__(self, in_channels, feat_size) -> None:
        super().__init__()
        self.us_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),  # padding F
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),  # F//4 => F//2, (f+2-2) * 2
            nn.LayerNorm(feat_size * 2),
            nn.PReLU(in_channels),
            #
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            nn.Conv2d(in_channels, in_channels, (1, 2)),
            nn.LayerNorm(feat_size * 4 + 1),
            nn.PReLU(in_channels),
        )

    def forward(self, x):
        x = self.us_conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, feat_size) -> None:
        super().__init__()
        self.us_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),  # padding F
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),  # F//4 => F//2, (f+2-2) * 2
            nn.LayerNorm(feat_size * 2),
            nn.PReLU(in_channels),
            #
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),
            nn.ConstantPad2d((1, 0, 0, 0), value=0.0),
            nn.LayerNorm(feat_size * 4 + 1),
            nn.PReLU(in_channels),
        )

    def forward(self, x):
        x = self.us_conv(x)
        return x


class DownSample_check(nn.Module):
    def __init__(self, in_channels, feat_size) -> None:
        super().__init__()
        self.ds_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # //2
                padding=(0, 1),
            ),
            Rearrange("b c t f->b t f c"),
            nn.LayerNorm(in_channels),
            Rearrange("b t f c->b c t f"),
            nn.PReLU(in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # no padding
                padding=(0, 1),
            ),
            Rearrange("b c t f->b t f c"),
            nn.LayerNorm(in_channels),
            Rearrange("b t f c->b c t f"),
            nn.PReLU(in_channels),
        )

    def forward(self, x):
        x = self.ds_conv(x)
        return x


# class UpSample_check(nn.Module):
#     def __init__(self, in_channels, feat_size) -> None:
#         super().__init__()
#         self.us_conv = nn.Sequential(
#             nn.ConvTranspose2d(in_channels, in_channels, (1, 3), (1, 2), (0, 1)),
#             Rearrange("b c t f->b t f c"),
#             nn.LayerNorm(in_channels),
#             Rearrange("b t f c->b c t f"),
#             # nn.BatchNorm2d(in_channels),
#             nn.PReLU(),
#             nn.ConvTranspose2d(in_channels, in_channels, (1, 3), (1, 2), (0, 1)),
#             Rearrange("b c t f->b t f c"),
#             nn.LayerNorm(in_channels),
#             Rearrange("b t f c->b c t f"),
#             # nn.BatchNorm2d(in_channels),
#             nn.PReLU(),
#             # Rearrange("b c t f -> b f t c"),
#             # nn.Conv2d(
#             #     in_channels=feat_size,
#             #     out_channels=feat_size,
#             #     kernel_size=(1, 1),
#             #     padding=(0, 0),
#             # ),
#             # nn.LayerNorm(in_channels),
#             # nn.PReLU(),
#             # Rearrange("b f t c-> b c t f"),
#         )

#     def forward(self, x):
#         x = self.us_conv(x)
#         return x


# class UpSample_check(nn.Module):
#     def __init__(self, in_channels, feat_size) -> None:
#         super().__init__()
#         self.us_conv = nn.Sequential(
#             nn.ConstantPad2d((1, 1, 0, 0), value=0.0),  # padding F
#             SPConvTranspose2d(
#                 in_channels=in_channels,
#                 out_channels=in_channels,
#                 kernel_size=(1, 3),
#                 r=2,
#             ),  # F//4 => F//2, (f+2-2) * 2
#             nn.LayerNorm(feat_size * 2),
#             nn.PReLU(in_channels),
#             nn.Conv2d(in_channels, in_channels, (1, 2)),
#             nn.PReLU(in_channels),
#             #
#             nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
#             SPConvTranspose2d(
#                 in_channels=in_channels,
#                 out_channels=in_channels,
#                 kernel_size=(1, 3),
#                 r=2,
#             ),
#             nn.LayerNorm(feat_size * 4 - 2),
#             nn.PReLU(in_channels),
#             nn.Conv2d(in_channels, in_channels, (1, 2)),
#         )

#     def forward(self, x):
#         x = self.us_conv(x)
#         return x


@tables.register("models", "mcse_3en_all_aff_wdefu_wfafu_c72_B2")
class aia_mcse_skip_sd_upch(nn.Module):
    """mcse_3en_all_aff_wdefu_wfafu
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(
        self, in_channels: int, feature_size: int, mid_channels: int, n_mic: int = 6, depth=4
    ):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = MMEBLK_ALL_AFF(in_channels * 2, mid_channels, feature_size, n_mic=n_mic)
        # self.up_ch = nn.Sequential(
        #     nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
        #     nn.LayerNorm(feature_size),
        #     nn.PReLU(mid_channels),
        # )

        self.encode = EncoderATT(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        # mic_array = [
        #     [-0.1, 0.095, 0],
        #     [0, 0.095, 0],
        #     [0.1, 0.095, 0],
        #     [-0.1, -0.095, 0],
        #     [0, -0.095, 0],
        #     [0.1, -0.095, 0],
        # ]
        # self.spf_alg = SpatialFeats(mic_array)
        # spf_in_channel = 93  # 15 * 5 + 18
        # self.en_spf = nn.Sequential(
        #     Encoder(
        #         inp_channels=spf_in_channel,
        #         mid_channels=mid_channels,
        #         feat_size=feature_size,
        #     ),  # B, mid_c, T, F // 4
        #     nn.Conv2d(mid_channels, mid_channels, (1, 1), (1, 1)),
        #     nn.Tanh(),
        # )

        # self.rnns = nn.Sequential(
        #     FTLSTM_RESNET(mid_channels, 128),
        #     FTLSTM_RESNET(mid_channels, 128),
        # )
        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = Decoder(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.fa = FactorizedAttn(mid_channels, nhead=8)
        self.fu = nn.Sequential(
            *[DConvBLK(mid_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        )

        # self.chri_de = CHRIDecoder(mid_channels, feature_size, depth=1)
        # self.attn = TConformerQKV(mid_channels, 4, r=2)

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

        # self.df_order = 5
        # self.df_bins = feature_size
        # self.DF_de = DF_dense_decoder(
        #     mid_channels, feature_size // 4, 2 * self.df_order
        # )

        # self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)
        # self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)
        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        x_enh = self.fa(x_enh, en_skip)
        x_enh = self.fu(x_enh)
        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_3en_all_att_wdefu_wfafu_c72_B2")
class aia_mcse_skip_sd_upch_att(nn.Module):
    """mcse_3en_all_att_wdefu_wfafu
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, n_mic=4, depth=4):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = MMEBLK_ALL_ATT(in_channels * 2, mid_channels, feature_size, n_mic=n_mic)
        # self.up_ch = nn.Sequential(
        #     nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
        #     nn.LayerNorm(feature_size),
        #     nn.PReLU(mid_channels),
        # )

        self.encode = EncoderATT(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        # mic_array = [
        #     [-0.1, 0.095, 0],
        #     [0, 0.095, 0],
        #     [0.1, 0.095, 0],
        #     [-0.1, -0.095, 0],
        #     [0, -0.095, 0],
        #     [0.1, -0.095, 0],
        # ]
        # self.spf_alg = SpatialFeats(mic_array)
        # spf_in_channel = 93  # 15 * 5 + 18
        # self.en_spf = nn.Sequential(
        #     Encoder(
        #         inp_channels=spf_in_channel,
        #         mid_channels=mid_channels,
        #         feat_size=feature_size,
        #     ),  # B, mid_c, T, F // 4
        #     nn.Conv2d(mid_channels, mid_channels, (1, 1), (1, 1)),
        #     nn.Tanh(),
        # )

        # self.rnns = nn.Sequential(
        #     FTLSTM_RESNET(mid_channels, 128),
        #     FTLSTM_RESNET(mid_channels, 128),
        # )
        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = Decoder(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.fa = FactorizedAttn(mid_channels, nhead=8)
        self.fu = nn.Sequential(
            *[DConvBLK(mid_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        )

        # self.chri_de = CHRIDecoder(mid_channels, feature_size, depth=1)
        # self.attn = TConformerQKV(mid_channels, 4, r=2)

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

        # self.df_order = 5
        # self.df_bins = feature_size
        # self.DF_de = DF_dense_decoder(
        #     mid_channels, feature_size // 4, 2 * self.df_order
        # )

        # self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)
        # self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)
        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        x_enh = self.fa(x_enh, en_skip)
        x_enh = self.fu(x_enh)
        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_conformer")
class aia_mcse_skip_conformer(nn.Module):
    """
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
            nn.LayerNorm(feature_size),
            nn.PReLU(mid_channels),
        )

        self.encode = EncoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.mid = nn.Sequential(
            *[TSCB(mid_channels) for _ in range(4)],
        )

        self.decode = DecoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF

        x_ri = self.mid(x_ri)

        x_enh = self.decode(x_ri, None)  # B,1,T,F

        # x_enh = self.fa(x_enh, en_skip)
        # x_enh = self.fu(x_enh)

        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


class RNN_FC(nn.Module):
    """
    feature_size: F of b,c,f,t
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = False,
        act: bool = True,
    ):
        super().__init__()
        # Sequence layer
        self.sequence_model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.sequence_model.flatten_parameters()

        # Fully connected layer

        if bidirectional:
            self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc_output_layer = nn.Linear(hidden_size, output_size)

        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, Feature]
        Returns:
            [B, T, Feature]
        """
        x, _ = self.sequence_model(x)
        x = self.act(self.fc_output_layer(x))

        return x


@tables.register("models", "mcse_baseline_bb_de_postpwc")
class aia_mcse_skip_baseline_bb_de_postpwc(nn.Module):
    """
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
            nn.LayerNorm(feature_size),
            nn.PReLU(mid_channels),
        )

        self.encode = EncoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample_check(mid_channels, feature_size)
        self.usLayer = UpSample_check(mid_channels, feature_size // 4 + 1)
        # self.usLayer = nn.Sequential(
        #     nn.Linear(feature_size // 4 + 1, feature_size),
        #     Rearrange("b c t f->b t c f"),
        #     nn.LayerNorm([mid_channels, feature_size]),
        #     Rearrange("b t c f->b c t f"),
        #     nn.PReLU(),
        # )

        # self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        # self.aham = AHAM(input_channel=mid_channels)
        self.rnns = nn.Sequential(
            FTLSTM_RESNET_Train(mid_channels, 128),
            FTLSTM_RESNET_Train(mid_channels, 128),
        )

        self.decode = DecoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size // 4 + 1,
            # feat_size=feature_size,
            depth=depth,
        )

        self.out_conv = nn.Sequential(
            # Rearrange("b c t f -> b f t c"),
            # nn.Conv2d(
            #     in_channels=feature_size,
            #     out_channels=feature_size,
            #     kernel_size=(1, 1),
            #     padding=(0, 0),
            # ),
            # nn.LayerNorm(mid_channels),
            # nn.PReLU(),
            # Rearrange("b f t c-> b c t f"),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=2,
                kernel_size=(1, 1),
            ),
            # nn.BatchNorm2d(2),
            # Rearrange("b c t f -> b f t c"),
            # nn.PReLU(),
            # nn.Conv2d(
            #     in_channels=feature_size,
            #     out_channels=feature_size,
            #     kernel_size=(1, 1),
            #     padding=(0, 0),
            # ),
            # Rearrange("b f t c-> b c t f"),
        )
        # bxt,f,c
        # self.out_conv = RNN_FC(
        #     input_size=mid_channels,
        #     hidden_size=128,
        #     output_size=2,
        #     num_layers=1,
        #     bidirectional=True,
        #     act=False,
        # )

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x = self.up_ch(x)

        x_ri, _ = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        x_ri = self.rnns(x_ri)
        # _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        # x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.decode(x_ri, None)  # B,1,T,F
        x_ri = self.usLayer(x_ri)

        # x_enh = self.fa(x_enh, en_skip)
        # x_enh = self.fu(x_enh)

        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_ri

        # feat = rearrange(feat, "b c t f->(b t) f c")
        feat = self.out_conv(feat)
        # feat = rearrange(feat, "(b t) f c->b c t f", b=nB)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_baseline_bb_postpwc")
class aia_mcse_skip_baseline_bb_postpwc(nn.Module):
    """
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
            nn.LayerNorm(feature_size),
            nn.PReLU(mid_channels),
        )

        self.encode = EncoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        # self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        # self.aham = AHAM(input_channel=mid_channels)
        self.rnns = nn.Sequential(
            FTLSTM_RESNET_Train(mid_channels, 128),
            FTLSTM_RESNET_Train(mid_channels, 128),
        )

        self.decode = DecoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 1),
            padding=(0, 0),
        )
        # bxt,f,c
        # self.out_conv = RNN_FC(
        #     input_size=mid_channels,
        #     hidden_size=128,
        #     output_size=2,
        #     num_layers=1,
        #     bidirectional=True,
        #     act=False,
        # )

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        x_ri = self.rnns(x_ri)
        # _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        # x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_ri = self.decode(x_ri, None)  # B,1,T,F

        # x_enh = self.fa(x_enh, en_skip)
        # x_enh = self.fu(x_enh)

        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_ri

        # feat = rearrange(feat, "b c t f->(b t) f c")
        feat = self.out_conv(feat)
        # feat = rearrange(feat, "(b t) f c->b c t f", b=nB)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_baseline_2_postpwc")
class aia_mcse_skip_baseline_postpwc(nn.Module):
    """
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
            nn.LayerNorm(feature_size),
            nn.PReLU(mid_channels),
        )

        self.encode = EncoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample_check(mid_channels, feature_size // 4)

        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = DecoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size // 4,
            depth=depth,
        )

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 1),
            padding=(0, 0),
        )

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x = self.up_ch(x)
        x_ri, _ = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_enh = self.usLayer(x_ri)
        x_ri = self.decode(x_ri, None)  # B,1,T,F

        # x_enh = self.fa(x_enh, en_skip)
        # x_enh = self.fu(x_enh)

        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_baseline")
class aia_mcse_skip_baseline(nn.Module):
    """
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
            nn.LayerNorm(feature_size),
            nn.PReLU(mid_channels),
        )

        self.encode = EncoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = DecoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        # x_enh = self.fa(x_enh, en_skip)
        # x_enh = self.fu(x_enh)

        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_baseline+3att")
class aia_mcse_baseline_att(nn.Module):
    """
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(
        self, in_channels: int, feature_size: int, mid_channels: int, n_mic: int = 6, depth=4
    ):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = MMEBLK_ALL_ATT(in_channels * 2, mid_channels, feature_size, n_mic=n_mic)

        self.encode = EncoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = DecoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)
        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        # x_enh = self.fa(x_enh, en_skip)
        # x_enh = self.fu(x_enh)

        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_baseline+3aff")
class aia_mcse_baseline_aff(nn.Module):
    """
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(
        self, in_channels: int, feature_size: int, mid_channels: int, n_mic: int = 6, depth=4
    ):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = MMEBLK_ALL_AFF(in_channels * 2, mid_channels, feature_size, n_mic=n_mic)

        self.encode = EncoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = DecoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)
        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        # x_enh = self.fa(x_enh, en_skip)
        # x_enh = self.fu(x_enh)

        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_baseline+3en")
class aia_mcse_skip_wodwfu_wofafu(nn.Module):
    """
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = MMEBLK(in_channels * 2, mid_channels, feature_size)

        self.encode = EncoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = DecoderBaseline(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)
        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        # x_enh = self.fa(x_enh, en_skip)
        # x_enh = self.fu(x_enh)

        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_baseline+TC+ED")
class aia_mcse_tc_ed(nn.Module):
    """
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = MMEBLK_ALL_ATT(in_channels * 2, mid_channels, feature_size)

        self.encode = EncoderATT(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = Decoder(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        # self.fa = FactorizedAttn(mid_channels, nhead=8)
        # self.fu = nn.Sequential(
        #     *[DConvBLK(mid_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        # )

        # self.chri_de = CHRIDecoder(mid_channels, feature_size, depth=1)
        # self.attn = TConformerQKV(mid_channels, 4, r=2)

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

        # self.df_order = 5
        # self.df_bins = feature_size
        # self.DF_de = DF_dense_decoder(
        #     mid_channels, feature_size // 4, 2 * self.df_order
        # )

        # self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)
        # self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)
        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        # x_enh = self.fa(x_enh, en_skip)
        # x_enh = self.fu(x_enh)

        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_baseline+3en+ED")
class aia_mcse_skip_wofafu(nn.Module):
    """mcse_3en_wdefu_wofafu
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = MMEBLK(in_channels * 2, mid_channels, feature_size)
        # self.up_ch = nn.Sequential(
        #     nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
        #     nn.LayerNorm(feature_size),
        #     nn.PReLU(mid_channels),
        # )

        self.encode = EncoderATT(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = Decoder(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        # self.fa = FactorizedAttn(mid_channels, nhead=8)
        # self.fu = nn.Sequential(
        #     *[DConvBLK(mid_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        # )

        # self.chri_de = CHRIDecoder(mid_channels, feature_size, depth=1)
        # self.attn = TConformerQKV(mid_channels, 4, r=2)

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

        # self.df_order = 5
        # self.df_bins = feature_size
        # self.DF_de = DF_dense_decoder(
        #     mid_channels, feature_size // 4, 2 * self.df_order
        # )

        # self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)
        # self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)
        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        # x_enh = self.fa(x_enh, en_skip)
        # x_enh = self.fu(x_enh)

        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_baseline+UC+ED")
class aia_mcse_skip_uc_ed(nn.Module):
    """mcse_3en_wdefu_wofafu
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        # self.up_ch = MMEBLK(in_channels * 2, mid_channels, feature_size)
        self.up_ch = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
            nn.LayerNorm(feature_size),
            nn.PReLU(mid_channels),
        )

        self.encode = EncoderATT(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = Decoder(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        # self.fa = FactorizedAttn(mid_channels, nhead=8)
        # self.fu = nn.Sequential(
        #     *[DConvBLK(mid_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        # )

        # self.chri_de = CHRIDecoder(mid_channels, feature_size, depth=1)
        # self.attn = TConformerQKV(mid_channels, 4, r=2)

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

        # self.df_order = 5
        # self.df_bins = feature_size
        # self.DF_de = DF_dense_decoder(
        #     mid_channels, feature_size // 4, 2 * self.df_order
        # )

        # self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)
        # self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        # x, en_ch, en_ri = self.up_ch(x)
        x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        # x_enh = self.fa(x_enh, en_skip)
        # x_enh = self.fu(x_enh)

        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_3en_wdefu_wfafu_inv_c72_B2")
class aia_mcse_skip_sd_cau_inv(nn.Module):
    """mcse_3en_wdefu_wfafu
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = MMEBLK(in_channels * 2, mid_channels, feature_size)
        # self.up_ch = nn.Sequential(
        #     nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
        #     nn.LayerNorm(feature_size),
        #     nn.PReLU(mid_channels),
        # )

        self.encode = EncoderATT(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        # mic_array = [
        #     [-0.1, 0.095, 0],
        #     [0, 0.095, 0],
        #     [0.1, 0.095, 0],
        #     [-0.1, -0.095, 0],
        #     [0, -0.095, 0],
        #     [0.1, -0.095, 0],
        # ]
        # self.spf_alg = SpatialFeats(mic_array)
        # spf_in_channel = 93  # 15 * 5 + 18
        # self.en_spf = nn.Sequential(
        #     Encoder(
        #         inp_channels=spf_in_channel,
        #         mid_channels=mid_channels,
        #         feat_size=feature_size,
        #     ),  # B, mid_c, T, F // 4
        #     nn.Conv2d(mid_channels, mid_channels, (1, 1), (1, 1)),
        #     nn.Tanh(),
        # )

        # self.rnns = nn.Sequential(
        #     FTLSTM_RESNET(mid_channels, 128),
        #     FTLSTM_RESNET(mid_channels, 128),
        # )
        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = Decoder(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.fa = FactorizedAttn(mid_channels, nhead=8)
        self.fu = nn.Sequential(
            *[DConvBLK(mid_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        )

        # self.chri_de = CHRIDecoder(mid_channels, feature_size, depth=1)
        # self.attn = TConformerQKV(mid_channels, 4, r=2)

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

        # self.df_order = 5
        # self.df_bins = feature_size
        # self.DF_de = DF_dense_decoder(
        #     mid_channels, feature_size // 4, 2 * self.df_order
        # )

        # self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)
        # self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)
        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        x_enh = self.fa(en_skip, x_enh)  # NOTE inv
        x_enh = self.fu(x_enh)
        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_baseline+3en+ED+FA")
class aia_mcse_skip_sd_fa(nn.Module):
    """mcse_3en_wdefu_wfafu
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = MMEBLK(in_channels * 2, mid_channels, feature_size)
        # self.up_ch = nn.Sequential(
        #     nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
        #     nn.LayerNorm(feature_size),
        #     nn.PReLU(mid_channels),
        # )

        self.encode = EncoderATT(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        # mic_array = [
        #     [-0.1, 0.095, 0],
        #     [0, 0.095, 0],
        #     [0.1, 0.095, 0],
        #     [-0.1, -0.095, 0],
        #     [0, -0.095, 0],
        #     [0.1, -0.095, 0],
        # ]
        # self.spf_alg = SpatialFeats(mic_array)
        # spf_in_channel = 93  # 15 * 5 + 18
        # self.en_spf = nn.Sequential(
        #     Encoder(
        #         inp_channels=spf_in_channel,
        #         mid_channels=mid_channels,
        #         feat_size=feature_size,
        #     ),  # B, mid_c, T, F // 4
        #     nn.Conv2d(mid_channels, mid_channels, (1, 1), (1, 1)),
        #     nn.Tanh(),
        # )

        # self.rnns = nn.Sequential(
        #     FTLSTM_RESNET(mid_channels, 128),
        #     FTLSTM_RESNET(mid_channels, 128),
        # )
        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = Decoder(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.fa = FactorizedAttn(mid_channels, nhead=8)
        # self.fu = nn.Sequential(
        #     *[DConvBLK(mid_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        # )

        # self.chri_de = CHRIDecoder(mid_channels, feature_size, depth=1)
        # self.attn = TConformerQKV(mid_channels, 4, r=2)

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

        # self.df_order = 5
        # self.df_bins = feature_size
        # self.DF_de = DF_dense_decoder(
        #     mid_channels, feature_size // 4, 2 * self.df_order
        # )

        # self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)
        # self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)
        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        x_enh = self.fa(x_enh, en_skip)
        # x_enh = self.fu(x_enh)
        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_3en_wdefu_wfafu_c72_B2")
class aia_mcse_skip_sd(nn.Module):
    """mcse_3en_wdefu_wfafu
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = MMEBLK(in_channels * 2, mid_channels, feature_size)
        # self.up_ch = nn.Sequential(
        #     nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
        #     nn.LayerNorm(feature_size),
        #     nn.PReLU(mid_channels),
        # )

        self.encode = EncoderATT(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        # mic_array = [
        #     [-0.1, 0.095, 0],
        #     [0, 0.095, 0],
        #     [0.1, 0.095, 0],
        #     [-0.1, -0.095, 0],
        #     [0, -0.095, 0],
        #     [0.1, -0.095, 0],
        # ]
        # self.spf_alg = SpatialFeats(mic_array)
        # spf_in_channel = 93  # 15 * 5 + 18
        # self.en_spf = nn.Sequential(
        #     Encoder(
        #         inp_channels=spf_in_channel,
        #         mid_channels=mid_channels,
        #         feat_size=feature_size,
        #     ),  # B, mid_c, T, F // 4
        #     nn.Conv2d(mid_channels, mid_channels, (1, 1), (1, 1)),
        #     nn.Tanh(),
        # )

        # self.rnns = nn.Sequential(
        #     FTLSTM_RESNET(mid_channels, 128),
        #     FTLSTM_RESNET(mid_channels, 128),
        # )
        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = Decoder(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.fa = FactorizedAttn(mid_channels, nhead=8)
        self.fu = nn.Sequential(
            *[DConvBLK(mid_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        )

        # self.chri_de = CHRIDecoder(mid_channels, feature_size, depth=1)
        # self.attn = TConformerQKV(mid_channels, 4, r=2)

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

        # self.df_order = 5
        # self.df_bins = feature_size
        # self.DF_de = DF_dense_decoder(
        #     mid_channels, feature_size // 4, 2 * self.df_order
        # )

        # self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)
        # self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)
        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        x_enh = self.fa(x_enh, en_skip)
        x_enh = self.fu(x_enh)
        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_3en_wdefu_wfafu_c72_B2_uncau")
class aia_mcse_skip_sd_2(nn.Module):
    """mcse_3en_wdefu_wfafu
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = MMEBLK(in_channels * 2, mid_channels, feature_size, cau=False)

        self.encode = EncoderATT(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = Decoder(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.fa = FactorizedAttn(mid_channels, nhead=8)
        self.fu = nn.Sequential(
            *[DConvBLK(mid_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        )

        # self.chri_de = CHRIDecoder(mid_channels, feature_size, depth=1)
        # self.attn = TConformerQKV(mid_channels, 4, r=2)

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

        # self.df_order = 5
        # self.df_bins = feature_size
        # self.DF_de = DF_dense_decoder(
        #     mid_channels, feature_size // 4, 2 * self.df_order
        # )

        # self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)
        # self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)
        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        x_enh = self.fa(x_enh, en_skip)
        x_enh = self.fu(x_enh)
        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_3en_wdefu_wfafu_cau_c72_B2")
class aia_mcse_skip_sd_cau(nn.Module):
    """mcse_3en_wdefu_wfafu
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        self.up_ch = MMEBLK(in_channels * 2, mid_channels, feature_size)

        self.encode = EncoderATT(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        self.dual_trans = AIA_Transformer_cau(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = Decoder(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.fa = FactorizedAttn(mid_channels, nhead=8)
        self.fu = nn.Sequential(
            *[DConvBLK(mid_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        )

        # self.chri_de = CHRIDecoder(mid_channels, feature_size, depth=1)
        # self.attn = TConformerQKV(mid_channels, 4, r=2)

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

        # self.df_order = 5
        # self.df_bins = feature_size
        # self.DF_de = DF_dense_decoder(
        #     mid_channels, feature_size // 4, 2 * self.df_order
        # )

        # self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)
        # self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)
        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        x_enh = self.fa(x_enh, en_skip)
        x_enh = self.fu(x_enh)
        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


@tables.register("models", "mcse_uc_wdefu_wfafu_c72_B2")
class aia_mcse_skip_sd_uc(nn.Module):
    """mcse_3en_wdefu_wfafu
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4, **kwargs):
        super().__init__()
        self.stft = STFT(512, 256)
        # self.up_ch = MMEBLK(in_channels * 2, mid_channels, feature_size)
        self.up_ch = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, (1, 1), (1, 1), (0, 0)),
            nn.LayerNorm(feature_size),
            nn.PReLU(mid_channels),
        )

        self.encode = EncoderATT(
            in_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
            depth=depth,
        )

        self.dsLayer = DownSample(mid_channels, feature_size)
        self.usLayer = UpSample(mid_channels, feature_size // 4)

        # mic_array = [
        #     [-0.1, 0.095, 0],
        #     [0, 0.095, 0],
        #     [0.1, 0.095, 0],
        #     [-0.1, -0.095, 0],
        #     [0, -0.095, 0],
        #     [0.1, -0.095, 0],
        # ]
        # self.spf_alg = SpatialFeats(mic_array)
        # spf_in_channel = 93  # 15 * 5 + 18
        # self.en_spf = nn.Sequential(
        #     Encoder(
        #         inp_channels=spf_in_channel,
        #         mid_channels=mid_channels,
        #         feat_size=feature_size,
        #     ),  # B, mid_c, T, F // 4
        #     nn.Conv2d(mid_channels, mid_channels, (1, 1), (1, 1)),
        #     nn.Tanh(),
        # )

        # self.rnns = nn.Sequential(
        #     FTLSTM_RESNET(mid_channels, 128),
        #     FTLSTM_RESNET(mid_channels, 128),
        # )
        self.dual_trans = AIA_Transformer(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.decode = Decoder(
            in_channels=mid_channels,
            feat_size=feature_size,
            depth=depth,
        )

        self.fa = FactorizedAttn(mid_channels, nhead=8)
        self.fu = nn.Sequential(
            *[DConvBLK(mid_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        )

        # self.chri_de = CHRIDecoder(mid_channels, feature_size, depth=1)
        # self.attn = TConformerQKV(mid_channels, 4, r=2)

        self.out_conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=2,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

        # self.df_order = 5
        # self.df_bins = feature_size
        # self.DF_de = DF_dense_decoder(
        #     mid_channels, feature_size // 4, 2 * self.df_order
        # )

        # self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)
        # self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        # x, en_ch, en_ri = self.up_ch(x)
        x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF
        x_ri = self.dsLayer(x_ri)

        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # x_ri = self.rnns(x_ri)
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF
        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        x_ri = self.usLayer(x_ri)
        x_enh = self.decode(x_ri, None)  # B,1,T,F

        x_enh = self.fa(x_enh, en_skip)
        x_enh = self.fu(x_enh)
        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        # x_chri = self.chri_de(en_ch, en_ri)
        # feat = self.attn(x_chri, x_enh, x_enh)
        feat = x_enh

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


if __name__ == "__main__":
    from thop import profile
    import warnings

    # tables.print()

    inp = torch.randn(1, 10, 3, 33)
    # net = SPConvTranspose2d(in_channels=10, out_channels=20, kernel_size=(1, 3), r=2)
    net = UpSample_check(in_channels=10, feat_size=33)
    out = net(inp)
    print(out.shape)

    # net = MMEBLK(12, 24, 65)
    # inp = torch.randn(1, 12, 10, 65)
    # out = net(inp)
    # print(out.shape)

    # warnings.filterwarnings("ignore")
    # warnings.filterwarnings("ignore", category=DeprecationWarning)

    # ch must be multiply of 4,6,8; 24
    # model = aia_mcse_skip_sd_upch_att(in_channels=6, feature_size=257, mid_channels=72)  # C,F,C'
    # model = aia_mcse_skip_conformer(in_channels=6, feature_size=257, mid_channels=72)  # C,F,C'

    model = tables.models.get("mcse_3en_wdefu_wfafu_c72_B2_uncau")
    model = model(6, 257, 72)

    # model = mcse_skip_sd(in_channels=6, feature_size=257, mid_channels=72)  # C,F,C'
    input_test = torch.FloatTensor(1, 16000, 6)  # B,T,M

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This API is being deprecated")
        flops, params = profile(model, inputs=(input_test,), verbose=False)
    print(f"FLOPs={flops / 1e9}, params={params/1e6:.2f}")

    out = model(input_test)
    print(out.shape)
    # net = Encoder(10, 20, 31)
    # inp = torch.randn(2, 10, 5, 31)
    # out = net(inp)
    # print(out.shape)
    # model = CHRIDecoder(20, 65)
    # inp = torch.randn(1, 20, 2, 65)
    # out = model(inp, inp)
    # print(out.shape)
