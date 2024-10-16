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
    DF_dense_decoder,
    DfOutputReshapeMF,
    SPConvTranspose2d,
)
from models.SpatialFeatures import SpatialFeats
from models.conv_stft import STFT
from models.multiframe import DF
from models.Fusion.ms_cam import AFF
from models.MHSA import FConformer, FTConformer, TConformerConv, TConformerQKV
from models.ft_lstm import FTLSTM_RESNET
from models.DConvs import DConvBLK

from models.FactorizedAttention import FactorizedAttn, FactorizedAttn2


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
        self.attn = TConformerQKV(out_channels, 4, r=2)

    def forward(self, x):
        xch = self.en_ch(x)
        xri = self.en_ri(x)
        xsp = self.en_spec(x)

        xff = self.aff(xch, xri)  # B,C,T,F
        xsp = self.attn(xff, xsp, xsp)

        return xsp, xch, xri


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
                    nn.Conv2d(
                        in_channels * (i + 1), in_channels, (1, 3), (1, 1), (0, 1)
                    ),
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
                    nn.Conv2d(
                        in_channels * (i + 1), in_channels, (1, 3), (1, 1), (0, 1)
                    ),
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
            ),  # F//4 => F//2
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


class PostDF(nn.Module):
    """
    Inpt: B,C,T,F
    Oupt: B,C,T,F
    """

    def __init__(self, inp_channels: int, feature_size: int, df_size=96):
        super().__init__()
        self.fa = FactorizedAttn2(inp_channels, df_size, nhead=8)
        self.fu = nn.Sequential(
            *[DConvBLK(inp_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        )

    def forward(self, x, df):
        x = self.fa(x, df)
        x = self.fu(x)
        return x


class aia_mcse_skip_sd_df(nn.Module):
    """mcse_df_3en_wdefu_wfafu
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4):
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

        mic_array = [
            [-0.1, 0.095, 0],
            [0, 0.095, 0],
            [0.1, 0.095, 0],
            [-0.1, -0.095, 0],
            [0, -0.095, 0],
            [0.1, -0.095, 0],
        ]
        self.df_extractor = SpatialFeats(mic_array)
        # spf_in_channel = 93  # 15 * 5 + 18
        # self.df_conv = nn.Sequential(
        #     nn.Conv2d(93, mid_channels, (1, 1), (1, 1)),
        #     nn.Tanh(),
        # )
        # self.df_fa = FactorizedAttn(mid_channels, nhead=6)
        self.df_fa = FactorizedAttn2(mid_channels, 93, nhead=8)
        # self.df_fu = nn.Sequential(
        #     *[DConvBLK(mid_channels, feature_size, (2, 3), 2**i) for i in range(3)]
        # )

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

        # FFT
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)

        # up channel
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)
        x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)

        # Directional Features
        # x_df = self.df_conv(self.df_extractor(x_))
        x_df = self.df_extractor(x_)
        x = x + self.df_fa(x, x_df)
        # x = self.df_fu(x)

        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF

        x_ri = self.dsLayer(x_ri)

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


class aia_mcse_skip_sd_df_tail(nn.Module):
    """mcse_3en_wdefu_wfafu_df
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(self, in_channels: int, feature_size: int, mid_channels: int, depth=4):
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

        mic_array = [
            [-0.1, 0.095, 0],
            [0, 0.095, 0],
            [0.1, 0.095, 0],
            [-0.1, -0.095, 0],
            [0, -0.095, 0],
            [0.1, -0.095, 0],
        ]
        self.df_extractor = SpatialFeats(mic_array)
        # spf_in_channel = 93  # 15 * 5 + 18
        self.df_post = PostDF(mid_channels, feature_size, 93)

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

        # FFT
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)

        # up channel
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)
        x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)

        x, en_ch, en_ri = self.up_ch(x)

        # Directional Features
        x_df = self.df_extractor(x_)

        # x = self.up_ch(x)
        x_ri, en_skip = self.encode(x)  # BCTF

        x_ri = self.dsLayer(x_ri)

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
        feat = self.df_post(feat, x_df)

        feat = self.out_conv(feat)
        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


if __name__ == "__main__":
    from thop import profile
    import warnings

    # net = MMEBLK(12, 24, 65)
    # inp = torch.randn(1, 12, 10, 65)
    # out = net(inp)
    # print(out.shape)

    # warnings.filterwarnings("ignore")
    # warnings.filterwarnings("ignore", category=DeprecationWarning)

    # ch must be multiply of 4,6,8; 24
    model = aia_mcse_skip_sd_df(
        in_channels=6, feature_size=257, mid_channels=72
    )  # C,F,C'

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
