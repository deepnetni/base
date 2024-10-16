import sys
from pathlib import Path

from google.protobuf import message


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
from models.MHSA import FTConformer, TConformerQKV
from models.ft_lstm import FTLSTM_RESNET


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


# class DenseMPBLK_EXP(nn.Module):
#     """Dense Multi-Path Block
#     Input: xspec, xch, xri with shape B,C,T,F
#     """

#     def __init__(
#         self,
#         depth=4,
#         in_channels=64,
#         input_size: int = 257,
#         kernel_size=(2, 3),
#     ):
#         super().__init__()
#         self.depth = depth
#         twidth = kernel_size[0]
#         for i in range(depth):
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
#             setattr(self, "att{}".format(i + 1), FTConformer(in_channels, 4))
#             setattr(
#                 self,
#                 "post{}".format(i + 1),
#                 nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
#             )

#             setattr(self, "norm{}".format(i + 1), nn.LayerNorm(input_size))
#             setattr(self, "prelu{}".format(i + 1), nn.PReLU(in_channels))

#     def forward(self, x, ch, ri):
#         skip = x
#         mid = []
#         for i in range(self.depth):
#             out = getattr(self, "pad{}".format(i + 1))(skip)
#             out_1 = getattr(self, "conv{}".format(i + 1))(out)
#             out_2 = getattr(self, "pwc{}".format(i + 1))(skip)
#             # out = torch.concat([out_1, out_2], dim=1)  # bctf
#             out = getattr(self, f"fu{i+1}")(out_1, out_2)
#             out = getattr(self, "norm{}".format(i + 1))(out)
#             out = getattr(self, "prelu{}".format(i + 1))(out)
#             out = getattr(self, "att{}".format(i + 1))(out)
#             mid.append(getattr(self, "post{}".format(i + 1))(out))
#             skip = torch.cat([out, skip], dim=1)

#         return out, mid[::-1]


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
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),  # lrtb
            )

            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Sequential(
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
            setattr(
                self,
                "post{}".format(i + 1),
                nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
            )

            setattr(self, "norm{}".format(i + 1), nn.LayerNorm(input_size))
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(in_channels))

    def forward(self, x):
        skip = x
        mid = []
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out_1 = getattr(self, "conv{}".format(i + 1))(out)
            out_2 = getattr(self, "pwc{}".format(i + 1))(skip)
            # out = torch.concat([out_1, out_2], dim=1)  # bctf
            out = getattr(self, f"fu{i+1}")(out_1, out_2)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            out = getattr(self, "att{}".format(i + 1))(out)
            mid.append(getattr(self, "post{}".format(i + 1))(out))
            skip = torch.cat([out, skip], dim=1)

        return out, mid


class EncoderATT(nn.Module):
    """input B,(CM),T,F"""

    def __init__(
        self, inp_channels: int, mid_channels: int, feat_size: int, return_steps=False
    ) -> None:
        super().__init__()

        self.return_steps = return_steps
        # self.pre_layer = nn.Sequential(
        #     nn.Conv2d(inp_channels, mid_channels, (1, 1), (1, 1), (0, 0)),
        #     nn.LayerNorm(feat_size),
        #     nn.PReLU(mid_channels),
        # )
        self.pre_layer = MMEBLK(inp_channels, mid_channels, feat_size)
        self.enc_dense = DenseMPBLK(
            in_channels=mid_channels,
            input_size=feat_size,
            kernel_size=(2, 3),
            depth=4,
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # //2
                padding=(0, 1),
            ),
            nn.LayerNorm(feat_size // 2 + 1),
            nn.PReLU(mid_channels),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # no padding
            ),
            nn.LayerNorm(feat_size // 4),
            nn.PReLU(mid_channels),
        )

    def forward(self, x):
        x, ch, ri = self.pre_layer(x)
        x, x_l = self.enc_dense(x)
        x = self.post_conv(x)

        if self.return_steps:
            return x, ch, ri, x_l
        else:
            return x, ch, ri


class DenseBLK(nn.Module):  # dilated dense block
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
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),  # lrtb
            )

            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Sequential(
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
            setattr(
                self,
                "post{}".format(i + 1),
                nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
            )

            setattr(self, "norm{}".format(i + 1), nn.LayerNorm(input_size))
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(in_channels))

    def forward(self, x):
        skip = x
        mid = []
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out_1 = getattr(self, "conv{}".format(i + 1))(out)
            out_2 = getattr(self, "pwc{}".format(i + 1))(skip)
            # out = torch.concat([out_1, out_2], dim=1)  # bctf
            out = getattr(self, f"fu{i+1}")(out_1, out_2)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            mid.append(getattr(self, "post{}".format(i + 1))(out))
            skip = torch.cat([out, skip], dim=1)

        return out, mid[::-1]


class Encoder(nn.Module):
    """input B,(CM),T,F"""

    def __init__(
        self, inp_channels: int, mid_channels: int, feat_size: int, return_steps=False
    ) -> None:
        super().__init__()

        self.return_steps = return_steps
        self.pre_layer = nn.Sequential(
            nn.Conv2d(inp_channels, mid_channels, (1, 1), (1, 1), (0, 0)),
            nn.LayerNorm(feat_size),
            nn.PReLU(mid_channels),
        )
        self.enc_dense = DenseBLK(
            in_channels=mid_channels,
            input_size=feat_size,
            kernel_size=(2, 3),
            depth=4,
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # //2
                padding=(0, 1),
            ),
            nn.LayerNorm(feat_size // 2 + 1),
            nn.PReLU(mid_channels),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # no padding
            ),
            nn.LayerNorm(feat_size // 4),
            nn.PReLU(mid_channels),
        )

    def forward(self, x):
        x = self.pre_layer(x)
        x, x_l = self.enc_dense(x)
        x = self.post_conv(x)

        if self.return_steps:
            return x, x_l
        else:
            return x


class DenseSkipBLK(nn.Module):  # dilated dense block
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
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )

            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels * (i + 2),
                        in_channels,
                        kernel_size=kernel_size,
                        dilation=(dil, 1),
                    ),
                    nn.LayerNorm(input_size),
                    nn.PReLU(in_channels),
                    nn.Conv2d(in_channels, in_channels, (1, 3), (1, 1), (0, 1)),
                    nn.LayerNorm(input_size),
                    nn.PReLU(in_channels),
                ),
            )
            setattr(
                self,
                f"pwc{i+1}",
                nn.Sequential(
                    nn.Conv2d(
                        in_channels * (i + 2), in_channels, (1, 3), (1, 1), (0, 1)
                    ),
                    nn.LayerNorm(input_size),
                    nn.PReLU(in_channels),
                ),
            )
            setattr(self, "fu{}".format(i + 1), AFF(in_channels, input_size, r=1))

            setattr(self, "norm{}".format(i + 1), nn.LayerNorm(input_size))
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(in_channels))

    def forward(self, x, y):
        """y is the skip conn"""
        skip = x

        for i in range(self.depth):
            skip_ = torch.concat([skip, y[i]], dim=1)

            out = getattr(self, "pad{}".format(i + 1))(skip_)
            out_1 = getattr(self, "conv{}".format(i + 1))(out)
            out_2 = getattr(self, "pwc{}".format(i + 1))(skip_)
            # out = torch.concat([out_1, out_2], dim=1)  # bctf
            out = getattr(self, "fu{}".format(i + 1))(out_1, out_2)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)

            skip = torch.cat([out, skip], dim=1)

        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, feat_size, depth=4) -> None:
        super().__init__()

        self.dec_conv1 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),  # padding F
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),  # B,C,T,F
            nn.LayerNorm(feat_size * 2),
            nn.PReLU(in_channels),
        )

        feat_size = feat_size * 2

        self.dec_conv2 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),
            nn.ConstantPad2d((1, 0, 0, 0), value=0.0),
            nn.LayerNorm(feat_size * 2 + 1),
            nn.PReLU(in_channels),
        )

        feat_size = feat_size * 2 + 1

        self.dec_dense = DenseSkipBLK(
            depth=depth,
            in_channels=in_channels,
            input_size=feat_size,
            kernel_size=(2, 3),
        )

        self.out_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

    def forward(self, x, skip):
        out = self.dec_conv1(x)
        out = self.dec_conv2(out)
        out = self.dec_dense(out, skip)
        out = self.out_conv(out)
        return out


class aia_mcse_skip_sd(nn.Module):
    """
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(
        self,
        in_channels: int,
        feature_size: int,
        mid_channels: int,
    ):
        super().__init__()
        self.stft = STFT(512, 256)
        self.en_ri = EncoderATT(
            inp_channels=in_channels * 2,
            mid_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
        )  # B, mid_c, T, F // 4

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

        self.dual_trans = AIA_Transformer_cau(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)

        self.de = Decoder(
            in_channels=mid_channels,
            out_channels=2,
            feat_size=feature_size // 4,
            depth=4,
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

        x_ri, uc_ch, uc_ri, mid_l = self.en_ri(x)  # BCTF
        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # * FTLSTM
        _, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF

        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        # real and imag decode
        x_enh = self.de(x_ri, mid_l)  # B,1,T,F
        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        feat = x_enh

        out_wav = self.stft.inverse(feat)  # B, T

        return out_wav


class mcse_skip_sd(nn.Module):
    """
    Inputs: B,C,T,F
    Args:
        - in_channels, `C`
        - feature_size, `F`
        - mid_channels, `Cout`
    """

    def __init__(
        self,
        in_channels: int,
        feature_size: int,
        mid_channels: int,
    ):
        super().__init__()
        self.stft = STFT(512, 256)
        self.en_ri = EncoderATT(
            inp_channels=in_channels * 2,
            mid_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
        )  # B, mid_c, T, F // 4

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

        self.rnns = nn.Sequential(
            FTLSTM_RESNET(mid_channels, 128),
            FTLSTM_RESNET(mid_channels, 128),
        )
        # self.rnns = nn.Identity()
        # self.aham = AHAM(input_channel=mid_channels)

        self.de = Decoder(
            in_channels=mid_channels,
            out_channels=2,
            feat_size=feature_size // 4,
            depth=4,
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

        x_ri, mid_l = self.en_ri(x)  # BCTF
        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)
        # x_spf = self.en_spf(self.spf_alg(x_))
        # x_ri = x_ri * x_spf

        # * FTLSTM
        # x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        # x_ri = self.aham(x_outputlist)  # BCTF
        x_ri = self.rnns(x_ri)

        # DF coeffs decoder
        # df_coefs = self.DF_de(x_ri)  # BCTF

        # df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        # df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        # real and imag decode
        x_enh = self.de(x_ri, mid_l)  # B,1,T,F
        # x_enh = rearrange(x_enh, "b c t f->b 1 t f c").contiguous()
        # df_spec = self.df_op(x_enh, df_coefs)  # B,1,T,F,2
        # feat = rearrange(df_spec, "b 1 t f c->b c t f")

        feat = x_enh

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

    model = aia_mcse_skip_sd(in_channels=6, feature_size=257, mid_channels=72)  # C,F,C'
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
