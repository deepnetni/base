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

from models.MHSA import FConformer, FTConformer, TConformer
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
                    nn.Conv2d(in_channels, in_channels, (1, 3), (1, 1), (0, 1)),
                    nn.LayerNorm(input_size),
                    nn.PReLU(in_channels),
                ),
            )

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
            out = getattr(self, "conv{}".format(i + 1))(out)
            # out = torch.concat([out_1, out_2], dim=1)  # bctf
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
                "post{}".format(i + 1),
                nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0)),
            )

            setattr(self, "norm{}".format(i + 1), nn.LayerNorm(input_size))
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(in_channels))

    def forward(self, x, y):
        """y is the skip conn"""
        skip = x

        mid = []
        for i in range(self.depth):
            skip_ = torch.concat([skip, y[i]], dim=1)

            out = getattr(self, "pad{}".format(i + 1))(skip_)
            out = getattr(self, "conv{}".format(i + 1))(out)
            # out = torch.concat([out_1, out_2], dim=1)  # bctf
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)

            mid.append(getattr(self, "post{}".format(i + 1))(out))
            skip = torch.cat([out, skip], dim=1)

        return out, mid[::-1]


class DecoderBLK(nn.Module):
    def __init__(self, in_channels, feat_size, depth=4) -> None:
        super().__init__()
        self.dec_dense = DenseSkipBLK(
            depth=depth,
            in_channels=in_channels,
            input_size=feat_size,
            kernel_size=(2, 3),
        )

        # self.dec_post = nn.Sequential(
        #     FConformer(in_channels, 2, r=3),
        #     TConformer(in_channels, 2, attn_wlen=20, r=3),
        # )

    def forward(self, x, y):
        out, mids = self.dec_dense(x, y)
        # out = self.dec_post(out)
        return out, mids


class Decoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, feat_size, depth=4, return_steps=True
    ) -> None:
        super().__init__()

        self.return_steps = return_steps

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

        self.dense_att = nn.ModuleList(
            [DecoderBLK(in_channels, feat_size, depth=depth) for _ in range(4)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(feat_size) for _ in range(4)])
        self.drops = nn.ModuleList([nn.Dropout(0.1) for _ in range(4)])

        self.out_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

    def forward(self, x, skip):
        out = self.dec_conv1(x)
        x = self.dec_conv2(out)
        for i, l in enumerate(self.dense_att):
            out, _ = l(x, skip)
            x = self.norms[i](out + self.drops[i](x))
        out = self.out_conv(out)
        return out


class aia_mcse_skip(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_size: int,
        mid_channels: int,
        ref_channel: int = 4,
    ):
        super().__init__()
        self.stft = STFT(512, 256)
        self.en_ri = Encoder(
            inp_channels=in_channels * 2,
            mid_channels=mid_channels,
            feat_size=feature_size,
            return_steps=True,
        )  # B, mid_c, T, F // 4

        mic_array = [
            [-0.1, 0.095, 0],
            [0, 0.095, 0],
            [0.1, 0.095, 0],
            [-0.1, -0.095, 0],
            [0, -0.095, 0],
            [0.1, -0.095, 0],
        ]
        self.spf_alg = SpatialFeats(mic_array)
        spf_in_channel = 93  # 15 * 5 + 18
        self.en_spf = nn.Sequential(
            Encoder(
                inp_channels=spf_in_channel,
                mid_channels=mid_channels,
                feat_size=feature_size,
            ),  # B, mid_c, T, F // 4
            nn.Conv2d(mid_channels, mid_channels, (1, 1), (1, 1)),
            nn.Tanh(),
        )

        self.dual_trans = AIA_Transformer_cau(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)
        self.ref_channel = ref_channel
        self.in_channels = in_channels

        self.de1 = Decoder(
            in_channels=mid_channels,
            out_channels=1,
            feat_size=feature_size // 4,
            depth=3,
        )
        self.de2 = Decoder(
            in_channels=mid_channels,
            out_channels=1,
            feat_size=feature_size // 4,
            depth=3,
        )

        self.df_order = 5
        self.df_bins = feature_size
        self.DF_de = DF_dense_decoder(
            mid_channels, feature_size // 4, 2 * self.df_order
        )

        self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)
        x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)

        # noisy_real = x[:, self.ref_channel, :, :]
        # noisy_imag = x[:, self.ref_channel + self.in_channels, :, :]

        x_spf = self.en_spf(self.spf_alg(x_))
        # ri components enconde+ aia_transformer
        x_ri, mid_l = self.en_ri(x)  # BCTF
        x_ri = x_ri * x_spf

        # * FTLSTM
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # real and imag decode
        x_real = self.de1(x_ri, mid_l)  # B,1,T,F
        x_imag = self.de2(x_ri, mid_l)  # B,1,T,F
        # x_real, x_imag = x_.chunk(2, dim=1)
        x_real = x_real.squeeze(dim=1)  # B,T,F
        x_imag = x_imag.squeeze(dim=1)
        # enh_real = noisy_real * x_real - noisy_imag * x_imag
        # enh_imag = noisy_real * x_imag + noisy_imag * x_real

        enhanced_D = torch.stack([x_real, x_imag], 3)  # B,T,F,2
        enhanced_D = enhanced_D.unsqueeze(1)  # B,1,T,F,2

        # DF coeffs decoder
        df_coefs = self.DF_de(x_ri)  # BCTF
        df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        DF_spec = self.df_op(enhanced_D, df_coefs)  # B,1,T,F,2
        DF_spec = DF_spec.squeeze(1)

        DF_real = DF_spec[:, :, :, 0]
        DF_imag = DF_spec[:, :, :, 1]

        feat = torch.stack([DF_real, DF_imag], dim=1)  # B,2,T,F

        out_wav = self.stft.inverse(feat)  # B, T
        # out_wav = torch.squeeze(out_wav, 1)
        # out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


if __name__ == "__main__":
    from thop import profile
    import warnings

    # warnings.filterwarnings("ignore")
    # warnings.filterwarnings("ignore", category=DeprecationWarning)

    model = aia_mcse_skip(in_channels=6, feature_size=257, mid_channels=64)  # C,F,C'
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
