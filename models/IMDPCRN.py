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
from models.ft_lstm import FTLSTM_RESNET, FTLSTM_RESNET_ATT, GroupFTLSTM
from models.multiframe import DF


class DensePWConvBlock(nn.Module):
    """Dilated Dense Block
    input_size the F dimension of b,c,t,f
    """

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
                nn.Sequential(
                    nn.Conv2d(
                        self.in_channels * (i + 1),
                        self.in_channels,
                        kernel_size=self.kernel_size,
                        dilation=(dil, 1),
                    ),
                    nn.LayerNorm(input_size),
                    nn.PReLU(self.in_channels),
                    nn.Conv2d(
                        self.in_channels,
                        self.in_channels,
                        kernel_size=(1, 3),
                        stride=(1, 1),
                        padding=(0, 1),
                    ),
                    nn.LayerNorm(input_size),
                    nn.PReLU(self.in_channels),
                ),
            )

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DenseNet(nn.Module):
    """Dilated Dense Block
    input_size the F dimension of b,c,t,f
    """

    def __init__(
        self,
        depth=4,
        in_channels=64,
        kernel_size: Tuple = (3, 3),
        causal: Tuple = (True, True),
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        # self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        t_width, f_width = (*kernel_size,)
        t_causal, f_causal = (*causal,)
        self.kernel_size = kernel_size
        for i in range(self.depth):
            dil = 2**i
            pad_t = t_width + (dil - 1) * (t_width - 1) - 1
            pad_f = f_width - 1

            if t_causal and f_causal:
                pad_info = (pad_f, 0, pad_t, 0)
            elif t_causal and not f_causal:
                pad_info = (pad_f // 2, pad_f // 2, pad_t, 0)
            elif not t_causal and f_causal:
                pad_info = (pad_f, 0, pad_t // 2, pad_t // 2)
            else:
                pad_info = (pad_f // 2, pad_f // 2, pad_t // 2, pad_t // 2)

            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d(pad_info, value=0.0),
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
            setattr(self, "norm{}".format(i + 1), nn.InstanceNorm2d(self.in_channels))
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


class DenseNetRes(nn.Module):
    """
    Input: b,c,t,f
    """

    def __init__(self, depth: int, in_channels: int) -> None:
        super(DenseNetRes, self).__init__()
        self.f_unit = nn.Sequential(
            Rearrange("b c t f->b c f t"),
            DenseNet(depth, in_channels, (3, 3), (False, True)),
            Rearrange("b c f t->b c t f"),
        )
        # self.f_post = nn.Sequential()
        self.t_unit = nn.Sequential(DenseNet(depth, in_channels, (3, 3), (True, False)))

    def forward(self, x):
        x = self.f_unit(x)
        x = self.t_unit(x)
        return x


class DenseBlock(nn.Module):
    """Dilated Dense Block
    input_size the F dimension of b,c,t,f
    """

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


class dense_encoder_2(nn.Module):
    """
    Input: B,C,T,F

    Arugments:
      - in_chanels: C of input;
      - feature_size: F of input
    """

    def __init__(
        self, in_channels: int, feature_size: int, out_channels, depth: int = 4
    ):
        super(dense_encoder_2, self).__init__()
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
        self.enc_dense = DenseNetRes(depth, out_channels)

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


class dense_pwc_encoder(nn.Module):
    """
    Input: B,C,T,F

    Arugments:
      - in_chanels: C of input;
      - feature_size: F of input
    """

    def __init__(
        self, in_channels: int, feature_size: int, out_channels, depth: int = 4
    ):
        super(dense_pwc_encoder, self).__init__()
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
        self.enc_dense = DensePWConvBlock(
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


class dense_pwc_decoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, feature_size: int, depth: int = 4
    ):
        super().__init__()
        self.out_channels = 1
        self.in_channels = in_channels
        # self.dec_dense = DenseBlock(
        #     depth=depth, in_channels=in_channels, input_size=feature_size
        # )
        self.dec_dense = DensePWConvBlock(
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


class dense_decoder_2(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, feature_size: int, depth: int = 4
    ):
        super().__init__()
        self.out_channels = 1
        self.in_channels = in_channels
        self.dec_dense = DenseNetRes(depth, in_channels)

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
    Output: B, C x (sub_freqs.sum() + 1), T, F
    """

    def __init__(self, sub_freqs: Tuple = (2, 3)) -> None:
        super(SubBands, self).__init__()
        self.sub_freqs = sub_freqs
        self.sub_size = sum(sub_freqs) + 1
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
        x = F.unfold(x, kernel_size=(self.sub_size, 1))
        x = einops.rearrange(x, "(b t) c f->b c t f", b=nB)

        return x


class FTLSTM_DENSE(nn.Module):
    """Input and output has the same dimension. Operation along the C dim.
    Input:  B,C,T,F
    Return: B,C,T,F

    Args:
        input_size: should be equal to C of input shape B,C,T,F
        hidden_size: input_size -> hidden_size
        batch_first: input shape is B,C,T,F if true
        use_fc: add fc layer after lstm
    """

    def __init__(self, input_size, hidden_size, batch_first=True, use_fc=True):
        super().__init__()

        assert (
            not use_fc and input_size == hidden_size
        ) or use_fc, f"hidden_size {hidden_size} should equals to input_size {input_size} when use_fc is True"

        self.f_pre = nn.Sequential(
            Rearrange("b c t f->b c f t"),
            DenseNet(4, input_size, (3, 3), (False, True)),
            Rearrange("b c f t->b c t f"),
        )
        self.f_unit = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,  # bi-directional LSTM output is 2xhidden_size
            batch_first=batch_first,
            bidirectional=True,
        )

        if use_fc:
            self.f_post = nn.Sequential(
                nn.Linear(hidden_size, input_size),
                nn.LayerNorm(input_size),
            )
        else:
            self.f_post = nn.Identity()

        self.t_pre = nn.Sequential(
            DenseNet(4, input_size, (3, 3), (True, False)),
        )
        self.t_unit = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
        )

        if use_fc:
            self.t_post = nn.Sequential(
                nn.Linear(hidden_size, input_size),
                nn.LayerNorm(input_size),
            )
        else:
            self.t_post = nn.Identity()

    def forward(self, inp: torch.Tensor):
        """
        Args:
            x: input shape should be B,C,T,F
        """
        nB = inp.shape[0]

        # step1. F-LSTM
        x = self.f_pre(inp)
        x = einops.rearrange(x, "b c t f-> (b t) f c")  # BxT,F,C
        # x = inp.permute(0, 2, 3, 1)  # B, T, F, C
        # x = x.reshape(-1, nF, nC)  # BxT,F,C
        x, _ = self.f_unit(x)  # BxT,F,C
        x = self.f_post(x)
        # BxT,F,C => B,C,T,F
        x = einops.rearrange(x, "(b t) f c-> b c t f", b=nB)
        # x = x.reshape(nB, nT, nF, nC)
        # x = x.permute(0, 3, 1, 2)  # B,C,T,F
        inp = inp + x

        # step2. T-LSTM
        x = self.t_pre(inp)
        x = einops.rearrange(x, "b c t f->(b f) t c")  # BxF,T,C
        x, _ = self.t_unit(x)
        x = self.t_post(x)
        x = einops.rearrange(x, "(b f) t c -> b c t f", b=nB)
        inp = inp + x

        return inp


class CRN_AEC_2(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [16, 32, 64, 64],
        stride: List = [2, 1, 2, 1],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.conv_l = nn.ModuleList()

        self.encoder_l = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        self.atten_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            # feat_num = (self.fft_dim >> idx + 1) + 1
            # batchCh = self.cnn_num[idx + 1] * ((self.fft_dim >> idx + 1) + 1)
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_l.append(
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, 2, 0), 0.0),
                    nn.Conv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 3),
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    nn.LayerNorm(nbin),
                    nn.PReLU(self.cnn_num[idx + 1]),
                )
            )

            # if idx == 0:
            #     self.decoder_l.append(
            #         nn.Sequential(
            #             nn.ConvTranspose2d(
            #                 in_channels=3 * self.cnn_num[-1 - idx],  # skip_connection
            #                 out_channels=self.cnn_num[-1 - idx - 1],
            #                 kernel_size=(1, 3),
            #                 padding=(0, 1),
            #                 stride=(1, stride[-1 - idx]),
            #             ),
            #             nn.LayerNorm(nbinT),
            #             nn.PReLU(self.cnn_num[-1 - idx - 1]),
            #         )
            #     )
            # elif idx != n_cnn_layer - 1:
            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 3),
                            padding=(0, 1),
                            stride=(1, stride[-1 - idx]),
                        ),
                        nn.LayerNorm(nbinT),
                        nn.PReLU(self.cnn_num[-1 - idx - 1]),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 3),
                            padding=(0, 1),
                            stride=(1, stride[-1 - idx]),
                        ),
                    )
                )

        self.rnns_r = nn.Sequential(
            # FTLSTM_DENSE(cnn_num[-1], rnn_hidden_num),
            # FTLSTM_DENSE(cnn_num[-1], rnn_hidden_num),
            # FTLSTM_RESNET(cnn_num[-1], rnn_hidden_num),
            DenseNetRes(4, cnn_num[-1]),
            DenseNetRes(4, cnn_num[-1]),
        )
        # self.rnns_i = nn.Sequential(
        #     FTLSTM_RESNET(cnn_num[-1], rnn_hidden_num),
        #     FTLSTM_RESNET(cnn_num[-1], rnn_hidden_num),
        # )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft.transform(mic)  # [B, 2, T, F]
        specs_ref = self.stft.transform(ref)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, F, T]

        x = specs_mix
        feat_store = []
        for idx, layer in enumerate(self.encoder_l):
            x = layer(x)  # x shape [B, C, T, F]
            feat_store.append(x)

        # x_r, x_i = torch.chunk(x, 2, dim=1)

        feat_r = self.rnns_r(x)
        # feat_i = self.rnns_i(x)

        # mask_r, mask_i = F.tanh(mask_r), F.tanh(mask_i)
        # cmask = torch.concatenate([mask_r, mask_i], dim=1)
        # feat_r, feat_i = complex_mask_multi(feat, cmask)

        # feat = torch.concat([feat_r, feat_i], dim=1)
        feat = feat_r

        # B,C,F,T
        x = feat
        for idx, layer in enumerate(self.decoder_l):
            x = torch.concat([x, feat_store[-idx - 1]], dim=1)
            x = layer(x)

        # feat_r, feat_i = complex_apply_mask(specs_mic, x)
        mask_r, mask_i = x.chunk(2, dim=1)
        feat_r = specs_mic_real * mask_r - specs_ref_imag * mask_i
        feat_i = specs_mic_real * mask_i + specs_ref_imag * mask_r
        feat = torch.concat([feat_r, feat_i], dim=1)  # b,2,t,f

        # feat = self.post_conv(feat)  # b,f,t
        # r, i = feat.permute(0, 2, 1).chunk(2, dim=-1)  # b,t,f
        # feat = torch.stack([r, i], dim=1)

        out_wav = self.stft.inverse(feat)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class CRN_AEC(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [16, 32, 64, 64],
        stride: List = [2, 1, 2, 1],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.conv_l = nn.ModuleList()

        self.encoder_l = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        self.atten_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            # feat_num = (self.fft_dim >> idx + 1) + 1
            # batchCh = self.cnn_num[idx + 1] * ((self.fft_dim >> idx + 1) + 1)
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_l.append(
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, 2, 0), 0.0),
                    nn.Conv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 3),
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    nn.LayerNorm(nbin),
                    nn.PReLU(self.cnn_num[idx + 1]),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 3),
                            padding=(0, 1),
                            stride=(1, stride[-1 - idx]),
                        ),
                        nn.LayerNorm(nbinT),
                        nn.PReLU(self.cnn_num[-1 - idx - 1]),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 3),
                            padding=(0, 1),
                            stride=(1, stride[-1 - idx]),
                        ),
                    )
                )

        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            # DenseNetRes(4, cnn_num[-1] // 2),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            # DenseNetRes(4, cnn_num[-1] // 2),
        )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft.transform(mic)  # [B, 2, T, F]
        specs_ref = self.stft.transform(ref)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, F, T]

        x = specs_mix
        feat_store = []
        for idx, layer in enumerate(self.encoder_l):
            x = layer(x)  # x shape [B, C, T, F]
            feat_store.append(x)

        x_r, x_i = torch.chunk(x, 2, dim=1)

        feat_r = self.rnns_r(x_r)
        feat_i = self.rnns_i(x_i)

        # mask_r, mask_i = F.tanh(mask_r), F.tanh(mask_i)
        # cmask = torch.concatenate([mask_r, mask_i], dim=1)
        # feat_r, feat_i = complex_mask_multi(feat, cmask)

        feat = torch.concat([feat_r, feat_i], dim=1)

        # B,C,F,T
        x = feat
        for idx, layer in enumerate(self.decoder_l):
            x = torch.concat([x, feat_store[-idx - 1]], dim=1)
            x = layer(x)

        # feat_r, feat_i = complex_apply_mask(specs_mic, x)
        mask_r, mask_i = x.chunk(2, dim=1)
        feat_r = specs_mic_real * mask_r - specs_ref_imag * mask_i
        feat_i = specs_mic_real * mask_i + specs_ref_imag * mask_r
        feat = torch.concat([feat_r, feat_i], dim=1)  # b,2,t,f

        # feat = self.post_conv(feat)  # b,f,t
        # r, i = feat.permute(0, 2, 1).chunk(2, dim=-1)  # b,t,f
        # feat = torch.stack([r, i], dim=1)

        out_wav = self.stft.inverse(feat)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class BaseCRN_2(nn.Module):
    def __init__(self, mid_channels: int = 128):
        super().__init__()
        self.stft = STFT(512, 256)
        self.encode_spec = dense_encoder_2(
            in_channels=4, feature_size=257, out_channels=mid_channels, depth=4
        )
        self.de_r = dense_decoder_2(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=65,  # 257//4+1
            depth=4,
        )
        self.de_i = dense_decoder_2(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=65,
            depth=4,
        )

        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(mid_channels, 128),
            FTLSTM_RESNET(mid_channels, 128),
            # DenseNetRes(4, mid_channels),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(mid_channels, 128),
            FTLSTM_RESNET(mid_channels, 128),
            # DenseNetRes(4, mid_channels),
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


class BaseCRNDensePWC(nn.Module):
    def __init__(self, mid_channels: int = 128):
        super().__init__()
        self.stft = STFT(512, 256)
        # self.encode_spec = dense_encoder(
        #     in_channels=4, feature_size=257, out_channels=mid_channels, depth=4
        # )
        # self.de_r = dense_decoder(
        #     in_channels=mid_channels,
        #     out_channels=1,
        #     feature_size=65,  # 257//4+1
        #     depth=4,
        # )
        # self.de_i = dense_decoder(
        #     in_channels=mid_channels,
        #     out_channels=1,
        #     feature_size=65,
        #     depth=4,
        # )
        self.encode_spec = dense_pwc_encoder(
            in_channels=4, feature_size=257, out_channels=mid_channels, depth=4
        )
        self.de_r = dense_pwc_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=65,  # 257//4+1
            depth=4,
        )
        self.de_i = dense_pwc_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=65,
            depth=4,
        )

        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(mid_channels, 128),
            FTLSTM_RESNET(mid_channels, 128),
            # DenseNetRes(4, mid_channels),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(mid_channels, 128),
            FTLSTM_RESNET(mid_channels, 128),
            # DenseNetRes(4, mid_channels),
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
            # DenseNetRes(4, mid_channels),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(mid_channels, 128),
            FTLSTM_RESNET(mid_channels, 128),
            # DenseNetRes(4, mid_channels),
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


class BaseCRNwGroupFT(nn.Module):
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
            GroupFTLSTM(mid_channels, 65, 128, 4),
        )
        self.rnns_i = nn.Sequential(
            GroupFTLSTM(mid_channels, 65, 128, 4),
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


# class DfOutputReshapeMF(nn.Module):
#     """Coefficients output reshape for multiframe/MultiFrameModule

#     Requires input of shape B, C, T, F, 2.
#     """

#     def __init__(self, df_order: int, df_bins: int):
#         super().__init__()
#         self.df_order = df_order
#         self.df_bins = df_bins

#     def forward(self, coefs: torch.Tensor) -> torch.Tensor:
#         # [B, T, F, O*2] -> [B, O, T, F, 2]
#         new_shape = list(coefs.shape)
#         new_shape[-1] = -1
#         new_shape.append(2)  # b,c,t,f -> b,c,t,-1,2
#         coefs = coefs.view(new_shape)
#         coefs = coefs.permute(0, 3, 1, 2, 4)
#         return coefs


class DFCRN_pwc(nn.Module):
    def __init__(self, mid_channels: int = 128):
        super().__init__()
        self.stft = STFT(512, 256)
        # self.encode_spec = dense_encoder(
        #     in_channels=4,
        #     out_channels=mid_channels,
        #     depth=4,
        #     feature_size=257,
        # )
        self.encode_spec = dense_pwc_encoder(
            in_channels=4, feature_size=257, out_channels=mid_channels, depth=4
        )
        self.de_r = dense_pwc_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=65,  # 257//4+1
            depth=4,
        )
        self.de_i = dense_pwc_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=65,
            depth=4,
        )

        df_bins = 257
        df_order = 5
        self.df_op = DF(num_freqs=df_bins, frame_size=5, lookahead=0)

        self.de_df = nn.Sequential(
            dense_decoder(
                in_channels=mid_channels * 2,
                out_channels=df_order * 2,
                feature_size=65,
                depth=4,
            ),
            Rearrange("b (m c) t f-> b c t f m", m=2),
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
        xk = torch.concat([xk_r, xk_i], dim=1)

        mask_r = self.de_r(xk_r)
        mask_i = self.de_i(xk_i)
        mask_df = self.de_df(xk).contiguous()  # b,df_order,t,f,2

        xk_r = xk_m_r * mask_r - xk_m_i * mask_i
        xk_i = xk_m_r * mask_i + xk_m_i * mask_r

        xk = (
            torch.concat([xk_r, xk_i], dim=1)
            .permute(0, 2, 3, 1)
            .unsqueeze(1)
            .contiguous()
        )  # B,1,T,F,C

        xk = self.df_op(xk, mask_df).squeeze(1)  # b,t,f,c
        xk = xk.permute(0, 3, 1, 2)

        out = self.stft.inverse(xk)

        return out


class DFCRN(nn.Module):
    def __init__(self, mid_channels: int = 128):
        super().__init__()
        self.stft = STFT(512, 256)
        self.encode_spec = dense_encoder(
            in_channels=4,
            out_channels=mid_channels,
            depth=4,
            feature_size=257,
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

        df_bins = 257
        df_order = 5
        self.df_op = DF(num_freqs=df_bins, frame_size=5, lookahead=0)

        self.de_df = nn.Sequential(
            dense_decoder(
                in_channels=mid_channels * 2,
                out_channels=df_order * 2,
                feature_size=65,
                depth=4,
            ),
            Rearrange("b (m c) t f-> b c t f m", m=2),
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
        xk = torch.concat([xk_r, xk_i], dim=1)

        mask_r = self.de_r(xk_r)
        mask_i = self.de_i(xk_i)
        mask_df = self.de_df(xk).contiguous()  # b,df_order,t,f,2

        xk_r = xk_m_r * mask_r - xk_m_i * mask_i
        xk_i = xk_m_r * mask_i + xk_m_i * mask_r

        xk = (
            torch.concat([xk_r, xk_i], dim=1)
            .permute(0, 2, 3, 1)
            .unsqueeze(1)
            .contiguous()
        )  # B,1,T,F,C

        xk = self.df_op(xk, mask_df).squeeze(1)  # b,t,f,c
        xk = xk.permute(0, 3, 1, 2)

        out = self.stft.inverse(xk)

        return out


class BaseCRNwSubBands(nn.Module):
    def __init__(self, mid_channels: int = 128, sub=(2, 3)):
        super().__init__()
        self.stft = STFT(512, 256)
        cin = 4 * (sum(sub) + 1)
        self.subf = nn.Sequential(
            SubBands(sub),
            nn.Conv2d(cin, 2 * cin, (1, 3), (1, 2), (0, 1)),
            nn.LayerNorm(129),
            nn.PReLU(2 * cin),
            nn.Conv2d(2 * cin, 4 * cin, (1, 3), (1, 2), (0, 1)),
            nn.LayerNorm(65),
            nn.PReLU(4 * cin),
            nn.Conv2d(4 * cin, 2 * mid_channels, (1, 1)),
            nn.Tanh(),
        )
        self.encode_spec = dense_encoder(
            in_channels=4, feature_size=257, out_channels=mid_channels, depth=4
        )
        # self.fusion = nn.Sequential(
        #     nn.Conv2d(mid_channels + 4 * cin, mid_channels, (1, 1)),
        #     nn.LayerNorm(65),
        #     nn.Tanh(),
        # )
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
            # FTLSTM_RESNET(mid_channels, 128),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(mid_channels, 128),
            # FTLSTM_RESNET(mid_channels, 128),
        )

        self.rnns_r_2 = nn.Sequential(
            FTLSTM_RESNET(2 * mid_channels, 128),
            # FTLSTM_RESNET(mid_channels, 128),
            Rearrange("b c t f->b f t c"),
            nn.Linear(2 * mid_channels, mid_channels),
            nn.LayerNorm(mid_channels),
            Rearrange("b f t c->b c t f"),
            nn.ReLU(),
        )
        self.rnns_i_2 = nn.Sequential(
            FTLSTM_RESNET(2 * mid_channels, 128),
            # FTLSTM_RESNET(mid_channels, 128),
            Rearrange("b c t f->b f t c"),
            nn.Linear(2 * mid_channels, mid_channels),
            nn.LayerNorm(mid_channels),
            Rearrange("b f t c->b c t f"),
            nn.ReLU(),
        )

    def forward(self, mic, ref):
        xk_m = self.stft.transform(mic)  # b,2,t,f
        xk_r = self.stft.transform(ref)

        xk_m_r, xk_m_i = xk_m.chunk(2, dim=1)
        xk_r_r, xk_r_i = xk_r.chunk(2, dim=1)
        # mag_m = (xk_m_r**2 + xk_m_i**2) ** 0.5
        # mag_r = (xk_m_r**2 + xk_m_i**2) ** 0.5

        xk = torch.concat([xk_m_r, xk_r_r, xk_m_i, xk_r_i], dim=1)
        # mag = torch.concat([mag_m, mag_r], dim=1)

        xk_subs = self.subf(xk)
        xk = self.encode_spec(xk)  # b,4,t,f

        # xk = xk + xk_subs
        # xk = self.fusion(xk_) * xk

        # xk_r, xk_i = xk.chunk(2, dim=1)
        xk_r = self.rnns_r(xk)
        xk_i = self.rnns_i(xk)

        r, i = xk_subs.chunk(2, dim=1)
        # xk_r_ = xk_r * r - xk_i * i
        # xk_i_ = xk_i * r + xk_r * i
        xk_r_ = torch.concat([xk_r, r], dim=1)
        xk_i_ = torch.concat([xk_i, i], dim=1)

        xk_r_ = self.rnns_r_2(xk_r_)
        xk_i_ = self.rnns_i_2(xk_i_)

        mask_r = self.de_r(xk_r_)
        mask_i = self.de_i(xk_i_)

        xk_r = xk_m_r * mask_r - xk_m_i * mask_i
        xk_i = xk_m_r * mask_i + xk_m_i * mask_r

        xk = torch.concat([xk_r, xk_i], dim=1)

        out = self.stft.inverse(xk)

        return out


if __name__ == "__main__":
    from thop import profile

    net = BaseCRN_2(64)
    net = DFCRN_pwc(64)
    # net = BaseCRNwSubBands(64)
    # net = DFCRN(64)
    # net = BaseCRNwGroupFT(64)
    inp = torch.randn(1, 16000)
    out = net(inp, inp)

    # flops, param = profile(net, inputs=(inp, inp))
    # print(flops / 1e9, param / 1e6)

    # net = CRN_AEC(
    #     nframe=512,
    #     nhop=256,
    #     nfft=512,
    #     cnn_num=[32, 64, 128, 128],
    #     stride=[2, 2, 1, 1],
    #     rnn_hidden_num=128,
    # )
    # out = net(inp, inp)
    # flops, param = profile(net, inputs=(inp, inp))
    # print(flops / 1e9, param / 1e6)
    flops, param = profile(net, inputs=(inp, inp))
    print(flops / 1e9, param / 1e6)

    # net = SubBands(sub_freqs=(1, 3))
    # inp = torch.randn(1, 2, 10, 257)

    # out = net(inp)
    # print(out.shape)
