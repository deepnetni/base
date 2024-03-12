import sys
from pathlib import Path

from einops import rearrange

sys.path.append(str(Path(__file__).parent.parent))
from typing import List, Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.complexnn import (
    ComplexConv2d,
    ComplexGateConvTranspose2d,
    InstanceNorm,
    complex_apply_mask,
    complex_cat,
    ComplexConv1d,
)
from models.conv_stft import STFT
from models.ft_lstm import FTLSTM_RESNET
from models.CMGAN.generator import DilatedDenseNet
from models.Fusion.ms_cam import AFF, MS_CAM, MS_SELF_CAM, MS_CAM_F


class TFusion(nn.Module):
    def __init__(self, inp_channel: int) -> None:
        super().__init__()
        self.l = nn.Sequential(
            nn.Conv2d(
                in_channels=inp_channel,
                out_channels=inp_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Tanh(),
        )

    def forward(self, x, y):
        return self.l(x) * y


class FTAttention(nn.Module):
    def __init__(self, inp_channel: int, winL: int = 10) -> None:
        super().__init__()
        self.attn_f = nn.MultiheadAttention(
            embed_dim=inp_channel, num_heads=2, batch_first=True
        )

        self.attn_t = nn.MultiheadAttention(
            embed_dim=inp_channel, num_heads=2, batch_first=True
        )
        self.attn_window = winL

    def forward(self, k, q, v):
        """b,c,t,f"""
        nB = k.shape[0]
        k_ = rearrange(k, "b c t f -> (b t) f c")
        q_ = rearrange(q, "b c t f -> (b t) f c")
        v_ = rearrange(v, "b c t f -> (b t) f c")

        vf, _ = self.attn_f(k_, q_, v_)
        v = v + rearrange(vf, "(b t) f c -> b c t f", b=nB)

        nT = v.shape[2]
        mask_1 = torch.ones(nT, nT, device=v.device).triu_(1).bool()  # TxT
        mask_2 = (
            torch.ones(nT, nT, device=v.device).tril_(-self.attn_window).bool()
        )  # TxT
        mask = mask_1 + mask_2

        k_ = rearrange(k, "b c t f -> (b f) t c")
        q_ = rearrange(q, "b c t f -> (b f) t c")
        v_ = rearrange(v, "b c t f -> (b f) t c")

        vt, _ = self.attn_t(k_, q_, v_, attn_mask=mask)
        v = v + rearrange(vt, "(b f) t c -> b c t f", b=nB)

        return v


class ChannelFreqAttention(nn.Module):
    def __init__(self, inp_channels: int, feature_size: int) -> None:
        super().__init__()

        self.layer_ch = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=(1, feature_size), stride=(1, feature_size)
            ),  # B,C,T,1
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            Rearrange("b c t f-> b t c f"),
            nn.LayerNorm(1, inp_channels),
            Rearrange("b t c f-> b c t f"),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            nn.Sigmoid(),
        )

        self.layer_freq = nn.Sequential(
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            Rearrange("b c t f-> b t c f"),
            nn.LayerNorm(feature_size, inp_channels),
            Rearrange("b t c f-> b c t f"),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * self.layer_ch(x)
        x = x * self.layer_freq(x)

        return x


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
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(2, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        # ComplexConvTranspose2d(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # nn.BatchNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # nn.InstanceNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # InstanceNorm(),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        # * ((self.fft_dim >> n_cnn_layer - idx - 1) + 1)
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # InstanceNorm(2 * self.fft_dim),
                        # nn.PReLU(),
                    )
                )

        # self.rnns_r = FTLSTM(cnn_num[-1] // 2, rnn_hidden_num)
        # self.rnns_i = FTLSTM(cnn_num[-1] // 2, rnn_hidden_num)
        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
        )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft.transform(mic)  # [B, 2, T, F]
        specs_ref = self.stft.transform(ref)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,2,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        feat = torch.stack([specs_mic_real, specs_mic_imag], dim=1)

        x = specs_mix
        feat_store = []
        for idx, layer in enumerate(self.encoder_l):
            x = layer(x)  # x shape [B, C, T, F]
            feat_store.append(x)

        nB, nC, nF, nT = x.shape
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
            x = complex_cat([x, feat_store[-idx - 1]], dim=1)
            # print("Tconv", idx, feat.shape)
            x = layer(x)
            # feat = feat[..., 1:]  # padding=(2, 0) will padding 1 col in Time dimension

        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        feat_r = feat_r.squeeze(dim=1).permute(0, 2, 1)  # b1tf -> bft
        feat_i = feat_i.squeeze(dim=1).permute(0, 2, 1)
        feat = torch.concat([feat_r, feat_i], dim=1)  # b,2f,t
        # print("Tconv", feat.shape)
        # B, 2, F, T -> B, F(r, i), T
        # feat = feat.reshape(nB, self.fft_dim * 2, -1)  # B, F, T
        feat = self.post_conv(feat)  # b,f,t
        r, i = feat.permute(0, 2, 1).chunk(2, dim=-1)  # b,t,f
        feat = torch.stack([r, i], dim=1)

        out_wav = self.stft.inverse(feat)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class ADPCRN(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [32, 64, 128, 128],
        stride: List = [2, 2, 2, 2],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.encoder_rel = nn.ModuleList()
        self.encoder_mic = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            # feat_num = (self.fft_dim >> idx + 1) + 1
            # batchCh = self.cnn_num[idx + 1] * ((self.fft_dim >> idx + 1) + 1)
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_rel.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(2, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            self.encoder_mic.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=2 if idx == 0 else self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(1, 5),
                        padding=(0, 2),
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1] // 2),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1] // 2),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        # ComplexConvTranspose2d(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # InstanceNorm(2 * self.fft_dim),
                        # nn.PReLU(),
                    )
                )

        # self.encoder_fusion = nn.Sequential(
        #     ComplexConv2d(
        #         in_channels=self.cnn_num[-1],
        #         out_channels=self.cnn_num[-1],
        #         kernel_size=(1, 1),
        #         stride=(1, 1),
        #     ),
        #     InstanceNorm(nbin * self.cnn_num[-1]),
        #     nn.Tanh(),
        # )
        self.encoder_fusion = MS_CAM_F(
            inp_channels=self.cnn_num[-1], feature_size=nbin, r=1
        )
        # self.encoder_fusion = TFusion(inp_channel=self.cnn_num[-1])
        # self.encoder_fusion = nn.Sequential(
        #     ComplexConv2d(
        #         in_channels=2 * self.cnn_num[-1],
        #         out_channels=self.cnn_num[-1],
        #         kernel_size=(1, 1),
        #         stride=(1, 1),
        #     ),
        #     InstanceNorm(nbin * self.cnn_num[-1]),
        #     nn.PReLU(),
        # )

        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
        )

        # self.post_conv = ComplexConv2d(
        #     in_channels=2,
        #     out_channels=2,
        #     kernel_size=(1, 1),
        #     stride=(1, 1),
        #     padding=(0, 0),
        # )

        # self.dilateds = nn.ModuleList()
        # for _ in range(3):
        #     self.dilateds.append(
        #         nn.Sequential(
        #             Rearrange("b c f t -> b c t f"),
        #             DilatedDenseNet(depth=4, in_channels=cnn_num[-1]),
        #             Rearrange("b c t f -> b c f t"),
        #         )
        #     )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft.transform(mic)  # [B, 2, T, F]
        specs_ref = self.stft.transform(ref)
        # print("##", specs_ref.shape)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        # mag_spec_mic = torch.sqrt(specs_mic_real**2 + specs_mic_imag**2)
        # phs_spec_mic = torch.atan2(specs_mic_imag, specs_mic_real)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, F, T]

        spec_store = []
        spec = specs_mic
        x = specs_mix
        for idx, (lm, lr) in enumerate(zip(self.encoder_mic, self.encoder_rel)):
            spec = lm(spec)
            x = lr(x)  # x shape [B, C, T, F]
            spec_store.append(spec)
            # spec_store.append(x)
            # mx = complex_cat([spec, x], dim=1)
            # spec_store.append(mx)

        # feat = self.dilateds[0](feat)

        # for idx, layer in enumerate(self.encoder_rel):
        #     x = layer(x)  # x shape [B, C, T, F]

        # x = self.dilateds[1](x)

        # Fusion
        # x = torch.concat([spec, x], dim=1)
        # x = complex_cat([spec, x], dim=1)
        # x = self.encoder_fusion(x)
        # x = x + spec
        # x = self.encoder_fusion(x) * spec
        x = self.encoder_fusion(x, spec)
        x_r, x_i = torch.chunk(x, 2, dim=1)

        x_r = self.rnns_r(x_r)
        x_i = self.rnns_i(x_i)

        # mask_r, mask_i = F.tanh(mask_r), F.tanh(mask_i)

        x = torch.concatenate([x_r, x_i], dim=1)

        # x = self.dilateds[2](x)

        # feat_r, feat_i = complex_mask_multi(feat, cmask)

        # feat = torch.concat([feat_r, feat_i], dim=1)

        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, spec_store[-idx - 1]], dim=1)
            x = layer(x)

        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        x = torch.concat([feat_r, feat_i], dim=1)

        # feat = self.post_conv(x)
        feat = x

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class ADPCRN_PLUS(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [32, 64, 128, 128],
        stride: List = [2, 2, 2, 2],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.encoder_rel = nn.ModuleList()
        self.encoder_mic = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            # feat_num = (self.fft_dim >> idx + 1) + 1
            # batchCh = self.cnn_num[idx + 1] * ((self.fft_dim >> idx + 1) + 1)
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_rel.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(2, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            self.encoder_mic.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=2 if idx == 0 else self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(1, 5),
                        padding=(0, 2),
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1] // 2),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1] // 2),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        # ComplexConvTranspose2d(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # InstanceNorm(2 * self.fft_dim),
                        # nn.PReLU(),
                    )
                )

        # self.encoder_fusion = nn.Sequential(
        #     ComplexConv2d(
        #         in_channels=self.cnn_num[-1],
        #         out_channels=self.cnn_num[-1],
        #         kernel_size=(1, 1),
        #         stride=(1, 1),
        #     ),
        #     InstanceNorm(nbin * self.cnn_num[-1]),
        #     nn.Tanh(),
        # )
        # self.encoder_fusion = TFusion(inp_channel=self.cnn_num[-1])
        # self.encoder_fusion = nn.Sequential(
        #     ComplexConv2d(
        #         in_channels=2 * self.cnn_num[-1],
        #         out_channels=self.cnn_num[-1],
        #         kernel_size=(1, 1),
        #         stride=(1, 1),
        #     ),
        #     InstanceNorm(nbin * self.cnn_num[-1]),
        #     nn.PReLU(),
        # )

        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
        )

        # self.post_conv = ComplexConv2d(
        #     in_channels=2,
        #     out_channels=2,
        #     kernel_size=(1, 1),
        #     stride=(1, 1),
        #     padding=(0, 0),
        # )

        # self.dilateds = nn.ModuleList()
        # for _ in range(3):
        #     self.dilateds.append(
        #         nn.Sequential(
        #             Rearrange("b c f t -> b c t f"),
        #             DilatedDenseNet(depth=4, in_channels=cnn_num[-1]),
        #             Rearrange("b c t f -> b c f t"),
        #         )
        #     )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft.transform(mic)  # [B, 2, T, F]
        specs_ref = self.stft.transform(ref)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        # mag_spec_mic = torch.sqrt(specs_mic_real**2 + specs_mic_imag**2)
        # phs_spec_mic = torch.atan2(specs_mic_imag, specs_mic_real)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, F, T]

        spec_store = []
        spec = specs_mic
        x = specs_mix
        for idx, (lm, lr) in enumerate(zip(self.encoder_mic, self.encoder_rel)):
            spec = lm(spec)
            x = lr(x)  # x shape [B, C, T, F]
            spec_store.append(spec)
            # mx = complex_cat([spec, x], dim=1)
            # spec_store.append(mx)

        # feat = self.dilateds[0](feat)

        # for idx, layer in enumerate(self.encoder_rel):
        #     x = layer(x)  # x shape [B, C, T, F]

        # x = self.dilateds[1](x)

        # Fusion
        # x = torch.concat([spec, x], dim=1)
        # x = complex_cat([spec, x], dim=1)
        # x = self.encoder_fusion(x)
        x = x + spec
        # x = self.encoder_fusion(x) * spec
        x_r, x_i = torch.chunk(x, 2, dim=1)

        x_r = self.rnns_r(x_r)
        x_i = self.rnns_i(x_i)

        # mask_r, mask_i = F.tanh(mask_r), F.tanh(mask_i)

        x = torch.concatenate([x_r, x_i], dim=1)

        # x = self.dilateds[2](x)

        # feat_r, feat_i = complex_mask_multi(feat, cmask)

        # feat = torch.concat([feat_r, feat_i], dim=1)

        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, spec_store[-idx - 1]], dim=1)
            x = layer(x)

        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        x = torch.concat([feat_r, feat_i], dim=1)

        # feat = self.post_conv(x)
        feat = x

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class ADPCRN_MS(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [32, 64, 128, 128],
        stride: List = [2, 2, 2, 2],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.encoder_rel = nn.ModuleList()
        self.encoder_mic = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        self.conn_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            # feat_num = (self.fft_dim >> idx + 1) + 1
            # batchCh = self.cnn_num[idx + 1] * ((self.fft_dim >> idx + 1) + 1)
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.conn_l.append(
                # MS_CAM(inp_channels=self.cnn_num[idx + 1], feature_size=nbin)
                MS_CAM_F(inp_channels=self.cnn_num[idx + 1], feature_size=nbin, r=1)
                # AFF(inp_channels=self.cnn_num[idx + 1], feature_size=nbin)
            )

            self.encoder_rel.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(2, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            self.encoder_mic.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=2 if idx == 0 else self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(1, 5),
                        padding=(0, 2),
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1] // 2),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1] // 2),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        # ComplexConvTranspose2d(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # nn.BatchNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # nn.InstanceNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # InstanceNorm(),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        # * ((self.fft_dim >> n_cnn_layer - idx - 1) + 1)
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # InstanceNorm(2 * self.fft_dim),
                        # nn.PReLU(),
                    )
                )

        # self.encoder_fusion = MS_CAM_F(
        #     inp_channels=self.cnn_num[-1], feature_size=nbin, r=1
        # )
        # self.encoder_fusion = nn.Sequential(
        #     ComplexConv2d(
        #         in_channels=2 * self.cnn_num[-1],
        #         out_channels=self.cnn_num[-1],
        #         kernel_size=(1, 1),
        #         stride=(1, 1),
        #     ),
        #     InstanceNorm(nbin * self.cnn_num[-1]),
        #     nn.PReLU(),
        # )

        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            # Rearrange("b c f t -> b t f c"),
            # nn.Linear(in_features=cnn_num[-1], out_features=cnn_num[-1] // 2),
            # Rearrange("b t f c -> b c f t"),
            # InstanceNorm(cnn_num[-1] // 2 * nbin),
            # nn.PReLU(),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            # Rearrange("b c f t -> b t f c"),
            # nn.Linear(in_features=cnn_num[-1], out_features=cnn_num[-1] // 2),
            # Rearrange("b t f c -> b c f t"),
            # InstanceNorm(cnn_num[-1] // 2 * nbin),
            # nn.PReLU(),
        )

        # self.post_conv = ComplexConv2d(
        #     in_channels=2,
        #     out_channels=2,
        #     kernel_size=(1, 1),
        #     stride=(1, 1),
        #     padding=(0, 0),
        # )

        # self.dilateds = nn.ModuleList()
        # for _ in range(3):
        #     self.dilateds.append(
        #         nn.Sequential(
        #             Rearrange("b c f t -> b c t f"),
        #             DilatedDenseNet(depth=4, in_channels=cnn_num[-1]),
        #             Rearrange("b c t f -> b c f t"),
        #         )
        #     )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft.transform(mic)  # [B, 2, T, F]
        specs_ref = self.stft.transform(ref)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        # mag_spec_mic = torch.sqrt(specs_mic_real**2 + specs_mic_imag**2)
        # phs_spec_mic = torch.atan2(specs_mic_imag, specs_mic_real)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, F, T]

        spec_store = []
        spec = specs_mic
        x = specs_mix
        for idx, (lm, lr) in enumerate(zip(self.encoder_mic, self.encoder_rel)):
            spec = lm(spec)
            x = lr(x)  # x shape [B, C, T, F]
            spec_store.append(spec)
            spec = self.conn_l[idx](x, spec)
            # spec_store.append(spec)
            # mx = complex_cat([spec, x], dim=1)
            # spec_store.append(mx)

        # feat = self.dilateds[0](feat)

        # for idx, layer in enumerate(self.encoder_rel):
        #     x = layer(x)  # x shape [B, C, T, F]

        # x = self.dilateds[1](x)

        # Fusion
        # x = torch.concat([spec, x], dim=1)
        # x = complex_cat([spec, x], dim=1)
        # x = self.encoder_fusion(x)
        # x = x + spec
        x = spec
        x_r, x_i = torch.chunk(x, 2, dim=1)

        x_r = self.rnns_r(x_r)
        x_i = self.rnns_i(x_i)

        # mask_r, mask_i = F.tanh(mask_r), F.tanh(mask_i)

        x = torch.concatenate([x_r, x_i], dim=1)

        # x = self.dilateds[2](x)

        # feat_r, feat_i = complex_mask_multi(feat, cmask)

        # feat = torch.concat([feat_r, feat_i], dim=1)

        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, spec_store[-idx - 1]], dim=1)
            x = layer(x)

        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        x = torch.concat([feat_r, feat_i], dim=1)

        # feat = self.post_conv(x)

        out_wav = self.stft.inverse(x)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class DPCRN_AEC(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [32, 64, 128, 128],
        stride: List = [2, 2, 2, 2],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.encoder_rel = nn.ModuleList()
        self.encoder_mic = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            # feat_num = (self.fft_dim >> idx + 1) + 1
            # batchCh = self.cnn_num[idx + 1] * ((self.fft_dim >> idx + 1) + 1)
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_rel.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(2, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            # self.encoder_mic.append(
            #     nn.Sequential(
            #         ComplexConv2d(
            #             in_channels=2 if idx == 0 else self.cnn_num[idx],
            #             out_channels=self.cnn_num[idx + 1],
            #             kernel_size=(1, 5),
            #             padding=(0, 2),
            #             stride=(1, stride[idx]),
            #         ),
            #         # nn.BatchNorm2d(self.cnn_num[idx + 1] // 2),
            #         # nn.InstanceNorm2d(self.cnn_num[idx + 1] // 2),
            #         # InstanceNorm(),
            #         InstanceNorm(self.cnn_num[idx + 1] * nbin),
            #         nn.PReLU(),
            #     )
            # )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        # ComplexConvTranspose2d(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # nn.BatchNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # nn.InstanceNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # InstanceNorm(),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        # * ((self.fft_dim >> n_cnn_layer - idx - 1) + 1)
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # InstanceNorm(2 * self.fft_dim),
                        # nn.PReLU(),
                    )
                )

        # self.encoder_fusion = nn.Sequential(
        #     ComplexConv2d(
        #         in_channels=2 * self.cnn_num[-1],
        #         out_channels=self.cnn_num[-1],
        #         kernel_size=(1, 1),
        #         stride=(1, 1),
        #     ),
        #     InstanceNorm(nbin * self.cnn_num[-1]),
        #     nn.PReLU(),
        # )

        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            # Rearrange("b c f t -> b t f c"),
            # nn.Linear(in_features=cnn_num[-1], out_features=cnn_num[-1] // 2),
            # Rearrange("b t f c -> b c f t"),
            # InstanceNorm(cnn_num[-1] // 2 * nbin),
            # nn.PReLU(),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            # Rearrange("b c f t -> b t f c"),
            # nn.Linear(in_features=cnn_num[-1], out_features=cnn_num[-1] // 2),
            # Rearrange("b t f c -> b c f t"),
            # InstanceNorm(cnn_num[-1] // 2 * nbin),
            # nn.PReLU(),
        )

        # self.post_conv = ComplexConv2d(
        #     in_channels=2,
        #     out_channels=2,
        #     kernel_size=(1, 1),
        #     stride=(1, 1),
        #     padding=(0, 0),
        # )

        # self.dilateds = nn.ModuleList()
        # for _ in range(3):
        #     self.dilateds.append(
        #         nn.Sequential(
        #             Rearrange("b c f t -> b c t f"),
        #             DilatedDenseNet(depth=4, in_channels=cnn_num[-1]),
        #             Rearrange("b c t f -> b c f t"),
        #         )
        #     )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft.transform(mic)  # [B, 2, T, F]
        specs_ref = self.stft.transform(ref)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        # mag_spec_mic = torch.sqrt(specs_mic_real**2 + specs_mic_imag**2)
        # phs_spec_mic = torch.atan2(specs_mic_imag, specs_mic_real)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, F, T]

        # spec = specs_mic
        spec_store = []
        # for idx, layer in enumerate(self.encoder_mic):
        #     spec = layer(spec)
        #     spec_store.append(spec)

        # feat = self.dilateds[0](feat)

        x = specs_mix
        for idx, layer in enumerate(self.encoder_rel):
            x = layer(x)  # x shape [B, C, T, F]
            spec_store.append(x)

        # x = self.dilateds[1](x)

        # Fusion
        # x = torch.concat([spec, x], dim=1)
        # x = complex_cat([spec, x], dim=1)
        # x = self.encoder_fusion(x)
        # x = x + spec
        x_r, x_i = torch.chunk(x, 2, dim=1)

        x_r = self.rnns_r(x_r)
        x_i = self.rnns_i(x_i)

        # mask_r, mask_i = F.tanh(mask_r), F.tanh(mask_i)

        x = torch.concatenate([x_r, x_i], dim=1)

        # x = self.dilateds[2](x)

        # feat_r, feat_i = complex_mask_multi(feat, cmask)

        # feat = torch.concat([feat_r, feat_i], dim=1)

        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, spec_store[-idx - 1]], dim=1)
            x = layer(x)

        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        x = torch.concat([feat_r, feat_i], dim=1)

        # feat = self.post_conv(x)

        out_wav = self.stft.inverse(x)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class ADPCRN_ATTN(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [32, 64, 128, 128],
        stride: List = [2, 2, 2, 2],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.encoder_rel = nn.ModuleList()
        self.encoder_mic = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            # feat_num = (self.fft_dim >> idx + 1) + 1
            # batchCh = self.cnn_num[idx + 1] * ((self.fft_dim >> idx + 1) + 1)
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_rel.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(2, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            self.encoder_mic.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=2 if idx == 0 else self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(1, 5),
                        padding=(0, 2),
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1] // 2),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1] // 2),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        # ComplexConvTranspose2d(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # nn.BatchNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # nn.InstanceNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # InstanceNorm(),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        # * ((self.fft_dim >> n_cnn_layer - idx - 1) + 1)
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # InstanceNorm(2 * self.fft_dim),
                        # nn.PReLU(),
                    )
                )

        self.encoder_fusion = MS_CAM_F(
            inp_channels=self.cnn_num[-1], feature_size=nbin, r=1
        )

        self.attn = FTAttention(cnn_num[-1], 10)

        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            # Rearrange("b c f t -> b t f c"),
            # nn.Linear(in_features=cnn_num[-1], out_features=cnn_num[-1] // 2),
            # Rearrange("b t f c -> b c f t"),
            # InstanceNorm(cnn_num[-1] // 2 * nbin),
            # nn.PReLU(),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            # Rearrange("b c f t -> b t f c"),
            # nn.Linear(in_features=cnn_num[-1], out_features=cnn_num[-1] // 2),
            # Rearrange("b t f c -> b c f t"),
            # InstanceNorm(cnn_num[-1] // 2 * nbin),
            # nn.PReLU(),
        )

        # self.dilateds = nn.ModuleList()
        # for _ in range(3):
        #     self.dilateds.append(
        #         nn.Sequential(
        #             Rearrange("b c f t -> b c t f"),
        #             DilatedDenseNet(depth=4, in_channels=cnn_num[-1]),
        #             Rearrange("b c t f -> b c f t"),
        #         )
        #     )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft.transform(mic)  # [B, 2, T, F]
        specs_ref = self.stft.transform(ref)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        # mag_spec_mic = torch.sqrt(specs_mic_real**2 + specs_mic_imag**2)
        # phs_spec_mic = torch.atan2(specs_mic_imag, specs_mic_real)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, F, T]

        spec_store = []
        spec = specs_mic
        x = specs_mix
        for idx, (lm, lr) in enumerate(zip(self.encoder_mic, self.encoder_rel)):
            spec = lm(spec)
            x = lr(x)  # x shape [B, C, T, F]
            spec_store.append(spec)
            # spec_store.append(spec)
            # mx = complex_cat([spec, x], dim=1)
            # spec_store.append(mx)

        # x = self.dilateds[1](x)

        # Fusion
        # x = self.encoder_fusion(x, spec)

        # x = torch.concat([spec, x], dim=1)
        # x = complex_cat([spec, x], dim=1)
        # x = self.encoder_fusion(x)
        x = self.encoder_fusion(x, spec)
        x = self.attn(x, x, x)
        # x = x + spec
        x_r, x_i = torch.chunk(x, 2, dim=1)

        x_r = self.rnns_r(x_r)
        x_i = self.rnns_i(x_i)

        # mask_r, mask_i = F.tanh(mask_r), F.tanh(mask_i)

        x = torch.concatenate([x_r, x_i], dim=1)

        # x = self.dilateds[2](x)

        # feat_r, feat_i = complex_mask_multi(feat, cmask)

        # feat = torch.concat([feat_r, feat_i], dim=1)

        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, spec_store[-idx - 1]], dim=1)
            x = layer(x)

        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        x = torch.concat([feat_r, feat_i], dim=1)

        out_wav = self.stft.inverse(x)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class ADPCRN_Dilated(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [32, 64, 128, 128],
        stride: List = [2, 2, 2, 2],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.encoder_rel = nn.ModuleList()
        self.encoder_mic = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            # feat_num = (self.fft_dim >> idx + 1) + 1
            # batchCh = self.cnn_num[idx + 1] * ((self.fft_dim >> idx + 1) + 1)
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_rel.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(2, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            self.encoder_mic.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=2 if idx == 0 else self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(1, 5),
                        padding=(0, 2),
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1] // 2),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1] // 2),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        # ComplexConvTranspose2d(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # nn.BatchNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # nn.InstanceNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # InstanceNorm(),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        # * ((self.fft_dim >> n_cnn_layer - idx - 1) + 1)
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # InstanceNorm(2 * self.fft_dim),
                        # nn.PReLU(),
                    )
                )

        self.encoder_fusion = nn.Sequential(
            ComplexConv2d(
                in_channels=2 * self.cnn_num[-1],
                out_channels=self.cnn_num[-1],
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
            InstanceNorm(nbin * self.cnn_num[-1]),
            nn.PReLU(),
        )

        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            # Rearrange("b c f t -> b t f c"),
            # nn.Linear(in_features=cnn_num[-1], out_features=cnn_num[-1] // 2),
            # Rearrange("b t f c -> b c f t"),
            # InstanceNorm(cnn_num[-1] // 2 * nbin),
            # nn.PReLU(),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            # Rearrange("b c f t -> b t f c"),
            # nn.Linear(in_features=cnn_num[-1], out_features=cnn_num[-1] // 2),
            # Rearrange("b t f c -> b c f t"),
            # InstanceNorm(cnn_num[-1] // 2 * nbin),
            # nn.PReLU(),
        )

        self.post_conv = ComplexConv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )

        self.dilateds = nn.ModuleList()
        for _ in range(2):
            self.dilateds.append(
                nn.Sequential(
                    DilatedDenseNet(depth=4, in_channels=cnn_num[-1]),
                )
            )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft.transform(mic)  # [B, 2, T, F]
        specs_ref = self.stft.transform(ref)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        # mag_spec_mic = torch.sqrt(specs_mic_real**2 + specs_mic_imag**2)
        # phs_spec_mic = torch.atan2(specs_mic_imag, specs_mic_real)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, F, T]

        spec = specs_mic
        spec_store = []
        for idx, layer in enumerate(self.encoder_mic):
            spec = layer(spec)
            spec_store.append(spec)

        spec = self.dilateds[0](spec)

        x = specs_mix
        for idx, layer in enumerate(self.encoder_rel):
            x = layer(x)  # x shape [B, C, T, F]

        # x = self.dilateds[1](x)

        # Fusion
        # x = torch.concat([spec, x], dim=1)
        x = complex_cat([spec, x], dim=1)
        x = self.encoder_fusion(x)
        # x = x + spec
        # x = self.dilateds[0](x)

        x_r, x_i = torch.chunk(x, 2, dim=1)

        x_r = self.rnns_r(x_r)
        x_i = self.rnns_i(x_i)

        # mask_r, mask_i = F.tanh(mask_r), F.tanh(mask_i)

        x = torch.concatenate([x_r, x_i], dim=1)

        x = self.dilateds[1](x)

        # feat_r, feat_i = complex_mask_multi(feat, cmask)

        # feat = torch.concat([feat_r, feat_i], dim=1)

        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, spec_store[-idx - 1]], dim=1)
            x = layer(x)

        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        x = torch.concat([feat_r, feat_i], dim=1)

        feat = self.post_conv(x)

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


if __name__ == "__main__":
    import time

    net = ADPCRN(
        nframe=512,
        nhop=256,
        nfft=512,
        cnn_num=[16, 32, 64, 128],
        stride=[2, 2, 1, 1],
        rnn_hidden_num=128,
    )
    inp = torch.randn(1, 16000)
    net = net.cpu()
    st = time.time()
    out = net(inp, inp)
    ed = time.time()
    print(out.shape)
    print(1000 * round(ed - st, 10) / 63)  # ms
    # torch.save(net.state_dict(), "31.pth")

    net = DPCRN_AEC(
        nframe=512,
        nhop=256,
        nfft=512,
        cnn_num=[16, 32, 64, 128],
        stride=[2, 2, 1, 1],
        rnn_hidden_num=128,
    )
    inp = torch.randn(1, 16000)
    out = net(inp, inp)
    print(out.shape)

    net = ADPCRN_ATTN(
        nframe=512,
        nhop=256,
        nfft=512,
        cnn_num=[16, 32, 64, 128],
        stride=[2, 2, 1, 1],
        rnn_hidden_num=128,
    )
    inp = torch.randn(1, 16000)
    out = net(inp, inp)
    print(out.shape)

    net = ADPCRN_MS(
        nframe=512,
        nhop=256,
        nfft=512,
        cnn_num=[16, 32, 64, 128],
        stride=[2, 2, 1, 1],
        rnn_hidden_num=128,
    )
    inp = torch.randn(1, 16000)
    out = net(inp, inp)
    print(out.shape)
