import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from typing import List, Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.complexnn import (
    ComplexConv1d,
    ComplexConv2d,
    ComplexGateConvTranspose2d,
    InstanceNorm,
    complex_apply_mask,
    complex_cat,
)
from models.conv_stft import STFT
from models.ft_lstm import FTLSTM_RESNET


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
                        InstanceNorm(2 * self.fft_dim),
                        nn.PReLU(),
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

        spec = specs_mic
        spec_store = []
        for idx, layer in enumerate(self.encoder_mic):
            spec = layer(spec)
            spec_store.append(spec)

        # feat = self.dilateds[0](feat)

        x = specs_mix
        for idx, layer in enumerate(self.encoder_rel):
            x = layer(x)  # x shape [B, C, T, F]

        # x = self.dilateds[1](x)

        # Fusion
        # x = torch.concat([spec, x], dim=1)
        x = complex_cat([spec, x], dim=1)
        x = self.encoder_fusion(x)
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

        feat = self.post_conv(x)

        out_wav = self.stft.inverse(feat)  # B, 1, T
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
                        InstanceNorm(2 * self.fft_dim),
                        nn.PReLU(),
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

        feat = self.post_conv(x)

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


if __name__ == "__main__":
    net = ADPCRN(
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
