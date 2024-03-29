import sys
from pathlib import Path

from einops import rearrange
from torch.nn.modules import padding

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
from models.Fusion.ms_cam import AFF, MS_SELF_CAM, MS_CAM_F


class DPCRN_REFINEMENT(nn.Module):
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
        self.refine = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        self.register_parameter(
            "mask_w", nn.Parameter(torch.ones(2 * (len(cnn_num) + 1), 1, 1))
        )

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

            self.refine.append(
                nn.Sequential(
                    Rearrange("b (m c) t f-> b m t (c f)", m=2),
                    ComplexConv2d(
                        in_channels=2,
                        out_channels=2,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                    ),
                    nn.Linear(
                        in_features=nbinT * self.cnn_num[-1 - idx - 1] // 2
                        if idx != n_cnn_layer - 1
                        else nbinT,
                        out_features=self.fft_dim,
                    ),
                    nn.Sigmoid(),
                )
            )
        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
        )

        self.rnn_dense_r = nn.Sequential(
            Rearrange("b c t f -> b 1 t (c f)"),
            nn.Linear(
                in_features=nbin * self.cnn_num[-1] // 2, out_features=self.fft_dim
            ),
            nn.Sigmoid(),
        )
        self.rnn_dense_i = nn.Sequential(
            Rearrange("b c t f -> b 1 t (c f)"),
            nn.Linear(
                in_features=nbin * self.cnn_num[-1] // 2, out_features=self.fft_dim
            ),
            nn.Sigmoid(),
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

        # Fusion
        # x = torch.concat([spec, x], dim=1)
        # x = complex_cat([spec, x], dim=1)
        # x = self.encoder_fusion(x)
        # x = x + spec
        x_r, x_i = torch.chunk(x, 2, dim=1)
        # x_r = rearrange(x_r, "b c t f->b t (c f)")
        # x_i = rearrange(x_i, "b c t f->b t (c f)")

        x_r = self.rnns_r(x_r)
        x_i = self.rnns_i(x_i)
        mask_rnn_r = self.rnn_dense_r(x_r)
        mask_rnn_i = self.rnn_dense_r(x_i)

        mask_rnn = torch.concat([mask_rnn_r, mask_rnn_i], dim=1)

        # mask_r, mask_i = F.tanh(mask_r), F.tanh(mask_i)

        x = torch.concat([x_r, x_i], dim=1)

        # x = self.dilateds[2](x)

        # feat_r, feat_i = complex_mask_multi(feat, cmask)

        # feat = torch.concat([feat_r, feat_i], dim=1)

        masks = [mask_rnn]
        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, spec_store[-idx - 1]], dim=1)
            x = layer(x)
            masks.append(self.refine[idx](x))

        masks = torch.concat(masks, dim=1) * self.mask_w
        masks_r, masks_i = masks.chunk(2, dim=1)
        masks_r = masks_r.sum(dim=1, keepdim=True)
        masks_i = masks_i.sum(dim=1, keepdim=True)
        x = torch.concat([masks_r, masks_i], dim=1)

        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        x = torch.concat([feat_r, feat_i], dim=1)

        # feat = self.post_conv(x)

        out_wav = self.stft.inverse(x)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


if __name__ == "__main__":
    from thop import profile

    net = DPCRN_REFINEMENT(
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
    flops, num = profile(net, inputs=(inp, inp))
    print(flops / 1e9, num / 1e6)
