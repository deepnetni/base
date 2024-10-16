from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import einops
from einops import rearrange
from einops.layers.torch import Rearrange

from models.conv_stft import STFT
from utils.register import tables


@dataclass
class RNNoiseConf:
    # fmt: off
    eband_48 = {
        "type": "BarkScale",
        "nframe": 960, # 50Hz per points
        "nhop": 480,
        "nbands": 22,
        # 0 200 400 600 800 1k 1.2k 1.4k 1.6k 2k 2.4k 2.8 3.2 4 4.8 5.6 6.8 8 9.6 12 15.6 20
        "bands": [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240, 312, 400]
    }
    #
    eband_16 = {
        "type": "MelScale",
        "nframe": 128, # 125 per points
        "nhop": 64,
        "nbands": 22,
        #        0 250 500 750 1 1.25 1.5 1.75 2 2.5 2.75 3k 3.25 3.5k 4k 4.5k 5k 5.5 6k 6.5k 7k 8k
        "bands_64": [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 26, 28, 32, 36, 40, 44, 48, 52, 56, 64],
        # 128w256
        "bands": [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 104, 112, 128],
    }
    # fmt: on


class FeatExtractor(nn.Module):
    """
    ret: B,T,D(22)
    """

    def __init__(self, nframe, nhop, conf: RNNoiseConf = RNNoiseConf()) -> None:
        super().__init__()
        self.stft = STFT(nframe, nhop, win="hann sqrt")
        self.conf = conf

    def compute_band_energy(self, x, spec=None):
        xk: torch.Tensor
        xk = self.stft.transform(x) if spec is None else spec
        pow = xk.pow(2).sum(1)  # B,T,F

        blist = self.conf.eband_16["bands"]
        # bands_bin = [list(range(m, s, e)) for m, s, e in zip(blist[1:-1], blist[:-2], blist[2:])]

        # first
        s, e = (*blist[:2],)
        w = 1 - torch.arange(0, 1, 1.0 / (e - s))
        w = w.to(x.device)
        power_f = 2 * (pow[..., s:e] * w).sum(-1, keepdim=True)

        # last
        s, e = (*blist[-2:],)
        w = torch.arange(0, 1, 1.0 / (e - s))
        w = w.to(x.device)
        power_l = 2 * (pow[..., s:e] * w).sum(-1, keepdim=True)

        # middle
        power_m = []
        for m, s, e in zip(blist[1:-1], blist[:-2], blist[2:]):
            eng = pow[..., s:e]  # B,T,Band
            w_up = torch.arange(0, 1, 1.0 / (m - s), dtype=torch.float32)
            w_down = 1.0 - torch.arange(0, 1, 1.0 / (e - m), dtype=torch.float32)
            w = torch.cat([w_up, w_down])
            w = w.to(x.device)
            power_m.append((eng * w).sum(-1, keepdim=True))

        # B,T,Bands
        bands_pow = torch.cat([power_f, *power_m, power_l], dim=-1)

        return bands_pow

    def compute_band_pitch_corr(self, x, y, norm=True):
        xk: torch.Tensor
        xk = self.stft.transform(x)  # B,C,T,F
        yk = self.stft.transform(y)  # B,C,T,F

        pow = (xk * yk).sum(1)

        blist = self.conf.eband_16["bands"]
        # bands_bin = [list(range(m, s, e)) for m, s, e in zip(blist[1:-1], blist[:-2], blist[2:])]

        # first
        s, e = (*blist[:2],)
        w = 1 - np.arange(0, 1, 1.0 / (e - s))
        power_f = 2 * (pow[..., s:e] * w).sum(-1, keepdim=True)

        # last
        s, e = (*blist[-2:],)
        w = 1 - np.arange(0, 1, 1.0 / (e - s))
        power_l = 2 * (pow[..., s:e] * w).sum(-1, keepdim=True)

        # middle
        power_m = []
        for m, s, e in zip(blist[1:-1], blist[:-2], blist[2:]):
            eng = pow[..., s:e]  # B,T,Band
            w_up = torch.arange(0, 1, 1.0 / (m - s), dtype=torch.float32)
            w_down = 1.0 - torch.arange(0, 1, 1.0 / (e - m), dtype=torch.float32)
            w = torch.concatenate([w_up, w_down])
            power_m.append((eng * w).sum(-1, keepdim=True))

        # B,T,Bands
        bands_pitch_corr = torch.concatenate([power_f, *power_m, power_l], dim=-1)

        if norm:
            x_band_pow = self.compute_band_energy(None, xk)
            y_band_pow = self.compute_band_energy(None, yk)
            # element wise divide
            bands_pitch_corr = bands_pitch_corr / torch.sqrt(x_band_pow * y_band_pow + 1e-6)

        return bands_pitch_corr

    def compute_dct(self, x):
        """
        x: B,T,Bands
        """
        nBands = x.size(-1)

        # Generate matix for DCT transformation
        n = torch.arange(nBands)  # N,
        k = torch.arange(nBands).unsqueeze(1)  # N,1
        dct = torch.cos(torch.pi / nBands * (n + 0.5) * k)  # N, N
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / nBands)
        dct = dct.to(x.device)
        # apply
        dct_out = x.float() @ dct.t()
        return dct_out

    def forward(self, x):
        # xk = self.stft.transform(x)
        bands_pow = self.compute_band_energy(x)
        lg_bands_pow = torch.log10(bands_pow)
        feat = self.compute_dct(lg_bands_pow)
        return feat


@tables.register("models", "rnnoise_vad")
class RNNoiseVAD(nn.Module):
    """
    input: B,T,M

    Args:
        input_size, M;
    """

    def __init__(self, input_size: int = 22) -> None:
        super().__init__()

        self.dense = nn.Sequential(nn.Linear(input_size, 24), nn.Tanh())
        self.gru = nn.GRU(
            input_size=24,
            hidden_size=24,
            num_layers=1,
            batch_first=True,
        )
        self.post = nn.Sequential(nn.Linear(24, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.dense(x)
        x2, _ = self.gru(x1)
        x2 = x2.relu()

        out = self.post(x2)
        # out, 1; x1, 24; x2,24;
        return out, (x1, x2)


@tables.register("models", "rnnoise")
class RNNoise(nn.Module):
    def __init__(self, conf: RNNoiseConf = RNNoiseConf()) -> None:
        super().__init__()
        self.conf = conf

        self.gru1 = nn.GRU(
            input_size=70,
            hidden_size=48,
            num_layers=1,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=94,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
        )

        self.dense = nn.Sequential(nn.Linear(96, 22), nn.Tanh())

    def interp_band_gain(self, gain):
        """
        gain: B,T,nbands
        """
        bands = self.conf.eband_16["bands"]

        interp_g = []

        for i, (st, ed) in enumerate(zip(bands[:-1], bands[1:])):
            N = ed - st
            frac = torch.arange(0, 1, 1 / N)[None, None, :]  # B(1),T(1),F
            frac = frac.to(gain.device)
            # B,T,Fbins
            g = gain[..., i].unsqueeze(-1) * (torch.tensor(1.0).to(gain.device) - frac) + gain[..., i + 1].unsqueeze(-1) * frac
            interp_g.append(g)

        # the last point
        interp_g.append(gain[..., -1].unsqueeze(-1))

        return torch.concat(interp_g, dim=-1).to(gain.device)

    def forward(self, inp, vad_state, spec):
        """
        inp: B,T,Bands
        spec: B,C,T,F
        """
        x1, x2 = vad_state

        x = torch.concat([x1, x2, inp], dim=-1)
        m1, _ = self.gru1(x)
        m1 = m1.relu()

        x = torch.concat([x2, m1, inp], dim=-1)
        m2, _ = self.gru2(x)
        m2 = m2.relu()

        bg = self.dense(m2)  ## B,T,bands(22)

        mask = self.interp_band_gain(bg)  # B,T,F
        # mask = mask.to(x.device)

        real, imag = spec[:, 0, ...], spec[:, 1, ...]
        spec_mags = torch.sqrt(real**2 + imag**2 + 1e-8)
        spec_phase = torch.atan2(imag + 1e-8, real)
        spec_mags = spec_mags * mask

        r = spec_mags * torch.cos(spec_phase)
        i = spec_mags * torch.sin(spec_phase)

        return torch.stack([r, i], dim=1)


if __name__ == "__main__":
    # conf = RNNoiseConf()
    # print(len(conf.eband_48["bands"]), len(conf.eband_16["bands"]))

    # x = torch.randn(1, 10, 42)
    vader = RNNoiseVAD()
    # out = net(x)

    fe = FeatExtractor(128, 64)
    x = torch.randn(1, 24000)
    # corr = fe.compute_band_pitch_corr(x, x)
    # fe.compute_dct(corr)
    feat, spec = fe(x)

    vad, state = vader(feat)

    net = RNNoise()

    out = net(feat, state, spec)
    print(out.shape)
    out = net.interp_band_gain(torch.arange(22)[None, None, :])
