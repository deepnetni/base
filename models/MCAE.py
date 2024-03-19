import sys
from pathlib import Path

from einops import rearrange
from scipy.signal import dimpulse

sys.path.append(str(Path(__file__).parent.parent))
from typing import List, Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from models.conv_stft import STFT
import numpy as np
from models.complexnn import *


class Encoder(nn.Module):
    """
    IN: B,C,T,F
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dims: List = [32, 64, 128],
    ):
        super().__init__()
        self.in_channels = in_channels // 2
        self.layers = nn.ModuleList()
        in_chs = [in_channels] + hidden_dims

        convs = []
        for ch_in, ch_out in zip(in_chs[:-1], in_chs[1:]):
            convs.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=ch_in,
                        out_channels=ch_out,
                        kernel_size=(3, 5),
                        stride=(1, 2),
                        padding=(2, 2),
                    ),
                    Rearrange("b c t f->b t f c"),
                    nn.LayerNorm(ch_out),
                    Rearrange("b t f c->b c t f"),
                    nn.PReLU(),
                )
            )
        self.layers.append(nn.Sequential(*convs))

    def forward(self, x):
        """B,C,T,F"""
        for l in self.layers:
            x = l(x)
        return x


class Decoder(nn.Module):
    """
    IN: B,C,T,F
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dims: List = [128, 64, 32],
    ):
        super().__init__()
        self.in_channels = in_channels // 2
        self.layers = nn.ModuleList()
        in_chs = [in_channels] + hidden_dims

        convs = []
        for ch_in, ch_out in zip(in_chs[:-1], in_chs[1:]):
            convs.append(
                nn.Sequential(
                    ComplexConvTranspose2d(
                        # ComplexGateConvTranspose2d(
                        in_channels=ch_in,
                        out_channels=ch_out,
                        kernel_size=(1, 5),
                        stride=(1, 2),
                        padding=(0, 2),
                    ),
                    Rearrange("b c t f->b t f c"),
                    nn.LayerNorm(ch_out),
                    Rearrange("b t f c->b c t f"),
                    nn.PReLU(),
                )
            )
        self.layers.append(nn.Sequential(*convs))

    def forward(self, x):
        """B,C,T,F"""
        for l in self.layers:
            x = l(x)
        return x


class MCAE_BLK(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        x_channel: int = 0,
        cnn_num: List = [32, 64, 128],
        latent_dim: int = 128,
    ):
        super().__init__()
        n_ch = in_channels // 2
        self.x_channel = torch.tensor([x_channel, x_channel + n_ch])
        n_chs = torch.arange(in_channels)
        self.n_channel = torch.tensor([x for x in n_chs if x not in self.x_channel])

        self.encoder = Encoder(in_channels - 2, cnn_num)
        cnn_num_inv = cnn_num[:-1][::-1] + [2]
        self.decoder = Decoder(cnn_num[-1], cnn_num_inv)
        self.fc_mu = nn.Sequential(
            Rearrange("b c t f->b t f c"),
            nn.Linear(cnn_num[-1], latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            Rearrange("b c t f->b t f c"),
            nn.Linear(cnn_num[-1], latent_dim),
        )
        self.fc_decode_inp = nn.Sequential(
            nn.Linear(latent_dim, cnn_num[-1]),
            Rearrange("b t f c -> b c t f"),
        )

    def encode(self, x):
        x = x[:, self.n_channel, ...]

        feats = self.encoder(x)  # b,c,t,f
        mu = self.fc_mu(feats)
        logvar = self.fc_logvar(feats)
        return mu, logvar  # b,t,f,c

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        x = self.fc_decode_inp(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        """
        b,c,t,f
        """
        x_hat = x[:, self.x_channel, ...]

        mu, logvar = self.encode(x)  # b,t,f,c
        z = self.reparameterize(mu, logvar)

        return (
            self.decode(z),
            x_hat,
            z.permute(0, 3, 1, 2),
            mu.permute(0, 3, 1, 2),
            logvar.permute(0, 3, 1, 2),
        )


class MCAE_BLK_Dense(nn.Module):
    def __init__(
        self,
        feature_size: int,
        in_channels: int = 12,
        x_channel: int = 0,
        depth: int = 4,
        kernel_size: Tuple = (3, 5),
        latent_dim: int = 128,
    ):
        super().__init__()
        n_ch = in_channels // 2
        self.x_channel = torch.tensor([x_channel, x_channel + n_ch])
        n_chs = torch.arange(in_channels)
        self.n_channel = torch.tensor([x for x in n_chs if x not in self.x_channel])

        self.encoder = nn.Sequential(
            ComplexDenseNet(
                depth=depth,
                in_channels=len(self.n_channel),
                kernel_size=kernel_size,
            )
        )
        self.decoder = nn.Sequential(
            ComplexDenseNet(
                depth=depth,
                in_channels=len(self.n_channel),
                kernel_size=kernel_size,
            )
        )
        self.fc_mu = nn.Sequential(
            Rearrange("b c t f->b t (f c)"),
            nn.Linear(len(self.n_channel) * feature_size, latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            Rearrange("b c t f->b t (f c)"),
            nn.Linear(len(self.n_channel) * feature_size, latent_dim),
        )
        self.fc_decode_inp = nn.Sequential(
            nn.Linear(latent_dim, len(self.n_channel) * feature_size),
            Rearrange("b t (f c)->b c t f", f=feature_size),
        )

    def encode(self, x):
        x = x[:, self.n_channel, ...]

        feats = self.encoder(x)  # b,c,t,f
        mu = self.fc_mu(feats)  # b,t,z
        logvar = self.fc_logvar(feats)  # b,t,z
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar  # b,t,z

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        x = self.fc_decode_inp(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        """
        b,c,t,f
        """
        x_hat = x[:, self.x_channel, ...]

        z, mu, logvar = self.encode(x)  # b,t,z

        return self.decode(z), x_hat, z, mu, logvar


class MCAE(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        # cnn_num: List = [32, 64, 128],
        in_channels: int = 6,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.in_channels = in_channels

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.blocks = nn.ModuleList()
        for i in range(in_channels):
            self.blocks.append(
                MCAE_BLK_Dense(
                    feature_size=self.fft_dim,
                    in_channels=in_channels * 2,
                    x_channel=i,
                    latent_dim=latent_dim,
                )
            )

    def encode(self, x):
        nC = x.size(-1)
        xks = []
        for i in range(nC):
            d = x[..., i]  # B,T
            xk = self.stft.transform(d)  # B,C,T,F
            xks.append(xk)

        xks = torch.concat(xks, dim=1)  # B,C,T,F
        # print(xks.shape)
        z_l, mu_l, var_l = [], [], []
        for l in self.blocks:
            z, mu, logvar = l.encode(xks)
            z_l.append(z)
            mu_l.append(mu)
            var_l.append(logvar)

        z = torch.stack(z_l, dim=-1)
        mu = torch.stack(mu_l, dim=-1)
        logvar = torch.stack(var_l, dim=-1)

        return z, mu, logvar

    def forward(self, x):
        """
        x: B,T,C
        """
        nC = x.size(-1)
        xks = []
        for i in range(nC):
            d = x[..., i]  # B,T
            xk = self.stft.transform(d)  # B,C,T,F
            xks.append(xk)

        xks = torch.concat(xks, dim=1)  # B,C,T,F
        # print(xks.shape)

        x_l, x_h_l, z_l, mu_l, var_l = [], [], [], [], []
        for l in self.blocks:
            x_, x_hat, z, mu, logvar = l(xks)
            x_ = self.stft.inverse(x_)
            x_l.append(x_)
            x_hat = self.stft.inverse(x_hat)
            x_h_l.append(x_hat)
            z_l.append(z)
            mu_l.append(mu)
            var_l.append(logvar)

        x_ = torch.stack(x_l, dim=-1)
        x_hat = torch.stack(x_h_l, dim=-1)
        z = torch.stack(z_l, dim=-1)
        mu = torch.stack(mu_l, dim=-1)
        logvar = torch.stack(var_l, dim=-1)

        return x_, x_hat, z, mu, logvar


if __name__ == "__main__":
    inp = torch.randn(2, 16000, 6)  # B,T,C
    # inp = torch.randn(2, 12, 30, 257)  # B,C,T
    net = MCAE(512, 256)
    out, lbl, z, _, _ = net(inp)
    print(out.shape, lbl.shape, z.shape)
