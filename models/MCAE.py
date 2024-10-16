import sys
from pathlib import Path

from einops import rearrange
from scipy.signal import dimpulse

sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange
from models.conv_stft import STFT
import numpy as np
from models.ft_lstm import FTLSTM_RESNET


class Encoder(nn.Module):
    """
    IN: B,C,T,F
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dims: List = [16, 32, 64, 128, 256],
        kernel_size: List = [(3, 5), (3, 5), (3, 5), (3, 3), (1, 1)],
        kernel_stride: List = [(1, 2), (1, 2), (1, 2), (1, 1), (1, 1)],
    ):
        super().__init__()
        self.in_channels = in_channels // 2
        self.layers = nn.ModuleList()
        in_chs = [in_channels] + hidden_dims

        convs = []
        for ch_in, ch_out, kernel, stride in zip(
            in_chs[:-1], in_chs[1:], kernel_size, kernel_stride
        ):
            kh, kw = (*kernel,)
            pad_w = (kw - 1) // 2
            pad_h = kh - 1
            convs.append(
                nn.Sequential(
                    nn.ConstantPad2d((pad_w, pad_w, pad_h, 0), value=0.0),
                    nn.Conv2d(
                        in_channels=ch_in,
                        out_channels=ch_out,
                        kernel_size=kernel,
                        stride=stride,
                    ),
                    nn.Dropout(p=0.05),
                    Rearrange("b c t f->b t f c"),
                    nn.LayerNorm(ch_out),
                    Rearrange("b t f c->b c t f"),
                    nn.PReLU(ch_out),
                )
            )
        self.layers.append(nn.Sequential(*convs))
        self.post = nn.Dropout(p=0.1)

    def forward(self, x):
        """B,C,T,F"""
        for l in self.layers:
            x = l(x)
        return self.post(x)


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
                    nn.ConvTranspose2d(
                        in_channels=ch_in,
                        out_channels=ch_out,
                        kernel_size=(1, 5),
                        stride=(1, 2),
                        padding=(0, 2),
                    ),
                    nn.Dropout(p=0.05),
                    Rearrange("b c t f->b t f c"),
                    nn.LayerNorm(ch_out),
                    Rearrange("b t f c->b c t f"),
                    nn.PReLU(ch_out),
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


class DenseNet(nn.Module):
    """
    Input
    -------
    B,C,T,F

    """

    def __init__(
        self,
        depth: int,
        in_channels: int,
        kernel_size: Tuple = (3, 5),
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels // 2
        kh, kw = (*kernel_size,)
        self.pad = nn.ConstantPad2d(
            ((kw - 1) // 2, (kw - 1) // 2, kh - 1, 0), value=0.0
        )  # lrud

        convs_r = []
        convs_i = []
        for i in range(depth):
            convs_r.append(
                nn.Sequential(
                    self.pad,
                    nn.Conv2d(
                        in_channels=self.in_channels * (i + 1),
                        out_channels=self.in_channels,
                        kernel_size=kernel_size,
                    ),
                    nn.Dropout(p=0.05),
                    nn.InstanceNorm2d(self.in_channels, affine=True),
                    nn.PReLU(self.in_channels),
                )
            )
            convs_i.append(
                nn.Sequential(
                    self.pad,
                    nn.Conv2d(
                        in_channels=self.in_channels * (i + 1),
                        out_channels=self.in_channels,
                        kernel_size=kernel_size,
                    ),
                    nn.Dropout(p=0.05),
                    nn.InstanceNorm2d(self.in_channels, affine=True),
                    nn.PReLU(self.in_channels),
                )
            )

        self.layers_r = nn.ModuleList(convs_r)
        self.layers_i = nn.ModuleList(convs_i)
        self.post = nn.Dropout(p=0.1)

    def forward(self, x):
        r, i = x.chunk(2, dim=1)
        skip_r = r
        skip_i = i
        for lr, li in zip(self.layers_r, self.layers_i):
            rr = lr(skip_r)
            ri = lr(skip_i)
            ii = li(skip_i)
            ir = li(skip_r)
            r = rr - ii
            i = ri + ir
            skip_r = torch.concat([skip_r, r], dim=1)
            skip_i = torch.concat([skip_i, i], dim=1)

        x = torch.concat([r, i], dim=1)
        return self.post(x)


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

        return z, mu, logvar  # z shape is B,T,H,C

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


class MCAE_BLK_Dense(nn.Module):
    def __init__(
        self,
        feature_size: int,
        in_channels: int = 12,
        up_channels: int = 64,
        x_channel: int = 0,
        depth: int = 4,
        kernel_size: Tuple = (3, 5),
        latent_dim: int = 512,
    ):
        super().__init__()
        n_ch = in_channels // 2
        self.x_channel = torch.tensor([x_channel, x_channel + n_ch])
        n_chs = torch.arange(in_channels)
        self.n_channel = torch.tensor([x for x in n_chs if x not in self.x_channel])

        self.encoder_x = nn.Sequential(
            nn.Conv2d(len(self.x_channel), up_channels, (1, 1), (1, 1)),
            nn.Dropout(p=0.05),
            nn.LayerNorm(feature_size),
            nn.PReLU(up_channels),
            DenseNet(
                depth=depth,
                in_channels=up_channels,
                kernel_size=kernel_size,
            ),
            nn.Conv2d(up_channels, up_channels, (1, 3), (1, 2), (0, 1)),
            nn.LayerNorm(feature_size // 2 + 1),
            nn.PReLU(up_channels),
            nn.Conv2d(up_channels, up_channels, (1, 3), (1, 2), (0, 1)),
            nn.LayerNorm(feature_size // 4 + 1),
            nn.PReLU(up_channels),
        )
        nbin = feature_size // 4 + 1
        self.x_transform = nn.Sequential(
            FTLSTM_RESNET(input_size=up_channels, hidden_size=128),
            nn.Dropout(p=0.05),
            FTLSTM_RESNET(input_size=up_channels, hidden_size=128),
            Rearrange("b c t f-> b t (c f)"),
            nn.Linear(up_channels * nbin, latent_dim),
            nn.Tanh(),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(len(self.n_channel), up_channels, (1, 1), (1, 1)),
            nn.Dropout(p=0.05),
            nn.LayerNorm(feature_size),
            nn.PReLU(up_channels),
            DenseNet(
                depth=depth,
                in_channels=up_channels,
                kernel_size=kernel_size,
            ),
            nn.Conv2d(up_channels, up_channels, (1, 3), (1, 2), (0, 1)),
            nn.LayerNorm(feature_size // 2 + 1),
            nn.PReLU(up_channels),
            nn.Conv2d(up_channels, up_channels, (1, 3), (1, 2), (0, 1)),
            nn.LayerNorm(feature_size // 4 + 1),
            nn.PReLU(up_channels),
        )
        # self.fc_mu = nn.Sequential(
        #     Rearrange("b c t f->b t (f c)"),
        #     nn.Linear(up_channels * feature_size, latent_dim),
        # )
        # self.fc_logvar = nn.Sequential(
        #     Rearrange("b c t f->b t (f c)"),
        #     nn.Linear(up_channels * feature_size, latent_dim),
        # )
        self.context = nn.Sequential(
            FTLSTM_RESNET(input_size=up_channels, hidden_size=128),
            nn.Dropout(p=0.05),
            FTLSTM_RESNET(input_size=up_channels, hidden_size=128),
            Rearrange("b c t f-> b t (c f)"),
            nn.Linear(up_channels * nbin, latent_dim),
            nn.Tanh(),
        )

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

    # def decode(self, z):
    #     x = self.fc_decode_inp(z)
    #     x = self.decoder(x)
    #     return x

    def encode(self, x):
        x = x[:, self.n_channel, ...]

        feats = self.encoder(x)  # b,c,t,f

        return feats
        # mu = self.fc_mu(feats)  # b,t,z
        # logvar = self.fc_logvar(feats)  # b,t,z
        # z = self.reparameterize(mu, logvar)
        # return z, mu, logvar  # b,t,z

    def forward(self, x):
        """
        b,c,t,f
        """

        z_hat = self.encode(x)  # B,C,T,F
        c_hat = self.context(z_hat)  # B,T,H

        x = x[:, self.x_channel, ...]
        z = self.encoder_x(x)
        c = self.x_transform(z)

        return z_hat, c_hat, c


class MCAE_2(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        # cnn_num: List = [32, 64, 128],
        in_channels: int = 6,
        latent_dim: int = 512,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.in_channels = in_channels
        self.eps = 1e-7

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.blocks = nn.ModuleList()
        for i in range(in_channels):
            self.blocks.append(
                MCAE_BLK_Dense(
                    feature_size=self.fft_dim,
                    in_channels=in_channels * 2,
                    up_channels=32,
                    x_channel=i,
                    latent_dim=latent_dim,
                )
            )

    def encode(self, x):
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        xks = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        # print(xks.shape)
        z_l, chat_l, c_l = [], [], []
        for l in self.blocks:
            z, chat, c = l(xks)  # B,T,H
            z_l.append(z)
            chat_l.append(chat)
            c_l.append(c)

        z = torch.stack(z_l, dim=-1)  # B,C,T,F,6
        chat = torch.stack(chat_l, dim=-1)  # B,T,H,6
        tgt = torch.stack(c_l, dim=-1)

        return z, chat, tgt  # B,T,H,6

    def contrastive_loss(self, x, t, tau=0.07):
        """
        x: b,t,h,channel(6)
        t: b,t,h,channel(6)
        """
        x = x.permute(0, 1, 3, 2).unsqueeze(-2)  # B,T,6,1,H
        t = t.permute(0, 1, 3, 2).unsqueeze(-3)  # B,T,1,6,H
        sim = F.cosine_similarity(x, t, dim=-1)  # B,T,6,6
        sim = torch.exp(sim / tau)
        pos_mask = torch.eye(self.in_channels, device=x.device).int()
        neg_mask = 1 - pos_mask
        positive = torch.sum(sim * pos_mask, dim=-1)  # B,T,6
        negative = torch.sum(sim * neg_mask, dim=-1)  # B,T,6
        loss = -torch.log(positive / (positive + negative))  # B,T,6
        loss = torch.mean(loss.sum(1))
        return loss

    def forward(self, x):
        """
        x: B,T,C
        return:
            - z, B,C(6x64),T,F
            - chat, B,T,H(512)
            - c, B,T,H
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        xks = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        z_l, chat_l, c_l = [], [], []
        for l in self.blocks:
            z, chat, c = l(xks)  # B,T,H
            z_l.append(z)
            chat_l.append(chat)
            c_l.append(c)

        z = torch.concat(z_l, dim=1)  # B,6C,T,F
        chat = torch.stack(chat_l, dim=-1)  # B,T,H,6
        tgt = torch.stack(c_l, dim=-1)

        return z, chat, tgt  # B,T,H,6


class DenseEncode(nn.Module):
    """
    x_channel: target_channel
    """

    def __init__(
        self,
        feature_size: int,
        in_channels: int = 12,
        up_channels: int = 64,
        x_channel: int = 0,
        depth: int = 4,
        kernel_size: Tuple = (2, 5),
    ):
        super().__init__()
        n_ch = in_channels // 2
        self.x_channel = torch.tensor([x_channel, x_channel + n_ch])
        n_chs = torch.arange(in_channels)
        self.n_channel = torch.tensor([x for x in n_chs if x not in self.x_channel])

        self.encoder_x = nn.Sequential(
            nn.Conv2d(len(self.x_channel), up_channels, (1, 1), (1, 1)),
            nn.Dropout(p=0.05),
            nn.LayerNorm(feature_size),
            nn.PReLU(up_channels),
            DenseNet(
                depth=depth,
                in_channels=up_channels,
                kernel_size=kernel_size,
            ),
            nn.Conv2d(up_channels, up_channels, (1, 3), (1, 2), (0, 1)),
            nn.LayerNorm(feature_size // 2 + 1),
            nn.PReLU(up_channels),
            nn.Conv2d(up_channels, up_channels, (1, 3), (1, 2), (0, 1)),
            nn.LayerNorm(feature_size // 4 + 1),
            nn.PReLU(up_channels),
        )

        self.encoder_n = nn.Sequential(
            nn.Conv2d(len(self.n_channel), up_channels, (1, 1), (1, 1)),
            nn.Dropout(p=0.05),
            nn.LayerNorm(feature_size),
            nn.PReLU(up_channels),
            DenseNet(
                depth=depth,
                in_channels=up_channels,
                kernel_size=kernel_size,
            ),
            nn.Conv2d(up_channels, up_channels, (1, 3), (1, 2), (0, 1)),
            nn.LayerNorm(feature_size // 2 + 1),
            nn.PReLU(up_channels),
            nn.Conv2d(up_channels, up_channels, (1, 3), (1, 2), (0, 1)),
            nn.LayerNorm(feature_size // 4 + 1),
            nn.PReLU(up_channels),
        )

        # self.context = nn.Sequential(
        #     FTLSTM_RESNET(input_size=up_channels, hidden_size=128),
        #     nn.Dropout(p=0.05),
        #     FTLSTM_RESNET(input_size=up_channels, hidden_size=128),
        #     Rearrange("b c t f-> b t (c f)"),
        #     nn.Linear(up_channels * nbin, latent_dim),
        #     nn.Tanh(),
        # )

    def forward(self, x):
        """
        Input: b,c,t,f
        output: b,c,t,f
            - zn, estimate
            - zx, target
        """
        x_n = x[:, self.n_channel, ...]
        zn = self.encoder_n(x_n)  # b,c,t,f

        x = x[:, self.x_channel, ...]
        zx = self.encoder_x(x)

        return zn, zx


class MCAE_3(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        # cnn_num: List = [32, 64, 128],
        in_channels: int = 6,
        mid_channel: int = 24,
        pred_steps: int = 10,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.in_channels = in_channels
        self.eps = 1e-7

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.blocks = nn.ModuleList()
        for i in range(in_channels):
            self.blocks.append(
                DenseEncode(
                    feature_size=self.fft_dim,
                    in_channels=in_channels * 2,
                    up_channels=mid_channel,
                    x_channel=i,
                )
            )

        self.autoregressive = nn.Sequential(
            FTLSTM_RESNET(input_size=mid_channel, hidden_size=128),
            nn.Dropout(p=0.05),
            FTLSTM_RESNET(input_size=mid_channel, hidden_size=128),
            nn.Dropout(p=0.05),
        )
        # nbin = self.fft_dim // 4 + 1
        self.drop_feat = nn.Dropout2d(p=0.25)

        self.fc = nn.ModuleList()
        for i in range(pred_steps):
            pred = nn.ModuleList(
                [
                    nn.Sequential(
                        Rearrange("b c t->b t c"),
                        nn.Linear(mid_channel, mid_channel),
                        Rearrange("b t c->b c t"),
                    )
                    for _ in range(self.in_channels)
                ]
            )
            self.fc.append(pred)

    def forward(self, x):
        """
        x: B,T,6
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        xks = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        z_l, x_l = [], []
        for l in self.blocks:
            zn, zx = l(xks)  # B,C,T,F
            z_l.append(zn)
            x_l.append(zx)

            z = torch.stack(z_l, dim=-1)  # B,C,T,F,6
        x = torch.stack(x_l, dim=-1)

        z = rearrange(z, "b c t f m->(b m) c t f")
        z = self.drop_feat(z)
        c = self.autoregressive(z)
        c = rearrange(c, "(b m) c t f -> b c t f m", m=self.in_channels)

        c = c.mean(dim=-2)  # b,c,t,m
        x = x.mean(dim=-2)  # b,c,t,m

        return c, x

    # def contrastive_loss(self, x, t, tau=0.07):
    #     """
    #     x: b,t,h,channel(6)
    #     t: b,t,h,channel(6)
    #     """
    #     x = x.permute(0, 1, 3, 2).unsqueeze(-2)  # B,T,6,1,H
    #     t = t.permute(0, 1, 3, 2).unsqueeze(-3)  # B,T,1,6,H
    #     sim = F.cosine_similarity(x, t, dim=-1)  # B,T,6,6
    #     sim = torch.exp(sim / tau)
    #     pos_mask = torch.eye(self.in_channels, device=x.device).int()
    #     neg_mask = 1 - pos_mask
    #     positive = 0.92 * torch.sum(sim * pos_mask, dim=-1)  # B,T,6
    #     negative = 0.08 * torch.sum(sim * neg_mask, dim=-1)  # B,T,6
    #     loss = -torch.log(positive / (positive + negative))  # B,T,6
    #     loss = torch.mean(loss.sum(1))
    #     return loss

    def batch_cpc_loss(self, z_est, z, tau=0.07):
        """
        x: b,c,t-k,6
        t: b,c,t-k,6
        """
        nb, nt = z.size(0), z.size(2)
        z_e = rearrange(z_est, "b c t m->t (b m) 1 c")
        z = rearrange(z, "b c t m->t 1 (b m) c")
        sim = F.cosine_similarity(z_e, z, dim=-1)  # T,bm,bm
        prob = torch.argmax(F.softmax(sim, dim=-1), dim=-1)  # T,bm

        correct = torch.sum(
            torch.eq(prob, torch.arange(nb * self.in_channels, device=z.device))
        )
        correct = correct / (nb * self.in_channels * nt)

        sim = torch.exp(sim / tau)

        pos_mask = torch.eye(self.in_channels * nb, device=z.device).int()
        neg_mask = 1 - pos_mask
        positive = torch.sum(sim * pos_mask, dim=-1)  # T,bm
        negative = torch.sum(sim * neg_mask, dim=-1)  # T,bm
        loss = -torch.log(positive / (positive + negative))  # T,bm
        # loss = loss.sum(0)
        return loss.mean(), correct.detach()

    def loss(self, c, x):
        """
        x: B,C,T,6
        return:
        """

        # c, x = self.forward(x)  # B,C,T,6

        loss = torch.tensor(0.0, device=x.device).float()
        cor = []

        for k, layer_l in enumerate(self.fc, start=1):  # steps
            z_est_l, z_l = [], []
            for ch, l in enumerate(layer_l):  # each channel
                ctx = c[..., :-k, ch]  # b,c,t
                z_ = l(ctx)  # B,C,T
                z = x[..., k:, ch]  # B,C,T-k
                z_est_l.append(z_)
                z_l.append(z)
            z_ = torch.stack(z_est_l, dim=-1)  # B,C,T-k,6
            z = torch.stack(z_l, dim=-1)
            lv, correct = self.batch_cpc_loss(z_, z)
            loss = loss + lv
            cor.append(correct)

        return loss.mean(), torch.tensor(cor).mean()


class MCAE_F(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        # cnn_num: List = [32, 64, 128],
        in_channels: int = 6,
        mid_channel: int = 40,
        pred_steps: int = 10,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.in_channels = in_channels
        self.eps = 1e-7

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.blocks = nn.ModuleList()
        for i in range(in_channels):
            self.blocks.append(
                DenseEncode(
                    feature_size=self.fft_dim,
                    in_channels=in_channels * 2,
                    up_channels=mid_channel,
                    x_channel=i,
                )
            )

        self.autoregressive = nn.Sequential(
            FTLSTM_RESNET(input_size=mid_channel, hidden_size=128),
            nn.Dropout(p=0.05),
            FTLSTM_RESNET(input_size=mid_channel, hidden_size=128),
            nn.Dropout(p=0.05),
        )
        nbin = self.fft_dim // 4 + 1

        self.fc = nn.ModuleList()
        for i in range(pred_steps):
            pred = nn.ModuleList(
                [
                    nn.Sequential(
                        Rearrange("b c t f->b t f c"),
                        nn.Linear(mid_channel, mid_channel),
                        # nn.LayerNorm(mid_channel),
                        # nn.PReLU(mid_channel),
                        Rearrange("b t f c->b t c f"),
                        nn.Linear(nbin, nbin),
                        Rearrange("b t c f->b c t f"),
                        nn.Dropout(0.1),
                    )
                    for _ in range(self.in_channels)
                ]
            )
            self.fc.append(pred)

    def forward(self, x):
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        xks = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        z_l, x_l = [], []
        for l in self.blocks:
            zn, zx = l(xks)  # B,C,T,F
            z_l.append(zn)
            x_l.append(zx)

            z = torch.stack(z_l, dim=-1)  # B,C,T,F,6
        x = torch.stack(x_l, dim=-1)

        z = rearrange(z, "b c t f m->(b m) c t f")
        c = self.autoregressive(z)
        c = rearrange(c, "(b m) c t f -> b c t f m", m=self.in_channels)

        # c = c.mean(dim=-2)  # b,c,t,m
        # x = x.mean(dim=-2)  # b,c,t,m

        # b,c,t,f,m
        return c, x

    def batch_cpc_loss(self, z_est, z, tau=0.07):
        """
        x: b,c,t-k,f,6
        t: b,c,t-k,f,6
        """
        nb, nt = z.size(0), z.size(2)
        z_e = rearrange(z_est, "b c t f m->t (b m) 1 (c f)")
        z = rearrange(z, "b c t f m->t 1 (b m) (c f)")
        sim = F.cosine_similarity(z_e, z, dim=-1)  # T,bm,bm
        prob = torch.argmax(F.softmax(sim, dim=-1), dim=-1)  # T,bm

        correct = torch.sum(
            torch.eq(prob, torch.arange(nb * self.in_channels, device=z.device))
        )
        correct = correct / (nb * self.in_channels * nt)

        sim = torch.exp(sim / tau)

        pos_mask = torch.eye(self.in_channels * nb, device=z.device).int()
        neg_mask = 1 - pos_mask
        positive = torch.sum(sim * pos_mask, dim=-1)  # T,bm
        negative = torch.sum(sim * neg_mask, dim=-1)  # T,bm
        loss = -torch.log(positive / (positive + negative))  # T,bm
        # loss = loss.sum(0)
        return loss.mean(), correct.detach()

    def contrastive_loss(self, x, t, tau=0.07):
        """
        x: b,t,h,channel(6)
        t: b,t,h,channel(6)
        """
        x = x.permute(0, 1, 3, 2).unsqueeze(-2)  # B,T,6,1,H
        t = t.permute(0, 1, 3, 2).unsqueeze(-3)  # B,T,1,6,H
        sim = F.cosine_similarity(x, t, dim=-1)  # B,T,6,6
        sim = torch.exp(sim / tau)
        pos_mask = torch.eye(self.in_channels, device=x.device).int()
        neg_mask = 1 - pos_mask
        positive = torch.sum(sim * pos_mask, dim=-1)  # B,T,6
        negative = torch.sum(sim * neg_mask, dim=-1)  # B,T,6
        loss = -torch.log(positive / (positive + negative))  # B,T,6
        loss = torch.mean(loss.sum(1))
        return loss

    def loss(self, c, x):
        """
        x: B,C,T,F,6
        return:
        """

        # c, x = self.forward(x)  # B,C,T,6

        loss = torch.tensor(0.0, device=x.device).float()
        cor = []

        for k, layer_l in enumerate(self.fc, start=1):  # steps
            z_est_l, z_l = [], []
            for ch, l in enumerate(layer_l):  # each channel
                ctx = c[..., :-k, :, ch]  # b,c,t,f
                z_ = l(ctx)  # B,C,T,F
                z = x[..., k:, :, ch]  # B,C,T-k,F
                z_est_l.append(z_)
                z_l.append(z)
            z_ = torch.stack(z_est_l, dim=-1)  # B,C,T-k,6
            z = torch.stack(z_l, dim=-1)
            lv, correct = self.batch_cpc_loss(z_, z)
            loss = loss + lv
            cor.append(correct)

        return loss.mean(), torch.tensor(cor).mean()


class MCAE_4(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        # cnn_num: List = [32, 64, 128],
        in_channels: int = 6,
        mid_channel: int = 40,
        pred_steps: int = 10,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.in_channels = in_channels
        self.eps = 1e-7

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.blocks = nn.ModuleList()
        for i in range(in_channels):
            self.blocks.append(
                DenseEncode(
                    feature_size=self.fft_dim,
                    in_channels=in_channels * 2,
                    up_channels=mid_channel,
                    x_channel=i,
                )
            )

        self.autoregressive = nn.Sequential(
            FTLSTM_RESNET(input_size=mid_channel, hidden_size=128),
            nn.Dropout(p=0.05),
            # FTLSTM_RESNET(input_size=mid_channel, hidden_size=128),
            # nn.Dropout(p=0.05),
        )
        # nbin = self.fft_dim // 4 + 1
        self.drop_feat = nn.Dropout2d(p=0.25)

        self.fc = nn.ModuleList()
        for i in range(pred_steps):
            pred = nn.ModuleList(
                [
                    nn.Sequential(
                        Rearrange("b c t->b t c"),
                        nn.Linear(mid_channel, mid_channel),
                        Rearrange("b t c->b c t"),
                    )
                    for _ in range(self.in_channels)
                ]
            )
            self.fc.append(pred)

    def forward(self, xks):
        # nB = x.size(0)
        # x = rearrange(x, "b t c-> (b c) t")
        # xk = self.stft.transform(x)
        # xks = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        z_l, x_l = [], []
        for l in self.blocks:
            zn, zx = l(xks)  # B,C,T,F
            z_l.append(zn)
            x_l.append(zx)

            z = torch.stack(z_l, dim=-1)  # B,C,T,F,6

        x = torch.stack(x_l, dim=-1)
        z = rearrange(z, "b c t f m->(b m) c t f")
        z = self.drop_feat(z)
        c = self.autoregressive(z)
        c = rearrange(c, "(b m) c t f -> b c t f m", m=self.in_channels)

        c = c.mean(dim=-2)  # b,c,t,m
        x = x.mean(dim=-2)  # b,c,t,m

        return c, x

    # def contrastive_loss(self, x, t, tau=0.07):
    #     """
    #     x: b,t,h,channel(6)
    #     t: b,t,h,channel(6)
    #     """
    #     x = x.permute(0, 1, 3, 2).unsqueeze(-2)  # B,T,6,1,H
    #     t = t.permute(0, 1, 3, 2).unsqueeze(-3)  # B,T,1,6,H
    #     sim = F.cosine_similarity(x, t, dim=-1)  # B,T,6,6
    #     sim = torch.exp(sim / tau)
    #     pos_mask = torch.eye(self.in_channels, device=x.device).int()
    #     neg_mask = 1 - pos_mask
    #     positive = 0.92 * torch.sum(sim * pos_mask, dim=-1)  # B,T,6
    #     negative = 0.08 * torch.sum(sim * neg_mask, dim=-1)  # B,T,6
    #     loss = -torch.log(positive / (positive + negative))  # B,T,6
    #     loss = torch.mean(loss.sum(1))
    #     return loss

    def batch_cpc_loss(self, z_est, z, tau=0.07):
        """
        x: b,c,t-k,6
        t: b,c,t-k,6
        """
        nb, nt = z.size(0), z.size(2)
        z_e = rearrange(z_est, "b c t m->t (b m) 1 c")
        z = rearrange(z, "b c t m->t 1 (b m) c")
        sim = F.cosine_similarity(z_e, z, dim=-1)  # T,bm,bm
        prob = torch.argmax(F.softmax(sim, dim=-1), dim=-1)  # T,bm

        correct = torch.sum(
            torch.eq(prob, torch.arange(nb * self.in_channels, device=z.device))
        )
        correct = correct / (nb * self.in_channels * nt)

        sim = torch.exp(sim / tau)

        pos_mask = torch.eye(self.in_channels * nb, device=z.device).int()
        neg_mask = 1 - pos_mask
        positive = torch.sum(sim * pos_mask, dim=-1)  # T,bm
        negative = torch.sum(sim * neg_mask, dim=-1)  # T,bm
        loss = -torch.log(positive / (positive + negative))  # T,bm
        # loss = loss.sum(0)
        return loss.mean(), correct.detach()

    def loss(self, c, x):
        """
        x: B,C,T,6
        return:
        """

        # c, x = self.forward(x)  # B,C,T,6

        loss = torch.tensor(0.0, device=x.device).float()
        cor = []

        for k, layer_l in enumerate(self.fc, start=1):  # steps
            z_est_l, z_l = [], []
            for ch, l in enumerate(layer_l):  # each channel
                ctx = c[..., :-k, ch]  # b,c,t
                z_ = l(ctx)  # B,C,T
                z = x[..., k:, ch]  # B,C,T-k
                z_est_l.append(z_)
                z_l.append(z)
            z_ = torch.stack(z_est_l, dim=-1)  # B,C,T-k,6
            z = torch.stack(z_l, dim=-1)
            lv, correct = self.batch_cpc_loss(z_, z)
            loss = loss + lv
            cor.append(correct)

        return loss.mean(), torch.tensor(cor).mean()


if __name__ == "__main__":
    from thop import profile
    import warnings

    inp = torch.randn(1, 16000, 6)  # B,T,C
    # inp = torch.randn(2, 12, 30, 257)  # B,C,T
    net = MCAE_3(512, 256, mid_channel=36,pred_steps=3)
    # net = MCAE_F(nframe=512, nhop=256, nfft=512, in_channels=6, mid_channel=16)
    # flops, param = profile(net, inputs=(inp,))
    # print(flops / 1e9, param / 1e6)
    # net.load_state_dict(
    #     torch.load(r"E:\github\base\trained_mcae\MCAE\checkpoints\epoch_0030.pth")
    # )
    c, z = net(inp)
    print(c.shape, z.shape)
    l = net.loss(c, z)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This API is being deprecated")
        flops, params = profile(net, inputs=(inp,), verbose=False)
    print(f"FLOPs={flops / 1e9}, params={params/1e6:.2f}")

    # net.contrastive_loss(chat, c)
    # z, mu, logvar = net.encode(inp)
    # print(z.shape, mu.shape, logvar.shape)
