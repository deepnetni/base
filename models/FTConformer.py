#!/usr/bin/env python3
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange


class FConformer(nn.Module):
    """
    Input: B,C,T,F

    Argus
    -----
    ndim: the `C` of the input;
    """

    def __init__(self, ndim, num_head, dropout=0.1, r=4, return_attn=False):
        super().__init__()
        self.num_head = num_head
        dim_ffn = r * ndim
        self.return_attn = return_attn

        self.self_attn = nn.MultiheadAttention(
            embed_dim=ndim, num_heads=num_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(ndim)

        self.ff = nn.Sequential(
            nn.Linear(ndim, dim_ffn),
            nn.GELU(),
            nn.Linear(dim_ffn, ndim),
        )

        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(ndim)

    def forward(self, input):
        nB = input.size(0)
        input = einops.rearrange(input, "b c t f->(b t) f c")
        att_out, w = self.self_attn(input, input, input, need_weights=True)
        norm_out = self.norm1(input + self.dropout1(att_out))

        ffw_out = self.ff(norm_out)
        output = self.norm2(norm_out + self.dropout3(ffw_out))
        output = einops.rearrange(output, "(b t) f c->b c t f", b=nB)

        return output if self.return_attn is False else (output, w)


class TConformer(nn.Module):
    """
    Input: B,C,T,F

    Argus
    -----
    ndim: `C` dim of input;
    attn_wlen: None using the all history frames if `causal=True`;
        otherwise, incoperates the `attn_wlen` the history frames.

    tconv: the dilated length of `T` in dilated convs
    """

    def __init__(
        self,
        ndim,
        num_head,
        dropout=0.1,
        causal=True,
        attn_wlen: Optional[int] = None,
        r=4,
        return_attn=False,
    ):
        super().__init__()
        dim_ffn = r * ndim
        self.attn_wlen = attn_wlen
        self.causal = causal

        self.self_attn = nn.MultiheadAttention(
            embed_dim=ndim, num_heads=num_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(ndim, eps=1e-5)
        self.return_attn = return_attn

        self.ff1 = nn.Sequential(
            nn.Linear(ndim, dim_ffn),
            nn.GELU(),
            nn.Linear(dim_ffn, ndim),
        )

        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(ndim)

    def forward(self, x):
        # tmask
        nT = x.shape[-2]
        nB = x.size(0)
        x = einops.rearrange(x, "b c t f->(b f) t c")

        if self.causal:
            mask_1 = torch.ones(nT, nT, device=x.device, dtype=torch.bool).triu_(1)  # TxT
            mask_2 = (
                torch.ones(nT, nT, device=x.device, dtype=torch.bool).tril_(-self.attn_wlen)
                if self.attn_wlen is not None
                else torch.zeros(nT, nT, device=x.device, dtype=torch.bool)
            )
            mask = mask_1 + mask_2
        else:
            mask = None

        att_out, w = self.self_attn(x, x, x, attn_mask=mask, need_weights=True)
        norm_out = self.norm1(x + self.dropout1(att_out))

        ffw_out = self.ff1(norm_out)  # btc
        output = self.norm2(norm_out + self.dropout3(ffw_out))
        output = einops.rearrange(output, "(b f) t c->b c t f", b=nB)

        return output if self.return_attn is False else (output, w)


class TConformerConv(nn.Module):
    """
    Input: B,C,T,F

    Argus
    -----
    ndim: `C` dim of input;
    attn_wlen: None using the all history frames if `causal=True`;
        otherwise, incoperates the `attn_wlen` the history frames.

    tconv: the dilated length of `T` in dilated convs
    """

    def __init__(
        self,
        ndim,
        num_head,
        dropout=0.1,
        depth=3,
        tconv=3,
        causal=True,
        attn_wlen: Optional[int] = None,
        r=4,
    ):
        super().__init__()
        dim_ffn = r * ndim
        self.attn_wlen = attn_wlen
        self.causal = causal

        self.self_attn = nn.MultiheadAttention(
            embed_dim=ndim, num_heads=num_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(ndim, eps=1e-5)

        self.ff1 = nn.Sequential(
            nn.Linear(ndim, dim_ffn),
            nn.GELU(),
        )
        convs = []
        for i in range(depth):
            # bct
            dil = 2**i
            pad_length = tconv + (dil - 1) * (tconv - 1) - 1
            convs.append(
                nn.Sequential(
                    nn.ConstantPad1d((pad_length, 0), value=0.0),  # lr
                    nn.Conv1d(
                        dim_ffn,
                        dim_ffn,
                        kernel_size=tconv,
                        dilation=dil,
                        groups=dim_ffn,
                    ),
                    nn.GroupNorm(1, dim_ffn),
                    nn.PReLU(dim_ffn),
                )
            )
        self.seq_conv = nn.Sequential(*convs)
        self.linear2 = nn.Linear(dim_ffn, ndim)

        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(ndim)

    def forward(self, x):
        # tmask
        nT = x.shape[-2]
        nB = x.size(0)
        x = einops.rearrange(x, "b c t f->(b f) t c")

        if self.causal:
            mask_1 = torch.ones(nT, nT, device=x.device, dtype=torch.bool).triu_(1)  # TxT
            mask_2 = (
                torch.ones(nT, nT, device=x.device, dtype=torch.bool).tril_(-self.attn_wlen)
                if self.attn_wlen is not None
                else torch.zeros(nT, nT, device=x.device, dtype=torch.bool)
            )
            mask = mask_1 + mask_2
        else:
            mask = None

        att_out, w = self.self_attn(x, x, x, attn_mask=mask)
        norm_out = self.norm1(x + self.dropout1(att_out))

        ffw_mid = self.ff1(norm_out).transpose(1, 2).contiguous()  # btc->bct
        ffw_out = self.linear2(self.seq_conv(ffw_mid).transpose(1, 2).contiguous())
        output = self.norm2(norm_out + self.dropout3(ffw_out))
        output = einops.rearrange(output, "(b f) t c->b c t f", b=nB)

        return output


class FTConformer(nn.Module):
    def __init__(self, ndim, num_head, dropout=0.1, attn_wlen=None, r=4, return_attn=False) -> None:
        super().__init__()
        self.return_attn = return_attn

        self.freq = FConformer(ndim, num_head, dropout, r=r, return_attn=return_attn)
        self.time = TConformer(
            ndim, num_head, dropout, attn_wlen=attn_wlen, causal=True, r=r, return_attn=return_attn
        )
        # self.time = TConformerConv(
        #     ndim, num_head, dropout, depth=depth, attn_wlen=attn_wlen, causal=True
        # )

    def forward(self, x):
        if self.return_attn:
            x, wf = self.freq(x)
            x, wt = self.time(x)
            return x, wf, wt
        else:
            x = self.freq(x)
            x = self.time(x)
            return x


class TConformerQKV(nn.Module):
    """
    Input: B,C,T,F

    Argus
    -----
    ndim: `C` dim of input;
    attn_wlen: None using the all history frames if `causal=True`;
        otherwise, incoperates the `attn_wlen` the history frames.

    tconv: the dilated length of `T` in dilated convs
    """

    def __init__(
        self,
        ndim,
        num_head,
        dropout=0.1,
        causal=True,
        attn_wlen: Optional[int] = None,
        r=4,
        return_attn: bool = False,
    ):
        super().__init__()
        dim_ffn = r * ndim
        self.attn_wlen = attn_wlen
        self.causal = causal

        self.self_attn = nn.MultiheadAttention(
            embed_dim=ndim, num_heads=num_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(ndim, eps=1e-5)

        self.ff1 = nn.Sequential(
            nn.Linear(ndim, dim_ffn),
            nn.GELU(),
            nn.Linear(dim_ffn, ndim),
        )

        self.return_attn = return_attn

        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(ndim)

    def forward(self, q, k, v):
        # tmask
        nT = q.shape[-2]
        nB = q.size(0)
        x = einops.rearrange(q, "b c t f->(b f) t c")
        k = einops.rearrange(k, "b c t f->(b f) t c")
        v = einops.rearrange(v, "b c t f->(b f) t c")

        if self.causal:
            mask_1 = torch.ones(nT, nT, device=x.device, dtype=torch.bool).triu_(1)  # TxT
            mask_2 = (
                torch.ones(nT, nT, device=x.device, dtype=torch.bool).tril_(-self.attn_wlen)
                if self.attn_wlen is not None
                else torch.zeros(nT, nT, device=x.device, dtype=torch.bool)
            )
            mask = mask_1 + mask_2
        else:
            mask = None

        att_out, w = self.self_attn(x, k, v, attn_mask=mask, need_weights=True)
        norm_out = self.norm1(x + self.dropout1(att_out))

        ffw_out = self.ff1(norm_out)  # btc
        output = self.norm2(norm_out + self.dropout3(ffw_out))
        output = einops.rearrange(output, "(b f) t c->b c t f", b=nB)

        return output if self.return_attn is False else (output, w)


if __name__ == "__main__":
    # net = FConformer(20, 4)
    # inp = torch.randn(2, 20, 4, 5)  # B,C,T,F
    # out = net(inp)
    # print(out.shape)

    # net = TConformer(20, 4, attn_wlen=2)
    net = FTConformer(20, 4)
    inp = torch.randn(2, 20, 4, 5)  # B,C,T,F
    out = net(inp)
    print(out.shape)
