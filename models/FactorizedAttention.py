#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange


class FactorizedAttn(nn.Module):
    """Factorized attention
    Input: B,C,T,F
        - x, contenxt embeddings;
        - y, used to calculate the weights.
    Argus:
        - ndim, `C` of inputs;
    """

    def __init__(self, ndim: int, nhead: int = 4) -> None:
        super().__init__()
        self.ctxToHeads = nn.Sequential(
            Rearrange("b c t f->(b f) t c"),
            nn.Linear(ndim, ndim * nhead),
            Rearrange("b t (c n)->b t c n", n=nhead),
        )
        self.wToHeads = nn.Sequential(
            Rearrange("b c t f->(b f) t c"),
            nn.Linear(ndim, nhead),
            nn.Softmax(-1),  # b,t,n
        )

    def forward(self, x, y):
        nB = x.size(0)
        ctx = self.ctxToHeads(x)
        w = self.wToHeads(y)
        refined = torch.einsum("btcn,btn->btc", ctx, w)
        refined = rearrange(refined, "(b f) t c->b c t f", b=nB)
        # return refined.sigmoid()
        return refined.tanh()


class FactorizedAttn2(nn.Module):
    """Factorized attention
    Input: B,C,T,F
        - x, contenxt embeddings;
        - y, used to calculate the weights.
    Argus:
        - ndim, `C` of inputs;
        - hdim, `h` of hidden states;
    """

    def __init__(
        self,
        ndim: int,
        hdim: int,
        nhead: int = 4,
    ) -> None:
        super().__init__()
        self.ctxToHeads = nn.Sequential(
            Rearrange("b c t f->(b f) t c"),
            nn.Linear(ndim, ndim * nhead),
            Rearrange("b t (c n)->b t c n", n=nhead),
        )
        self.wToHeads = nn.Sequential(
            Rearrange("b c t f->(b f) t c"),
            nn.Linear(hdim, nhead),
            nn.Softmax(-1),  # b,t,n
        )

    def forward(self, x, y):
        nB = x.size(0)
        ctx = self.ctxToHeads(x)
        w = self.wToHeads(y)
        refined = torch.einsum("btcn,btn->btc", ctx, w)
        refined = rearrange(refined, "(b f) t c->b c t f", b=nB)
        # return refined.sigmoid()
        return refined.tanh()


if __name__ == "__main__":
    net = FactorizedAttn(2, 2)
    inp = torch.randn(1, 2, 1, 4)
    out = net(inp, inp)
    print(out.shape)
