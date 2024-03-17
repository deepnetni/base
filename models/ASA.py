import torch
import torch.nn as nn
import einops


class ASA(nn.Module):
    """
    Input: B, C, T, F
    Return: B, C, T, F

    Args:
        in_channels: the Channel dim of input shape B,C,T,F

    Proposed by
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746610
    called Axial Self-Attention (ASA)
    """

    def __init__(self, in_channels, causal=True):
        """ """
        super().__init__()
        # self.atten_ch = in_channels // 4
        self.atten_ch = in_channels
        self.f_qkv = nn.Sequential(
            nn.Conv2d(in_channels, self.atten_ch * 3, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.atten_ch * 3),
            nn.PReLU(),
        )

        self.t_qk = nn.Sequential(
            nn.Conv2d(in_channels, self.atten_ch * 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.atten_ch * 2),
            nn.PReLU(),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(self.atten_ch, in_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )
        self.causal = causal

    def forward(self, inp):
        # step 1. F-attention
        f_qkv = self.f_qkv(inp)
        # qf, kf, v shape B,C,T,F
        qf, kf, v = tuple(einops.rearrange(f_qkv, "b (c k) t f -> k b c t f", k=3))
        # b,t,f,c matmul b,t,c,y -> b,t,f_query,f_key
        f_score = torch.einsum("bctf,bcty->btfy", qf, kf) / (self.atten_ch**0.5)
        f_score = f_score.softmax(dim=-1)
        # b,t,fq,fk matmul b,t,fk,c -> b,t,fq,c -> b,c,t,fq
        f_out = torch.einsum("btfy,bcty->bctf", f_score, v)

        # step 2. T-attention
        t_qk = self.t_qk(inp)
        qt, kt = tuple(einops.rearrange(t_qk, "b (c k) t f ->k b c t f", k=2))
        # b,f,t,c matmul b,f,c,y(tk)) -> b,f,t,tk
        t_score = torch.einsum("bctf,bcyf->bfty", qt, kt) / (self.atten_ch**0.5)

        if self.causal is True:
            mask_v = -torch.finfo(t_score.dtype).max
            # t_score shape is b,f,qt,kt
            # i is the query time axis, which is the row of t_score.
            # j - i + 1 is the upper triangle
            i, j = t_score.shape[-2:]  # i->qt, j->kt
            mask = torch.ones(i, j, device=t_score.device).triu_(j - i + 1).bool()
            # modify elements to mask_v when mask is true
            t_score.masked_fill(mask, mask_v)

        t_score = t_score.softmax(dim=-1)
        # b,f,tq,y(tk) matmul b,f,y(tk),c
        t_out = torch.einsum("bfty,bcyf->bctf", t_score, f_out).contiguous()
        out = self.proj(t_out)
        return out + inp
