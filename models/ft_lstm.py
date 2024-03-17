import einops
import torch
import torch.nn as nn


class MH_FTLSTM(nn.Module):
    """Input and output has the same dimension.
    Input:  B,C,T,F
    Output: B,C,T,F
    Args:
        input_size: should be equal to C of input shape B,C,F,T
    """

    def __init__(self, input_ch, hidden_size, batch_first=True, use_fc=True):
        super().__init__()

        assert input_ch == hidden_size if use_fc is False else True

        self.attn = nn.MultiheadAttention(
            embed_dim=input_ch, num_heads=4, batch_first=True
        )

        self.f_unit = nn.LSTM(
            input_size=input_ch,
            hidden_size=hidden_size // 2,
            batch_first=batch_first,
            bidirectional=True,
        )

        self.f_post = (
            nn.Sequential(
                nn.Linear(hidden_size, input_ch),
                nn.LayerNorm(input_ch),
            )
            if use_fc
            else nn.Identity()
        )

        self.t_unit = nn.LSTM(
            input_size=input_ch,
            hidden_size=hidden_size,
            batch_first=batch_first,
        )

        self.t_post = (
            nn.Sequential(
                nn.Linear(hidden_size, input_ch),
                nn.LayerNorm(input_ch),
            )
            if use_fc
            else nn.Identity()
        )

    def forward(self, inp: torch.Tensor):
        """
        Args:
            x: input shape should be B,C,T,F
        """
        nB, nC, nT, nF = inp.shape

        x = inp.permute(0, 2, 3, 1).flatten(0, 1)  # BxT, F, C
        # x = x.reshape(-1, nF, nC)  # BxT,F,C
        x, _ = self.attn(x, x, x, need_weights=False)
        x, _ = self.f_unit(x)  # BxT,F,C
        x = self.f_post(x)
        x = x.reshape(nB, nT, nF, nC).permute(0, 3, 1, 2)  # B,C,T,F
        inp = inp + x

        x = inp.permute(0, 3, 2, 1).flatten(0, 1)  # BxF,T,C
        # x = x.reshape(-1, nT, nC)  # BxF,T,C
        x, _ = self.t_unit(x)
        x = self.t_post(x)

        x = x.reshape(nB, nF, nT, nC)
        x = x.permute(0, 3, 2, 1)  # B,C,T,F

        return x + inp


class FTLSTM_RESNET(nn.Module):
    """Input and output has the same dimension.
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
        x = einops.rearrange(inp, "b c t f-> (b t) f c")  # BxT,F,C
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
        x = einops.rearrange(inp, "b c t f->(b f) t c")  # BxF,T,C
        x, _ = self.t_unit(x)
        x = self.t_post(x)
        x = einops.rearrange(x, "(b f) t c -> b c t f", b=nB)
        inp = inp + x

        return inp


class FTLSTM_RESNET_ATT(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, atten_wlen=10):
        """
        Args:
            input_size: should be equal to C of input shape B,C,T,F
        """
        super().__init__()
        self.att_wlen = atten_wlen

        self.f_att = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=2, batch_first=True
        )

        self.f_unit = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            batch_first=batch_first,
            bidirectional=True,
        )

        self.f_post = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.LayerNorm(input_size),
        )

        self.t_att = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=2, batch_first=True
        )

        self.t_unit = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
        )

        self.t_post = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.LayerNorm(input_size),
        )

    def forward(self, inp: torch.Tensor):
        """
        Args:
            x: input shape should be B,C,T,F
        """
        nB, nC, nF, nT = inp.shape
        x = einops.rearrange(inp, "b c t f -> (b t) f c")

        xf, _ = self.f_att(x, x, x)
        x = x + xf
        x, _ = self.f_unit(x)  # BxT,F,C
        x = self.f_post(x)
        x = einops.rearrange(x, "(b t) f c -> b c t f", b=nB)
        inp = inp + x
        x = einops.rearrange(inp, "b c t f -> (b f) t c")

        # tmask
        nT = x.shape[1]
        mask_1 = torch.ones(nT, nT, device=x.device).triu_(1).bool()  # TxT
        mask_2 = torch.ones(nT, nT, device=x.device).tril_(-self.att_wlen).bool()  # TxT
        mask = mask_1 + mask_2

        xt, w = self.t_att(x, x, x, attn_mask=mask)
        # print(w.shape, mask, mask.shape)
        # print(xt[..., 0, :], xt.shape)
        x = x + xt
        x, _ = self.t_unit(x)
        x = self.t_post(x)
        x = einops.rearrange(x, "(b f) t c -> b c t f", b=nB)
        x += inp

        return x
