from typing import *

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


def forgetting_normalization(XrMag: Tensor, sliding_window_len: int = 192) -> Tensor:
    # https://github.com/Audio-WestlakeU/FullSubNet/blob/e97448375cd1e883276ad583317b1828318910dc/audio_zen/model/base_model.py#L103C19-L103C19
    alpha = (sliding_window_len - 1) / (sliding_window_len + 1)
    mu = 0
    mu_list = []
    B, _, F, T = XrMag.shape
    XrMM = XrMag.mean(dim=2, keepdim=True).detach().cpu()  # [B,1,1,T]
    for t in range(T):
        if t < sliding_window_len:
            alpha_this = min((t - 1) / (t + 1), alpha)
        else:
            alpha_this = alpha
        mu = alpha_this * mu + (1 - alpha_this) * XrMM[..., t]
        mu_list.append(mu)

    XrMM = torch.stack(mu_list, dim=-1).to(XrMag.device)
    return XrMM


# 雨洁的实现
# def cumulative_normalization(original_signal_mag: Tensor, sliding_window_len: int = 192) -> Tensor:
#     alpha = (sliding_window_len - 1) / (sliding_window_len + 1)
#     eps = 1e-10
#     mu = 0
#     mu_list = []
#     batch_size, frame_num, freq_num = original_signal_mag.shape
#     for frame_idx in range(frame_num):
#         if frame_idx < sliding_window_len:
#             alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
#             mu = alp * mu + (1 - alp) * torch.mean(original_signal_mag[:, frame_idx, :], dim=-1).reshape(batch_size, 1)
#         else:
#             current_frame_mu = torch.mean(original_signal_mag[:, frame_idx, :], dim=-1).reshape(batch_size, 1)
#             mu = alpha * mu + (1 - alpha) * current_frame_mu
#         mu_list.append(mu)

#     XrMM = torch.stack(mu_list, dim=-1).permute(0, 2, 1).reshape(batch_size, frame_num, 1, 1)
#     return XrMM


class Norm(nn.Module):
    def __init__(
        self,
        mode: Optional[Literal["utterance", "frequency", "forgetting", "none"]],
        online: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.online = online
        assert mode != "forgetting" or online == True, "forgetting is one online normalization"

    def forward(self, X: Tensor, norm_paras: Any = None, inverse: bool = False) -> Any:
        if not inverse:
            return self.norm(X, norm_paras=norm_paras)
        else:
            return self.inorm(X, norm_paras=norm_paras)

    def norm(
        self, X: Tensor, norm_paras: Any = None, ref_channel: int = None, eps: float = 1e-6
    ) -> Tuple[Tensor, Any]:
        """normalization
        Args:
            X: [B, Chn, F, T], complex
            norm_paras: the paramters for inverse normalization or for the normalization of other X's
            eps: 1e-6!=0 when dtype=float16

        Returns:
            the normalized tensor and the paramters for inverse normalization
        """
        if self.mode == "none" or self.mode is None:
            Xr = X[:, [ref_channel], :, :].clone()
            return X, (Xr, None)

        B, C, F, T = X.shape
        if norm_paras is None:
            Xr = X[:, [ref_channel], :, :].clone()  # [B,1,F,T], complex

            if self.mode == "frequency":
                if self.online:
                    XrMM = torch.abs(Xr) + eps  # [B,1,F,T]
                else:
                    XrMM = (
                        torch.abs(Xr).mean(dim=3, keepdim=True) + eps
                    )  # Xr_magnitude_mean, [B,1,F,1]
            elif self.mode == "forgetting":
                XrMM = forgetting_normalization(XrMag=torch.abs(Xr)) + eps  # [B,1,1,T]
            else:
                assert self.mode == "utterance", self.mode
                if self.online:
                    XrMM = (
                        torch.abs(Xr).mean(dim=(2,), keepdim=True) + eps
                    )  # Xr_magnitude_mean, [B,1,1,T]
                else:
                    XrMM = (
                        torch.abs(Xr).mean(dim=(2, 3), keepdim=True) + eps
                    )  # Xr_magnitude_mean, [B,1,1,1]
        else:
            Xr, XrMM = norm_paras
        X[:, :, :, :] /= XrMM
        return X, (Xr, XrMM)

    def inorm(self, X: Tensor, norm_paras: Any) -> Tensor:
        """inverse normalization
        Args:
            X: [B, Chn, F, T], complex
            norm_paras: the paramters for inverse normalization

        Returns:
            the normalized tensor and the paramters for inverse normalization
        """

        Xr, XrMM = norm_paras
        return X * XrMM

    def extra_repr(self) -> str:
        return f"{self.mode}, online={self.online}"


class LayerNorm(nn.LayerNorm):
    def __init__(self, seq_last: bool, **kwargs) -> None:
        """
        Arg s:
            seq_last (bool): whether the sequence dim is the last dim
        """
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last:
            input = input.transpose(-1, 1)  # [B, H, Seq] -> [B, Seq, H], or [B,H,w,h] -> [B,h,w,H]
        o = super().forward(input)
        if self.seq_last:
            o = o.transpose(-1, 1)
        return o


class GlobalLayerNorm(nn.Module):
    """gLN in convtasnet"""

    def __init__(self, dim_hidden: int, seq_last: bool, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim_hidden = dim_hidden
        self.seq_last = seq_last
        self.eps = eps

        if seq_last:
            self.weight = Parameter(torch.empty([dim_hidden, 1]))
            self.bias = Parameter(torch.empty([dim_hidden, 1]))
        else:
            self.weight = Parameter(torch.empty([dim_hidden]))
            self.bias = Parameter(torch.empty([dim_hidden]))
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): shape [B, Seq, H] or [B, H, Seq]
        """
        var, mean = torch.var_mean(input, dim=(1, 2), unbiased=False, keepdim=True)

        output = (input - mean) / torch.sqrt(var + self.eps)
        output = output * self.weight + self.bias
        return output

    def extra_repr(self) -> str:
        return "{dim_hidden}, seq_last={seq_last}, eps={eps}".format(**self.__dict__)


class BatchNorm1d(nn.Module):
    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__()
        self.seq_last = seq_last
        self.bn = nn.BatchNorm1d(**kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if not self.seq_last:
            input = input.transpose(-1, -2)  # [B, Seq, H] -> [B, H, Seq]
        o = self.bn.forward(input)  # accepts [B, H, Seq]
        if not self.seq_last:
            o = o.transpose(-1, -2)
        return o


class GroupNorm(nn.GroupNorm):
    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last == False:
            input = input.transpose(-1, 1)  # [B, ..., H] -> [B, H, ...]
        o = super().forward(input)  # accepts [B, H, ...]
        if self.seq_last == False:
            o = o.transpose(-1, 1)
        return o


class GroupBatchNorm(Module):
    """Applies Group Batch Normalization over a group of inputs

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    see: `Changsheng Quan, Xiaofei Li. NBC2: Multichannel Speech Separation with Revised Narrow-band Conformer. arXiv:2212.02076.`

    """

    dim_hidden: int
    group_size: int
    eps: float
    affine: bool
    seq_last: bool
    share_along_sequence_dim: bool

    def __init__(
        self,
        dim_hidden: int,
        group_size: Optional[int],
        share_along_sequence_dim: bool = False,
        seq_last: bool = False,
        affine: bool = True,
        eps: float = 1e-5,
        dims_norm: List[int] = None,
        dim_affine: int = None,
    ) -> None:
        """
        Args:
            dim_hidden (int): hidden dimension
            group_size (int): the size of group, optional
            share_along_sequence_dim (bool): share statistics along the sequence dimension. Defaults to False.
            seq_last (bool): whether the shape of input is [B, Seq, H] or [B, H, Seq]. Defaults to False, i.e. [B, Seq, H].
            affine (bool): affine transformation. Defaults to True.
            eps (float): Defaults to 1e-5.
            dims_norm: the dims for normalization
            dim_affine: the dims for affine transformation
        """
        super(GroupBatchNorm, self).__init__()

        self.dim_hidden = dim_hidden
        self.group_size = group_size
        self.eps = eps
        self.affine = affine
        self.seq_last = seq_last
        self.share_along_sequence_dim = share_along_sequence_dim
        if self.affine:
            if seq_last:
                weight = torch.empty([dim_hidden, 1])
                bias = torch.empty([dim_hidden, 1])
            else:
                self.weight = torch.empty([dim_hidden])
                self.bias = torch.empty([dim_hidden])

        assert (dims_norm is not None and dim_affine is not None) or (dims_norm is not None), (
            dims_norm,
            dim_affine,
            "should be none at the time",
        )
        self.dims_norm, self.dim_affine = dims_norm, dim_affine
        if dim_affine is not None:
            assert dim_affine < 0, dim_affine
            weight = weight.squeeze()
            bias = bias.squeeze()
            while dim_affine < -1:
                weight = weight.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
                dim_affine += 1

        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: Tensor, group_size: int = None) -> Tensor:
        """
        Args:
            x: shape [B, Seq, H] if seq_last=False, else shape [B, H, Seq] , where B = num of groups * group size.
            group_size: the size of one group. if not given anywhere, the input must be 4-dim tensor with shape [B, group_size, Seq, H] or [B, group_size, H, Seq]
        """
        if self.group_size != None:
            assert group_size == None or group_size == self.group_size, (
                group_size,
                self.group_size,
            )
            group_size = self.group_size

        if group_size is not None:
            assert (
                x.shape[0] // group_size
            ) * group_size, f"batch size {x.shape[0]} is not divisible by group size {group_size}"

        original_shape = x.shape
        if self.dims_norm is not None:
            var, mean = torch.var_mean(x, dim=self.dims_norm, unbiased=False, keepdim=True)
            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias
        elif self.seq_last == False:
            if x.ndim == 4:
                assert group_size is None or group_size == x.shape[1], (group_size, x.shape)
                B, group_size, Seq, H = x.shape
            else:
                B, Seq, H = x.shape
                x = x.reshape(B // group_size, group_size, Seq, H)

            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(x, dim=(1, 3), unbiased=False, keepdim=True)

            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias

            output = output.reshape(original_shape)
        else:
            if x.ndim == 4:
                assert group_size is None or group_size == x.shape[1], (group_size, x.shape)
                B, group_size, H, Seq = x.shape
            else:
                B, H, Seq = x.shape
                x = x.reshape(B // group_size, group_size, H, Seq)

            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(x, dim=(1, 2), unbiased=False, keepdim=True)

            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias

            output = output.reshape(original_shape)

        return output

    def extra_repr(self) -> str:
        return (
            "{dim_hidden}, {group_size}, share_along_sequence_dim={share_along_sequence_dim}, seq_last={seq_last}, eps={eps}, "
            "affine={affine}".format(**self.__dict__)
        )


def new_norm(
    norm_type: str,
    dim_hidden: int,
    seq_last: bool,
    group_size: int = None,
    num_groups: int = None,
    dims_norm: List[int] = None,
    dim_affine: int = None,
) -> nn.Module:
    if norm_type.upper() == "LN":
        norm = LayerNorm(normalized_shape=dim_hidden, seq_last=seq_last)
    elif norm_type.upper() == "GBN":
        norm = GroupBatchNorm(
            dim_hidden=dim_hidden,
            seq_last=seq_last,
            group_size=group_size,
            share_along_sequence_dim=False,
            dims_norm=dims_norm,
            dim_affine=dim_affine,
        )
    elif norm_type == "GBNShare":
        norm = GroupBatchNorm(
            dim_hidden=dim_hidden,
            seq_last=seq_last,
            group_size=group_size,
            share_along_sequence_dim=True,
            dims_norm=dims_norm,
            dim_affine=dim_affine,
        )
    elif norm_type.upper() == "BN":
        norm = BatchNorm1d(num_features=dim_hidden, seq_last=seq_last)
    elif norm_type.upper() == "GN":
        norm = GroupNorm(num_groups=num_groups, num_channels=dim_hidden, seq_last=seq_last)
    elif norm == "gLN":
        norm = GlobalLayerNorm(dim_hidden, seq_last=seq_last)
    else:
        raise Exception(norm_type)
    return norm
