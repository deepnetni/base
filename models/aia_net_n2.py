import copy
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.rnn import GRU, LSTM
from torch.nn.parameter import Parameter


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, bidirectional=True, dropout=0, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.gru = GRU(d_model, d_model * 2, 1, bidirectional=bidirectional)
        self.dropout = Dropout(dropout)
        if bidirectional:
            self.linear2 = Linear(d_model * 2 * 2, d_model)
        else:
            self.linear2 = Linear(d_model * 2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src_norm = self.norm3(src)
        # src_norm = src
        src2 = self.self_attn(
            src_norm,
            src_norm,
            src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        src2 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoderLayer_cau(nn.Module):
    """
    Arguments:
        - feature_size: F of input B,C,T,F
    """

    def __init__(self, feature_size: int, nhead, bidirectional=True, dropout=0, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(feature_size, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.gru = GRU(
            input_size=feature_size,
            hidden_size=feature_size * 2,
            num_layers=1,
            bidirectional=bidirectional,
        )

        self.norm1 = LayerNorm(feature_size)
        self.norm2 = LayerNorm(feature_size)
        self.norm3 = LayerNorm(feature_size)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        if bidirectional:
            self.post = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(feature_size * 2 * 2, feature_size),
            )
        else:
            self.post = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(feature_size * 2, feature_size),
            )

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal: bool = False):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        """

        x_norm = self.norm3(x)

        # src_norm = x

        # attn_mask_in = nn.Transformer.generate_square_subsequent_mask(x.shape[0])
        # attn_mask_in = nn.Transformer.generate_square_subsequent_mask(
        #     x.shape[0], x.device
        # )
        # print(attn_mask_in)
        # attn_mask_in = None

        x_, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        x = x + self.dropout1(x_)
        x = self.norm1(x)
        # self.gru.flatten_parameters()
        out, h_n = self.gru(x)
        # x_ = self.fc2(self.dropout(self.activation(out)))
        x_ = self.post(out)
        x = x + self.dropout2(x_)
        x = self.norm2(x)
        return x


class TransformerEncoderLayer_new(nn.Module):
    def __init__(self, d_model, nhead, bidirectional=True, dropout=0, activation="relu"):
        super(TransformerEncoderLayer_new, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.gru = GRU(d_model, d_model, 1, bidirectional=bidirectional)
        self.dropout = Dropout(dropout)
        if bidirectional:
            self.linear2 = Linear(d_model * 2, d_model)
        else:
            self.linear2 = Linear(d_model, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src_norm = self.norm3(src)
        # src_norm = src
        src2 = self.self_attn(
            src_norm,
            src_norm,
            src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        src2 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class AIA_Transformer_cau(nn.Module):
    """
    Adaptive time-frequency attention Transformer without interaction on maginitude path and complex path.
    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
    """

    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.k1 = Parameter(torch.ones(1))
        self.k2 = Parameter(torch.ones(1))

        self.pre_conv = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1), nn.PReLU()
        )

        # dual-path RNN
        self.F_trans = nn.ModuleList([])
        self.T_trans = nn.ModuleList([])
        self.F_norm = nn.ModuleList([])
        self.T_norm = nn.ModuleList([])

        for _ in range(num_layers):
            self.F_trans.append(
                TransformerEncoderLayer_cau(
                    feature_size=input_size // 2,
                    nhead=4,
                    dropout=dropout,
                    bidirectional=True,
                )
            )
            self.T_trans.append(
                TransformerEncoderLayer_cau(
                    feature_size=input_size // 2,
                    nhead=4,
                    dropout=dropout,
                    bidirectional=False,
                )
            )
            self.F_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))
            self.T_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))

        # output layer
        self.post_conv = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size // 2, output_size, 1))

    def forward(self, x):
        """
        # x: B,C,T,F
        """
        nB = x.size(0)

        output_list = []
        inp = self.pre_conv(x)

        for i in range(len(self.F_trans)):
            x = rearrange(inp, "b c t f -> f (b t) c")
            x = self.F_trans[i](x, is_causal=False)  # [F, B*T, c]
            x = rearrange(x, "f (b t) c-> b c t f", b=nB)
            x_f = self.F_norm[i](x)  # [B, C, T, F]

            x = rearrange(x_f, "b c t f -> t (b f) c")
            x = self.T_trans[i](x, is_causal=True)
            x = rearrange(x, "t (b f) c-> b c t f", b=nB)
            x_t = self.T_norm[i](x)
            inp = inp + self.k1 * x_f + self.k2 * x_t
            out_i = self.post_conv(inp)
            output_list.append(out_i)

        return out_i, output_list


class AIA_Transformer(nn.Module):
    """
    Adaptive time-frequency attention Transformer without interaction on maginitude path and complex path.
    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
    """

    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.k1 = Parameter(torch.ones(1))
        self.k2 = Parameter(torch.ones(1))

        self.pre_conv = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1), nn.PReLU()
        )

        # dual-path RNN
        self.F_trans = nn.ModuleList([])
        self.T_trans = nn.ModuleList([])
        self.F_norm = nn.ModuleList([])
        self.T_norm = nn.ModuleList([])

        for _ in range(num_layers):
            self.F_trans.append(
                TransformerEncoderLayer_cau(
                    feature_size=input_size // 2,
                    nhead=4,
                    dropout=dropout,
                    bidirectional=True,
                )
            )
            self.T_trans.append(
                TransformerEncoderLayer_cau(
                    feature_size=input_size // 2,
                    nhead=4,
                    dropout=dropout,
                    bidirectional=False,
                )
            )
            self.F_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))
            self.T_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))

        # output layer
        self.post_conv = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size // 2, output_size, 1))

    def forward(self, x):
        """
        # x: B,C,T,F
        """
        nB = x.size(0)

        output_list = []
        inp = self.pre_conv(x)

        for i in range(len(self.F_trans)):
            x = rearrange(inp, "b c t f -> f (b t) c")
            x = self.F_trans[i](x, is_causal=False)  # [F, B*T, c]
            x = rearrange(x, "f (b t) c-> b c t f", b=nB)
            x_f = self.F_norm[i](x)  # [B, C, T, F]

            x = rearrange(x_f, "b c t f -> t (b f) c")
            x = self.T_trans[i](x, is_causal=False)
            x = rearrange(x, "t (b f) c-> b c t f", b=nB)
            x_t = self.T_norm[i](x)
            inp = inp + self.k1 * x_f + self.k2 * x_t
            out_i = self.post_conv(inp)
            output_list.append(out_i)

        return out_i, output_list


# class AIA_Transformer(nn.Module):
#     """
#     Adaptive time-frequency attention Transformer without interaction on maginitude path and complex path.
#     args:
#         input_size: int, dimension of the input feature. The input should have shape
#                     (batch, seq_len, input_size).
#         hidden_size: int, dimension of the hidden state.
#         output_size: int, dimension of the output size.
#         dropout: float, dropout ratio. Default is 0.
#         num_layers: int, number of stacked RNN layers. Default is 1.
#     """
#
#     def __init__(self, input_size, output_size, dropout=0, num_layers=1):
#         super(AIA_Transformer, self).__init__()
#
#         self.input_size = input_size
#         self.output_size = output_size
#         self.k1 = Parameter(torch.ones(1))
#         self.k2 = Parameter(torch.ones(1))
#
#         self.input = nn.Sequential(
#             nn.Conv2d(input_size, input_size // 2, kernel_size=1), nn.PReLU()
#         )
#
#         # dual-path RNN
#         self.row_trans = nn.ModuleList([])
#         self.col_trans = nn.ModuleList([])
#         self.row_norm = nn.ModuleList([])
#         self.col_norm = nn.ModuleList([])
#         for i in range(num_layers):
#             self.row_trans.append(
#                 TransformerEncoderLayer(
#                     d_model=input_size // 2,
#                     nhead=4,
#                     dropout=dropout,
#                     bidirectional=True,
#                 )
#             )
#             self.col_trans.append(
#                 TransformerEncoderLayer(
#                     d_model=input_size // 2,
#                     nhead=4,
#                     dropout=dropout,
#                     bidirectional=True,
#                 )
#             )
#             self.row_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))
#             self.col_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))
#
#         # output layer
#         self.output = nn.Sequential(
#             nn.PReLU(), nn.Conv2d(input_size // 2, output_size, 1)
#         )
#
#     def forward(self, input):
#         #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
#         b, c, dim2, dim1 = input.shape
#         output_list = []
#         output = self.input(input)
#         for i in range(len(self.row_trans)):
#             row_input = (
#                 output.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)
#             )  # [F, B*T, c]
#             row_output = self.row_trans[i](row_input)  # [F, B*T, c]
#             row_output = (
#                 row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()
#             )  # [B, C, T, F]
#             row_output = self.row_norm[i](row_output)  # [B, C, T, F]
#
#             col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)
#             col_output = self.col_trans[i](col_input)
#             col_output = (
#                 col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()
#             )
#             col_output = self.col_norm[i](col_output)
#             output = output + self.k1 * row_output + self.k2 * col_output
#             output_i = self.output(output)
#             output_list.append(output_i)
#         del row_input, row_output, col_input, col_output
#
#         return output_i, output_list


class AIA_Transformer_new(nn.Module):
    """
    Adaptive time-frequency attention Transformer without interaction on maginitude path and complex path.
    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
    """

    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super(AIA_Transformer_new, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.k1 = Parameter(torch.ones(1))
        self.k2 = Parameter(torch.ones(1))

        # self.input = nn.Sequential(
        #     nn.Conv2d(input_size, input_size // 2, kernel_size=1),
        #     nn.PReLU()
        # )

        # dual-path RNN
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(
                TransformerEncoderLayer_new(
                    d_model=input_size, nhead=4, dropout=dropout, bidirectional=True
                )
            )
            self.col_trans.append(
                TransformerEncoderLayer_new(
                    d_model=input_size, nhead=4, dropout=dropout, bidirectional=True
                )
            )
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # output layer
        # self.output = nn.Sequential(nn.PReLU(),
        #                             nn.Conv2d(input_size//2, output_size, 1)
        #                             )

    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input.shape
        output_list = []
        output = input
        for i in range(len(self.row_trans)):
            row_input = (
                output.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)
            )  # [F, B*T, c]
            row_output = self.row_trans[i](row_input)  # [F, B*T, c]
            row_output = (
                row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()
            )  # [B, C, T, F]
            row_output = self.row_norm[i](row_output)  # [B, C, T, F]

            col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)
            col_output = self.col_trans[i](col_input)
            col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()
            col_output = self.col_norm[i](col_output)
            output = output + self.k1 * row_output + self.k2 * col_output
            # output_i = self.output(output)
            output_list.append(output)
        del row_input, row_output, col_input, col_output

        return output, output_list


class AIA_Transformer_merge(nn.Module):
    """
    Adaptive time-frequency attention Transformer with interaction on two branch
    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
    """

    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super(AIA_Transformer_merge, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.k1 = Parameter(torch.ones(1))
        self.k2 = Parameter(torch.ones(1))

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1), nn.PReLU()
        )

        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(
                TransformerEncoderLayer(
                    d_model=input_size // 2,
                    nhead=4,
                    dropout=dropout,
                    bidirectional=True,
                )
            )
            self.col_trans.append(
                TransformerEncoderLayer(
                    d_model=input_size // 2,
                    nhead=4,
                    dropout=dropout,
                    bidirectional=True,
                )
            )
            self.row_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size // 2, output_size, 1))

    def forward(self, input1, input2):
        #  input --- [B,  C,  T, F]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input1.shape
        output_list_mag = []
        output_list_ri = []
        input_merge = torch.cat((input1, input2), dim=1)
        input_mag = self.input(input_merge)
        input_ri = self.input(input_merge)
        for i in range(len(self.row_trans)):
            if i >= 1:
                output_mag_i = output_list_mag[-1] + output_list_ri[-1]
            else:
                output_mag_i = input_mag
            AFA_input_mag = (
                output_mag_i.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)
            )  # [F, B*T, c]
            AFA_output_mag = self.row_trans[i](AFA_input_mag)  # [F, B*T, c]
            AFA_output_mag = (
                AFA_output_mag.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()
            )  # [B, C, T, F]
            AFA_output_mag = self.row_norm[i](AFA_output_mag)  # [B, C, T, F]

            ATA_input_mag = (
                output_mag_i.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)
            )  # [T, B*F, C]
            ATA_output_mag = self.col_trans[i](ATA_input_mag)  # [T, B*F, C]
            ATA_output_mag = (
                ATA_output_mag.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()
            )  # [B, C, T, F]
            ATA_output_mag = self.col_norm[i](ATA_output_mag)  # [B, C, T, F]
            output_mag_i = (
                input_mag + self.k1 * AFA_output_mag + self.k2 * ATA_output_mag
            )  # [B, C, T, F]
            output_mag_i = self.output(output_mag_i)
            output_list_mag.append(output_mag_i)

            if i >= 1:
                output_ri_i = output_list_ri[-1] + output_list_mag[-2]
            else:
                output_ri_i = input_ri
            # input_ri_mag = output_ri_i + output_list_mag[-1]
            AFA_input_ri = (
                output_ri_i.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)
            )  # [F, B*T, c]
            AFA_output_ri = self.row_trans[i](AFA_input_ri)  # [F, B*T, c]
            AFA_output_ri = (
                AFA_output_ri.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()
            )  # [B, C, T, F]
            AFA_output_ri = self.row_norm[i](AFA_output_ri)  # [B, C, T, F]

            ATA_input_ri = (
                output_ri_i.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)
            )  # [T, B*F, C]
            ATA_output_ri = self.col_trans[i](ATA_input_ri)  # [T, B*F, C]
            ATA_output_ri = (
                ATA_output_ri.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()
            )  # [B, C, T, F]
            ATA_output_ri = self.col_norm[i](ATA_output_ri)  # [B, C, T, F]
            output_ri_i = (
                input_ri + self.k1 * AFA_output_ri + self.k2 * ATA_output_ri
            )  # [B, C, T, F]
            output_ri_i = self.output(output_ri_i)
            output_list_ri.append(output_ri_i)

        del (
            AFA_input_mag,
            AFA_output_mag,
            ATA_input_mag,
            ATA_output_mag,
            AFA_input_ri,
            AFA_output_ri,
            ATA_input_ri,
            ATA_output_ri,
        )
        # [b, c, dim2, dim1]

        return output_mag_i, output_list_mag, output_ri_i, output_list_ri


class AIA_DCN_Transformer_merge(nn.Module):
    """
    Adaptive time-frequency attention Transformer with interaction on two branch
    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
    """

    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super(AIA_DCN_Transformer_merge, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.k1 = Parameter(torch.ones(1))
        self.k2 = Parameter(torch.ones(1))

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1), nn.PReLU()
        )

        self.row_trans = nn.ModuleList([])
        self.row_DCN = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.col_DCN = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(
                TransformerEncoderLayer(
                    d_model=input_size // 2,
                    nhead=4,
                    dropout=dropout,
                    bidirectional=True,
                )
            )
            self.row_DCN.append(DilatedDenseNet(depth=4, in_channels=64))

            self.col_trans.append(
                TransformerEncoderLayer(
                    d_model=input_size // 2,
                    nhead=4,
                    dropout=dropout,
                    bidirectional=True,
                )
            )
            self.col_DCN.append(DilatedDenseNet(depth=4, in_channels=64))

            self.row_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size // 2, output_size, 1))

    def forward(self, input1, input2):
        #  input --- [B,  C,  T, F]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input1.shape
        output_list_mag = []
        output_list_ri = []
        input_merge = torch.cat((input1, input2), dim=1)
        input_mag = self.input(input_merge)
        input_ri = self.input(input_merge)
        for i in range(len(self.row_trans)):
            if i >= 1:
                output_mag_i = output_list_mag[-1] + output_list_ri[-1]
            else:
                output_mag_i = input_mag

            ###### input DCN

            intra_in_DCN_i = output_mag_i.permute(0, 1, 3, 2)
            intra_out_DCN_i = self.row_DCN[i](intra_in_DCN_i)

            intra_out_DCN_i = intra_out_DCN_i.permute(0, 1, 3, 2)

            AFA_input_mag = (
                intra_out_DCN_i.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)
            )  # [F, B*T, c]
            AFA_output_mag = self.row_trans[i](AFA_input_mag)  # [F, B*T, c]
            AFA_output_mag = (
                AFA_output_mag.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()
            )  # [B, C, T, F]
            AFA_output_mag = self.row_norm[i](AFA_output_mag)  # [B, C, T, F]

            ATA_input_mag = (
                intra_out_DCN_i.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)
            )  # [T, B*F, C]
            ATA_output_mag = self.col_trans[i](ATA_input_mag)  # [T, B*F, C]
            ATA_output_mag = (
                ATA_output_mag.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()
            )  # [B, C, T, F]
            ATA_output_mag = self.col_norm[i](ATA_output_mag)  # [B, C, T, F]
            ATA_output_temp_i = (
                input_mag + self.k1 * AFA_output_mag + self.k2 * ATA_output_mag
            )  # [B, C, T, F]

            output_mag_i = self.col_DCN[i](ATA_output_temp_i)

            output_mag_i = self.output(output_mag_i)
            output_list_mag.append(output_mag_i)

            if i >= 1:
                output_ri_i = output_list_ri[-1] + output_list_mag[-2]
            else:
                output_ri_i = input_ri

            intra_in_DCN_ri = output_ri_i.permute(0, 1, 3, 2)
            intra_out_DCN_ri = self.row_DCN[i](intra_in_DCN_ri)

            intra_out_DCN_ri = intra_out_DCN_ri.permute(0, 1, 3, 2)

            # input_ri_mag = output_ri_i + output_list_mag[-1]
            AFA_input_ri = (
                intra_out_DCN_ri.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)
            )  # [F, B*T, c]
            AFA_output_ri = self.row_trans[i](AFA_input_ri)  # [F, B*T, c]
            AFA_output_ri = (
                AFA_output_ri.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()
            )  # [B, C, T, F]
            AFA_output_ri = self.row_norm[i](AFA_output_ri)  # [B, C, T, F]

            ATA_input_ri = (
                intra_out_DCN_ri.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)
            )  # [T, B*F, C]
            ATA_output_ri = self.col_trans[i](ATA_input_ri)  # [T, B*F, C]
            ATA_output_ri = (
                ATA_output_ri.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()
            )  # [B, C, T, F]
            ATA_output_ri = self.col_norm[i](ATA_output_ri)  # [B, C, T, F]
            ATA_output_temp_ri = (
                input_ri + self.k1 * AFA_output_ri + self.k2 * ATA_output_ri
            )  # [B, C, T, F]

            output_ri_i = self.col_DCN[i](ATA_output_temp_ri)

            output_ri_i = self.output(output_ri_i)
            output_list_ri.append(output_ri_i)

        del (
            AFA_input_mag,
            AFA_output_mag,
            ATA_input_mag,
            ATA_output_mag,
            AFA_input_ri,
            AFA_output_ri,
            ATA_input_ri,
            ATA_output_ri,
        )
        # [b, c, dim2, dim1]

        return output_mag_i, output_list_mag, output_ri_i, output_list_ri


class AHAM(nn.Module):  # aham merge
    def __init__(self, input_channel=64, kernel_size=(1, 1), bias=True):
        super(AHAM, self).__init__()

        self.k3 = Parameter(torch.zeros(1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        self.conv1 = nn.Conv2d(input_channel, 1, kernel_size, (1, 1), bias=bias)

    def merge(self, x, y):
        batch, channel, height, width, blocks = x.size()
        input_x = x
        y = self.softmax(y)
        context = torch.matmul(input_x, y)
        context = context.view(batch, channel, height, width)

        return context

    def forward(self, input_list):  # X:BCTFG Y:B11G1
        batch, channel, frames, frequency = input_list[-1].size()
        # x_list = []
        y_list = []
        for i in range(len(input_list)):
            input = self.avg_pool(input_list[i])  # BCTF->BC11
            y = self.conv1(input)  # B111
            y = y.unsqueeze(-2)  # B1111
            # x = input_list[i].unsqueeze(-1)  # BCTF1
            # x_list.append(x)
            y_list.append(y)
        # x_merge = torch.cat((x_list[0], x_list[1], x_list[2], x_list[3]), dim=-1)
        # y_merge = torch.cat((y_list[0], y_list[1], y_list[2], y_list[3]), dim=-2)
        x_merge = torch.stack(input_list, dim=-1)  # BCTF->BCTF4
        y_merge = torch.cat(y_list, dim=-2)  # B1111->B1141

        y_softmax = self.softmax(y_merge)  # B114_1

        # aham = torch.matmul(x_merge, y_softmax)  # B
        aham = x_merge @ y_softmax  # BCTF_4 x B114_1 => BCTF1
        aham = aham.view(batch, channel, frames, frequency)
        aham_output = input_list[-1] + aham
        return aham_output


class AHAM_ori(
    nn.Module
):  # aham merge, share the weights on 1D CNN get better performance and lower parameter
    def __init__(self, input_channel=64, kernel_size=(1, 1), bias=True, act=nn.ReLU(True)):
        super(AHAM_ori, self).__init__()

        self.k3 = Parameter(torch.zeros(1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        self.conv1 = nn.Conv2d(input_channel, 1, kernel_size, (1, 1), bias=bias)

    def merge(self, x, y):
        batch, channel, height, width, blocks = x.size()
        input_x = x  # B*C*T*F*G
        y = self.softmax(y)
        context = torch.matmul(input_x, y)
        context = context.view(batch, channel, height, width)  # B*C*T*F

        return context

    def forward(self, input_list):  # X:BCTFG Y:B11G1
        batch, channel, frames, frequency = input_list[-1].size()
        x_list = []
        y_list = []
        for i in range(len(input_list)):
            input = self.avg_pool(input_list[i])
            y = self.conv1(input)
            x = input_list[i].unsqueeze(-1)
            y = y.unsqueeze(-2)
            x_list.append(x)
            y_list.append(y)

        # x_merge = torch.cat((x_list[0],x_list[1], x_list[2], x_list[3]), dim=-1)
        # y_merge = torch.cat((y_list[0],y_list[1], y_list[2], y_list[3]), dim=-2)

        x_merge = torch.cat((x_list[0], x_list[1]), dim=-1)
        y_merge = torch.cat((y_list[0], y_list[1]), dim=-2)

        y_softmax = self.softmax(y_merge)
        aham = torch.matmul(x_merge, y_softmax)
        aham = aham.view(batch, channel, frames, frequency)
        aham_output = input_list[-1] + aham
        # print(str(aham_output.shape))
        return aham_output


# x = torch.rand(4, 64, 100, 161)
# # model2 = AHAM(64)
# model = AIA_Transformer_merge(128, 64, num_layers=4)
# model2 = AHAM(64)
# output_mag, output_mag_list, output_ri, output_ri_list = model(x, x)
# aham = model2(output_mag_list)
# print(str(aham.shape))
