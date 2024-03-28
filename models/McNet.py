import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.conv_stft import STFT

from models.DenseNet import DenseNet
from models.MCAE import MCAE


class FuseBLK(nn.Module):
    """
    input
    -----
    B,C,T,F
    """

    def __init__(self, in_channels: int, depth: int = 4, dense_kernel: Tuple = (3, 5)):
        super().__init__()

        self.layers = nn.Sequential(
            DenseNet(depth=depth, in_channels=in_channels, kernel_size=dense_kernel)
        )

        pass

    def forward(self, x):
        pass


class RNN_FC(nn.Module):
    """
    feature_size: F of b,c,f,t
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = False,
        act: bool = True,
    ):
        super().__init__()
        # Sequence layer
        self.sequence_model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.sequence_model.flatten_parameters()

        # Fully connected layer

        if bidirectional:
            self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc_output_layer = nn.Linear(hidden_size, output_size)

        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, Feature]
        Returns:
            [B, T, Feature]
        """
        x, _ = self.sequence_model(x)
        x = self.act(self.fc_output_layer(x))

        return x


class MCNet_wED(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        ref_channel: int = 5,
        sub_freqs: Tuple = (3, 2),
        past_ahead: Tuple = (5, 0),
    ):
        super().__init__()
        self.stft = STFT(512, 256, 512)
        self.ed = MCAE(512, 256)
        self.ed.load_state_dict(
            torch.load(r"E:\github\base\trained_mcae\MCAE\checkpoints\epoch_0030.pth")
        )
        for p in self.ed.parameters():
            p.requires_grad = False

        self.ed_fc = nn.Sequential(
            Rearrange("b t h c -> b c t h"),
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                kernel_size=(1, 3),
                stride=1,
            ),
            Rearrange("b (c r) t h -> b c t (h r)", c=in_channels),
            nn.ConstantPad2d((1, 0, 0, 0), value=0.0),
            nn.LayerNorm(257),
            # nn.BatchNorm2d(num_features=in_channels),  # B,12,T,H
            # nn.Linear(128, 257),
            nn.Tanh(),  # B,2,T,F
            Rearrange("b c t h-> b t h c"),
        )

        # self.ed = RNN_FC(
        #     input_size=128,
        #     output_size=64,
        #     hidden_size=128,
        #     num_layers=2,
        #     bidirectional=True,
        #     act=True,
        # )

        self.freq = RNN_FC(
            input_size=in_channels * 3,
            output_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            act=True,
        )
        self.narr = nn.Sequential(
            RNN_FC(
                input_size=76,  # 64 + 12
                output_size=64,
                hidden_size=128,
                num_layers=2,
                bidirectional=False,
                act=True,
            ),
            # Rearrange("(b f) t c->b c t f", f=257),
            # nn.Conv2d(in_channels=,out_channels=64,kernel_size=1),
            # nn.BatchNorm2d(),
            # nn.PReLU(),
        )
        self.sub = nn.Sequential(
            RNN_FC(
                input_size=327,  # 1x(2x3+1)(7) + 64x(2x2+1)(320)
                output_size=64,
                hidden_size=128,
                num_layers=2,
                bidirectional=False,
                act=True,
            )
        )
        self.full = RNN_FC(
            input_size=70,  # 64 + past_ahead + 1
            output_size=2,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            act=False,
        )
        self.ref_channel_idx = (ref_channel - 1) * 2
        self.sub_freqs = sub_freqs
        self.past_ahead = past_ahead

    def forward(self, x):
        """
        x:B,T,M
        xk_inp: B,T,H,M
        """
        nB = x.size(0)
        x_ = rearrange(x, "b t m ->(b m) t")
        xk = self.stft.transform(x_)  # B,2,T,F
        # print(xk.shape)
        # m c, ((r,i), (r,i), ...)
        xk = rearrange(xk, "(b m) c t f->b t f (m c)", b=nB)
        xk_ref = xk[..., self.ref_channel_idx : self.ref_channel_idx + 2]
        mag_ref = torch.sqrt(torch.sum(xk_ref**2, dim=-1) + 1e-7).unsqueeze(
            -1
        )  # b t f 1

        z, _, _ = self.ed.encode(x)  # B,T,128,M
        z = self.ed_fc(z)  # B,T,F,C

        xk_ = torch.concat([xk, z], dim=-1)
        # freq part
        x_ = rearrange(xk_, "b t f c-> (b t) f c")
        x_ = self.freq(x_)
        x_ = rearrange(x_, "(b t) f c->b t f c", b=nB)

        # narr
        x_ = torch.concat([x_, xk], dim=-1)  # b,t,f,(64+12)
        x_ = rearrange(x_, "b t f c->(b f) t c")
        x_ = self.narr(x_)
        x_ = rearrange(x_, "(b f) t c->b t f c", b=nB)

        # sub
        x_ = rearrange(x_, "b t f c-> (b t) c f 1")  # BT,C,F,1
        x_ = torch.concat(
            [x_[..., -self.sub_freqs[1] :, :], x_, x_[..., : self.sub_freqs[1], :]],
            dim=2,
        )  # BT,C,F+2n,1
        # kernel_size is the number of sub-bands
        x_ = F.unfold(
            x_, kernel_size=(self.sub_freqs[1] * 2 + 1, 1)
        )  # BT,Cx(2n[1]+1),F

        mag_ = rearrange(mag_ref, "b t f c->(b t) c f 1")
        mag_ = torch.concat(
            [
                mag_[..., -self.sub_freqs[0] :, :],
                mag_,
                mag_[..., : self.sub_freqs[0], :],
            ],
            dim=2,
        )
        mag_ = F.unfold(
            mag_, kernel_size=(self.sub_freqs[0] * 2 + 1, 1)
        )  # B,1x(2n[0]+1),F
        x_ = torch.concat([x_, mag_], dim=1)
        x_ = rearrange(x_, "(b t) c f-> (b f) t c", b=nB)
        x_ = self.sub(x_)
        x_ = rearrange(x_, "(b f) t c->b t f c", b=nB)

        # full
        x_ = rearrange(x_, "b t f c->(b t) f c")
        mag_ = rearrange(mag_ref, "b t f c->b f c t")  # b f 1 t
        mag_ = F.pad(mag_, pad=self.past_ahead, mode="constant", value=0.0)
        mag_ = rearrange(mag_, "b f c t->(b f) 1 (c t) 1")
        mag_ = F.unfold(
            mag_, kernel_size=(sum(self.past_ahead) + 1, 1)
        )  # BF, 1xpost_ahead, T
        mag_ = rearrange(mag_, "(b f) c t-> (b t) f c", b=nB)
        x_ = torch.concat([x_, mag_], dim=-1)
        x_ = self.full(x_)
        x_ = rearrange(x_, "(b t) f c-> b c t f", b=nB)

        out = self.stft.inverse(x_)

        return out


class MCNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        ref_channel: int = 5,
        sub_freqs: Tuple = (3, 2),
        past_ahead: Tuple = (5, 0),
    ):
        super().__init__()
        self.stft = STFT(512, 256, 512)
        self.freq = RNN_FC(
            input_size=in_channels * 2,
            output_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            act=True,
        )
        self.narr = nn.Sequential(
            RNN_FC(
                input_size=76,  # 64 + 12
                output_size=64,
                hidden_size=128,
                num_layers=2,
                bidirectional=False,
                act=True,
            ),
            # Rearrange("(b f) t c->b c t f", f=257),
            # nn.Conv2d(in_channels=,out_channels=64,kernel_size=1),
            # nn.BatchNorm2d(),
            # nn.PReLU(),
        )
        self.sub = nn.Sequential(
            RNN_FC(
                input_size=327,  # 1x(2x3+1)(7) + 64x(2x2+1)(320)
                output_size=64,
                hidden_size=128,
                num_layers=2,
                bidirectional=False,
                act=True,
            )
        )
        self.full = RNN_FC(
            input_size=70,  # 64 + past_ahead + 1
            output_size=2,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            act=False,
        )
        self.ref_channel_idx = (ref_channel - 1) * 2
        self.sub_freqs = sub_freqs
        self.past_ahead = past_ahead

    def forward(self, x):
        """
        x:B,T,M
        """
        nB = x.size(0)
        x = rearrange(x, "b t m ->(b m) t")
        xk = self.stft.transform(x)  # B,2,T,F
        # print(xk.shape)
        # m c, ((r,i), (r,i), ...)
        xk = rearrange(xk, "(b m) c t f->b t f (m c)", b=nB)
        xk_ref = xk[..., self.ref_channel_idx : self.ref_channel_idx + 2]
        mag_ref = torch.sqrt(torch.sum(xk_ref**2, dim=-1) + 1e-7).unsqueeze(
            -1
        )  # b t f 1

        # freq part
        x_ = rearrange(xk, "b t f c-> (b t) f c")
        x_ = self.freq(x_)
        x_ = rearrange(x_, "(b t) f c->b t f c", b=nB)

        # narr
        x_ = torch.concat([x_, xk], dim=-1)  # b,t,f,(64+12)
        x_ = rearrange(x_, "b t f c->(b f) t c")
        x_ = self.narr(x_)
        x_ = rearrange(x_, "(b f) t c->b t f c", b=nB)

        # sub
        x_ = rearrange(x_, "b t f c-> (b t) c f 1")  # BT,C,F,1
        x_ = torch.concat(
            [x_[..., -self.sub_freqs[1] :, :], x_, x_[..., : self.sub_freqs[1], :]],
            dim=2,
        )  # BT,C,F+2n,1
        # kernel_size is the number of sub-bands
        x_ = F.unfold(
            x_, kernel_size=(self.sub_freqs[1] * 2 + 1, 1)
        )  # BT,Cx(2n[1]+1),F

        mag_ = rearrange(mag_ref, "b t f c->(b t) c f 1")
        mag_ = torch.concat(
            [
                mag_[..., -self.sub_freqs[0] :, :],
                mag_,
                mag_[..., : self.sub_freqs[0], :],
            ],
            dim=2,
        )
        mag_ = F.unfold(
            mag_, kernel_size=(self.sub_freqs[0] * 2 + 1, 1)
        )  # B,1x(2n[0]+1),F
        x_ = torch.concat([x_, mag_], dim=1)
        x_ = rearrange(x_, "(b t) c f-> (b f) t c", b=nB)
        x_ = self.sub(x_)
        x_ = rearrange(x_, "(b f) t c->b t f c", b=nB)

        # full
        x_ = rearrange(x_, "b t f c->(b t) f c")
        mag_ = rearrange(mag_ref, "b t f c->b f c t")  # b f 1 t
        mag_ = F.pad(mag_, pad=self.past_ahead, mode="constant", value=0.0)
        mag_ = rearrange(mag_, "b f c t->(b f) 1 (c t) 1")
        mag_ = F.unfold(
            mag_, kernel_size=(sum(self.past_ahead) + 1, 1)
        )  # BF, 1xpost_ahead, T
        mag_ = rearrange(mag_, "(b f) c t-> (b t) f c", b=nB)
        x_ = torch.concat([x_, mag_], dim=-1)
        x_ = self.full(x_)
        x_ = rearrange(x_, "(b t) f c-> b c t f", b=nB)

        out = self.stft.inverse(x_)

        return out


if __name__ == "__main__":
    from thop import profile

    inp = torch.randn(4, 16000 * 3, 6)
    net = MCNet_wED(6, 5, (3, 2), (5, 0))
    out = net(inp)
    print(out.shape)

    flops, parm = profile(net, inputs=(inp,))
    print(flops / 1e9, parm / 1e6)
