from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


class MS_CAM_F(nn.Module):
    """
    input: B,C,T,F

    Arguments
    ---------
    feature_size, the `F` size of B,C,T,F
    """

    def __init__(self, inp_channels: int, feature_size: int, r=2) -> None:
        super().__init__()
        assert feature_size % r == 0, f"{feature_size}%{r}"

        self.layer_global = nn.Sequential(
            Rearrange("b c t f->b f t c"),
            nn.AvgPool2d(
                kernel_size=(1, inp_channels), stride=(1, inp_channels)
            ),  # B,F,T,1
            nn.Conv2d(
                in_channels=feature_size,
                out_channels=feature_size // r,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=feature_size,
            ),  # b,t,f,1
            nn.LayerNorm(1, feature_size),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=feature_size // r,
                out_channels=feature_size,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=feature_size // r,
            ),
            Rearrange("b f t c->b c t f"),
        )

        self.layer_local = nn.Sequential(
            Rearrange("b c t f->b f t c"),
            nn.Conv2d(
                in_channels=feature_size,
                out_channels=feature_size // r,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=feature_size,
            ),
            nn.LayerNorm(inp_channels, feature_size),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=feature_size // r,
                out_channels=feature_size,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=feature_size // r,
            ),
            Rearrange("b f t c->b c t f"),
            # Rearrange("b c t f-> b t c f"),
            # nn.LayerNorm(inp_channels, 1)
            # Rearrange("b t c f-> b c t f"),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_ = x + y
        x_ = self.layer_global(x_) + self.layer_local(x_)
        w = x_.sigmoid()
        return x * w + (1 - w) * y


class MS_CAM(nn.Module):
    """
    input: B,C,T,F

    Arguments
    ---------
    feature_size, the `F` size of B,C,T,F
    """

    def __init__(self, inp_channels: int, feature_size: int) -> None:
        super().__init__()

        self.layer_global = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=(1, feature_size), stride=(1, feature_size)
            ),  # B,C,T,1
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            Rearrange("b c t f-> b t c f"),
            nn.LayerNorm(1, inp_channels),
            Rearrange("b t c f-> b c t f"),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            # Rearrange("b c t f-> b t c f"),
            # nn.LayerNorm(inp_channels, 1)
            # Rearrange("b t c f-> b c t f"),
        )

        self.layer_local = nn.Sequential(
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            Rearrange("b c t f-> b t c f"),
            nn.LayerNorm(feature_size, inp_channels),
            Rearrange("b t c f-> b c t f"),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            # Rearrange("b c t f-> b t c f"),
            # nn.LayerNorm(inp_channels, 1)
            # Rearrange("b t c f-> b c t f"),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_ = self.layer_global(x) + self.layer_local(x)
        return y * x_.sigmoid()


class MS_CAM_BLK(nn.Module):
    """
    input: B,C,T,F

    Arguments
    ---------
    feature_size, the `F` size of B,C,T,F
    """

    def __init__(self, inp_channels: int, feature_size: int) -> None:
        super().__init__()

        self.layer_global = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=(1, feature_size), stride=(1, feature_size)
            ),  # B,C,T,1
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            Rearrange("b c t f-> b t c f"),
            nn.LayerNorm(1, inp_channels),
            Rearrange("b t c f-> b c t f"),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            # Rearrange("b c t f-> b t c f"),
            # nn.LayerNorm(inp_channels, 1)
            # Rearrange("b t c f-> b c t f"),
        )

        self.layer_local = nn.Sequential(
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            Rearrange("b c t f-> b t c f"),
            nn.LayerNorm(feature_size, inp_channels),
            Rearrange("b t c f-> b c t f"),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            # Rearrange("b c t f-> b t c f"),
            # nn.LayerNorm(inp_channels, 1)
            # Rearrange("b t c f-> b c t f"),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_ = x + y
        x_ = self.layer_global(x_) + self.layer_local(x_)
        return x_.sigmoid()


class AFF(nn.Module):
    """
    input: B,C,T,F

    Arguments
    ---------
    feature_size, the `F` size of B,C,T,F
    """

    def __init__(self, inp_channels: int, feature_size: int) -> None:
        super().__init__()
        self.layer = MS_CAM_BLK(inp_channels, feature_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        w = self.layer(x, y)
        return x * w + (1.0 - w) * y


class MS_SELF_CAM(nn.Module):
    """
    input: B,C,T,F

    Arguments
    ---------
    feature_size, the `F` size of B,C,T,F
    """

    def __init__(self, inp_channels: int, feature_size: int) -> None:
        super().__init__()

        self.layer_global = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=(1, feature_size), stride=(1, feature_size)
            ),  # B,C,T,1
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            Rearrange("b c t f-> b t c f"),
            nn.LayerNorm(1, inp_channels),
            Rearrange("b t c f-> b c t f"),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            # Rearrange("b c t f-> b t c f"),
            # nn.LayerNorm(inp_channels, 1)
            # Rearrange("b t c f-> b c t f"),
        )

        self.layer_local = nn.Sequential(
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            Rearrange("b c t f-> b t c f"),
            nn.LayerNorm(feature_size, inp_channels),
            Rearrange("b t c f-> b c t f"),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            # Rearrange("b c t f-> b t c f"),
            # nn.LayerNorm(inp_channels, 1)
            # Rearrange("b t c f-> b c t f"),
        )

    def forward(self, x: torch.Tensor):
        x_ = self.layer_global(x) + self.layer_local(x)
        return x * x_.sigmoid()


if __name__ == "__main__":
    l = MS_CAM(inp_channels=2, feature_size=251)
    inp = torch.randn(1, 2, 3, 251)
    out = l(inp, inp)
    print(out.shape)
