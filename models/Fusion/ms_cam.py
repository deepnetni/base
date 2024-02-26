from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


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
