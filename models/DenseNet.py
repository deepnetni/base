from typing import Tuple
import torch
import torch.nn as nn


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
        self.in_channels = in_channels
        kh, kw = (*kernel_size,)
        self.pad = nn.ConstantPad2d(
            ((kw - 1) // 2, (kw - 1) // 2, kh - 1, 0), value=0.0
        )  # lrud

        convs = []
        for i in range(depth):
            convs.append(
                nn.Sequential(
                    self.pad,
                    nn.Conv2d(
                        in_channels=in_channels * (i + 1),
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                    ),
                    nn.BatchNorm2d(in_channels, affine=True),
                    nn.PReLU(),
                )
            )

        self.layers = nn.ModuleList(convs)

    def forward(self, x):
        skip = x
        for l in self.layers:
            x = l(skip)
            skip = torch.concat([skip, x], dim=1)

        return x


if __name__ == "__main__":
    inp = torch.randn(1, 2, 10, 5)
    net = DenseNet(4, 2, (3, 5))
    out = net(inp)
    print(out.shape)
