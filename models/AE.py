from typing import Tuple
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    """
    xk: shape b,c,t,f
    """

    def __init__(
        self,
        in_features: int = 257,
        latent_dim: int = 64,
        channels=[2, 16, 32, 64, 128],
        kernel_size=(1, 3),
        fstride: Tuple = (2, 2, 2, 1),
    ):
        super().__init__()

        # Build Encoder
        modules = []
        for cin, cout, step in zip(channels[:-1], channels[1:], fstride):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=cin,
                        out_channels=cout,
                        kernel_size=kernel_size,
                        stride=(1, step),
                        padding=(kernel_size[0] - 1, (kernel_size[1] - 1) // 2),
                    ),
                    nn.BatchNorm2d(cout),
                    nn.LeakyReLU(),
                )
            )

        self.fbin = (in_features >> fstride.count(2)) + 1

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(
            in_features=channels[-1] * self.fbin, out_features=latent_dim
        )

        # Build Decoder
        modules = []

        self.pre_process = nn.Linear(latent_dim, channels[-1] * self.fbin)
        channels = channels[::-1]

        for cin, cout, step in zip(channels[:-1], channels[1:-1], fstride[::-1]):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=cin,
                        out_channels=cout,
                        kernel_size=kernel_size,
                        stride=(1, step),
                        padding=(kernel_size[0] - 1, (kernel_size[1] - 1) // 2),
                    ),
                    nn.BatchNorm2d(cout),
                    nn.LeakyReLU(),
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.post_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=kernel_size,
                stride=(1, fstride[0]),
                padding=(kernel_size[0] - 1, (kernel_size[1] - 1) // 2),
            ),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(channels[-1], channels[-1], kernel_size=(1, 3), padding=(0, 1)),
            nn.Tanh(),
        )

    def generate(self, x):
        return self.forward(x)[0]

    def encode(self, x):
        """
        x, b,c,t,f
        """
        x = self.encoder(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        z = self.fc(x)

        return z

    def decode(self, z):
        x = self.pre_process(z)
        x = rearrange(x, "b t (c f) -> b c t f", f=self.fbin)
        x = self.decoder(x)
        x = self.post_layer(x)

        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, xk):
        z = self.encode(xk)
        return self.decode(z), xk

    def loss_fn(self, *args):
        recons, inp = args
        recons_loss = F.mse_loss(recons, inp)

        loss = recons_loss
        return {
            "loss": loss,
            "recons_loss": recons_loss.detach(),
        }


if __name__ == "__main__":
    net = AE(in_features=257, latent_dim=64)
    inp = torch.randn(3, 2, 10, 257)
    ret = net(inp)
    print(len(ret))
    l = net.loss_fn(*ret)
    print(l)
    out, xk = ret
    print(out.shape, xk.shape)

    out = net.generate(inp)
    print(out.shape)

    z = net.encode(inp)
    print(z.shape)
