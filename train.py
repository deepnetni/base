import torch

from utils.Engine import Engine
from utils.losses import loss_compressed_mag, loss_sisnr
from models.VAE import VAE
from utils.ini_opts import read_ini
from utils.trunk import NSTrunk
from torch.utils.data import DataLoader
from models.STFT import STFT
from utils.record import REC
from tqdm import tqdm
from typing import Dict


class Train(Engine):
    def __init__(self, train_dset, valid_dset, batch_sz, **kwargs):
        super().__init__(**kwargs)
        self.train_loader = DataLoader(
            train_dset,
            batch_size=batch_sz,
            num_workers=6,
            pin_memory=True,
            shuffle=True,
        )
        self.valid_loader = DataLoader(
            valid_dset,
            batch_size=batch_sz,
            num_workers=6,
            pin_memory=True,
            shuffle=True,
        )

        self.stft = STFT(filter_length=512, hop_length=320).to(self.device)

    def loss_fn(self, xk, xk_est, mu, logvar) -> Dict:
        mse_mag, mse_specs = loss_compressed_mag(xk, xk_est)
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=-1)
        )
        wav = self.stft.inverse(xk)
        wav_est = self.stft.inverse(xk_est)
        sisnr = loss_sisnr(wav, wav_est)
        loss = 0.03 * sisnr + mse_mag + mse_specs + kl_loss

        return {
            "loss": loss,
            "sisnr": 0.03 * sisnr.detach(),
            "mse_mag_loss": mse_mag.detach(),
            "mse_specs_loss": mse_specs.detach(),
            "kld": kl_loss.detach(),
        }

    def _fit_each_epoch(self, epoch):
        losses_rec = REC()

        pbar = tqdm(
            self.train_loader,
            # ncols=120,
            leave=True,
            desc=f"Epoch-{epoch}/{self.epochs}",
        )
        for noisy in pbar:
            noisy = noisy.to(self.device)  # b,c,t,f
            xk = self.stft.transform(noisy)
            self.optimizer.zero_grad()
            out = self.net(xk)

            gen, xk, mu, logvar = out
            loss_dict = self.loss_fn(xk, gen, mu, logvar)
            loss = loss_dict["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()

            # record the loss
            losses_rec.update(loss_dict)
            pbar.set_postfix(**losses_rec.state_dict())

        return losses_rec.state_dict()

    def valid_fn(self, xk, xk_, mu, logvar) -> Dict:
        wav = self.stft.inverse(xk)
        wav_ = self.stft.inverse(xk_)
        sisnr = -loss_sisnr(wav, wav_)
        mu_mean = torch.mean(mu)
        var_mean = torch.mean(logvar.exp())
        score = sisnr

        state = {
            "score": score.detach(),
            "sisnr": sisnr.detach(),
            "mu": mu_mean.detach(),
            "var": var_mean.detach(),
        }
        return state

    def _valid_each_epoch(self, epoch):
        metric_rec = REC()

        pbar = tqdm(
            self.valid_loader,
            # ncols=120,
            leave=False,
            desc=f"Epoch-{epoch}/{self.epochs}",
        )
        for noisy in pbar:
            noisy = noisy.to(self.device)  # b,c,t,f
            xk = self.stft.transform(noisy)

            out = self.net(xk)

            gen, xk, mu, logvar = out
            metric_dict = self.valid_fn(xk, gen, mu, logvar)

            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())

        return metric_rec.state_dict()


if __name__ == "__main__":
    cfg = read_ini("config.ini")

    net = VAE(in_features=257, latent_dim=64)
    init = cfg["config"]
    eng = Train(
        NSTrunk(
            r"\\192.168.110.31\dataset\vae_dns",
            "**/*_nearend.wav",
        ),
        NSTrunk(
            r"\\192.168.110.31\dataset\vae_val",
            "**/*.wav",
        ),
        net=net,
        batch_sz=64,
        **init,
    )
    print(eng)
    eng.fit()
