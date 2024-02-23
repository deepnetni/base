import torch

from utils.Engine import Engine
from utils.losses import loss_compressed_mag, loss_sisnr
from models.AE import AE
from utils.ini_opts import read_ini
from utils.trunk import NSTrunk
from torch.utils.data import DataLoader
from models.conv_stft import STFT
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

        self.stft = STFT(nframe=512, nhop=256).to(self.device)

    def loss_fn(self, xk, xk_est) -> Dict:
        mse_mag, mse_specs = loss_compressed_mag(xk, xk_est)

        wav = self.stft.inverse(xk)
        wav_est = self.stft.inverse(xk_est)
        sisnr = loss_sisnr(wav, wav_est)
        # snr = loss_snr(wav, wav_est)
        loss = 0.03 * sisnr + mse_mag + mse_specs

        return {
            "loss": loss,
            "si_snr": sisnr.detach(),
            "mse_mag_loss": mse_mag.detach(),
            "mse_specs_loss": mse_specs.detach(),
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
            xk = self.stft.transform(noisy)  # b,2,t,f

            self.optimizer.zero_grad()
            out = self.net(xk)

            gen, xk = out
            loss_dict = self.loss_fn(xk, gen)
            loss = loss_dict["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()

            # record the loss
            losses_rec.update(loss_dict)
            pbar.set_postfix(**losses_rec.state_dict())

        return losses_rec.state_dict()

    def valid_fn(self, xk, xk_) -> Dict:
        wav = self.stft.inverse(xk)
        wav_ = self.stft.inverse(xk_)
        sisnr = -loss_sisnr(wav, wav_)
        score = sisnr

        state = {
            "score": score.detach(),
        }
        return state

    def _valid_each_epoch(self, epoch):
        metric_rec = REC()

        pbar = tqdm(
            self.valid_loader,
            # ncols=120,
            leave=False,
            desc=f"Valid-{epoch}/{self.epochs}",
        )

        draw = True

        for noisy in pbar:
            noisy = noisy.to(self.device)  # b,c,t,f
            xk = self.stft.transform(noisy)

            out = self.net(xk)

            gen, xk = out

            metric_dict = self.valid_fn(xk, gen)
            if draw is True:
                self._draw_spectrogram(
                    epoch, xk, gen, titles=("inp", "reconst"), fs=16000
                )
                draw = False

            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())

        return metric_rec.state_dict()


if __name__ == "__main__":
    cfg = read_ini("config/config_ae.ini")

    net = AE(in_features=257, latent_dim=64)
    init = cfg["config"]
    eng = Train(
        NSTrunk(
            cfg["dataset"]["train_dset"],
            "**/*_nearend.wav",
        ),
        NSTrunk(
            cfg["dataset"]["valid_dset"],
            "**/*.wav",
        ),
        batch_sz=64,
        net=net,
        **init,
    )
    print(eng)
    eng.fit()
