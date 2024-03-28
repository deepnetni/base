import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from utils.Engine import Engine
from models.AE import AE
from utils.ini_opts import read_ini
from utils.trunk import NSTrunk
from utils.record import REC
from models.conv_stft import STFT
from tqdm import tqdm
from typing import Dict
from models.APC_SNR.apc_snr import APC_SNR_multi_filter
from models.DPCRN import DPCRN_Model_new

# from models.pase.models.frontend import wf_builder


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

        self.stft = STFT(nframe=480, nhop=160, center=False, nfft=480).to(self.device)

        self.APC_criterion = APC_SNR_multi_filter(
            model_hop=128,
            model_winlen=512,
            mag_bins=256,
            theta=0.01,
            hops=[8, 16, 32, 64],
        ).to(self.device)

        # self.pase = wf_builder("config/frontend/PASE+.cfg").eval()
        # self.pase.load_pretrained(
        #     "./pretrained/pase_e199.ckpt", load_last=True, verbose=True
        # )
        # self.pase.to("cuda")

    @staticmethod
    def _config_optimizer(name: str, params, **kwargs):
        return super(Train, Train)._config_optimizer(
            name, filter(lambda p: p.requires_grad, params, **kwargs)
        )

    def loss_fn(self, clean: Tensor, enh: Tensor) -> Dict:
        # apc loss
        loss_APC_SNR, loss_pmsqe = self.APC_criterion(enh + 1e-8, clean + 1e-8)

        # pase loss
        # clean = clean.unsqueeze(1)
        # enh = enh.unsqueeze(1)
        # clean_pase = self.pase(clean)
        # clean_pase = clean_pase.flatten(1)
        # enh_pase = self.pase(enh)
        # enh_pase = enh_pase.flatten(1)
        # loss_pase = F.mse_loss(clean_pase, enh_pase)

        # loss final
        loss = 0.05 * loss_APC_SNR + loss_pmsqe  # + loss_pase

        return {
            "loss": loss,
            "apc_snr": loss_APC_SNR.detach(),
            "pmsqe": loss_pmsqe.detach(),
            # "pase": loss_pase.detach(),
        }

    def _fit_each_epoch(self, epoch):
        losses_rec = REC()

        pbar = tqdm(
            self.train_loader,
            # ncols=120,
            leave=True,
            desc=f"Epoch-{epoch}/{self.epochs}",
        )
        for noisy, clean in pbar:
            noisy = noisy.to(self.device)  # b,c,t,f
            clean = clean.to(self.device)  # b,c,t,f
            xk = self.stft.transform(noisy)

            self.optimizer.zero_grad()
            out = self.net(xk)
            enh = self.stft.inverse(out)
            loss_dict = self.loss_fn(clean[:, : enh.shape[-1]], enh)

            loss = loss_dict["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()

            # record the loss
            losses_rec.update(loss_dict)
            pbar.set_postfix(**losses_rec.state_dict())

        return losses_rec.state_dict()

    def valid_fn(self, clean: Tensor, enh: Tensor) -> Dict:
        sisnr = self._si_snr(clean, enh)
        sisnr = np.mean(sisnr)

        pesq = self._pesq(
            clean.cpu().detach().numpy(),
            enh.cpu().detach().numpy(),
            fs=16000,
            norm=False,
        )
        pesq = np.mean(pesq)
        # composite = self._eval(clean, enh, 16000)
        # composite = {k: np.mean(v) for k, v in composite.items()}
        # pesq = composite.pop("pesq")

        stoi = self._stoi(
            clean.cpu().detach().numpy(),
            enh.cpu().detach().numpy(),
            fs=16000,
        )
        stoi = np.mean(stoi)

        score = (pesq - 1) / 3.5 + stoi

        state = {"score": score, "pesq": pesq, "stoi": stoi, "sisnr": sisnr}

        # return dict(state, **composite)
        return dict(state)

    def _valid_each_epoch(self, epoch):
        metric_rec = REC()

        pbar = tqdm(
            self.valid_loader,
            # ncols=120,
            leave=False,
            desc=f"Valid-{epoch}/{self.epochs}",
        )

        draw = True

        for noisy, clean in pbar:
            noisy = noisy.to(self.device)  # b,c,t,f
            clean = clean.to(self.device)  # b,c,t,f
            xk = self.stft.transform(noisy)
            clean_xk = self.stft.transform(clean)

            out = self.net(xk)
            enh = self.stft.inverse(out)

            metric_dict = self.valid_fn(clean[:, : enh.shape[-1]], enh)
            if draw is True:
                self._draw_spectrogram(
                    epoch, xk, out, clean_xk, titles=("noisy", "enh", "clean"), fs=16000
                )
                draw = False

            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())

        return metric_rec.state_dict()


if __name__ == "__main__":
    cfg = read_ini("config/config.ini")

    # net = DPCRN_Model_new()

    net = DPCRN_Model_new(use_ae=False)

    init = cfg["config"]
    eng = Train(
        NSTrunk(
            cfg["dataset"]["train_dset"],
            "**/*_nearend.wav",
            keymap=("nearend.wav", "target.wav"),
        ),
        NSTrunk(
            cfg["dataset"]["valid_dset"],
            "**/*_nearend.wav",
            keymap=("nearend.wav", "target.wav"),
        ),
        net=net,
        batch_sz=6,
        valid_first=True,
        **init,
    )
    print(eng)
    eng.fit()
