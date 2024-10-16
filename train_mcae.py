import argparse
import os
from typing import Dict, List, Optional, Union
from einops import rearrange

import numpy as np
import torch
from matplotlib import shutil
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# from utils.conv_stft_loss import MultiResolutionSTFTLoss
from tqdm import tqdm

from utils.audiolib import audiowrite
from utils.Engine import Engine
from utils.ini_opts import read_ini
from utils.losses import loss_compressed_mag, loss_pmsqe, loss_sisnr
from utils.record import REC, RECDepot
from utils.stft_loss import MultiResolutionSTFTLoss
from utils.trunk import CHiMe3, pad_to_longest
from models.conv_stft import STFT
from models.MCAE import MCAE, MCAE_3, MCAE_F


class Train(Engine):
    def __init__(
        self,
        train_dset: Dataset,
        valid_dset: Dataset,
        vtest_dset: Dataset,
        batch_sz: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_loader = DataLoader(
            train_dset,
            batch_size=batch_sz,
            num_workers=6,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
        )
        # g = torch.Generator()
        # g.manual_seed(0)
        self.valid_loader = DataLoader(
            valid_dset,
            # batch_size=batch_sz,
            batch_size=1,
            num_workers=6,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
            collate_fn=pad_to_longest,
            # generator=g,
        )
        self.valid_dset = valid_dset

        self.vtest_loader = DataLoader(
            vtest_dset,
            # batch_size=batch_sz,
            batch_size=1,
            num_workers=6,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
            collate_fn=pad_to_longest,
            # generator=g,
        )
        self.vtest_dset = vtest_dset

        # self.vtest_loader = (
        #     [vtest_dset] if isinstance(vtest_dset, Dataset) else vtest_dset
        # )

        self.stft = STFT(nframe=512, nhop=256).to(self.device)
        self.stft.eval()

        self.ms_stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 512, 256],
            hop_sizes=[512, 256, 128],
            win_lengths=[1024, 512, 256],
        ).to(self.device)
        self.ms_stft_loss.eval()

    @staticmethod
    def _config_optimizer(name: str, params, **kwargs):
        return super(Train, Train)._config_optimizer(
            name, filter(lambda p: p.requires_grad, params, **kwargs)
        )

    def loss_fn(self, c, x, nlen=None) -> Dict:
        """
        clean: B,T,M
        """
        # apc loss
        # loss_APC_SNR, loss_pmsqe = self.APC_criterion(enh + 1e-8, clean + 1e-8)
        # loss = 0.05 * loss_APC_SNR + loss_pmsqe  # + loss_pase
        # return {
        #     "loss": loss,
        #     "pmsqe": loss_pmsqe.detach(),
        #     "apc_snr": loss_APC_SNR.detach(),
        # }
        # sisnr_lv = loss_sisnr(clean, enh)
        # specs_enh = self.stft.transform(enh)
        # specs_sph = self.stft.transform(clean)
        # mse_mag, mse_pha = loss_compressed_mag(specs_sph, specs_enh)
        # pmsqe_score = loss_pmsqe(specs_sph, specs_enh)
        # loss = 0.05 * sisnr_lv + mse_pha + mse_mag + pmsq_score
        # return {
        #     "loss": loss,
        #     "sisnr": sisnr_lv.detach(),
        #     "mag": mse_mag.detach(),
        #     "pha": mse_pha.detach(),
        #     "pmsq": pmsq_score.detach(),
        # }

        if nlen is None:
            loss, cor = self.net.loss(c, x)
        else:
            c_ = c[0, : nlen[0], ...].unsqueeze(0)
            x_ = x[0, : nlen[0], ...].unsqueeze(0)
            c_loss, cor = self.net.loss(c_, x_)
            for idx, n in enumerate(nlen[1:], start=1):
                c_ = c[idx, :n, ...].unsqueeze(0)
                x_ = x[idx, :n, ...].unsqueeze(0)
                tmp, tmp2 = self.net.loss(c_, x_)
                c_loss = c_loss + tmp
                cor += tmp2
            loss = c_loss / len(nlen)  # + 0.05 * pmsqe_score
            cor /= len(nlen)

        return {
            "loss": loss,
            "cor": cor,
        }

        # pase loss
        # clean = clean.unsqueeze(1)
        # enh = enh.unsqueeze(1)
        # clean_pase = self.pase(clean)
        # clean_pase = clean_pase.flatten(1)
        # enh_pase = self.pase(enh)
        # enh_pase = enh_pase.flatten(1)
        # loss_pase = F.mse_loss(clean_pase, enh_pase)

    def _fit_each_epoch(self, epoch):
        losses_rec = REC()

        pbar = tqdm(
            self.train_loader,
            ncols=120,
            leave=True,
            desc=f"Epoch-{epoch}/{self.epochs}",
        )
        for mic, _ in pbar:
            mic = mic.to(self.device)  # B,T,6

            self.optimizer.zero_grad()
            c, x = self.net(mic)
            loss_dict = self.loss_fn(c, x)

            loss = loss_dict["loss"]
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()

            # record the loss
            losses_rec.update(loss_dict)
            pbar.set_postfix(**losses_rec.state_dict())

        return losses_rec.state_dict()

    def valid_fn(self, c, x, nlen_list) -> Dict:
        """
        B,T,M
        """

        loss_dict = self.loss_fn(c, x, nlen_list)
        state = {"score": -loss_dict["loss"]}

        # return dict(state, **composite)
        return dict(state, **loss_dict)

    def _valid_each_epoch(self, epoch):
        metric_rec = REC()

        pbar = tqdm(
            self.valid_loader,
            ncols=120,
            leave=True,
            desc=f"Valid-{epoch}/{self.epochs}",
        )

        draw = True

        for mic, sph, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            # sph = sph.to(self.device)  # b,c,t,f
            nlen = nlen.to(self.device)  # B
            nlen = self.stft.nLen(nlen)

            with torch.no_grad():
                c, x = self.net(mic)  # B,T,M

            metric_dict = self.valid_fn(c, x, nlen)

            # if draw is True:
            #     with torch.no_grad():
            #         exk = self.stft.transform(enh[..., 0])
            #         sxk = self.stft.transform(lbl[..., 0])
            #     self._draw_spectrogram(epoch, exk, sxk, titles=("enh-c0", "sph-c0"))
            #     draw = False

            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())

        return metric_rec.state_dict()

    def vtest_fn(self, sph: Tensor, enh: Tensor) -> Dict:
        sisnr = self._si_snr(sph, enh)
        sisnr = np.mean(sisnr)

        pesq = self._pesq(
            sph.cpu().detach().numpy(),
            enh.cpu().detach().numpy(),
            fs=16000,
            norm=False,
        )
        pesq = np.mean(pesq)
        # composite = self._eval(clean, enh, 16000)
        # composite = {k: np.mean(v) for k, v in composite.items()}
        # pesq = composite.pop("pesq")

        stoi = self._stoi(
            sph.cpu().detach().numpy(),
            enh.cpu().detach().numpy(),
            fs=16000,
        )
        stoi = np.mean(stoi)

        state = {"pesq": pesq, "stoi": stoi, "sisnr": sisnr}

        # return dict(state, **composite)
        return state

    def vtest_aecmos(self, src_dir, out_dir):
        import json

        out = os.popen(f"python scripts/aecmos.py --src {src_dir}  --est {out_dir}")
        return json.loads(out.read())

    def _vtest_each_epoch(self, epoch):
        vtest_metric = {}

        metric_rec = RECDepot()
        use_aecmos = False
        vtest_loader = self.vtest_loader

        dirname = os.path.split(self.vtest_dset.dirname)[-1]
        pbar = tqdm(
            vtest_loader,
            total=len(vtest_loader),
            ncols=120,
            leave=False,
            desc=f"vTest-{epoch}/{dirname}",
        )
        vtest_outdir = os.path.join(self.vtest_outdir, dirname)
        shutil.rmtree(vtest_outdir) if os.path.exists(vtest_outdir) else None

        for mic, sph, nlen in pbar:
            mic = mic.to(self.device)  # b,c,t,f
            # sph = sph.to(self.device)  # b,c,t,f
            nlen = nlen.to(self.device)  # B
            nlen = self.stft.nLen(nlen)

            item = os.path.split(os.path.dirname(fname))[-1]

            with torch.no_grad():
                enh = self.net(mic, ref)

            audiowrite(
                os.path.join(self.vtest_outdir, fname),
                enh.cpu().squeeze().numpy(),
                16000,
            )

            if sph is None:
                use_aecmos = True
                continue

            metric_dict = self.vtest_fn(sph[:, : enh.shape[-1]], enh)
            # record the loss
            metric_rec.update(item, metric_dict)
            # pbar.set_postfix(**metric_rec.state_dict())

        if use_aecmos:
            metric_dict = self.vtest_aecmos(vtest_loader.dirname, vtest_outdir)
            vtest_metric[dirname] = metric_dict
        else:
            vtest_metric[dirname] = metric_rec.state_dict()

        return vtest_metric

    def _net_flops(self) -> int:
        from thop import profile
        import copy

        x = torch.randn(1, 16000, 6)
        flops, _ = profile(
            copy.deepcopy(self.net).cpu(),
            inputs=(x,),
            verbose=False,
        )
        return flops


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crn", help="crn aec model", action="store_true")
    parser.add_argument("--wo-sfp", help="without SFP path mode", action="store_true")
    parser.add_argument(
        "--test", help="fusion with the atten method", action="store_true"
    )
    parser.add_argument("-T", "--train", help="train mode", action="store_true")

    parser.add_argument("--ckpt", help="ckpt path", type=str)
    parser.add_argument("--src", help="input directory", type=str)
    parser.add_argument("--dst", help="predicting output directory", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    cfg_fname = "config/config_mcae.ini"
    # cfg_fname = "config/config_mcae_win.ini"
    cfg = read_ini(cfg_fname)
    net = MCAE_3(nframe=512, nhop=256, nfft=512, in_channels=6, mid_channel=24)
    # net = MCAE_F(nframe=512, nhop=256, nfft=512, in_channels=6, mid_channel=16)

    print("##", cfg_fname)

    # vtests = list(
    #     map(lambda x: x.strip("\n"), cfg["dataset"]["vtest_dset"].strip(",").split(","))
    # )

    train_dset = CHiMe3(
        cfg["dataset"]["train_dset"], subdir="train", nlen=2.5, min_len=1.0
    )
    valid_dset = CHiMe3(cfg["dataset"]["valid_dset"], subdir="dev")
    test_dsets = CHiMe3(cfg["dataset"]["vtest_dset"], subdir="test")

    init = cfg["config"]
    eng = Train(
        train_dset,
        valid_dset,
        test_dsets,
        net=net,
        batch_sz=4,
        # valid_first=True,
        valid_first=False,
        **init,
    )
    print(eng)
    if args.test:
        eng.test()
    else:
        eng.fit()
