import argparse
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# from matplotlib import pyplot as plt
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# from utils.conv_stft_loss import MultiResolutionSTFTLoss
from tqdm import tqdm

from utils.Engine import Engine
from utils.ini_opts import read_ini
from utils.losses import loss_compressed_mag, loss_pmsqe, loss_sisnr
from utils.record import REC, RECDepot
from utils.stft_loss import MultiResolutionSTFTLoss
from utils.trunk import CHiMe3, pad_to_longest
from models.conv_stft import STFT
from models.MCAE import MCAE, MCAE_2, MCAE_3, MCAE_F
from models.aia_trans import (
    DF_AIA_TRANS,
    DF_AIA_TRANS_AE,
    DF_AIA_TRANS_AE_F,
    dual_aia_trans_chime,
    DF_AIA_TRANS_densepwc,
)
from models.McNet import MCNet, MCNet_wED, DenseMCNet, MCNetwDense
from torchmetrics.functional.audio import signal_distortion_ratio as SDR
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio as si_sdr,
)
from models.APC_SNR.apc_snr import APC_SNR_multi_filter
from models.pase.models.frontend import wf_builder


class Train(Engine):
    def __init__(
        self,
        train_dset: Dataset,
        valid_dset: Dataset,
        vtest_dset: Dataset,
        net_ae: torch.nn.Module,
        batch_sz: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.net_ae = net_ae.to(self.device)
        self.net_ae.eval()

        self.train_loader = DataLoader(
            train_dset,
            batch_size=batch_sz,
            num_workers=6,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
        )
        self.train_dset = train_dset
        # g = torch.Generator()
        # g.manual_seed(0)
        self.valid_loader = DataLoader(
            valid_dset,
            batch_size=2,
            # batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
            collate_fn=pad_to_longest,
            # generator=g,
        )
        self.valid_dset = valid_dset

        self.vtest_loader = DataLoader(
            vtest_dset,
            batch_size=2,
            # batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
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

        self.raw_metrics = self._load_dsets_metrics(self.dsets_mfile)
        self.pase = wf_builder("config/frontend/PASE+.cfg")
        self.pase.cuda()
        self.pase.eval()
        self.pase.load_pretrained(
            "pretrained/pase_e199.ckpt", load_last=True, verbose=False
        )

        # self.APC_criterion = APC_SNR_multi_filter(
        #     model_hop=128,
        #     model_winlen=512,
        #     mag_bins=256,
        #     theta=0.01,
        #     hops=[8, 16, 32, 64],
        # ).to(self.device)

    @staticmethod
    def _config_optimizer(name: str, params, **kwargs):
        return super(Train, Train)._config_optimizer(
            name, filter(lambda p: p.requires_grad, params, **kwargs)
        )

    def loss_fn(self, clean: Tensor, enh: Tensor, nlen=None) -> Dict:
        """
        clean: B,T
        """
        # apc loss
        # loss_APC_SNR, loss_pmsqe = self.APC_criterion(enh + 1e-8, clean + 1e-8)
        # loss = 0.05 * loss_APC_SNR + loss_pmsqe  # + loss_pase
        # return {
        #     "loss": loss,
        #     "pmsqe": loss_pmsqe.detach(),
        #     "apc_snr": 0.05 * loss_APC_SNR.detach(),
        # }
        # sisdr_lv = loss_sisnr(clean, enh)
        specs_enh = self.stft.transform(enh)
        specs_sph = self.stft.transform(clean)
        pmsqe_score = 0.3 * loss_pmsqe(specs_sph, specs_enh)
        # mse_mag, mse_pha = loss_compressed_mag(specs_sph, specs_enh)
        # loss = 0.05 * sisnr_lv + mse_pha + mse_mag + pmsq_score
        # return {
        #     "loss": loss,
        #     "sisnr": sisnr_lv.detach(),
        #     "mag": mse_mag.detach(),
        #     "pha": mse_pha.detach(),
        #     "pmsq": pmsq_score.detach(),
        # }
        # sdr_lv = -SDR(preds=enh, target=clean).mean()
        sc_loss, mag_loss = self.ms_stft_loss(enh, clean)
        # else:
        #     cln_ = clean[0, : nlen[0]]  # B,T
        #     enh_ = enh[0, : nlen[0]]
        #     sc_loss, mag_loss = self.ms_stft_loss(enh_, cln_)
        #     for idx, n in enumerate(nlen[1:], start=1):
        #         cln_ = clean[idx, :n]  # B,T
        #         enh_ = enh[idx, :n]
        #         sc_, mag_ = self.ms_stft_loss(enh_, cln_)
        #         sc_loss = sc_loss + sc_
        #         mag_loss = mag_loss + mag_

        # * pase loss
        # clean = clean.unsqueeze(1)  # B,1,T
        # enh = enh.unsqueeze(1)
        # clean_pase = self.pase(clean)
        # clean_pase = clean_pase.flatten(0)
        # enh_pase = self.pase(enh)
        # enh_pase = enh_pase.flatten(0)
        # loss_pase = F.mse_loss(clean_pase, enh_pase)

        loss = sc_loss + mag_loss + pmsqe_score  # + loss_pase  # + 0.05 * sdr_lv

        return {
            "loss": loss,
            "sc": sc_loss.detach(),
            "mag": mag_loss.detach(),
            "pmsqe": pmsqe_score.detach(),
            # "pase": loss_pase.detach(),
            # "sdr": 0.05 * sdr_lv.detach(),
        }

    def _fit_each_epoch(self, epoch):
        losses_rec = REC()

        pbar = tqdm(
            self.train_loader,
            ncols=160,
            leave=True,
            desc=f"Epoch-{epoch}/{self.epochs}",
        )
        for mic, sph in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # B,T
            # print("##", mic.shape)

            self.optimizer.zero_grad()
            c, _ = self.net_ae(mic)  # B,C,T,6
            enh = self.net(mic, c.detach())
            loss_dict = self.loss_fn(sph[:, : enh.size(-1)], enh)

            loss = loss_dict["loss"]
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()

            # record the loss
            losses_rec.update(loss_dict)
            pbar.set_postfix(**losses_rec.state_dict())

        return losses_rec.state_dict()

    def _valid_dsets(self):
        dset_dict = {}
        # -----------------------#
        ##### valid dataset  #####
        # -----------------------#
        metric_rec = REC()
        pbar = tqdm(
            self.valid_loader,
            ncols=120,
            leave=False,
            desc=f"v-{self.valid_dset.dirname}",
        )

        for mic, sph, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)  # B,

            metric_dict = self.valid_fn(sph, mic[..., 4], nlen, return_loss=False)
            metric_dict.pop("score")
            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())

        dset_dict["valid"] = metric_rec.state_dict()

        # -----------------------#
        ##### vtest dataset ######
        # -----------------------#
        metric_rec = REC()
        pbar = tqdm(
            self.vtest_loader,
            ncols=120,
            leave=False,
            desc=f"v-{self.vtest_dset.dirname}",
        )

        for mic, sph, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)  # B,

            metric_dict = self.valid_fn(sph, mic[..., 4], nlen, return_loss=False)
            metric_dict.pop("score")

            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())

        dset_dict["vtest"] = metric_rec.state_dict()

        return dset_dict

    def valid_fn(
        self, sph: Tensor, enh: Tensor, nlen_list: List, return_loss: bool = True
    ) -> Dict:
        """
        B,T
        """
        # sisnr_l = []
        sdr_l = []

        B = sph.size(0)
        sph_ = sph[0, : nlen_list[0]]  # B,T
        enh_ = enh[0, : nlen_list[0]]
        # sisnr_l.append(self._si_snr(sph_, enh_))
        np_l_sph = [sph_.cpu().numpy()]
        np_l_enh = [enh_.cpu().numpy()]
        sdr_l.append(SDR(preds=enh_, target=sph_).cpu().numpy())

        for i in range(1, B):
            sph_ = sph[i, : nlen_list[i]]  # B,T
            enh_ = enh[i, : nlen_list[i]]
            np_l_sph.append(sph_.cpu().numpy())
            np_l_enh.append(enh_.cpu().numpy())

            # sisnr_l.append(self._si_snr(sph_, enh_))
            sdr_l.append(SDR(preds=enh_, target=sph_).cpu().numpy())

        # sisnr_sc = np.array(sisnr_l).mean()
        sdr_sc = np.array(sdr_l).mean()
        pesq_wb_sc = self._pesq(np_l_sph, np_l_enh, fs=16000).mean()
        pesq_nb_sc = self._pesq(np_l_sph, np_l_enh, fs=16000, mode="nb").mean()
        stoi_sc = self._stoi(np_l_sph, np_l_enh, fs=16000).mean()

        # composite = self._eval(clean, enh, 16000)
        # composite = {k: np.mean(v) for k, v in composite.items()}
        # pesq = composite.pop("pesq")

        state = {
            "score": pesq_wb_sc,
            # "sisnr": sisnr_sc,
            "sdr": sdr_sc,
            "pesq": pesq_wb_sc,
            "pesq_nb": pesq_nb_sc,
            "stoi": stoi_sc,
        }

        if return_loss:
            loss_dict = self.loss_fn(sph[..., : enh.size(-1)], enh, nlen_list)
        else:
            loss_dict = {}

        # return dict(state, **composite)
        return dict(state, **loss_dict)

    def _valid_each_epoch(self, epoch):
        metric_rec = REC()

        pbar = tqdm(
            self.valid_loader,
            ncols=160,
            leave=True,
            desc=f"Valid-{epoch}/{self.epochs}",
        )

        draw = False

        for mic, sph, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)
            # nlen = nlen.to(self.device)  # B

            with torch.no_grad():
                c, _ = self.net_ae(mic)
                enh = self.net(mic, c)

            metric_dict = self.valid_fn(sph, enh, nlen)

            if draw is True:
                with torch.no_grad():
                    sxk = self.stft.transform(sph)
                    exk = self.stft.transform(enh)
                self._draw_spectrogram(epoch, sxk, exk, titles=("sph", "enh"))
                draw = False

            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())
            # break

        out = {}
        for k, v in metric_rec.state_dict().items():
            if k in self.raw_metrics["valid"]:
                out[k] = {"raw": self.raw_metrics["valid"][k], "enh": v}
            else:
                out[k] = v
        # return metric_rec.state_dict()
        return out

    def vtest_fn(self, sph: Tensor, enh: Tensor) -> Dict:
        sisnr = self._si_snr(sph, enh)
        sisnr = np.mean(sisnr)

        pesq_sc = self._pesq(
            sph.cpu().detach().numpy(),
            enh.cpu().detach().numpy(),
            fs=16000,
            norm=False,
        ).mean()
        # pesq = np.mean(pesq)
        # composite = self._eval(clean, enh, 16000)
        # composite = {k: np.mean(v) for k, v in composite.items()}
        # pesq = composite.pop("pesq")

        stoi_sc = self._stoi(
            sph.cpu().detach().numpy(),
            enh.cpu().detach().numpy(),
            fs=16000,
        ).mean()
        # stoi = np.mean(stoi)

        state = {"pesq": pesq_sc, "stoi": stoi_sc, "sisnr": sisnr}

        # return dict(state, **composite)
        return state

    def _vtest_each_epoch(self, epoch):
        out = {}

        metric_rec = REC()
        dirname = os.path.split(self.vtest_dset.dirname)[-1]
        pbar = tqdm(
            self.vtest_loader,
            ncols=120,
            leave=False,
            desc=f"vTest-{epoch}/{dirname}",
        )
        # vtest_outdir = os.path.join(self.vtest_outdir, dirname)
        # shutil.rmtree(vtest_outdir) if os.path.exists(vtest_outdir) else None

        for mic, sph, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)

            with torch.no_grad():
                c, _ = self.net_ae(mic)
                enh = self.net(mic, c)

            metric_dict = self.valid_fn(sph, enh, nlen, return_loss=False)
            # record the loss
            metric_rec.update(metric_dict)
            # pbar.set_postfix(**metric_rec.state_dict())
            # break

        dirn = {}
        for k, v in metric_rec.state_dict().items():
            if k in self.raw_metrics["vtest"]:
                dirn[k] = {"raw": self.raw_metrics["vtest"][k], "enh": v}
            else:
                dirn[k] = v
        out[dirname] = dirn
        return out

    def _net_flops(self) -> int:
        from thop import profile
        import copy

        x = torch.randn(1, 16000, 6)
        net_ae = copy.deepcopy(self.net_ae).cpu()
        y, _ = net_ae(x)
        flops, _ = profile(
            copy.deepcopy(self.net).cpu(),
            inputs=(x, y),
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

    cfg_fname = "config/config_mcse_ae.ini"
    cfg = read_ini(cfg_fname)
    print("##", cfg_fname)

    # net_ae = MCAE_2(nframe=512, nhop=256, nfft=512, in_channels=6, latent_dim=256)
    # net_ae.load_state_dict(
    #     torch.load("trained_mcae/MCAE_infonce/checkpoints/epoch_0004.pth")
    # )
    net_ae = MCAE_3(nframe=512, nhop=256, nfft=512, in_channels=6, mid_channel=40)
    net_ae.load_state_dict(
        torch.load("trained_mcae/MCAE_3_drop/checkpoints/epoch_0050.pth")
    )
    net = DF_AIA_TRANS_AE(
        in_channels=6, feature_size=257, mid_channels=80
    )  # df_aia_trans_spf_pmsqe

    # net_ae = MCAE_F(nframe=512, nhop=256, nfft=512, in_channels=6, mid_channel=16)
    # net_ae.load_state_dict(torch.load("trained_mcae/MCAE_F/checkpoints/epoch_0050.pth"))
    # net = DF_AIA_TRANS_AE_F(
    #     in_channels=6, feature_size=257, mid_channels=80
    # )  # df_aia_trans_spf_pmsqe

    train_dset = CHiMe3(
        cfg["dataset"]["train_dset"], subdir="train", nlen=3.0, min_len=1.0
    )
    valid_dset = CHiMe3(cfg["dataset"]["valid_dset"], subdir="dev")
    test_dsets = CHiMe3(cfg["dataset"]["vtest_dset"], subdir="test")

    # net = DF_AIA_TRANS_densepwc(
    #     in_channels=6, feature_size=257, mid_channels=128
    # )  # df_aia_trans_spf_densepwc_pmsqe
    # net = McNet_DF_AIA_TRANS(in_channels=6, feature_size=257, mid_channels=96)  # C,F,C'
    # net = dual_aia_trans_chime(
    #     in_channels=6, feature_size=257, mid_channels=96
    # )  # C,F,C'
    # net = MCNet(
    #     in_channels=6, ref_channel=5, sub_freqs=(3, 2), past_ahead=(5, 0)
    # )  # C,F,C'
    # net = MCNetwDense(
    #     in_channels=6, ref_channel=5, sub_freqs=(3, 2), past_ahead=(5, 0)
    # )  # C,F,C'
    # net = DenseMCNet(
    #     in_channels=6, ref_channel=5, sub_freqs=(3, 2), past_ahead=(5, 0)
    # )  # C,F,C'
    # net = MCNet_wED(
    #     in_channels=6, ref_channel=5, sub_freqs=(3, 2), past_ahead=(5, 0)
    # )  # C,F,C'

    init = cfg["config"]
    eng = Train(
        train_dset,
        valid_dset,
        test_dsets,
        net_ae=net_ae,
        net=net,
        batch_sz=2,
        valid_first=False,
        **init,
    )
    print(eng)
    if args.test:
        eng.test("")
    else:
        eng.fit()
