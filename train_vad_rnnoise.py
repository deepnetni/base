import argparse
import os
import sys
import yaml
from typing import Dict, List, Optional

import warnings
import numpy as np
import torch
import torch.nn.functional as F

# from matplotlib import pyplot as plt
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# from utils.conv_stft_loss import MultiResolutionSTFTLoss
from tqdm import tqdm
from rebuild.rnnoise import FeatExtractor, RNNoise

from utils.Engine import Engine
from utils.ini_opts import read_ini
from utils.losses import loss_compressed_mag, loss_pmsqe, loss_sisnr
from utils.record import REC, RECDepot
from utils.stft_loss import MultiResolutionSTFTLoss
from utils.trunk import CHiMe3, VADSet, SpatialedDNS, pad_to_longest, L3DAS22
from utils.audiolib import audioread, audiowrite
from utils.register import tables
from utils.logger import CPrint
from models.conv_stft import STFT
from models.DeFT_AN import Network
from torchmetrics.functional.audio import signal_distortion_ratio as SDR
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio as si_sdr,
)
from models.APC_SNR.apc_snr import APC_SNR_multi_filter
from models.pase.models.frontend import wf_builder
from models.VADModel import *
from utils.focal_loss import BCEFocalLoss
from rebuild import *


class Train(Engine):
    def __init__(
        self,
        train_dset: Dataset,
        valid_dset: Dataset,
        vtest_dset: Optional[Dataset],
        # net_ae: torch.nn.Module,
        batch_sz: int,
        denoise: nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.net_ae = net_ae.to(self.device)
        # self.net_ae.eval()
        self.nframe = kwargs.get("nframe", 0)
        self.nhop = kwargs.get("nhop", 0)
        # self.denoise_net: nn.Module
        self.denoise_net = denoise
        assert self.denoise_net is not None
        self.denoise_net.cuda()

        self.optimizer.add_param_group(
            {"params": self.denoise_net.parameters(), "lr": 5e-4, "amsgrad": False}
        )

        self.feat_extract = FeatExtractor(self.nframe, self.nhop).eval().to(self.device)

        assert self.nframe is not None and self.nhop is not None

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
            batch_size=1,
            # batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
            # collate_fn=pad_to_longest,
            # generator=g,
        )
        self.valid_dset = valid_dset

        self.vtest_loader = (
            DataLoader(
                vtest_dset,
                batch_size=1,
                # batch_size=1,
                num_workers=4,
                pin_memory=True,
                shuffle=False,
                worker_init_fn=self._worker_set_seed,
                generator=self._set_generator(),
                # collate_fn=pad_to_longest,
                # generator=g,
            )
            if vtest_dset is not None
            else None
        )
        self.vtest_dset = vtest_dset

        # self.vtest_loader = (
        #     [vtest_dset] if isinstance(vtest_dset, Dataset) else vtest_dset
        # )

        # self.stft = STFT(nframe=128, nhop=64, win="hann sqrt").to(self.device)
        self.stft = STFT(nframe=self.nframe, nhop=self.nhop, win="hann sqrt").to(self.device)
        self.stft.eval()

        self.focal = BCEFocalLoss(gamma=1, alpha=0.5).to(self.device)
        self.focal.eval()

        self.ms_stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 512, 256],
            hop_sizes=[512, 256, 128],
            win_lengths=[1024, 512, 256],
        ).to(self.device)
        self.ms_stft_loss.eval()

        self.raw_metrics = self._load_dsets_metrics(self.dsets_mfile)
        # self.pase = wf_builder("config/frontend/PASE+.cfg")
        # self.pase.cuda()
        # self.pase.eval()
        # self.pase.load_pretrained(
        #     "pretrained/pase_e199.ckpt", load_last=True, verbose=False
        # )

        # self.APC_criterion = APC_SNR_multi_filter(
        #     model_hop=128,
        #     model_winlen=512,
        #     mag_bins=256,
        #     theta=0.01,
        #     hops=[8, 16, 32, 64],
        # ).to(self.device)

    # def config_optimizer(self):
    #     return torch.optim.Adam(
    #         [
    #             {"params": self.net.parameters(), "lr": 5e-4, "amsgrad": False},
    #             {"params": self.denoise_net.parameters(), "lr": 5e-4, "amsgrad": False},
    #         ]
    #     )

    def post_save_ckpt(self, ckpt_dict):
        ckpt_dict.update({"denoise_net": self.denoise_net.state_dict()})
        return ckpt_dict

    def post_load_ckpt(self, ckpt_dict):
        self.denoise_net.load_state_dict(ckpt_dict["denoise_net"])

    @staticmethod
    def _config_optimizer(name: str, params, **kwargs):
        return super(Train, Train)._config_optimizer(
            name, filter(lambda p: p.requires_grad, params, **kwargs)
        )

    def loss_fn(self, vad, vad_pred, clean, enh) -> Dict:
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

        # specs_enh = self.stft.transform(enh)
        # specs_sph = self.stft.transform(clean)
        # pmsqe_score = 0.3 * loss_pmsqe(specs_sph, specs_enh)

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

        # loss = sc_loss + mag_loss + pmsqe_score  # + loss_pase  # + 0.05 * sdr_lv

        fc_loss = self.focal(vad, vad_pred)
        loss = fc_loss + sc_loss + mag_loss

        return {
            "loss": loss,
            "focal": fc_loss.detach(),
            "mag": mag_loss.detach(),
            "sc": sc_loss.detach(),
        }

    def _fit_each_epoch(self, epoch):
        losses_rec = REC()
        self.denoise_net.train()

        pbar = tqdm(
            self.train_loader,
            ncols=160,
            leave=True,
            desc=f"Epoch-{epoch}/{self.epochs}",
        )
        for mic, sph, vad in pbar:
            mic = mic.to(self.device)  # B,T
            sph = sph.to(self.device)  # B,T
            vad = vad_to_frames(vad, self.nframe, self.nhop)
            vad = vad.to(self.device)  # B,T
            spec = self.stft.transform(mic).to(self.device)

            self.optimizer.zero_grad()
            xk_band = self.feat_extract(mic)
            vad_pred, state = self.net(xk_band)

            enh = self.denoise_net(xk_band, state, spec)
            enh = self.stft.inverse(enh)

            loss_dict = self.loss_fn(vad, vad_pred, sph, enh)

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
        return dset_dict

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

            metric_dict = self.valid_fn(sph, mic[..., 0], nlen, return_loss=False)
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

            metric_dict = self.valid_fn(sph, mic[..., 0], nlen, return_loss=False)
            metric_dict.pop("score")

            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())

        dset_dict["vtest"] = metric_rec.state_dict()

        return dset_dict

    def valid_fn(
        self, sph: Tensor, enh: Tensor, nlen_list: Tensor, return_loss: bool = True
    ) -> Dict:
        """
        B,T
        """
        sisnr_l = []
        sdr_l = []

        B = sph.size(0)
        sph_ = sph[0, : nlen_list[0]]  # B,T
        enh_ = enh[0, : nlen_list[0]]
        sisnr_l.append(self._si_snr(sph_.cpu().numpy(), enh_.cpu().numpy()))
        np_l_sph = [sph_.cpu().numpy()]
        np_l_enh = [enh_.cpu().numpy()]
        sdr_l.append(SDR(preds=enh_, target=sph_).cpu().numpy())

        for i in range(1, B):
            sph_ = sph[i, : nlen_list[i]]  # B,T
            enh_ = enh[i, : nlen_list[i]]
            np_l_sph.append(sph_.cpu().numpy())
            np_l_enh.append(enh_.cpu().numpy())

            sisnr_l.append(self._si_snr(sph_.cpu().numpy(), enh_.cpu().numpy()))
            sdr_l.append(SDR(preds=enh_, target=sph_).cpu().numpy())

        sisnr_sc = np.array(sisnr_l).mean()
        # sisnr_sc_ = self._si_snr(sph, enh).mean()
        sdr_sc = np.array(sdr_l).mean()
        pesq_wb_sc = self._pesq(np_l_sph, np_l_enh, fs=16000).mean()
        pesq_nb_sc = self._pesq(np_l_sph, np_l_enh, fs=16000, mode="nb").mean()
        stoi_sc = self._stoi(np_l_sph, np_l_enh, fs=16000).mean()

        # composite = self._eval(clean, enh, 16000)
        # composite = {k: np.mean(v) for k, v in composite.items()}
        # pesq = composite.pop("pesq")

        state = {
            "score": pesq_wb_sc,
            "sisnr": sisnr_sc,
            # "sisnr_pad": sisnr_sc_.cpu().detach().numpy(),
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
        self.denoise_net.eval()

        pbar = tqdm(
            self.valid_loader,
            ncols=160,
            leave=True,
            desc=f"Valid-{epoch}/{self.epochs}",
        )

        draw = False

        for mic, sph, vad in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # B,T
            vad = vad_to_frames(vad, self.nframe, self.nhop)
            vad = vad.to(self.device)  # b,c,t,f
            # nlen = self.stft.nLen(nlen).to(self.device)
            spec = self.stft.transform(mic).to(self.device)

            with torch.no_grad():
                xk_band = self.feat_extract(mic)
                vad_pred, state = self.net(xk_band)
                enh = self.denoise_net(xk_band, state, spec)
                enh = self.stft.inverse(enh)

            metric_dict = self.loss_fn(vad, vad_pred, sph, enh)

            if draw is True:
                with torch.no_grad():
                    sxk = self.stft.transform(vad)
                    exk = self.stft.transform(enh)
                self._draw_spectrogram(epoch, sxk, exk, titles=("sph", "enh"))
                draw = False

            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())
            # break

        out = {}
        for k, v in metric_rec.state_dict().items():
            if "valid" in self.raw_metrics and k in self.raw_metrics["valid"]:
                out[k] = {"raw": self.raw_metrics["valid"][k], "enh": v}
            else:
                out[k] = v
        # return metric_rec.state_dict()
        return out

    # def vtest_fn(self, sph: Tensor, enh: Tensor) -> Dict:
    #     sisnr = self._si_snr(sph.cpu().detach().numpy(), enh.cpu().detach().numpy())
    #     sisnr = np.mean(sisnr)

    #     pesq_sc = self._pesq(
    #         sph.cpu().detach().numpy(),
    #         enh.cpu().detach().numpy(),
    #         fs=16000,
    #         norm=False,
    #     ).mean()
    #     # pesq = np.mean(pesq)
    #     # composite = self._eval(clean, enh, 16000)
    #     # composite = {k: np.mean(v) for k, v in composite.items()}
    #     # pesq = composite.pop("pesq")

    #     stoi_sc = self._stoi(
    #         sph.cpu().detach().numpy(),
    #         enh.cpu().detach().numpy(),
    #         fs=16000,
    #     ).mean()
    #     # stoi = np.mean(stoi)

    #     state = {"pesq": pesq_sc, "stoi": stoi_sc, "sisnr": sisnr}

    #     # return dict(state, **composite)
    #     return state

    def _vtest_each_epoch(self, epoch):
        out = {}

        metric_rec = REC()
        # dirname = os.path.split(self.vtest_dset.dirname)[-1]
        dirname = self.vtest_dset.dirname
        pbar = tqdm(
            self.vtest_loader,
            ncols=120,
            leave=False,
            desc=f"vTest-{epoch}/{dirname}",
        )
        # vtest_outdir = os.path.join(self.vtest_outdir, dirname)
        # shutil.rmtree(vtest_outdir) if os.path.exists(vtest_outdir) else None

        for mic, vad, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            vad = vad_to_frames(vad, self.nframe, self.nhop)
            vad = vad.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)

            with torch.no_grad():
                enh = self.net(mic)

            metric_dict = self.loss_fn(vad, enh, nlen)
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

        # x = torch.randn(1, 2, 250, self.nframe // 2 + 1)
        x = torch.randn(1, 16000)
        x = copy.deepcopy(self.feat_extract).cpu()(x)
        x = x.cpu()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="This API is being deprecated")
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
    parser.add_argument("--test", help="test mode", action="store_true")
    parser.add_argument("--train", help="train mode", action="store_true", default=True)
    parser.add_argument("-P", "--pred", help="predict mode", action="store_true")

    parser.add_argument("--ckpt", help="ckpt path", type=str)
    parser.add_argument("--src", help="input directory", type=str)
    parser.add_argument("--dst", help="predicting output directory", type=str)

    args = parser.parse_args()
    args.train = False if args.test or args.pred else True
    return args


if __name__ == "__main__":
    args = parse()
    cp = CPrint()

    cfg_fname = "config/config_vad.yaml"
    md_conf = {}
    # cfg = read_ini(cfg_fname)
    with open(cfg_fname) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print("##", cfg_fname)
    md_name = cfg["config"]["name_vad"]
    tables.print()
    cp.r(f"current: {md_name}")

    model = tables.models.get(md_name)
    assert model is not None
    net_vad = model(**md_conf)

    denoise = RNNoise()

    if args.pred is False:
        cfg["config"]["info_dir"] = f"{cfg['config']['info_dir']}"
        cfg["config"]["vtest_per_epoch"] = 0

        train_dset = VADSet(
            dirname=cfg["dataset"]["train_dset"],
            patten="**/*mic.wav",
            keymap=("mic", "target"),
            flist="vad.csv",
            # min_len=1.0,
            # nlen=5.0,
        )
        valid_dset = VADSet(
            dirname=cfg["dataset"]["valid_dset"],
            patten="**/*mic.wav",
            keymap=("mic", "vad"),
            flist="vad_val.csv",
            # min_len=1.0,
            # nlen=5.0,
        )
        # test_dsets = VADSet(
        #     dirname=cfg["dataset"]["vtest_dset"],
        #     patten="**/*mic.wav",
        #     keymap=('mic', 'vad')
        #     flist="vad_tst.csv",
        #     min_len=1.0,
        #     nlen=3.0,
        # )

        init = cfg["config"]
        eng = Train(
            train_dset,
            valid_dset,
            # test_dsets,
            None,
            net=net_vad,
            denoise=denoise,
            batch_sz=8,
            valid_first=False,
            nframe=256,
            nhop=128,
            **init,
        )
        print(eng)

        if args.test:
            eng.test()
        else:
            eng.fit()

    else:  # pred
        assert args.ckpt is not None
        assert args.src is not None
        net.load_state_dict(torch.load(args.ckpt))
        net.cuda()
        net.eval()
        d, fs = audioread(args.src)
        d = d[None, :]
        d = torch.from_numpy(d).float().cuda()
        with torch.no_grad():
            out = net(d)
        out = out.cpu().squeeze().detach()
        out = out.numpy()
        print(out.shape)
        audiowrite("bout.wav", out, sample_rate=fs)
