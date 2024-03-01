import os
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np

from utils.Engine import Engine
from utils.ini_opts import read_ini
from utils.trunk import AECTrunk
from utils.record import REC, RECDepot
from utils.losses import loss_compressed_mag, loss_sisnr, loss_pmsqe
from tqdm import tqdm
from typing import Dict
from models.APC_SNR.apc_snr import APC_SNR_multi_filter
from models.ADPCRN import (
    ADPCRN,
    ADPCRN_ATTN,
    CRN_AEC,
    DPCRN_AEC,
    ADPCRN_Dilated,
    ADPCRN_MS,
)
from models.conv_stft import STFT

# from models.pase.models.frontend import wf_builder
import argparse


class Train(Engine):
    def __init__(self, train_dset, valid_dset, vtest_dset, batch_sz, **kwargs):
        super().__init__(**kwargs)
        self._set_seed()
        self.train_loader = DataLoader(
            train_dset,
            batch_size=batch_sz,
            num_workers=6,
            pin_memory=True,
            shuffle=True,
        )
        self._set_seed()
        self.valid_loader = DataLoader(
            valid_dset,
            batch_size=batch_sz,
            num_workers=6,
            pin_memory=True,
            shuffle=True,
        )

        self.vtest_loader = vtest_dset

        self.stft = STFT(nframe=512, nhop=256).to(self.device)
        self.stft.eval()

        # self.APC_criterion = APC_SNR_multi_filter(
        #     model_hop=128,
        #     model_winlen=512,
        #     mag_bins=256,
        #     theta=0.01,
        #     hops=[8, 16, 32, 64],
        # ).to(self.device)

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
        # loss_APC_SNR, loss_pmsqe = self.APC_criterion(enh + 1e-8, clean + 1e-8)
        # loss = 0.05 * loss_APC_SNR + loss_pmsqe  # + loss_pase
        # return {
        #     "loss": loss,
        #     "pmsqe": loss_pmsqe.detach(),
        #     "apc_snr": loss_APC_SNR.detach(),
        # }

        sisnr_lv = loss_sisnr(clean, enh)
        with torch.no_grad():
            specs_enh = self.stft.transform(enh)
            specs_sph = self.stft.transform(clean)
        mse_mag, mse_pha = loss_compressed_mag(specs_sph, specs_enh)
        # pmsq_score = loss_pmsqe(specs_sph, specs_enh)
        loss = 0.05 * sisnr_lv + mse_pha + mse_mag  # + pmsq_score
        return {
            "loss": loss,
            "sisnr": sisnr_lv.detach(),
            "mag": mse_mag.detach(),
            "pha": mse_pha.detach(),
            # "pmsq": pmsq_score.detach(),
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
        for mic, ref, sph, _ in pbar:
            mic = mic.to(self.device)  # b,c,t,f
            ref = ref.to(self.device)  # b,c,t,f
            sph = sph.to(self.device)  # b,c,t,f

            self.optimizer.zero_grad()
            enh = self.net(mic, ref)
            loss_dict = self.loss_fn(sph[:, : enh.shape[-1]], enh)

            loss = loss_dict["loss"]
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()

            # record the loss
            losses_rec.update(loss_dict)
            pbar.set_postfix(**losses_rec.state_dict())

        return losses_rec.state_dict()

    def valid_fn(self, sph: Tensor, enh: Tensor, scenario: Tensor) -> Dict:
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

        score = (pesq - 1) / 3.5 + stoi

        state = {"score": score, "pesq": pesq, "stoi": stoi, "sisnr": sisnr}

        loss_dict = self.loss_fn(sph, enh)

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

        for mic, ref, sph, scenario in pbar:
            mic = mic.to(self.device)  # b,c,t,f
            ref = ref.to(self.device)  # b,c,t,f
            sph = sph.to(self.device)  # b,c,t,f
            sce = scenario.to(self.device)  # b,1

            with torch.no_grad():
                enh = self.net(mic, ref)

            metric_dict = self.valid_fn(sph[:, : enh.shape[-1]], enh, sce)

            if draw is True:
                with torch.no_grad():
                    mxk = self.stft.transform(mic)
                    exk = self.stft.transform(enh)
                    sxk = self.stft.transform(sph)
                self._draw_spectrogram(
                    epoch, mxk, exk, sxk, titles=("mic", "enh", "sph")
                )
                draw = False

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

    def _vtest_each_epoch(self, epoch):
        metric_rec = RECDepot()

        pbar = tqdm(
            self.vtest_loader,
            total=len(self.vtest_loader),
            ncols=120,
            leave=False,
            desc=f"vTest-{epoch}/{self.epochs}",
        )

        for mic, ref, sph, fname in pbar:
            mic = mic.to(self.device)  # b,c,t,f
            ref = ref.to(self.device)  # b,c,t,f
            sph = sph.to(self.device)  # b,c,t,f

            item, fname = os.path.split(fname)

            with torch.no_grad():
                enh = self.net(mic, ref)

            metric_dict = self.vtest_fn(sph[:, : enh.shape[-1]], enh)

            # record the loss
            metric_rec.update(item, metric_dict)
            # pbar.set_postfix(**metric_rec.state_dict())

        return metric_rec.state_dict()

    def _net_flops(self) -> int:
        from thop import profile
        import copy

        x = torch.randn(1, 16000)
        flops, _ = profile(
            copy.deepcopy(self.net).cpu(),
            inputs=(x, x),
            verbose=False,
        )
        return flops


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crn", help="crn aec model", action="store_true")
    parser.add_argument("--wo-sfp", help="without SFP path mode", action="store_true")
    parser.add_argument(
        "--wfusion-att", help="fusion with the atten method", action="store_true"
    )
    parser.add_argument(
        "--wfusion-ms", help="fusion with the atten method", action="store_true"
    )
    parser.add_argument(
        "--wfusion-dilated", help="fusion with the atten method", action="store_true"
    )
    parser.add_argument("-T", "--train", help="train mode", action="store_true")
    parser.add_argument("-P", "--predict", help="predict mode", action="store_true")

    parser.add_argument("--ckpt", help="ckpt path", type=str)
    parser.add_argument("--src", help="input directory", type=str)
    parser.add_argument("--dst", help="predicting output directory", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    if args.wo_sfp:  # 31
        cfg_fname = "config/config_adpcrn_wo_sfp.ini"
        cfg = read_ini(cfg_fname)
        net = DPCRN_AEC(
            nframe=512,
            nhop=256,
            nfft=512,
            cnn_num=[16, 32, 64, 128],
            stride=[2, 2, 1, 1],
            rnn_hidden_num=128,
        )
    elif args.crn:  # 212
        cfg_fname = "config/config_adpcrn_crn.ini"
        cfg = read_ini(cfg_fname)
        net = CRN_AEC(
            nframe=512,
            nhop=256,
            nfft=512,
            cnn_num=[16, 32, 64, 128],
            stride=[2, 2, 1, 1],
            rnn_hidden_num=128,
        )
    elif args.wfusion_ms:  # 212
        cfg_fname = "config/config_adpcrn_w_fusion_ms.ini"
        cfg = read_ini(cfg_fname)
        net = ADPCRN_MS(
            nframe=512,
            nhop=256,
            nfft=512,
            cnn_num=[16, 32, 64, 128],
            stride=[2, 2, 1, 1],
            rnn_hidden_num=128,
        )
    elif args.wfusion_att:  # 212
        cfg_fname = "config/config_adpcrn_w_fusion_att.ini"
        cfg = read_ini(cfg_fname)
        net = ADPCRN_ATTN(
            nframe=512,
            nhop=256,
            nfft=512,
            cnn_num=[16, 32, 64, 128],
            stride=[2, 2, 1, 1],
            rnn_hidden_num=128,
        )
    elif args.wfusion_dilated:  # 212
        cfg_fname = "config/config_adpcrn_w_fusion_dilated.ini"
        cfg = read_ini(cfg_fname)
        net = ADPCRN_Dilated(
            nframe=512,
            nhop=256,
            nfft=512,
            cnn_num=[16, 32, 64, 128],
            stride=[2, 2, 1, 1],
            rnn_hidden_num=128,
        )
    else:  # 212  baseline
        cfg_fname = "config/config_adpcrn.ini"
        cfg = read_ini(cfg_fname)
        net = ADPCRN(
            nframe=512,
            nhop=256,
            nfft=512,
            cnn_num=[16, 32, 64, 128],
            stride=[2, 2, 1, 1],
            rnn_hidden_num=128,
        )

    print("##", cfg_fname)

    init = cfg["config"]
    eng = Train(
        AECTrunk(
            cfg["dataset"]["train_dset"],
            flist="gene-aec-100-30.csv",
            # flist="gene-aec-train-test.csv",
            patten="**/*mic.wav",
            keymap=("mic", "ref", "sph"),
            align=True,
        ),
        AECTrunk(
            cfg["dataset"]["valid_dset"],
            flist="gene-aec-4-1.csv",
            patten="**/*mic.wav",
            keymap=("mic", "ref", "sph"),
            align=True,
        ),
        AECTrunk(
            cfg["dataset"]["vtest_dset"],
            flist="aec-test-set.csv",
            patten="**/*mic.wav",
            keymap=("mic", "lpb", "sph"),
            align=True,
        ),
        net=net,
        batch_sz=4,
        valid_first=False,
        **init,
    )
    print(eng)
    eng.fit()
