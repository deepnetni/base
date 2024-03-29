import json
import os
import random
import re
from collections import Counter
from itertools import repeat
from pathlib import Path
from typing import Dict, Optional

import matplotlib

from utils.audiolib import audioread

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from torch.optim import Optimizer, lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from utils.composite_metrics import eval_composite
from utils.logger import get_logger
from utils.metrics import *

log = get_logger("eng", mode="console")


def setup_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # set seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # set seed for current GPU
        torch.cuda.manual_seed_all(seed)  # set seed for all GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


setup_seed()


class Engine(object):
    def __init__(
        self,
        name: str,
        net: nn.Module,
        epochs: int,
        desc: str = "",
        info_dir: str = "",
        resume: bool = False,
        optimizer: str = "adam",
        scheduler: str = "stepLR",
        seed: int = 0,
        valid_per_epoch: int = 1,
        vtest_per_epoch: int = 0,
        valid_first: bool = False,
        vtest_outdir: str = "vtest",
        dsets_raw_metrics: str = "",
        **kwargs,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.net = net.to(self.device)
        self.optimizer = self._config_optimizer(
            optimizer, filter(lambda p: p.requires_grad, self.net.parameters())
        )
        self.scheduler = self._config_scheduler(scheduler, self.optimizer)
        self.epochs = epochs
        self.start_epoch = 1
        self.valid_per_epoch = valid_per_epoch
        self.vtest_per_epoch = vtest_per_epoch
        self.vtest_outdir = vtest_outdir
        self.best_score = torch.finfo(torch.float32).min
        self.seed = seed
        # self.dsets_metrics = self._load_dset_valid_info()

        # checkpoints
        if os.path.isabs(info_dir):  # abspath
            self.info_dir = Path(info_dir)
        else:  # relative path
            self.info_dir = (
                Path(__file__).parent.parent / info_dir
                if info_dir != ""
                else Path(__file__).parent.parent / "trained"
            )
        log.info(f"info dirname: {self.info_dir}")

        self.ckpt_dir = self.info_dir / name / "checkpoints"
        self.ckpt_file = self.ckpt_dir / "ckpt.pth"
        self.ckpt_best_file = self.ckpt_dir / "best.pth"
        self.valid_first = valid_first
        self.dsets_mfile = (
            self.info_dir / dsets_raw_metrics
            if dsets_raw_metrics != ""
            else self.info_dir / name / "dset_metrics.json"
        )

        self.ckpt_dir.mkdir(
            parents=True, exist_ok=True
        )  # create directory if not exists

        if resume is True:
            self._load_ckpt() if self.ckpt_file.exists() else log.warning(
                f"ckpt file: {self.ckpt_file} is not existed."
            )

        # tensorboard
        self.tfb_dir = self.info_dir / name / "tfb"
        self.writer = SummaryWriter(log_dir=self.tfb_dir.as_posix())
        self.writer.add_text("Description", desc, global_step=1)

    @staticmethod
    def _config_optimizer(name: str, params, **kwargs) -> Optimizer:
        alpha = kwargs.get("alpha", 1.0)
        supported = {
            "adam": lambda p: torch.optim.Adam(p, lr=alpha * 5e-4, amsgrad=False),
            "adamw": lambda p: torch.optim.AdamW(p, lr=alpha * 5e-4, amsgrad=False),
            "rmsprop": lambda p: torch.optim.RMSprop(p, lr=alpha * 5e-4),
        }
        return supported[name](params)

    @staticmethod
    def _config_scheduler(name: str, optimizer: Optimizer):
        supported = {
            "stepLR": lambda p: lr_scheduler.StepLR(p, step_size=20, gamma=0.8),
            "reduceLR": lambda p: lr_scheduler.ReduceLROnPlateau(
                p, mode="min", factor=0.5, patience=1
            ),
        }
        return supported[name](optimizer)

    @staticmethod
    def _si_snr(
        sph: Union[torch.Tensor, np.ndarray],
        enh: Union[torch.Tensor, np.ndarray],
        zero_mean: bool = True,
    ) -> np.ndarray:
        # if isinstance(sph, np.ndarray) or isinstance(enh, np.ndarray):
        #     sph = torch.from_numpy(sph)
        #     enh = torch.from_numpy(enh)

        return compute_si_snr(sph, enh, zero_mean).cpu().detach().numpy()

    @staticmethod
    def _snr(
        sph: Union[np.ndarray, list, torch.Tensor],
        enh: Union[np.ndarray, list, torch.Tensor],
        njobs: int = 10,
    ) -> np.ndarray:
        scores = np.array(
            Parallel(n_jobs=njobs)(delayed(compute_snr)(s, e) for s, e in zip(sph, enh))
        )
        return scores

    @staticmethod
    def _pesq(
        sph: Union[np.ndarray, list, torch.Tensor],
        enh: Union[np.ndarray, list, torch.Tensor],
        fs: int,
        norm: bool = False,
        njobs: int = 10,
        mode: str = "wb",
    ) -> np.ndarray:
        if isinstance(sph, torch.Tensor):
            sph = sph.cpu().detach().numpy()
            enh = sph.cpu().detach().numpy()

        scores = np.array(
            Parallel(n_jobs=njobs)(
                delayed(compute_pesq)(s, e, fs, norm, mode) for s, e in zip(sph, enh)
            )
        )
        return scores

    @staticmethod
    def _stoi(
        sph: Union[np.ndarray, list, torch.Tensor],
        enh: Union[np.ndarray, list, torch.Tensor],
        fs: int,
        njobs: int = 10,
    ) -> np.ndarray:
        if isinstance(sph, torch.Tensor):
            sph = sph.cpu().detach().numpy()
            enh = enh.cpu().detach().numpy()

        scores = np.array(
            Parallel(n_jobs=njobs)(
                delayed(compute_stoi)(s, e, fs) for s, e in zip(sph, enh)
            )
        )
        return scores

    @staticmethod
    def _eval(
        sph: Union[torch.Tensor, np.ndarray],
        enh: Union[torch.Tensor, np.ndarray],
        fs: int,
        njobs: int = 10,
    ) -> Dict:
        if isinstance(sph, torch.Tensor):
            sph = sph.cpu().detach().numpy()
            enh = enh.cpu().detach().numpy()

        scores = Parallel(n_jobs=njobs)(
            delayed(eval_composite)(s, e, fs) for s, e in zip(sph, enh)
        )  # [{"pesq":..},{...}]
        # score = Counter(scores[0])
        # for s in scores[1:]:
        #     score += Counter(s)
        out = {}
        for _ in scores:
            for k, v in _.items():
                out.setdefault(k, []).append(v)
        return {
            k: np.array(v) for k, v in out.items()
        }  # {"pesq":,"csig":,"cbak","cvol"}

    @staticmethod
    def _worker_set_seed(worker_id):
        # seed = torch.initial_seed() % 2**32
        np.random.seed(worker_id)
        random.seed(worker_id)

    def _load_dsets_metrics(self, fname: Optional[Path] = None) -> Dict:
        """load dataset metrics provided by `self._valid_dsets`"""
        metrics = {}
        fname = self.dsets_mfile if fname is None else fname

        if fname.exists() is True:
            with open(str(fname), "r") as fp:
                metrics = json.load(fp)
        else:  # file not exists
            metrics = self._valid_dsets()
            with open(str(fname), "w+") as fp:
                json.dump(metrics, fp, indent=2)

        return metrics

    def _set_generator(self, seed: int = 0) -> torch.Generator:
        # make sure the dataloader return the same series under different PCs
        # torch.manual_seed(seed if seed is not None else self.seed)
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    def _draw_spectrogram(self, epoch, *args, **kwargs):
        """
        draw spectrogram with args

        :param args: (xk, xk, ...), xk with shape (b,2,t,f) or (2,t,f)
        :param kwargs: fs
        :return:
        """
        N = len(args)
        fs = kwargs.get("fs", 0)
        titles = kwargs.get("titles", repeat(None))

        fig, ax = plt.subplots(N, 1, constrained_layout=True, figsize=(16.0, 9.0))
        for xk, axi, title in zip(args, ax.flat, titles):
            xk = xk.cpu().detach().numpy() if isinstance(xk, torch.Tensor) else xk

            if xk.ndim > 3:  # B,C,T,F
                r, i = xk[0, 0, ...], xk[0, 1, ...]  # r, i shape t,f
            else:  # C,T,F
                r, i = xk[0, ...], xk[1, ...]

            mag = (r**2 + i**2) ** 0.5
            spec = 10 * np.log10(mag**2 + 1e-10).transpose(1, 0)  # f,t

            if fs != 0:
                nbin = spec.shape[0]
                ylabel = np.arange(
                    0, fs // 2 + 1, 1000 if fs <= 16000 else 3000
                )  # 1000, 2000, ..., Frequency
                yticks = nbin * ylabel * 2 // fs
                axi.set_yticks(yticks)
                axi.set_yticklabels(ylabel)

            axi.set_title(title) if title is not None else None
            axi.imshow(spec, origin="lower", aspect="auto", cmap="jet")

        self.writer.add_figure(
            f"spectrogram/{epoch}", fig, global_step=None, close=True, walltime=None
        )

    def __str__(self):
        content = "\n"
        ncol = 6
        total, trainable, total_sz = self._net_info()
        content += "=" * 60 + "\n"
        content += f"{'ckpt':<{ncol}}: {self.ckpt_file}\n"
        content += f"{'Total':<{ncol}}: {total_sz/1024**2:.3f}MB\n"
        content += f"{'nTotal':<{ncol}}: {total:<{ncol},d}\n"
        content += f"nTrainable: {trainable: <{ncol},d}, "
        content += f"nNon-Trainable: {total-trainable: <{ncol},d}\n"

        try:
            flops = self._net_flops()
            content += f"FLOPS: {flops / 1024**3:.3f}G\n"
        except NotImplementedError:
            # content += "\n"
            pass

        content += "=" * 60
        return content

    def _net_info(self):
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        size = sum(p.numel() * p.element_size() for p in self.net.parameters())
        return total, trainable, size

    def _save_ckpt(self, epoch, is_best=False):
        """Could be overwritten by the subclass"""

        if is_best:
            torch.save(self.net.state_dict(), self.ckpt_best_file)
        else:
            torch.save(
                self.net.state_dict(),
                self.ckpt_dir / f"epoch_{str(epoch).zfill(4)}.pth",
            )

            state_dict = {
                "epoch": epoch,
                "best_score": self.best_score,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "net": self.net.state_dict(),
            }
            torch.save(state_dict, self.ckpt_file)

    def _load_ckpt(self):
        ckpt = torch.load(self.ckpt_file, map_location=self.device)

        self.start_epoch = (
            ckpt["epoch"] + 1 if self.valid_first is False else ckpt["epoch"]
        )
        self.best_score = ckpt["best_score"]
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.net.load_state_dict(ckpt["net"])

    def _print(self, tag: str, state_dict: Dict, epoch: int):
        """
        :param state_dict: {"loss1":1, "loss2":2} or {"i1":{"k1":v1,"k2":v2},"i2":{..}}
        :param epoch:
        :return:
        """
        for k, v in state_dict.items():
            if isinstance(v, dict):
                self.writer.add_scalars(f"{tag}/{k}", v, epoch)
            else:
                self.writer.add_scalar(f"{tag}/{k}", v, epoch)

    def fit(self):
        for i in range(self.start_epoch, self.epochs + 1):
            if self.valid_first is False:
                self.net.train()

                loss = self._fit_each_epoch(i)
                self.scheduler.step()
                self._print("Loss", loss, i)
                self._save_ckpt(i, is_best=False)

            self.valid_first = False
            # valid_first is True
            if self.valid_per_epoch != 0 and i % self.valid_per_epoch == 0:
                self.net.eval()
                score = self._valid_each_epoch(i)
                self._print("Eval", score, i)
                if score["score"] > self.best_score:
                    self.best_score = score["score"]
                    self._save_ckpt(i, is_best=True)

            if self.vtest_per_epoch != 0 and i % self.vtest_per_epoch == 0:
                self.net.eval()
                scores = self._vtest_each_epoch(i)
                for name, score in scores.items():
                    out = ""
                    # score {"-5":{"pesq":v,"stoi":v},"0":{...}}
                    for k, v in score.items():
                        out += f"{k}:{v} " + "\n"
                    self.writer.add_text(f"Test-{name}", out, i)
                    self._print(f"Test-{name}", score, i)

    def test(self, name: str, epoch: Optional[int] = None):
        if epoch is None:
            epoch = (
                self.start_epoch - 1 if self.valid_first is False else self.start_epoch
            )
        self.net.eval()
        scores = self._vtest_each_epoch(epoch)
        # {"dir1":{"-5":{"pesq":v,"stoi":v},"0":{...}},"dir2":{...}}
        # out = ""
        # for k, v in score.items():
        #     out += f"{k}:{v} " + "\n"
        # self.writer.add_text("Test", out, i)
        for name, score in scores.items():
            self.writer.add_text(f"Test-{name}", json.dumps(score), epoch)
            self._print(f"Test-{name}", score, epoch)

    def _net_flops(self) -> int:
        # from thop import profile
        # import copy
        # x = torch.randn(1, 16000)
        # flops, _ = profile(copy.deepcopy(self.net), inputs=(x,), verbose=False)
        # return flops
        raise NotImplementedError

    def _valid_dsets(self) -> Dict:
        """return metrics of valid & test dataset
        Return:
            {'valid':{"pesq":xx, "STOI":xx,...}, "test":{...}}
        """
        return {}
        # return {"loss": 0}

    def _fit_each_epoch(self, epoch: int) -> Dict:
        raise NotImplementedError
        # return {"loss": 0}

    def _valid_each_epoch(self, epoch: int) -> Dict:
        raise NotImplementedError
        # return {"score": 0}

    def _vtest_each_epoch(self, epoch: int) -> Dict[str, Dict[str, Dict]]:
        # {"dir1":{"metric":v,..}, "d2":{..}}
        # or {"dir1":{"subd1":{"metric":v,...},"sub2":{...}}, "dir2":{...}}
        raise NotImplementedError
