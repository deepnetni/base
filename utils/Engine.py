import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from torch.optim import Optimizer, lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path
from typing import Dict
import numpy as np
from itertools import repeat
from joblib import Parallel, delayed
from utils.metrics import *
from utils.composite_metrics import eval_composite
from collections import Counter


def setup_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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
        valid_per_epoch: int = 1,
        valid_first: bool = False,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.net = net.to(self.device)
        self.optimizer = self._config_optimizer(optimizer, self.net.parameters())
        self.scheduler = self._config_scheduler(scheduler, self.optimizer)
        self.epochs = epochs
        self.start_epoch = 1
        self.valid_per_epoch = valid_per_epoch
        self.best_score = torch.finfo(torch.float32).min

        # checkpoints
        self.info_dir = Path(info_dir) if info_dir != "" else Path(__file__).parent
        self.ckpt_dir = self.info_dir / name / "checkpoints"
        self.ckpt_file = self.ckpt_dir / "ckpt.pth"
        self.ckpt_best_file = self.ckpt_dir / "best.pth"
        self.valid_first = valid_first

        self.ckpt_dir.mkdir(
            parents=True, exist_ok=True
        )  # create directory if not exists

        if resume is True:
            assert self.ckpt_file.exists()
            self._load_ckpt()

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
        if isinstance(sph, np.ndarray):
            # sph = sph.cpu().detach().numpy()
            # enh = enh.cpu().detach().numpy()
            sph = torch.from_numpy(sph)
            enh = torch.from_numpy(enh)

        return compute_si_snr(sph, enh, zero_mean).cpu().detach().numpy()

    @staticmethod
    def _pesq(
        sph: np.ndarray, enh: np.ndarray, fs: int, norm: bool = False, njobs: int = 10
    ) -> np.ndarray:
        scores = np.array(
            Parallel(n_jobs=10)(
                delayed(compute_pesq)(s, e, fs, norm) for s, e in zip(sph, enh)
            )
        )
        return scores

    @staticmethod
    def _stoi(sph: np.ndarray, enh: np.ndarray, fs: int, njobs: int = 10) -> np.ndarray:
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

    def _draw_spectrogram(self, epoch, *args, **kwargs):
        """
        draw spectrogram with args

        :param args: (xk, xk, ...)
        :param kwargs: fs
        :return:
        """
        N = len(args)
        fs = kwargs.get("fs", 0)
        titles = kwargs.get("titles", repeat(None))

        fig, ax = plt.subplots(N, 1, constrained_layout=True, figsize=(16.0, 9.0))
        for xk, axi, title in zip(args, ax.flat, titles):
            xk = xk.cpu().detach().numpy() if isinstance(xk, torch.Tensor) else xk

            if xk.ndim > 2:
                r, i = xk[0, 0, ...], xk[0, 1, ...]  # r, i shape t,f
            else:
                r, i = xk[0, ...], xk[1, ...]

            mag = (r**2 + i**2) ** 0.5
            spec = 10 * np.log10(mag**2 + 1e-10).transpose(1, 0)  # f,t

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
        content += f"{'nTotal':<{ncol}}: {total:<{ncol},d}, "
        content += f"nTrainable: {trainable: <{ncol},d}, "
        try:
            flops = self._net_flops()
            content += f"FLOPS: {flops / 1024**3:.3f}G\n"
        except NotImplementedError:
            content += "\n"

        content += "=" * 60
        return content

    def _net_info(self):
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        size = sum(p.numel() * p.element_size() for p in self.net.parameters())
        return total, trainable, size

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

    def _net_flops(self) -> int:
        # x = torch.randn(1, 16000)
        # flops = profile(copy.deepcopy(self.net), inputs=(x,), verbose=False)
        raise NotImplementedError

    def _fit_each_epoch(self, epoch: int) -> Dict:
        raise NotImplementedError
        # return {"loss": 0}

    def _valid_each_epoch(self, epoch: int) -> Dict:
        raise NotImplementedError
        # return {"score": 0}
