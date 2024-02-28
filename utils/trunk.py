import os
import sys
import json
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.gcc_phat import gcc_phat

from utils.audiolib import audioread
from utils.logger import get_logger


class NSTrunk(Dataset):
    """Dataset class
    Args:
        dirname: directory contains the (train, label) wav files.
        patten: search patten to get noisy wav file;
        data_len: length of training file in seconds;
        clip_len: seconds;
        keymap: replace patten to get corresponding clean wav file.
                ["str in noisy", "corresponding str in clean"]
        under_same_dir: the train and label under the same directory if `True`; otherwise,
                        the train and label under different directory with the same file name.
        return_only_noisy: only used in `__getitem__` method, default False;
                        return (mic, zeros_like(mic)) if True, otherwise return (mic, sph).
        return_abspath: only used in `__next__` method, which return (data, fname), default False;
        norm: the normalization value for audio data, default None, while SIG set to -27;
        seed: random seed.

    Return:
        1. call by torch.utils.data.DataLoader
            (mic, sph) torch.tensor pair.

        2. call by iterator, e.g.
            ```
            for data, fname in NSTrunk():
                ...
            ```
            (data, fname)

    Examples:
        1. noisy and clean audio file under the same directory;
            NSTrunk(
                dirname=xx,
                patten="**/*mic.wav",
                keymap=("mic", "target"),
            )

        2. noisy and clean audio file under different directory with the same name;
            NSTrunk(
                dirname=xx,
                clean_dirname=yy,
                patten="**/*.wav",
                data_len=10.0,      # split audio files
                clip_len=2.0,       # only affects the dataloader
            )

        3. only return noisy data under validation mode if label is not used;
            NSTrunk(
                dirname=xx,
                patten="**/*mic.wav",
                return_only_noisy=True,
            )
    """

    def __init__(
        self,
        dirname: str,
        patten: str = "**/*.wav",
        keymap: Optional[Tuple[str, str]] = None,
        flist: Optional[str] = None,
        clean_dirname: Optional[str] = None,
        data_len: int = -1,
        clip_len: int = -1,
        seed: Optional[int] = None,
        norm: Optional[int] = None,
        return_abspath: bool = False,
    ):
        super().__init__()
        self.dir = Path(dirname)
        self.logger = get_logger(dirname)

        self.f_list = self._prepare(flist, patten, keymap, clean_dirname)
        if seed is not None:
            random.seed(seed)
            random.shuffle(self.f_list)

        self.keymap = keymap
        self.clean_dir = clean_dirname
        self.norm = norm
        self.return_abspath = return_abspath
        self.dataL = data_len
        self.clipL = clip_len
        assert data_len % clip_len == 0
        assert data_len != -1 and clip_len != -1 or data_len == -1 and clip_len == -1
        self.n_clip = int(data_len // clip_len)

        self.logger.info(f"Get {dirname} {len(self.f_list)} files.")

    def _prepare(
        self,
        fname: Optional[str],
        patten: str,
        keymap: Optional[Tuple[str, str]] = None,
        clean_dirname: Optional[str] = None,
    ) -> List:
        """
        fname: file path of a file list
        """
        if fname is not None and os.path.exists(fname):
            self.logger.info(f"Load flist {fname}")
            f_list = self.load_f_list(fname)
        else:
            f_list = []
            f_mic_list = list(map(str, self.dir.glob(patten)))
            # if keymap is not None:
            #     f_sph = f_mic.replace(str(self.dir), self.clean_dir)
            for f_mic in f_mic_list:
                dirp, f_mic_name = os.path.split(f_mic)
                f_sph = f_mic_name.replace(keymap[0], keymap[2])
                f_sph = os.path.join(dirp, f_sph)
                f_list.append((f_mic, f_sph))

            self.save_f_list(fname, f_list) if fname is not None else None
        return f_list

    def load_f_list(self, fname: str) -> List:
        f_list = []
        with open(fname, "r+") as fp:
            ctx = csv.reader(fp)
            for element in ctx:
                f_list.append(element)

        return f_list

    def save_f_list(self, fname: str, f_list: List):
        with open(fname, "w+") as fp:
            writer = csv.writer(fp)
            for f in f_list:
                writer.writerow(f)

    def __len__(self):
        return len(self.f_list) * self.n_clip

    def _split(self, data, fs, slice_idx):
        st = fs * slice_idx * self.clipL
        ed = st + fs * self.clipL if self.clipL != -1 else None
        return data[st:ed]

    def __getitem__(self, index) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        idx, slice_idx = index // self.n_clip, index % self.n_clip
        (f_mic,) = self.f_list[idx]
        if self.keymap is not None:
            if self.clean_dir is None:  # under same directory with different name
                assert self.keymap is not None
                dirp, fname_mic = os.path.split(f_mic)
                fname_sph = fname_mic.replace(*self.keymap)
                f_sph = os.path.join(dirp, fname_sph)
            else:  # under different directory with same name
                # assert self.clean_dir is not None
                f_sph = f_mic.replace(str(self.dir), self.clean_dir)

            d_mic, fs_1 = audioread(f_mic)
            d_sph, fs_2 = audioread(f_sph)
            assert fs_1 == fs_2
            d_mic = self._split(d_mic, fs_1, slice_idx)
            d_sph = self._split(d_sph, fs_2, slice_idx)

            return torch.from_numpy(d_mic).float(), torch.from_numpy(d_sph).float()
        else:  # without keymap only return noisy mic data.
            d_mic, fs_1 = audioread(f_mic)
            d_mic = self._split(d_mic, fs_1, slice_idx)
            d_mic = torch.from_numpy(d_mic).float()

            return d_mic

    def __iter__(self):
        self.pick_idx = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, str]:
        """used for predict api"""
        if self.pick_idx < len(self.f_list):
            fname = self.f_list[self.pick_idx]
            if self.norm is not None:
                data, _ = audioread(fname, norm=True, target_level=self.norm)
            else:
                data, _ = audioread(fname)
            self.pick_idx += 1

            fname = str(fname) if self.return_abspath else fname.name
            return torch.from_numpy(data).float()[None, :], fname
        else:
            raise StopIteration


class AECTrunk(Dataset):
    """Dataset class, the data of mic, ref, sph must under the same directory.
    Args:
        dirname: directory contains the (train, label) wav files.
        patten: search patten to get noisy wav file;
        data_len: length of training file in seconds;
        clip_len: seconds;
        keymap: ["str in mic", "corresponding str in ref", "str in sph"]
        return_abspath: only used in `__next__` method, which return (data, fname), default False;
        norm: the normalization value for audio data, default None, while SIG set to -27;
        seed: random seed.

    Return:
        1. call by torch.utils.data.DataLoader
            (mic, ref, sph, scenario) torch.tensor pair.

        2. call by iterator, e.g.
            ```
            for mic, ref, fname in NSTrunk():
                ...
            ```

    Examples:
        1. ref, mic and sph audio file must under the same directory;
            NSTrunk(
                dirname=xx,
                patten="**/*mic.wav",
                keymap=("mic", "ref", "sph"),
            )
    """

    NE = 0
    FE = 1
    DT = 2

    def __init__(
        self,
        dirname: str,
        patten: str,  # = "**/*.wav",
        keymap: Tuple[str, str, str],
        flist: Optional[str] = None,
        data_len: int = -1,
        clip_len: int = -1,
        tgt_fs: int = 16000,
        seed: Optional[int] = None,
        norm: Optional[int] = None,
        return_abspath: bool = False,
        align: bool = False,
        ne_flag=["NE"],
        dt_flag=["DT"],
        fe_flag=["FE"],
    ):
        super().__init__()
        self.dir = Path(dirname)
        self.logger = get_logger(dirname)

        self.f_list = self._prepare(flist, patten, keymap)
        if seed is not None:
            random.seed(seed)
            random.shuffle(self.f_list)

        self.keymap = keymap
        self.norm = norm
        self.return_abspath = return_abspath
        self.dataL = data_len
        self.clipL = clip_len
        assert data_len % clip_len == 0
        assert data_len != -1 and clip_len != -1 or data_len == -1 and clip_len == -1
        self.n_clip = int(data_len // clip_len)
        self.align = align
        self.ne_flag = ne_flag
        self.dt_flag = dt_flag
        self.fe_flag = fe_flag
        self.tgt_fs = tgt_fs

        self.logger.info(f"Get {dirname} {len(self.f_list)} files.")

    def _prepare(
        self, fname: Optional[str], patten: str, keymap: Tuple[str, str, str]
    ) -> List:
        """
        fname: file path of a file list
        """
        if fname is not None and os.path.exists(fname):
            f_list = self.load_f_list(fname, str(self.dir))
            self.logger.info(f"Load flist {fname}")
        else:
            f_list = []
            f_mic_list = list(map(str, self.dir.glob(patten)))
            for f_mic in f_mic_list:
                dirp, f_mic_name = os.path.split(f_mic)
                f_ref = f_mic_name.replace(keymap[0], keymap[1])
                f_ref = os.path.join(dirp, f_ref)
                f_sph = f_mic_name.replace(keymap[0], keymap[2])
                f_sph = os.path.join(dirp, f_sph)
                f_list.append((f_mic, f_ref, f_sph))

            self.save_f_list(
                fname, f_list, str(self.dir)
            ) if fname is not None else None
        return f_list

    def load_f_list(self, fname: str, relative: Optional[str] = None) -> List:
        f_list = []
        with open(fname, "r+") as fp:
            ctx = csv.reader(fp)
            for element in ctx:
                f_list.append(
                    element
                    if relative is None
                    else tuple(map(lambda x: os.path.join(relative, x), element))
                )

        return f_list

    def save_f_list(self, fname: str, f_list: List, relative: Optional[str] = None):
        with open(fname, "w+") as fp:
            writer = csv.writer(fp)
            for f in f_list:
                writer.writerow(
                    f
                    if relative is None
                    else tuple(map(lambda x: os.path.relpath(x, relative), f))
                )

    def __len__(self):
        return len(self.f_list) * self.n_clip

    def _split(self, data, fs, slice_idx):
        st = fs * slice_idx * self.clipL
        ed = st + fs * self.clipL if self.clipL != -1 else None
        return data[st:ed]

    def __getitem__(self, index) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        idx, slice_idx = index // self.n_clip, index % self.n_clip

        f_mic, f_ref, f_sph = self.f_list[idx]
        dirp = os.path.dirname(f_mic)

        dirname = Path(dirp)
        if any(item in dirname.parts for item in self.ne_flag):
            cond = AECTrunk.NE
        elif any(item in dirname.parts for item in self.fe_flag):
            cond = AECTrunk.FE
        elif any(item in dirname.parts for item in self.dt_flag):
            cond = AECTrunk.DT
        else:
            raise RuntimeError("Scenario is not specified.")

        d_mic, fs_1 = audioread(f_mic, sub_mean=True)
        d_ref, fs_2 = audioread(f_ref, sub_mean=True)
        d_sph, fs_3 = audioread(f_sph, sub_mean=True)

        assert fs_1 == fs_2 == fs_3

        if self.align is True:
            tau, _ = gcc_phat(d_mic, d_ref, fs=fs_1, interp=1)
            tau = max(0, int((tau - 0.001) * fs_1))
            d_ref = np.concatenate([np.zeros(tau), d_ref], axis=-1, dtype=np.float32)[
                : d_mic.shape[-1]
            ]

        d_mic = self._split(d_mic, fs_1, slice_idx)
        d_ref = self._split(d_ref, fs_2, slice_idx)
        d_sph = self._split(d_sph, fs_3, slice_idx)

        return (
            torch.from_numpy(d_mic).float(),
            torch.from_numpy(d_ref).float(),
            torch.from_numpy(d_sph).float(),
            torch.tensor(cond).int(),
        )

    def __iter__(self):
        self.pick_idx = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """used for predict api"""
        if self.pick_idx > len(self.f_list):
            raise StopIteration

        mic_fname, ref_fname = self.f_list[self.pick_idx]
        d_mic, fs_1 = audioread(mic_fname, sub_mean=True, target_level=self.norm)
        d_ref, fs_2 = audioread(ref_fname, sub_mean=True, target_level=self.norm)
        assert fs_1 == fs_2

        if fs_1 != self.tgt_fs:
            d_mic = librosa.resample(d_mic, orig_sr=fs_1, target_sr=self.tgt_fs)
            d_ref = librosa.resample(d_ref, orig_sr=fs_2, target_sr=self.tgt_fs)

        if self.align is True:
            tau, _ = gcc_phat(d_mic, d_ref, fs=fs_1, interp=1)
            tau = max(0, int((tau - 0.001) * fs_1))
            d_ref = np.concatenate([np.zeros(tau), d_ref], axis=-1, dtype=np.float32)[
                : d_mic.shape[-1]
            ]

        self.pick_idx += 1

        fname = (
            mic_fname
            if self.return_abspath
            else str(Path(mic_fname).relative_to(self.dir))
        )

        N = min(len(d_ref), len(d_mic))
        return (
            torch.from_numpy(d_mic[:N]).float()[None, :],
            torch.from_numpy(d_ref[:N]).float()[None, :],
            fname,
        )


if __name__ == "__main__":
    dset = AECTrunk(
        "/home/deepnetni/trunk/gene-AEC-train-100-30",
        flist="list.csv",
        patten="**/*mic.wav",
        keymap=("mic", "ref", "sph"),
        align=True,
    )
