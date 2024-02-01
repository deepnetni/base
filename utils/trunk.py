import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

from typing import Optional, Tuple
from .audiolib import audioread
from .logger import get_logger

from models.STFT import STFT


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
        clean_dirname: Optional[str] = None,
        data_len: int = -1,
        clip_len: int = -1,
        seed: Optional[int] = None,
        norm: Optional[int] = None,
        return_abspath: bool = False,
    ):
        super().__init__()
        self.dir = Path(dirname)

        self.f_list = list(self.dir.glob(patten))
        if seed is not None:
            random.seed(seed)
            random.shuffle(self.f_list)

        self.keymap = keymap
        self.clean_dir = clean_dirname
        self.logger = get_logger(dirname)
        self.norm = norm
        self.return_abspath = return_abspath
        self.dataL = data_len
        self.clipL = clip_len
        assert data_len % clip_len == 0
        assert data_len != -1 and clip_len != -1 or data_len == -1 and clip_len == -1
        self.n_clip = int(data_len // clip_len)

        self.logger.info(f"Get {dirname} {len(self.f_list)} mic files.")

    def __len__(self):
        return len(self.f_list) * self.n_clip

    def _split(self, data, fs, slice_idx):
        st = fs * slice_idx * self.clipL
        ed = st + fs * self.clipL if self.clipL != -1 else None
        return data[st:ed]

    def __getitem__(self, index) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        idx, slice_idx = index // self.n_clip, index % self.n_clip
        f_mic = str(self.f_list[idx])
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
