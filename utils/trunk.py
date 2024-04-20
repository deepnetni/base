import csv
import os
import re
import sys
from pathlib import Path

import librosa

sys.path.append(str(Path(__file__).parent.parent))

import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence

from utils.audiolib import audioread
from utils.gcc_phat import gcc_phat
from utils.logger import get_logger
from models.conv_stft import STFT
from tqdm import tqdm


def load_f_list(
    fname: str,
    relative: Union[str, Tuple[str, ...], None] = None,
    # dirname: Optional[str] = None,
) -> List:
    """
    relative: str or tuple of str each corresponding to element in the file.
    return: [(f1, f2..), (...)]
    """
    f_list = []
    # fname = os.path.join(dirname, fname) if dirname is not None else fname
    with open(fname, "r+") as fp:
        ctx = csv.reader(fp)
        for element in ctx:
            if isinstance(relative, tuple):
                f_list.append(
                    tuple(map(lambda x, y: os.path.join(x, y), relative, element))
                )
            else:
                f_list.append(
                    element
                    if relative is None
                    else tuple(map(lambda x: os.path.join(relative, x), element))
                )

    return f_list


def save_f_list(
    fname: str,
    f_list: List,
    relative: Union[str, Tuple[str, ...], None] = None,
):
    """
    fname: csv file contains the training files path with format [(f1, f2..), (f1, f2)]
    relative: str or tuple of str each corresponding to element in f_list
    """
    dirname = os.path.dirname(fname)
    os.makedirs(dirname) if not os.path.exists(dirname) else None

    with open(fname, "w+", newline="") as fp:
        writer = csv.writer(fp)
        for f in f_list:
            if isinstance(relative, tuple):
                writer.writerow(
                    tuple(map(lambda x, y: os.path.relpath(x, y), f, relative))
                )
            else:
                writer.writerow(
                    f
                    if relative is None
                    else tuple(map(lambda x: os.path.relpath(x, relative), f))
                )


def load_f_list_len(
    fname: str,
    relative: Union[str, Tuple[str, ...], None] = None,
) -> List:
    """
    relative: str or tuple of str each corresponding to element in the file.
    return: [(f1, f2..), (...)]
    """
    f_list = []
    with open(fname, "r+") as fp:
        ctx = csv.reader(fp)
        for element in ctx:
            element, num = element[:-1], element[-1]
            if isinstance(relative, tuple):
                f_items = list(
                    map(
                        lambda x, y: os.path.join(x, y.replace("\\", "/")),
                        relative,
                        element,
                    )
                )
                # ([...], num)
                f_list.append((f_items, num))

            else:
                if relative is not None:
                    f_items = list(
                        map(
                            lambda x: os.path.join(relative, x.replace("\\", "/")),
                            element,
                        )
                    )
                else:
                    f_items = element
                f_list.append((f_items, num))

                # f_list.append(
                #     element
                #     if relative is None
                #     else tuple(
                #         (list(map(lambda x: os.path.join(relative, x), element)), num)
                #     )
                # )

    return f_list


def save_f_list_len(
    fname: str,
    f_list: List,
    relative: Union[str, Tuple[str, ...], None] = None,
):
    """
    fname: csv file contains the training files path with format [(f1, f2..), (f1, f2)]
    relative: str or tuple of str each corresponding to element in f_list
    """
    dirname = os.path.dirname(fname)
    os.makedirs(dirname) if not os.path.exists(dirname) else None

    with open(fname, "w+", newline="") as fp:
        writer = csv.writer(fp)
        for f, num in f_list:
            if isinstance(relative, tuple):
                writer.writerow(
                    tuple(
                        list(map(lambda x, y: os.path.relpath(x, y), f, relative))
                        + [num]
                    )
                )
            else:
                writer.writerow(
                    f
                    if relative is None
                    else tuple(
                        list(map(lambda x: os.path.relpath(x, relative), f)) + [num]
                    )
                )


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
        csv_dir: str = "csvs",
    ):
        super().__init__()
        self.dir = Path(dirname)
        self.logger = get_logger(dirname)
        self.csv_dir = csv_dir

        flist = (
            os.path.join(csv_dir, os.path.split(dirname)[-1] + ".csv")
            if flist is None
            else flist
        )
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
        assert keymap is None or clean_dirname is None

        if fname is not None and os.path.exists(fname):
            self.logger.info(f"Load flist {fname}")
            f_list = load_f_list(
                fname,
                (
                    str(self.dir),
                    str(self.dir) if clean_dirname is None else clean_dirname,
                ),
            )
        else:
            self.logger.info(f"flist {fname} not exist, regenerating.")
            f_list = []
            f_mic_list = list(map(str, self.dir.glob(patten)))
            # if keymap is not None:
            #     f_sph = f_mic.replace(str(self.dir), self.clean_dir)
            for f_mic in f_mic_list:
                if keymap is not None:
                    # under same directory with different name
                    dirp, f_mic_name = os.path.split(f_mic)
                    f_sph = f_mic_name.replace(*keymap)
                    f_sph = os.path.join(dirp, f_sph)
                elif clean_dirname is not None:
                    # under different directory with same name
                    f_sph = f_mic.replace(str(self.dir), clean_dirname)
                else:
                    raise RuntimeError("keymap and clean_dirname are both None.")

                f_list.append((f_mic, f_sph))

            save_f_list(
                fname,
                f_list,
                (
                    str(self.dir),
                    str(self.dir) if clean_dirname is None else clean_dirname,
                ),
            ) if fname is not None else None
        return f_list

    def __len__(self):
        return len(self.f_list) * self.n_clip

    def _split(self, data, fs, slice_idx):
        st = fs * slice_idx * self.clipL
        ed = st + fs * self.clipL if self.clipL != -1 else None
        return data[st:ed]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        idx, slice_idx = index // self.n_clip, index % self.n_clip
        f_mic, f_sph = self.f_list[idx]

        d_mic, fs_1 = audioread(f_mic, sub_mean=True, target_level=self.norm)
        d_sph, fs_2 = audioread(f_sph, sub_mean=True, target_level=self.norm)
        assert fs_1 == fs_2
        d_mic = self._split(d_mic, fs_1, slice_idx)
        d_sph = self._split(d_sph, fs_2, slice_idx)

        return torch.from_numpy(d_mic).float(), torch.from_numpy(d_sph).float()

    def __iter__(self):
        self.pick_idx = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, str]:
        """used for predict api
        return: data, relative path
        """
        if self.pick_idx < len(self.f_list):
            fname, _ = self.f_list[self.pick_idx]
            data, _ = audioread(fname, sub_mean=True, target_level=self.norm)
            self.pick_idx += 1

            fname = (
                fname
                if self.return_abspath
                else str(Path(fname).relative_to(self.dir.parent))
            )
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
        csv_dir: str = "csvs",
        ne_flag=["NE"],
        dt_flag=["DT"],
        fe_flag=["FE"],
    ):
        super().__init__()
        self.dir = Path(dirname)
        self.logger = get_logger(dirname)
        self.csv_dir = csv_dir

        flist = (
            os.path.join(csv_dir, os.path.split(dirname)[-1] + ".csv")
            if flist is None
            else flist
        )
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

        self.logger.info(f"    Get {dirname} {len(self.f_list)} files.")

    @property
    def dirname(self):
        return str(self.dir)

    def _prepare(
        self, fname: Optional[str], patten: str, keymap: Tuple[str, str, str]
    ) -> List:
        """
        fname: file path of a file list
        """
        if fname is not None and os.path.exists(fname):
            f_list = load_f_list(fname, str(self.dir))
            self.logger.info(f"Load flist {fname}")
        else:
            self.logger.info(f"Regenerating flist {fname} not exist.")
            f_list = []
            f_mic_list = list(map(str, self.dir.glob(patten)))
            for f_mic in f_mic_list:
                dirp, f_mic_name = os.path.split(f_mic)
                f_ref = f_mic_name.replace(keymap[0], keymap[1])
                f_ref = os.path.join(dirp, f_ref)
                f_sph = f_mic_name.replace(keymap[0], keymap[2])
                f_sph = os.path.join(dirp, f_sph)
                f_list.append((f_mic, f_ref, f_sph))

            save_f_list(fname, f_list, str(self.dir)) if fname is not None else None
        return f_list

    def __len__(self):
        return len(self.f_list) * self.n_clip

    def _split(self, data, fs, slice_idx):
        st = fs * slice_idx * self.clipL
        ed = st + fs * self.clipL if self.clipL != -1 else None
        return data[st:ed]

    def __getitem__(self, index) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        idx, slice_idx = index // self.n_clip, index % self.n_clip

        f_mic, f_ref, f_sph = self.f_list[idx]
        # print("##", idx, index, f_mic)

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

        d_mic, fs_1 = audioread(f_mic, sub_mean=True, target_level=self.norm)
        d_ref, fs_2 = audioread(f_ref, sub_mean=True, target_level=self.norm)
        d_sph, fs_3 = audioread(f_sph, sub_mean=True, target_level=self.norm)

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

    def __next__(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str]:
        """used for predict api"""
        if self.pick_idx >= len(self.f_list):
            raise StopIteration

        mic_fname, ref_fname, sph_fname = self.f_list[self.pick_idx]
        d_mic, fs_1 = audioread(mic_fname, sub_mean=True, target_level=self.norm)
        d_ref, fs_2 = audioread(ref_fname, sub_mean=True, target_level=self.norm)
        try:
            d_sph, _ = audioread(sph_fname, sub_mean=True, target_level=self.norm)
        except Exception as e:
            d_sph = None

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
            else str(Path(mic_fname).relative_to(self.dir.parent))
        )

        N = min(len(d_ref), len(d_mic))
        return (
            torch.from_numpy(d_mic[:N]).float()[None, :],
            torch.from_numpy(d_ref[:N]).float()[None, :],
            torch.from_numpy(d_sph[:N]).float()[None, :] if d_sph is not None else None,
            fname,
        )


class CHiMe3(Dataset):
    """
    subdir: train, test, dev
    """

    def __init__(
        self,
        dirname,
        subdir: str = "train",
        nlen: float = 0.0,
        min_len: float = 0.0,
        fs: int = 16000,
        flist: Optional[str] = None,
        csv_dir: str = "csvs",
        seed: Optional[int] = None,
        norm: Optional[int] = None,
        return_abspath: bool = False,
    ) -> None:
        super().__init__()
        self.dir = Path(dirname) / "data" / "audio" / "16kHz" / "isolated" / subdir

        if subdir == "train":
            self.clean_dir = self.dir / "tr05_org"
            self.pattern = ("(CAF|PED|STR|BUS).CH1.wav", "ORG.wav")
        elif subdir == "test":
            self.clean_dir = self.dir / "et05_CH0"
            self.pattern = ("(CAF|PED|STR|BUS).CH1.wav", "BTH.CH0.wav")
            # self.clean_dir = self.dir / "et05_bth"
            # self.pattern = ("(CAF|PED|STR|BUS).CH1.wav", "BTH.CH5.wav")
        elif subdir == "dev":
            self.clean_dir = self.dir / "dt05_CH0"
            self.pattern = ("(CAF|PED|STR|BUS).CH1.wav", "BTH.CH0.wav")
        else:
            raise RuntimeError(f"{subdir} not supported.")

        self.logger = get_logger(f"{dirname}-{subdir}")
        self.csv_dir = csv_dir
        self.N = int(nlen * fs)
        self.minN = int(min_len * fs)

        flist = (
            os.path.join(csv_dir, os.path.split(dirname)[-1] + f"-{subdir}.csv")
            if flist is None
            else flist
        )

        self.f_list = self._prepare(flist)

        if seed is not None:
            random.seed(seed)
            random.shuffle(self.f_list)

        self.norm = norm
        self.return_abspath = return_abspath

        self.logger.info(f"dirname {str(self.dir)} {len(self.f_list)} files.")

    @property
    def dirname(self):
        return str(self.dir)

    def _rearange(self, flist):
        f_list = []

        for f, nlen in flist:
            if self.N != 0 and self.minN != 0:
                st = 0
                nlen = int(nlen)
                if nlen < self.minN:
                    continue

                while nlen - st >= self.N:
                    f_list.append({"f": f, "start": st, "end": st + self.N, "pad": 0})
                    st += self.N

                if nlen - st >= self.minN:
                    f_list.append(
                        {"f": f, "start": st, "end": nlen, "pad": self.N - (nlen - st)}
                    )
            else:
                f_list.append({"f": f, "start": 0, "end": int(nlen), "pad": 0})

        return f_list

    def _sort(self, flist: List) -> List:
        l = []
        for f in tqdm(flist, ncols=80, leave=False):
            d, _ = audioread(f[0], sub_mean=True)
            l.append(len(d))

        tmp = zip(flist, l)
        flist = sorted(tmp, key=lambda x: x[-1], reverse=True)
        # flist, l = zip(*tmp)
        return flist

    def _prepare(self, fname: Optional[str]) -> List:
        """
        fname: file path of a file list
        """

        if fname is not None and os.path.exists(fname):
            self.logger.info(f"Load flist {fname}")
            f_list = load_f_list_len(fname, str(self.dir))
        else:
            self.logger.info(f"Regenerating flist {fname}.")
            f_list = []

            for f in self.dir.iterdir():
                if not f.is_dir() or not f.match("*simu"):
                    continue

                # searching simu directory
                ch1 = list(map(str, f.rglob("*CH1.wav")))
                f_list += [
                    (
                        x,
                        x.replace("CH1.wav", "CH2.wav"),
                        x.replace("CH1.wav", "CH3.wav"),
                        x.replace("CH1.wav", "CH4.wav"),
                        x.replace("CH1.wav", "CH5.wav"),
                        x.replace("CH1.wav", "CH6.wav"),
                        re.sub(
                            *self.pattern,
                            os.path.join(str(self.clean_dir), os.path.split(x)[-1]),
                        ),
                    )
                    for x in ch1
                ]

            f_list = self._sort(f_list)

            save_f_list_len(fname, f_list, str(self.dir)) if fname is not None else None

        return self._rearange(f_list)

    def __len__(self):
        return len(self.f_list)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        el = self.f_list[index]
        f1, f2, f3, f4, f5, f6, f_sph = el["f"]
        st, ed, pd = el["start"], el["end"], el["pad"]

        d1, _ = audioread(f1, target_level=self.norm)
        d2, _ = audioread(f2, target_level=self.norm)  # T,
        d3, _ = audioread(f3, target_level=self.norm)
        d4, _ = audioread(f4, target_level=self.norm)
        d5, _ = audioread(f5, target_level=self.norm)
        d6, _ = audioread(f6, target_level=self.norm)
        d_sph, _ = audioread(f_sph, target_level=self.norm)

        d1 = np.pad(d1[st:ed], (0, pd), "constant", constant_values=0)
        d2 = np.pad(d2[st:ed], (0, pd), "constant", constant_values=0)
        d3 = np.pad(d3[st:ed], (0, pd), "constant", constant_values=0)
        d4 = np.pad(d4[st:ed], (0, pd), "constant", constant_values=0)
        d5 = np.pad(d5[st:ed], (0, pd), "constant", constant_values=0)
        d6 = np.pad(d6[st:ed], (0, pd), "constant", constant_values=0)
        d_sph = np.pad(d_sph[st:ed], (0, pd), "constant", constant_values=0)

        d_mic = np.stack([d1, d2, d3, d4, d5, d6], axis=-1)

        return torch.from_numpy(d_mic).float(), torch.from_numpy(d_sph).float()

    def __iter__(self):
        self.pick_idx = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """used for predict api
        return: data, relative path
        """
        if self.pick_idx < len(self.f_list):
            el = self.f_list[self.pick_idx]
            f1, f2, f3, f4, f5, f6, f_sph = el["f"]
            st, ed, pd = el["start"], el["end"], el["pad"]
            d1, _ = audioread(f1, target_level=self.norm)
            d2, _ = audioread(f2, target_level=self.norm)
            d3, _ = audioread(f3, target_level=self.norm)
            d4, _ = audioread(f4, target_level=self.norm)
            d5, _ = audioread(f5, target_level=self.norm)
            d6, _ = audioread(f6, target_level=self.norm)
            d_sph, _ = audioread(f_sph, target_level=self.norm)
            d1 = np.pad(d1[st:ed], (0, pd), "constant", constant_values=0)
            d2 = np.pad(d2[st:ed], (0, pd), "constant", constant_values=0)
            d3 = np.pad(d3[st:ed], (0, pd), "constant", constant_values=0)
            d4 = np.pad(d4[st:ed], (0, pd), "constant", constant_values=0)
            d5 = np.pad(d5[st:ed], (0, pd), "constant", constant_values=0)
            d6 = np.pad(d6[st:ed], (0, pd), "constant", constant_values=0)
            d_sph = np.pad(d_sph[st:ed], (0, pd), "constant", constant_values=0)

            data = np.stack([d1, d2, d3, d4, d5, d6], axis=-1)
            self.pick_idx += 1

            fname = (
                f1
                if self.return_abspath
                else str(Path(f1).relative_to(self.dir.parent))
            )
            return (
                torch.from_numpy(data).float()[None, :],
                torch.from_numpy(d_sph).float()[None, :],
                fname,
            )
        else:
            raise StopIteration


def clip_to_shortest(batch: List):
    batch.sort(key=lambda x: x[0].shape[-1], reverse=True)

    for x in batch:
        print(x[0].shape, x[1].shape)
    print("done")


def pad_to_longest(batch):
    """
    batch: [(data, label), (...), ...], B,T,C
    the input data, label must with shape (T,C) if time domain
    """
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)  # data length

    seq_len = [d.size(0) for d, _ in batch]
    data, label = zip(*batch)  # B,T,C
    data = pad_sequence(data, batch_first=True).float()
    label = pad_sequence(label, batch_first=True).float()

    # data = pack_padded_sequence(data, seq_len, batch_first=True, enforce_sorted=True)

    return data, label, torch.tensor(seq_len)


if __name__ == "__main__":
    # from torchmetrics.functional.audio import signal_noise_ratio as SDR
    from torchmetrics.functional.audio import signal_distortion_ratio as SDR
    from tqdm import tqdm
    import fast_bss_eval
    from pesq import pesq

    # dset = AECTrunk(
    #     "/home/deepnetni/trunk/gene-AEC-train-100-30",
    #     flist="list.csv",
    #     patten="**/*mic.wav",
    #     keymap=("mic", "ref", "sph"),
    #     align=True,
    # )

    # dset = NSTrunk(
    #     "/home/deepnetni/trunk/vae_dns_p07",
    #     flist="list.csv",
    #     patten="**/*_nearend.wav",
    #     # keymap=("nearend.wav", "target.wav"),
    #     clean_dirname="/home/deepnetni/trunk/vae_dns",
    # )

    # dset = CHiMe3(
    #     "/home/deepnetni/trunk/CHiME3",
    #     subdir="train",
    #     nlen=5.0,
    #     min_len=1.0,
    # )
    dset = CHiMe3(
        "E:\datasets\CHiME3",
        subdir="test",
    )
    train_loader = DataLoader(
        dset,
        batch_size=1,
        pin_memory=True,
        shuffle=True,
        collate_fn=pad_to_longest,
    )

    sdr_l = []
    pesq_sc = []
    for mic, sph, nlen in tqdm(train_loader):
        mic = mic[..., 4]
        sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(sph.numpy(), mic.numpy())
        sdr_l.append(sdr)
        # sdr_l.append(SDR(preds=mic, target=sph, zero_mean=True))
        # pesq_sc.append(pesq(16000, sph[0].numpy(), mic[0].numpy(), "wb"))

    print(np.array(sdr_l).mean())
    # print(np.array(pesq_sc).mean())
    # rnn = nn.LSTM(input_size=6, hidden_size=10, num_layers=1, batch_first=True)
    # out, (h, c) = rnn(inp)
    # out, len = pad_packed_sequence(out, batch_first=True)
    # print(out.shape)
