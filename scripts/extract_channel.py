#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
from itertools import repeat
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.audiolib import audioread, audiowrite


def work(fpath: str, args):
    _, fname = os.path.split(fpath)
    name, _ = fname.split(".")
    data, fs = audioread(fpath)
    assert fs == args.fs
    N = int(args.nlen * fs)
    data = data[:, args.ch]
    nlen = len(data)

    idx = 0
    st = 0
    while nlen > N:
        fout = os.path.join(args.dst, f"{name}_{idx}.wav")
        audiowrite(fout, data[st : st + N], args.fs)
        idx += 1
        st += N
        nlen -= N

    return 1


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="input file or directory", type=str)
    parser.add_argument("--dst", help="output file or directory", type=str)
    parser.add_argument(
        "--nlen", help="output file or directory", type=float, default=10.0
    )
    parser.add_argument(
        "--patten", help="output file or directory", type=str, default="*.wav"
    )
    parser.add_argument("--ch", help="channel", type=int)
    parser.add_argument("--fs", help="channel", default=16000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    if os.path.isfile(args.src):
        pass
    elif os.path.isdir(args.src):
        if not os.path.exists(args.dst):
            os.makedirs(args.dst)
        # assert os.path.isdir(args.dst), f"{args.dst} is not a directory"

        mp.freeze_support()
        p = mp.Pool(processes=30)
        path = Path(args.src)
        wavs = list(map(str, path.glob(f"**/{args.patten}")))
        print(f"len {len(wavs)}")
        out = p.starmap(
            work,
            tqdm(
                zip(wavs, repeat(args)),
                ncols=80,
                total=len(wavs),
                leave=False,
            ),
        )
        v = np.array(out).sum()
    else:
        print("pass")
