import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def segment(fpath: str, ntime: float = 3.0, fs: int = 16000):
    # data: [ch, T]
    data, sr = librosa.load(fpath, sr=fs, mono=False)
    assert sr == fs
    if data.ndim == 1:  # mono
        nChs = 1
        nL = len(data)
    elif data.ndim == 2:  # stere
        nChs, nL = data.shape
    else:
        raise RuntimeError("shape error")

    nLen = int(ntime * fs)
    # nSegs = (nL - nLen) // nLen
    nSegs_idx = np.arange(0, nL - nLen, nLen)
    nSegs_idx = nSegs_idx.reshape(-1, 1) + np.arange(nLen)
    if nChs != 1:
        segs = data[:, nSegs_idx]  # ch, segs, T
    else:
        segs = data[nSegs_idx]  # ch, segs, T
    segs = segs.reshape(-1, nLen)

    return segs


def run(inp: str, out: str, ntime: float, fs: int):
    if os.path.isdir(inp):
        dirname = Path(inp)
        dirout = Path(out)
        if not dirout.exists():
            os.makedirs(out)

        files = list(dirname.rglob("[!.]*.wav"))
        for fpath in tqdm(files, ncols=120):
            fname, suffix = fpath.stem, fpath.suffix

            segs = segment(str(fpath), ntime, fs)

            for idx in range(len(segs)):
                d = segs[idx]
                wname = dirout / f"{fname}_{idx}{suffix}"
                sf.write(wname, d, fs)
    elif os.path.isfile(inp):
        pass
    else:
        raise RuntimeError("type error")


def parse():
    parser = argparse.ArgumentParser(
        description="segment wav audios."
        "\n\nExample: python segment_speech.py --src xx --out yy --ntime 10.0",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src", help="src file or directory", type=str)
    parser.add_argument("--out", help="out file or directory", type=str)
    parser.add_argument("--ntime", help="segment time", type=float, default=10.0)
    parser.add_argument("--fs", help="sample rate", type=int, default=16000)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse()
    # fname = "/home/deepni/disk/noise_3dquest_48/Cafeteria_Noise_binaural ( 0.00-30.00 s).wav"
    # fname = "/home/deepni/disk/dns_48/english/clean_fullband/read_speech/book_00701_chp_0012_reader_02603_4_seg_1.wav"
    # segment(fname)
    run(args.src, args.out, args.ntime, args.fs)
