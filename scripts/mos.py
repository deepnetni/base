import argparse
import multiprocessing as mp
import numpy as np
import os
import sys
from itertools import repeat
from pathlib import Path
from typing import Dict

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from utils.audiolib import audioread
from utils.metrics import compute_pesq, compute_stoi, compute_si_snr


def work(
    sph_path,
    enh_path,
):
    sph, fs = audioread(sph_path)
    enh, fs_ = audioread(enh_path)

    assert fs == fs_

    pesq_sc = compute_pesq(sph, enh, fs)
    stoi_sc = compute_stoi(sph, enh, fs)
    sisnr_sc = compute_si_snr(sph, enh)
    return (pesq_sc, stoi_sc, sisnr_sc)  # (echo_mos, other_mos)


def parse():
    parser = argparse.ArgumentParser(
        description="compute the pesq,stoi,si-snr mos score with input file or directory."
        "\n\nExample: python mos.py --src xx --est yy",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src", help="src file or directory", type=str)
    parser.add_argument("--est", help="dst file or directory", type=str)
    # parser.add_argument("--fs", help="sample rate", type=int, default=16000)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse()
    if os.path.isfile(args.src):
        pass

    elif os.path.isdir(args.src):
        mp.freeze_support()
        p = mp.Pool(processes=30)
        subdirs = os.listdir(args.src)
        for sc in subdirs:
            path = Path(args.src) / sc
            sph_list = list(map(str, path.glob("**/*sph.wav")))
            enh_list = list(map(lambda f: f.replace("sph.wav", "mic.wav"), sph_list))
            enh_list = list(map(lambda f: f.replace(args.src, args.est), enh_list))

            if len(sph_list) == 0:
                continue

            # print("##", sc, len(sph_list), len(enh_list))
            out = p.starmap(
                work,
                tqdm(
                    zip(sph_list, enh_list),
                    ncols=80,
                    total=len(sph_list),
                    leave=False,
                ),
            )
            v, n = np.array(out).sum(axis=0), len(out)
            print(
                f"{sc} {len(out)} PESQ, STOI, SI-SNR:\t{v[0]/n:.4f}, {v[1]/n:.4f}, {v[2]/n:.4f}."
            )

        p.close()
    else:
        raise RuntimeError(f"args.src {args.src} not a file or directory")
