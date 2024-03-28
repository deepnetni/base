import os
import sys
from pathlib import Path
import numpy as np
import argparse
from matplotlib import pyplot as plt
from librosa import stft

sys.path.append(str(Path(__file__).parent.parent))

from utils.audiolib import audioread


def parse():
    parser = argparse.ArgumentParser(
        description="compute the aecmos score with input file or directory."
        "\n\nExample: python sigmos.py --src xx --est yy --fs 16000",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src", help="src file or directory", type=str)
    parser.add_argument("--fs", help="sample rate", type=int, default=16000)

    args = parser.parse_args()

    return args


def draw_file(fname: str):
    data, fs = audioread(fname)

    xk = stft(  # B,F,T
        data,
        win_length=512,
        n_fft=512,
        hop_length=256,
        window="hann",
        center=True,
    )  # F,T
    mag = np.abs(xk)
    spec = 10 * np.log10(mag**2 + 1e-10)

    fname, suffix = os.path.splitext(fname)
    fname = fname + ".svg"
    plt.imshow(spec, origin="lower", aspect="auto", cmap="jet")
    # plt.specgram(data, Fs=fs, cmap="jet")
    # plt.show()
    plt.savefig(fname, bbox_inches="tight")


if __name__ == "__main__":
    args = parse()

    if os.path.isfile(args.src):  # file
        draw_file(args.src)
    elif os.path.isdir(args.src):  # dir
        pass
