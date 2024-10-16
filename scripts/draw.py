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


def draw_file(fname: str, nframe=512, nhop=256):
    data, fs = audioread(fname)
    data = data[..., 0] if data.ndim > 1 else data
    # data = data[int(0.3 * fs) : int(1.3 * fs)]
    nbin = nhop + 1

    xlabel = np.arange(0, fs // 2 + 1, 1600 if fs <= 16000 else 3000)  # 1000, 2000, ..., Frequency
    xticks = (nbin - 1) * xlabel * 2 // fs

    xk = stft(  # B,F,T
        data,
        win_length=nframe,
        n_fft=nframe,
        hop_length=nhop,
        window="hann",
        center=True,
    )  # F,T complex
    mag = np.abs(xk)
    spec = 10 * np.log10(mag**2 + 1e-10)

    fname, suffix = os.path.splitext(fname)
    fname = fname + ".svg"
    # _, fname = os.path.split(fname)
    print(f"out:{fname}")
    # plt.imshow(spec, origin="lower", aspect="auto", cmap="viridis")
    # plt.imshow(spec, origin="lower", aspect="auto", cmap="winter")
    plt.imshow(spec, origin="lower", aspect="auto", cmap="jet")
    plt.xticks([])
    plt.yticks([])
    # plt.specgram(data, Fs=fs, cmap="jet")
    # plt.ylabel("Frequency / Hz")
    # plt.xlabel("Frame Index")
    # plt.yticks(ticks=xticks, labels=xlabel)
    # plt.axis("off")
    plt.savefig(fname, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    args = parse()

    if os.path.isfile(args.src):  # file
        draw_file(args.src)
    elif os.path.isdir(args.src):  # dir
        for f in os.listdir(args.src):
            if f.startswith(".") or not f.endswith(".wav"):
                continue

            f = os.path.join(args.src, f)
            draw_file(f)
