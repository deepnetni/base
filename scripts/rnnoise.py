import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from itertools import repeat

sys.path.append(str(Path(__file__).parent.parent))

from utils.audiolib import wav2pcm, pcm2wav


def work(fname: str, src: str, dst: str, fs):
    fout = fname.replace(src, dst)

    if not os.path.exists(dst):
        os.makedirs(dst)

    pcm_p = fout.split(".")[0] + ".pcm"
    pcm_p_post = fout.split(".")[0] + "_rnn.pcm"
    wav2pcm(fname, pcm_p)
    # rnnoise
    try:
        store = os.getcwd()
        os.chdir("./rnnoise_48")
        # os.system(f"./rnnoise_48/rnnoise_demo {pcm_p} {pcm_p_post}")
        os.system(f"./rnnoise_demo {pcm_p} {pcm_p_post}")
        os.chdir(store)
    except Exception as e:
        print(f"{fname} with error ", e)
        os.remove(pcm_p)
        os.remove(pcm_p_post)
        return 0

    pcm2wav(pcm_p_post, fout, fs=fs)
    os.remove(pcm_p)
    os.remove(pcm_p_post)

    return 1


def process(src: str, dst: str, fs: int):
    if os.path.isdir(src):
        files = list(map(str, Path(src).glob("**/*.wav")))
        mp.freeze_support()
        p = mp.Pool(processes=30)

        out = list(
            p.starmap(
                work,
                tqdm(
                    zip(files, repeat(src), repeat(dst), repeat(fs)),
                    ncols=80,
                    total=len(files),
                ),
            )
        )
        return sum(out)
    elif os.path.isfile(src):
        pcm_p = src.split(".")[0] + ".pcm"
        pcm_post_p = src.split(".")[0] + "_rnn.pcm"
        wav2pcm(src, pcm_p)
        try:
            os.system(f"./rnnoise_48/rnnoise_demo {pcm_p} {pcm_post_p}")
        except Exception as e:
            os.remove(pcm_p)
            os.remove(pcm_post_p)
            print(f"{src} with error ", e)
            return 0

        pcm2wav(pcm_post_p, dst, fs=fs)
        os.remove(pcm_p)
        os.remove(pcm_post_p)
    else:
        raise RuntimeError(f"{src} is not a file or directory.")


def parse():
    parser = argparse.ArgumentParser(
        description="using rnnoise to post process the label data."
        "\n\nExample: python sigmos.py --src xxx --dst xxx --fs 48000",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src", help="src directory", type=str)
    parser.add_argument("--dst", help="dst directory", type=str)
    parser.add_argument("--fs", help="sample rate", type=int, default=48000)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse()
    process(args.src, args.dst, args.fs)
