import argparse
import json
import multiprocessing as mp
import os
import sys
from itertools import repeat
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from utils.audiolib import audioread
from models.sigmos.sigmos import SigMOS
import pandas as pd
import re


def parse():
    parser = argparse.ArgumentParser(
        description="compute the sigmos score with input file."
        "\n\nExample: python sigmos.py --src xxx.wav --fs 48000",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src", help="src file or directory", type=str)
    parser.add_argument("--fs", help="sample rate", type=int, default=48000)
    parser.add_argument("--out", help="out json file path", default="")

    args = parser.parse_args()

    return args


def work(fname, fs):
    sig = SigMOS()
    audio, fs_ = audioread(fname)
    assert fs == fs_
    score = sig.run(audio, sr=fs)
    # print("##", fname)
    return fname, score


if __name__ == "__main__":
    args = parse()
    if os.path.isfile(args.src):
        # audio, fs = audioread(args.src)
        # score = sig.run(audio, sr=fs)
        _, score = work(args.src, args.fs)

        for k, v in score.items():
            print(f"{k: <12}", f"{v:.4f}")
    elif os.path.isdir(args.src):
        # sig = SigMOS()
        files = list(map(str, Path(args.src).glob("**/*nearend.wav")))
        mp.freeze_support()
        p = mp.Pool(processes=30)
        out = p.starmap_async(
            work,
            tqdm(zip(files, repeat(args.fs)), ncols=80, total=len(files)),
            # zip(files, repeat(args.fs)),
        )
        p.close()
        score = {}
        print("# waiting results...")
        # for f in tqdm(files, ncols=80):
        #     audio, fs = audioread(f)
        #     mos = sig.run(audio, sr=args.fs)
        #     for k, v in mos.items():
        #         hs, hn = score.get(k, [0.0, 0])
        #         hn += 1
        #         hs += v
        #         score.update({k: [hs, hn]})
        record = {}

        for fname, mos in out.get():
            for k, v in mos.items():
                hs, hn = score.get(k, [0.0, 0])
                hn += 1
                hs += v
                score.update({k: [hs, hn]})
            record.update(
                {os.path.basename(fname): {k: round(v, 4) for k, v in mos.items()}}
            )

        for k, v in score.items():
            val, num = v
            print(f"{k: <12}", f"{val/num:.4f}")

        if args.out != "":
            avg = {"sigmos_avg": {k: round(v[0] / v[1], 4) for k, v in score.items()}}
            record = dict(avg, **record)
            with open(args.out, "w") as fp:
                json.dump(record, fp, indent=2)

            ctx = {}
            for k, v in record.items():
                idx = re.findall("^\d+\.?\d*", k)
                if idx != []:
                    idx = int(idx[0])
                    ctx.update({idx: dict(name=k, **v)})

            # print(ctx)
            df = pd.DataFrame(ctx).T
            df = df.sort_index()
            fout = os.path.splitext(args.out) + ".xlsx"
            df.to_excel(fout, index=False)
    else:
        raise RuntimeError(f"args.src {args.src} not a file or directory")
