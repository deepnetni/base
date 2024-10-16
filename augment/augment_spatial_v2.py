import argparse
import multiprocessing as mp
import os
from itertools import repeat
import pickle

import numpy as np
import yaml
from tqdm import tqdm

from synthesizer.spatialGenerator import SpatialGenerater
from utils.audiolib import audiowrite
from utils.mp_decoder import mpStarMap


@mpStarMap(10)
def worker(filenum, outdir, conf, fs):
    gene = SpatialGenerater(conf)
    audio = gene.generate()

    audiowrite(f"{outdir}/{filenum}_target.wav", audio["target"] - audio["target"].mean(), fs)

    mic = audio["mic"].T  # to T,C
    audiowrite(f"{outdir}/{filenum}_mic.wav", mic - mic.mean(axis=0, keepdims=True), fs)
    noise = audio["noise"].T  # to T,C
    audiowrite(f"{outdir}/{filenum}_noise.wav", noise, fs)
    reverb = audio["reverb"].T  # to T,C
    audiowrite(f"{outdir}/{filenum}_reverb.wav", reverb, fs)

    with open(f"{outdir}/{filenum}_meta.pkl", "wb") as fp:
        pickle.dump(audio["meta"], fp)

    return 1


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml", help="config yaml file", default="./template/spatialGenerator.yaml"
    )
    parser.add_argument("--num", help="number of generated rirs", type=int)
    parser.add_argument("--rir", help="", action="store_true")

    parser.add_argument("--wav", help="", action="store_true")
    parser.add_argument("--hour", help="", type=int)
    parser.add_argument("--out", help="output directory", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    with open(args.yaml) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.rir:
        assert args.num
        out_dir = args.out if args.out is not None else cfg["rir_configs"]["rir_out"]
        os.makedirs(out_dir) if not os.path.exists(out_dir) else None

        gene = SpatialGenerater(args.yaml)
        gene.generate_rirs(args.num, out_dir)
    elif args.wav:
        os.makedirs(args.out) if not os.path.exists(args.out) else None
        # mp.freeze_support()
        # p = mp.Pool(processes=30)

        assert args.out and args.hour
        nfile = int(args.hour * 3600 // cfg["synth_duration"])
        out = {}
        worker(range(nfile), outdir=args.out, conf=args.yaml, fs=cfg["synth_sampling_rate"])
    else:
        raise RuntimeError("Choose one option.")
