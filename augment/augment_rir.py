import argparse
import multiprocessing as mp
import os

# from itertools import repeat
import pickle

import numpy as np
import yaml
from tqdm import tqdm

from synthesizer.rirGenerator import RIRGenerator
from utils.audiolib import audiowrite
from utils.mp_decoder import mpStarMap
import matplotlib

matplotlib.use("tkagg")
from matplotlib import pyplot as plt


@mpStarMap()
def worker(filenum, outdir, conf):
    gene = RIRGenerator(**conf)
    meta = gene.sample()

    fname = f"{outdir}/{filenum}_m{conf['array']['mics']}_rir.pkl"
    with open(fname, "wb") as fp:
        pickle.dump(meta, fp)

    return 1


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", help="config yaml file", default="./template/spatialReverb.yaml")
    parser.add_argument("--fs", help="sample rate", type=int, default=16000)
    parser.add_argument("--num", help="number of generated rirs", type=int)
    parser.add_argument("--out", help="output directory", type=str)

    parser.add_argument("--show", help="show pkl contents", type=str, default=None)

    args = parser.parse_args()
    return args


def show(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(data.keys())

    h = data["h"]  # M,T
    for i in range(h.shape[0]):
        plt.subplot()
        plt.plot(h[i])
    plt.show()


if __name__ == "__main__":
    args = parse()

    if not args.show:
        with open(args.yaml) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        assert args.num
        out_dir = args.out if args.out is not None else cfg["rir_configs"]["rir_out"]
        os.makedirs(out_dir) if not os.path.exists(out_dir) else None

        worker(np.arange(args.num), outdir=out_dir, conf=cfg["rir_configs"])
    else:
        show(args.show)
