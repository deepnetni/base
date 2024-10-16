import argparse
import yaml
import multiprocessing as mp
import shutil
import os
import soundfile as sf
import numpy as np
from itertools import repeat

from tqdm import tqdm

# from synthesizer.synthesizer import Synthesizer
from synthesizer.synthesizer_bgnoise import Synthesizer


def aug(generator: Synthesizer, filenum: int, outdir, fs):
    audio = generator.generate()
    # print("#", len(audio["target"]), audio["target"][0], os.getpid())

    sf.write(f"{outdir}/{filenum}_target.wav", audio["target"], fs)
    sf.write(f"{outdir}/{filenum}_nearend.wav", audio["nearend"], fs)
    # sf.write(f"{outdir}/{filenum}_mic.wav", audio["mic"], fs)

    return 1


def aug_2(yaml, filenum: int, outdir, fs):
    generator = Synthesizer(yaml)
    audio = generator.generate()

    sf.write(f"{outdir}/{filenum}_target.wav", audio["target"] - audio["target"].mean(), fs)
    sf.write(f"{outdir}/{filenum}_nearend.wav", audio["nearend"], fs)
    # sf.write(f"{outdir}/{filenum}_mic.wav", audio["mic"], fs)

    return 1


def parser():
    parser = argparse.ArgumentParser(
        description="python augment_sig.py --yaml xx.yaml --outdir /yy/zz --time 50"
    )
    parser.add_argument(
        "--time", help="augment audio length based on hours", type=float, default=100.0
    )
    parser.add_argument(
        "--yaml",
        help="path to the configure yaml file",
        type=str,
        default="./demo_synthesizer/synthesizer_config.yaml",
    )
    parser.add_argument("--outdir", help="out dirname")

    return parser.parse_args()


if __name__ == "__main__":
    """
    Example:
        >>> python augment_sig.py --yaml xx.yaml --outdir /home/deepni/datasets/dns_p09_50h/ --time 50
    """
    args = parser()

    # with open(args.yaml) as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)

    worker = Synthesizer(args.yaml)
    fs = worker.cfg["onlinesynth_sampling_rate"]
    nlen = worker.cfg["onlinesynth_duration"]
    nfile = int(args.time * 3600 // (nlen))

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    else:
        shutil.rmtree(args.outdir)
        os.makedirs(args.outdir)

    mp.freeze_support()
    p = mp.Pool(processes=30)

    # worker_l = [Synthesizer(args.yaml) for _ in range(30)] # not work, still on the main thread
    out = {}
    out["aug"] = list(
        p.starmap(
            aug_2,
            tqdm(
                # zip(repeat(worker), range(nfile), repeat(args.outdir), repeat(fs)),
                zip(repeat(args.yaml), range(nfile), repeat(args.outdir), repeat(fs)),
                ncols=80,
                total=nfile,
            ),
        )
    )

    num = np.array(out["aug"])
    print(f"Generating {num.sum() * nlen / 3600:.2f} hours.")
