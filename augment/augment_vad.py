#!/usr/bin/env python3
import argparse
import yaml
import pickle

from synthesizer.vadLibriSpeech import NoisyVADGenerator
from utils.mp_decoder import mpMap
from utils.audiolib import audiowrite


@mpMap()
def generate(idx, conf, outdir):
    gene = NoisyVADGenerator(conf)
    meta = gene.sample()

    mic = meta["mic"]
    mic = mic - mic.mean()
    vad = meta["vad"]
    cln = meta["target"]
    cln = cln - cln.mean()
    fs = meta["fs"]
    meta_rir = meta["rir"]

    audiowrite(f"{outdir}/{idx}_target.wav", cln, fs)
    audiowrite(f"{outdir}/{idx}_mic.wav", mic, fs)
    audiowrite(f"{outdir}/{idx}_vad.wav", vad, fs)

    if meta_rir is not None:
        with open(f"{outdir}/{idx}_vad.pkl", "wb") as fp:
            pickle.dump(meta_rir, fp)

    return 1


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", help="", type=str, default="template/vad_librispeech.yaml")
    parser.add_argument("--hour", help="", type=int)
    parser.add_argument("--out", help="output directory", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    with open(args.yaml) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    nfile = int(args.hour * 3600 // cfg["synth_duration"])

    out = generate(
        range(nfile),
        conf=args.yaml,
        outdir=args.out if args.out is not None else cfg["synth_output_dir"],
    )
    print(sum(out))
