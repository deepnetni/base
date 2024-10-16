import argparse
import json
import multiprocessing as mp
import os
import sys
from itertools import repeat
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_distortion_ratio as SDR
from tqdm import tqdm

from utils.audiolib import audioread
from utils.composite_metrics import eval_composite
from utils.metrics import *
from utils.mp_decoder import mpStarMap
from l3das.metrics import task1_metric


def parse():
    parser = argparse.ArgumentParser(
        description="compute the metrics score with input file or directory."
        "\n\nExample: python compute_metrics.py --src xx --out yy --sisnr ",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src", help="src file or directory", type=str)
    parser.add_argument("--out", help="dst file or directory", type=str)
    parser.add_argument("--fs", help="dst file or directory", type=int, default=16000)

    parser.add_argument("--sisnr", help="si-snr", action="store_true")
    parser.add_argument("--snr", help="snr", action="store_true")
    parser.add_argument("--sdr", help="sdr", action="store_true")
    parser.add_argument("--pesqw", help="wide band pesq", action="store_true")
    parser.add_argument("--pesqn", help="narrow band pesq", action="store_true")
    parser.add_argument("--stoi", help="stoi", action="store_true")
    parser.add_argument("--asr", help="csig, cbak, covl", action="store_true")

    parser.add_argument("--l3das", help="l3das", action="store_true")

    parser.add_argument("--excel", help="excel name", default=None)

    parser.add_argument("--pattern", help="noisy files", default=".wav")

    parser.add_argument(
        "--map", help="pattern between mic and sph, --map mic.wav sph.wav", nargs="+"
    )
    parser.add_argument("--metrics", help="compute multi-metrics", nargs="+")

    # the output folder contain multi-types
    parser.add_argument(
        "--sub",
        help="the output dir contains sub-class each corresponds to the source files",
        action="store_true",
    )

    args = parser.parse_args()

    return args


def to_excel(data: dict, excel_f: str):
    """
    data:{'caf': [{'pesq':v, 'stoi':v,...}, {...}], 'bus':[], 'ped':[]}
    """
    if excel_f is None:
        return

    tag = excel_f.split(".")[0]

    save = {}
    for env, metrics in data.items():
        # env: caf, bus, ...
        # metrics: [{'pesq':v, ...}, {...}]

        dic = {}  # {'pesq': [...], 'stoi': [...]}
        for d in metrics:
            for k, v in d.items():
                dic.setdefault(k, []).append(v)

        for k, v in dic.items():
            # k: pesq, stoi, ...
            # v: list[...]
            save.setdefault(k, {"type": [tag] * len(v)}).update({env: v})
            save.setdefault(f"all_{k}", []).extend(v)

    # save: {'pesq':{'caf':[..], 'bus':[..]}, "stoi":{..}}
    with pd.ExcelWriter(excel_f) as writer:
        for m, d in save.items():
            df = pd.DataFrame(d)
            # df.to_excel(writer, sheet_name=m, index=False, header=None)
            df.to_excel(writer, sheet_name=m, index=False)


@mpStarMap(5)
def compute_score(spath, epath, args):
    """
    return: [v,v..] or [{'v1':v,'v2'..}, {...}]
    """
    # print(spath, epath)
    edata, _ = audioread(epath)
    sdata, _ = audioread(spath)
    # print(edata.shape) if edata.ndim > 1 else None
    # edata = edata.mean(-1) if edata.ndim > 1 else edata
    edata = edata[:, 0] if edata.ndim > 1 else edata
    sdata = sdata[: len(edata)]
    # edata = np.concatenate([np.zeros(15), sdata[: len(edata) - 15]])

    if args.sisnr:
        return compute_si_snr(sdata, edata)
    elif args.snr:
        return compute_snr(sdata, edata)
    elif args.pesqw:
        return compute_pesq(sdata, edata, fs=args.fs, mode="wb")
    elif args.pesqn:
        return compute_pesq(sdata, edata, fs=args.fs, mode="nb")
    elif args.stoi:
        return compute_stoi(sdata, edata, fs=args.fs)
    elif args.asr:
        return eval_composite(sdata, edata, sample_rate=args.fs)
    elif args.sdr:
        edata = torch.from_numpy(edata)
        sdata = torch.from_numpy(sdata)
        sdr = SDR(preds=edata, target=sdata)
        # sisdr = si_sdr(preds=edata, target=sdata)
        return sdr.cpu().numpy()
    elif args.l3das:
        score, wer, stoi_sc = task1_metric(sdata, edata)
        return {"score": score, "wer": wer, "l3das_stoi": stoi_sc}

    elif args.metrics:
        metrics = {}

        if "sisnr" in args.metrics:
            metrics["sisnr"] = compute_si_snr(sdata, edata)
        if "snr" in args.metrics:
            metrics["snr"] = compute_snr(sdata, edata)
        if "pesqw" in args.metrics:
            metrics["pesqw"] = compute_pesq(sdata, edata, fs=args.fs, mode="wb")
        if "pesqn" in args.metrics:
            metrics["pesqn"] = compute_pesq(sdata, edata, fs=args.fs, mode="nb")
        if "stoi" in args.metrics:
            metrics["stoi"] = compute_stoi(sdata, edata, fs=args.fs)
        if "sdr" in args.metrics:
            edata = torch.from_numpy(edata)
            sdata = torch.from_numpy(sdata)
            sdr = SDR(preds=edata, target=sdata)
            # sisdr = si_sdr(preds=edata, target=sdata)
            metrics["sdr"] = sdr.cpu().numpy()
        if "asr" in args.metrics:
            metrics.update(eval_composite(sdata, edata, sample_rate=args.fs))

        if "l3das" in args.metrics:
            score, wer, stoi_sc = task1_metric(sdata, edata)
            metrics.update({"l3das_score": score, "l3das_wer": wer * 100, "l3das_stoi": stoi_sc})

        return metrics

    else:
        raise RuntimeError("metric not supported.")


def compute_box(result: List) -> Dict:
    """
    result: [v,v..] or [{'v1':v,'v2'..}, {...}]
    return: {'m1':v, 'm2':v2,...}
    """
    if len(result) == 0:
        raise RuntimeWarning("empty result.")

    if isinstance(result[0], float):
        sc = np.array(result)
        sc = [
            np.percentile(sc, 25).round(4),
            np.percentile(sc, 50).round(4),
            np.percentile(sc, 75).round(4),
        ]
    elif isinstance(result[0], dict):
        dic = {}
        for d in result:
            for k, v in d.items():
                dic.setdefault(k, []).append(v)

        sc = {}
        for k, v in dic.items():
            d = np.array(v)
            d = [
                np.percentile(d, 25).round(4),
                np.percentile(d, 50).round(4),
                np.percentile(d, 75).round(4),
            ]
            sc[k] = d
    else:
        raise RuntimeError("not supported.")

    return sc


def compute_mean_std(result: List) -> Dict:
    """
    result: [v,v..] or [{'v1':v,'v2'..}, {...}]
    return: {'m1':v, 'm2':v2,...}
    """
    if len(result) == 0:
        raise RuntimeWarning("empty result.")

    if isinstance(result[0], float):
        sc = np.array(result)
        sc = [sc.mean().round(3), sc.std().round(2)]
    elif isinstance(result[0], dict):
        dic = {}
        for d in result:
            for k, v in d.items():
                dic.setdefault(k, []).append(v)

        sc = {}
        for k, v in dic.items():
            d = np.array(v)
            d = [d.mean().round(3), d.std().round(2)]
            sc[k] = d
    else:
        raise RuntimeError("not supported.")

    return sc


if __name__ == "__main__":
    """
    python compute_metrics.py --out ../trained_mcse_spdns/pred_mcse_50/test --src ~/datasets/spatialReverbNoise/test --map target.wav mic.wav --pesq

    Note: the `out` and `src` given the top different path, which means the ungiven path is the same.
        e,g,. `--out /a/b --src /c/d` means /a/b/e/f/..wav corresponds to /c/d/e/f/..wav
    """
    args = parse()

    if os.path.isfile(args.src):
        score = compute_score([args.src], [args.out], args=args)
        score = compute_mean_std(score)

    elif os.path.isdir(args.src) and not args.sub:
        subitem_full = {}
        # dst_l = list(map(str, Path(args.out).glob("**/*.wav")))
        dst_l = list(map(str, Path(args.out).glob(f"**/*{args.pattern}")))
        src_l = [f.replace(args.out, args.src) for f in dst_l]

        if args.map is not None:
            src_l = [f.replace(*args.map) for f in src_l]

        score_l = compute_score(src_l, dst_l, args=args)
        score = compute_mean_std(score_l)

        subitem_full["allenv"] = score_l if args.excel else None
        to_excel(subitem_full, args.excel)
    elif os.path.isdir(args.src) and args.sub:
        # out/sub1->src, out/sub2->src
        subitem = {}
        full = []
        subitem_full = {}

        for subd in os.listdir(args.out):
            if subd.startswith("."):
                continue

            dirp = Path(args.out) / subd
            dst_l = list(map(str, dirp.glob("**/*.wav")))
            src_l = [f.replace(str(dirp), args.src) for f in dst_l]

            if args.map is not None:
                dst_l = [f.replace(*args.map) for f in dst_l]

            score_l = compute_score(src_l, dst_l, args=args)
            score = compute_mean_std(score_l)
            # scbox = compute_box(score_l)
            subitem[subd] = score
            # subitem[subd + "-box"] = scbox
            subitem_full[subd] = score_l if args.excel else None
            full += score_l
        score = dict(total=compute_mean_std(full))
        score.update(subitem)

        to_excel(subitem_full, args.excel)
    else:
        raise RuntimeError("")

    if args.sub:
        for k, v in score.items():
            print(k)
            print(v)
    else:
        print(score)
