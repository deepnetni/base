import argparse
import multiprocessing as mp
import numpy as np
import os
import sys
from itertools import repeat
from pathlib import Path
from typing import Dict
import json

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from utils.audiolib import audioread
from utils.AECMOS.AECMOS_local.aecmos import AECMOSEstimator

fpath = Path(__file__).parent.parent
mos_p = fpath / "utils/AECMOS/AECMOS_local/Run_1663915512_Stage_0.onnx"
mos_p_48k = fpath / "utils/AECMOS/AECMOS_local/Run_1668423760_Stage_0.onnx"


def parse():
    parser = argparse.ArgumentParser(
        description="compute the aecmos score with input file or directory."
        "\n\nExample: python sigmos.py --src xx --est yy --fs 16000",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src", help="src file or directory", type=str)
    parser.add_argument("--est", help="dst file or directory", type=str)
    parser.add_argument("--fs", help="sample rate", type=int, default=16000)

    args = parser.parse_args()

    return args


def work(
    mic_path,
    ref_path,
    enh_path,
    fs,
    scenario: Dict = {
        "dt": ["doubletalk"],
        "nst": ["nearend_singletalk", "nearend-singletalk"],
        "st": ["farend_singletalk", "farend-singletalk"],
    },
):
    if fs == 16000:
        model_path = str(mos_p)
    else:
        model_path = str(mos_p_48k)

    aecmos = AECMOSEstimator(model_path)
    lpb_sig, mic_sig, enh_sig = aecmos.read_and_process_audio_files(
        ref_path, mic_path, enh_path
    )
    # print("###", len(lpb_sig), len(enh_sig))

    # sc_type = [k for k, v in scenario.items() if i in str(mic_path) for i in v]
    sc_type = []
    for k, v in scenario.items():
        for i in v:
            if i in str(mic_path):
                sc_type.append(k)
                break

    if len(sc_type) != 1:
        raise RuntimeError(
            f"file {mic_path} scenario type is unclear, may be {sc_type}, {mic_path}"
        )
    else:
        sc_type = sc_type[0]

    scores = aecmos.run(sc_type, lpb_sig, mic_sig, enh_sig)

    return scores  # (echo_mos, other_mos)


if __name__ == "__main__":
    args = parse()
    ret = {}
    if os.path.isfile(args.src):
        # audio, fs = audioread(args.src)
        # score = sig.run(audio, sr=fs)
        # score = work(args.src, args.fs)

        # for k, v in score.items():
        #     print(f"{k: <12}", f"{v:.4f}")
        pass

    elif os.path.isdir(args.src):
        mp.freeze_support()
        p = mp.Pool(processes=30)
        subdirs = os.listdir(args.src)
        for sc in subdirs:
            path = Path(args.src) / sc
            mic_list = list(map(str, path.glob("**/*mic.wav")))
            ref_list = list(map(lambda f: f.replace("mic.wav", "lpb.wav"), mic_list))
            enh_list = list(map(lambda f: f.replace(args.src, args.est), mic_list))
            # enh_list = list(map(lambda f: f.replace("mic.wav", "enh.wav"), enh_list))

            for m, r, e in zip(mic_list, ref_list, enh_list):
                if (
                    not os.path.exists(m)
                    or not os.path.exists(r)
                    or not os.path.exists(e)
                ):
                    mic_list.remove(m)
                    ref_list.remove(r)
                    enh_list.remove(e)

            if len(mic_list) == 0:
                continue

            # print("##", sc, len(mic_list), len(enh_list))
            out = p.starmap(
                work,
                tqdm(
                    zip(mic_list, ref_list, enh_list, repeat(args.fs)),
                    ncols=80,
                    total=len(mic_list),
                    leave=False,
                ),
            )
            v, n = np.array(out).sum(axis=0), len(out)
            # print(f"{sc} {len(out)} AECMOS, OtherMOS:\t{v[0]/n:.4f}, {v[1]/n:.4f}")
            ret[sc] = {
                "aecmos": np.round(v[0] / n, 4),
                "othermos": np.round(v[1] / n, 5),
            }

        p.close()
        print(json.dumps(ret))
        # score = {}
        # # for f in tqdm(files, ncols=80):
        # #     audio, fs = audioread(f)
        # #     mos = sig.run(audio, sr=args.fs)
        # #     for k, v in mos.items():
        # #         hs, hn = score.get(k, [0.0, 0])
        # #         hn += 1
        # #         hs += v
        # #         score.update({k: [hs, hn]})

        # for mos in out.get():
        #     for k, v in mos.items():
        #         hs, hn = score.get(k, [0.0, 0])
        #         hn += 1
        #         hs += v
        #         score.update({k: [hs, hn]})

        # for k, v in score.items():
        #     val, num = v
        #     print(f"{k: <12}", f"{val/num:.4f}")
    else:
        raise RuntimeError(f"args.src {args.src} not a file or directory")
