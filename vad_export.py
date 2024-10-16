import shutil

import numpy as np
import soundfile as sf
import torch
import os

from einops import rearrange
from models.conv_stft import STFT
from models.VADModel import CRNN_VAD_new, pack_frames_vad, CRNN_VAD_new_origin
from rebuild.rnnoise import *


def export(export_dir, ckpt):
    net = RNNoiseVAD()
    # extractor = FeatExtractor(128, 64).eval()
    net.load_state_dict(torch.load(ckpt))
    net.eval()

    # print(net)
    save_space = f'{export_dir}\\dense_params'
    if os.path.exists(save_space):
        shutil.rmtree(save_space)
        os.makedirs(save_space)
    elif not os.path.exists(save_space):
        os.makedirs(save_space)

    for n, v in net.named_parameters():
        print(v.flatten().shape)
        if n.split('.')[0][0:5] == 'dense':
            v = v.T if v.ndim > 1 else v
            param = v.flatten().detach().numpy()
            name = n.split('.')[0] + '_' + n.split('.')[1] + '_' + n.split('.')[2]
            np.savetxt(f'{save_space}\\{name}.txt', param.reshape(-1, 6), fmt="%.8f", delimiter=',')
        elif n.split('.')[0][0:4] == 'post':
            v = v.T if v.ndim > 1 else v
            param = v.flatten().detach().numpy()
            name = n.split('.')[0] + '_' + n.split('.')[1] + '_' + n.split('.')[2]
            L = len(param)
            n1 = (L // 4) * 4
            if n1 > 0:
                with open(f'{save_space}\\{name}.txt', 'a') as f:
                    np.savetxt(f, param[:n1].reshape(-1, 4), fmt="%.8f", delimiter=',')
                    np.savetxt(f, param[n1:].reshape(1, -1), fmt="%.8f", delimiter=',')
                    # np.savetxt(f, np.ones(12).reshape(1, -1), fmt="%.8f", delimiter=', ')
            else:
                np.savetxt(f'{save_space}\\{name}.txt', param, fmt="%.8f", delimiter=',')

    save_space = f"{export_dir}\\gru_params"
    if not os.path.exists(save_space):
        os.makedirs(save_space)
    for n, v in net.named_parameters():
        if n.split('.')[0] == 'gru':
            param = v.flatten().detach().numpy()
            key_list = n.split('.')
            name = n.replace(".", "_")
            np.savetxt(f'{save_space}\\{name}.txt', param.reshape(-1, 6), fmt="%.8f", delimiter=',')

def toH(dirname):
    outdir = os.path.join(dirname, "inc")

    shutil.rmtree(outdir) if os.path.exists(outdir) else None
    os.makedirs(outdir)

    for dirn in os.listdir(dirname):
        if dirn == "inc":
            continue
        with open(os.path.join(outdir, dirn) + '.h', "a+", encoding="utf-8") as fp:
            fp.write(f"#ifndef __{dirn.upper()}_H_\n")
            fp.write(f"#define __{dirn.upper()}_H_\n\n")

        dpath = os.path.join(dirname, dirn)
        for f in os.listdir(dpath):
            fpath = os.path.join(dpath, f)
            data = np.loadtxt(fpath, delimiter=',').flatten()

            with open(os.path.join(outdir, dirn)+'.h', "a+", encoding="utf-8") as fp:
                fp.write(f"static const float {f.split('.')[0]}[] = ")
                fp.write("{\n\t")
                N = len(data)
                for idx, d in enumerate(data):
                    fp.write(f"{d: .6f},")
                    if (idx + 1) % 8 == 0 and (idx + 8) < N:
                        fp.write("\n\t")

                fp.write("\n};\n\n")
            # print(data.shape)
        with open(os.path.join(outdir, dirn) + '.h', "a+", encoding="utf-8") as fp:
            fp.write(f"#endif")


if __name__ == "__main__":
    export_dir = "params\\export"
    ckpt = "trained_vad_300\\rnnoise_vad\\checkpoints\\epoch_0046.pth"
    export(export_dir, ckpt)
    toH(export_dir)