import os
from pyvad import vad
from matplotlib import pyplot as plt
import soundfile as sf
import numpy as np


# wavfile = "/Users/deepni/AEC/DT/0/echo.wav"
# plt.plot(data, label="data")
# plt.plot(vact, color="r", label="vad")
# plt.legend()
# plt.show()

Path = "/home/deepnetni/dataset/AEC"


def clean(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith("-vad.wav"):
                fname = os.path.join(root, f)
                os.remove(fname)


def gen_vad(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if not f.endswith(".wav") or f.endswith("-vad.wav"):
                continue
            fname = os.path.join(root, f)
            data, fs = sf.read(fname)
            vact = vad(data, fs)

            sdata = np.stack([data, vact], axis=-1)
            print(sdata)
            # print(sdata.shape, vact.shape, data.shape)
            fout = os.path.join(root, f.split(".")[0] + "-vad.wav")

            sf.write(fout, sdata, fs)

            # plt.plot(data, label="data")
            # plt.plot(vact, color="r", label="vad")
            # plt.legend()
            # plt.show()


if __name__ == "__main__":
    gen_vad(Path)
