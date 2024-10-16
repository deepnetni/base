import pickle

import matplotlib
import numpy as np
from scipy import signal as sig

matplotlib.use("tkagg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # rir_f = "/home/deepni/disk/rnnoise_vad/rirs/0_rir.pkl"
    rir_f = "/home/deepni/disk/spatial/rirs_norm/rir_m4_0.pkl"

    with open(rir_f, "rb") as fp:
        meta = pickle.load(fp)

    d = meta["h"]
    print(d.shape)

    lag = np.min(np.where(np.abs(d) >= 0.5 * np.max(np.abs(d)))[-1])

    for i in range(4):
        plt.figure()
        plt.subplot(211)
        plt.plot(d[i])
        plt.subplot(212)
        plt.plot(d[i, lag:])

    plt.show()

    # a = np.random.randn(1, 16000)

    # b = sig.fftconvolve(a, d, "full", axes=-1)
    # print(b.shape, a.shape[-1] + d.shape[-1] - 1)
