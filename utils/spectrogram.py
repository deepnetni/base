from matplotlib.pyplot import yticks
import numpy as np
from scipy.signal import get_window


def spectrum(data: np.ndarray, nframe, nhop):
    L = len(data)
    # frames = (L - nframe) // nhop + 1
    win = get_window("hann", nframe, fftbins=True) ** 0.5
    indx = np.arange(0, L - nframe, nhop).reshape(-1, 1) + np.arange(nframe)
    enframe = data[indx] * win  # T, F

    xk = np.abs(np.fft.rfft(enframe, axis=-1))
    xk = xk.transpose(1, 0)
    xk = 10 * np.log10(xk**2 + 1e-10)

    return xk  # F, T


def specgram(data: np.ndarray, nframe, nhop, fs):
    xk = spectrum(data, nframe, nhop)
    frames = xk.shape[1]
    ylabel = np.arange(
        0, fs // 2 + 1, 1000 if fs <= 16000 else 3000
    )  # 1000, 2000, ..., Frequency
    yticks = (nframe // 2 + 1) * ylabel * 2 // fs
    xlabel = np.arange(0, ((frames - 1) * nhop + nframe) / fs + 1)  # 1s, 2s, 3s
    xticks = xlabel * fs // nhop

    return xk, (xticks, xlabel), (yticks, ylabel)


if __name__ == "__main__":
    fs = 16000
    nt = 5
    a = np.random.randn(fs * nt)
    xk, (xticks, xlabels), (yticks, ylabels) = specgram(a, 512, 256, fs)
    print(yticks, ylabels, xk.shape, xticks, xlabels)
