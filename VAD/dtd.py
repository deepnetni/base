import numpy as np
from pyvad import vad
import soundfile as sf

from matplotlib import pyplot as plt

SILENT_MODE = 0
ST_FAR_MODE = 1
ST_NEAR_MODE = 2
DT_MODE = 3


def is_farend_active(frame_vad: np.ndarray, nfrm: int):
    """
    return True if one third of a frame is activate
    """
    necho_active = len(frame_vad[frame_vad == 1])
    return bool(necho_active > nfrm // 3)


def is_nearend_active(frame_vad: np.ndarray, nfrm: int):
    """
    return True if one third of a frame is activate
    """
    n_active = len(frame_vad[frame_vad == 1])
    return bool(n_active > nfrm // 3)


def label_DTD_with_echo(echo: np.ndarray, sph: np.ndarray, nfrm: int, nhop: int, fs=16000):
    noverlap = nfrm - nhop
    echo_vact = vad(echo, fs)
    sph_vact = vad(sph, fs)

    # * padding will infect the result of webrtc vad algorithm
    echo_vact_pad = np.pad(echo_vact, pad_width=(noverlap, noverlap))
    sph_vact_pad = np.pad(sph_vact, pad_width=(noverlap, noverlap))

    L = len(echo_vact_pad)
    frames = int((L - nfrm) / nhop) + 1

    frame_st = np.arange(0, frames * nhop, nhop).reshape(-1, 1)
    frame_idx = frame_st + np.arange(nfrm)

    mix_frame_label = []

    for _, (ec, sp) in enumerate(
        zip(echo_vact_pad[frame_idx], sph_vact_pad[frame_idx])
    ):
        farend_active = False
        nearend_active = False

        if is_farend_active(ec, nfrm):
            farend_active = True

        if is_nearend_active(sp, nfrm):
            nearend_active = True

        if farend_active and nearend_active:
            mix_frame_label.append(DT_MODE)
        elif farend_active:
            mix_frame_label.append(ST_FAR_MODE)
        elif nearend_active:
            mix_frame_label.append(ST_NEAR_MODE)
        else:
            mix_frame_label.append(SILENT_MODE)

    return np.array(mix_frame_label), echo_vact, sph_vact


if __name__ == "__main__":
    echo_f = "/Users/deepni/AEC/DT/2/echo.wav"
    sph_f = "/Users/deepni/AEC/DT/2/sph.wav"
    mix_f = "/Users/deepni/AEC/DT/2/mix.wav"

    echo, fs = sf.read(echo_f)
    sph, _ = sf.read(sph_f)
    label, _, _ = label_DTD_with_echo(echo, sph, 512, 128)
    label = label.reshape(1, -1)
    label = np.repeat(label, 256, axis=0)
    print(label.shape, label)
    t = np.arange(0, len(echo)) / 16000

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    ax0.plot(t, echo)
    ax0.set_title("echo")
    # ax0.set_xticks(np.arange(0, len(sph)) + 10)
    # print(np.arange(0, len(sph)) / 16000)
    ax1.plot(t, sph)
    ax1.set_title("sph")

    im = ax2.imshow(label, aspect="auto", cmap="Greens")
    ax2.set_title("0: silent, 1: ST_FAR, 2: ST_NEAR, 3: DT")
    ax2.set_yticks([])
    fig.colorbar(im, ax=ax2, label="VAD")
    plt.tight_layout()
    plt.show()
