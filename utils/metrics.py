import torch
import numpy as np
import librosa
from pesq import pesq
from pystoi import stoi

from typing import Callable, Dict, List, Optional, Union


def l2_norm(s, keepdim=False):
    # sqrt(|vec| * 2) value
    return torch.linalg.norm(s, dim=-1, keepdim=keepdim)


def compute_sdr(
    sph: Union[torch.Tensor, np.ndarray],
    est: Union[torch.Tensor, np.ndarray],
    zero_mean=True,
):
    """
    input: B,T
    """
    is_numpy = False
    if isinstance(sph, np.ndarray) or isinstance(est, np.ndarray):
        sph = torch.from_numpy(sph)
        est = torch.from_numpy(est)
        is_numpy = True
    eps = torch.finfo(sph.dtype).eps
    if zero_mean is True:
        s = sph - torch.mean(sph, dim=-1, keepdim=True)
        s_hat = est - torch.mean(est, dim=-1, keepdim=True)
    else:
        s = sph
        s_hat = est
    pass


def compute_si_snr(
    sph: Union[torch.Tensor, np.ndarray],
    est: Union[torch.Tensor, np.ndarray],
    zero_mean=True,
):
    """
    s1 is the est signal, s2 represent for clean speech
    """
    is_numpy = False
    if isinstance(sph, np.ndarray) or isinstance(est, np.ndarray):
        sph = torch.from_numpy(sph)
        est = torch.from_numpy(est)
        is_numpy = True

    eps = torch.finfo(sph.dtype).eps

    if zero_mean is True:
        s = sph - torch.mean(sph, dim=-1, keepdim=True)
        s_hat = est - torch.mean(est, dim=-1, keepdim=True)
    else:
        s = sph
        s_hat = est

    s_target = (
        (torch.sum(s_hat * s, dim=-1, keepdim=True) + eps)
        * s
        / (l2_norm(s, keepdim=True) ** 2 + eps)
    )
    e_noise = s_hat - s_target
    # sisnr = 10 * torch.log10(
    #     (l2_norm(s_target) ** 2 + eps) / (l2_norm(e_noise) ** 2 + eps)
    # )
    sisnr = 10 * torch.log10(
        (torch.sum(s_target**2, dim=-1) + eps)
        / (torch.sum(e_noise**2, dim=-1) + eps)
    )
    return sisnr.cpu().detach().numpy() if is_numpy else sisnr


def compute_erle(mic, est):
    pow_est = np.sum(est**2, axis=-1, keepdims=True)
    pow_mic = np.sum(mic**2, axis=-1, keepdims=True)

    erle_score = 10 * np.log10(pow_mic / (pow_est + np.finfo(np.float32).eps))
    return erle_score


def compute_pesq(lbl, est, fs=16000, norm=False):
    assert isinstance(lbl, np.ndarray)
    assert isinstance(est, np.ndarray)

    if fs > 16000:
        lbl = librosa.resample(lbl, orig_sr=fs, target_sr=16000)
        est = librosa.resample(est, orig_sr=fs, target_sr=16000)

    try:
        score = pesq(
            16000 if fs > 16000 else fs, lbl, est, "nb" if fs == 8000 else "wb"
        )
    except Exception as e:
        score = 0
        print(e)

    if norm:
        score = (score - 1.0) / 3.5

    return score  # scaler


def compute_stoi(lbl, est, fs=16000):
    try:
        score = stoi(lbl, est, fs)
    except Exception as e:
        score = 0
        print(e)

    return score  # scaler


if __name__ == "__main__":
    inp = np.random.randn(16000) + 10
    lbl = np.random.randn(16000) + 10

    l = compute_pesq(lbl, inp)
    print(l)

    inp = np.concatenate([inp, torch.zeros(20000)], axis=-1)
    lbl = np.concatenate([lbl, torch.zeros(20000)], axis=-1)
    l = compute_pesq(lbl, inp)
    print(l)
    pass
