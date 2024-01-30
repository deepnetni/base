import torch
import numpy as np
import librosa
from pesq import pesq
from pystoi import stoi

from typing import Callable, Dict, List, Optional, Union


def l2_norm(s, keepdim=False):
    # sqrt(|vec| * 2) value
    return torch.linalg.norm(s, dim=-1, keepdim=keepdim)


def compute_si_snr(sph, est, zero_mean=True):
    """
    s1 is the est signal, s2 represent for clean speech
    """
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
    return sisnr


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

    return score


def compute_stoi(lbl, est, fs=16000):
    try:
        score = stoi(lbl, est, fs)
    except Exception as e:
        score = 0
        print(e)

    return score
