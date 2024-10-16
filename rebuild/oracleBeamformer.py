import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from utils.audiolib import audiowrite, audioread
from scipy.linalg import LinAlgError, eigh, solve
from scipy.signal import istft as _istft
from scipy.signal import stft as _stft

eps = 1e-15

#### https://github.com/Enny1991/beamformers start ####


def stft(x, frame_len=2048, frame_step=512):
    return _stft(x, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]


def istft(x, frame_len=2048, frame_step=512, input_len=None):
    _reconstructed = _istft(x, noverlap=(frame_len - frame_step))[1].astype(
        "float32" if x.dtype == "complex64" else "float64"
    )
    if input_len is None:
        return _reconstructed
    else:
        rec_len = len(_reconstructed)
        if input_len <= rec_len:
            return _reconstructed[:input_len]
        else:
            return np.append(_reconstructed, np.zeros((input_len - rec_len,), dtype=x.dtype))


def oracleMVDR(mixture, noise, target=None, frame_len=512, frame_step=256):
    """
    ftp://ftp.esat.kuleuven.ac.be/stadius/spriet/reports/08-211.pdf
    Frequency domain Minimum Variance Distortionless Response (MVDR) beamformer
    :param mixture: nd_array (n_mics, time) of the mixture recordings
    :param noise: nd_array (n_mics, time) of the noise recordings
    :param target: nd_array (n_mics, time) of the target recordings
    :param frame_len: int (self explanatory)
    :param frame_step: int (self explanatory)
    :return: the enhanced signal
    """
    # calculate stft
    mixture_stft = stft(mixture, frame_len=frame_len, frame_step=frame_step)

    # estimate steering vector for desired speaker (depending if target is available)
    if target is not None:
        target_stft = stft(target, frame_len=frame_len, frame_step=frame_step)
        h = estimate_steering_vector(target_stft=target_stft)
    else:
        noise_spec = stft(noise, frame_len=frame_len, frame_step=frame_step)
        h = estimate_steering_vector(mixture_stft=mixture_stft, noise_stft=noise_spec)

    # calculate weights
    w = mvdr_weights(mixture_stft, h)

    # apply weights
    sep_spec = apply_beamforming_weights(mixture_stft, w)

    # reconstruct wav
    recon = istft(sep_spec, frame_len=frame_len, frame_step=frame_step, input_len=None)

    return recon


def estimate_steering_vector(target_stft=None, mixture_stft=None, noise_stft=None):
    """
    Estimation of steering vector based on microphone recordings. The eigenvector technique used is described in
    Sarradj, E. (2010). A fast signal subspace approach for the determination of absolute levels from phased microphone
    array measurements. Journal of Sound and Vibration, 329(9), 1553-1569.
    The steering vector is represented by the leading eigenvector of the covariance matrix calculated for each
    frequency separately.
    :param target_stft: nd_array (channels, time, freq_bins)
    :param mixture_stft: nd_array (channels, time, freq_bins)
    :param noise_stft: nd_array (channels, time, freq_bins)
    :return: h: nd_array (freq_bins, ): steering vector
    """

    if target_stft is None:
        if mixture_stft is None or noise_stft is None:
            raise ValueError(
                "If no target recordings are provided you need to provide both mixture recordings "
                "and noise recordings"
            )
        C, F, T = mixture_stft.shape  # (channels, freq_bins, time)
    else:
        C, F, T = target_stft.shape  # (channels, freq_bins, time)

    eigen_vec, eigen_val, h = [], [], []

    for f in range(F):  # Each frequency separately
        # covariance matrix
        if target_stft is None:
            assert mixture_stft is not None and noise_stft is not None
            # covariance matrix estimated by subtracting mixture and noise covariances
            _R0 = mixture_stft[:, f].dot(np.conj(mixture_stft[:, f].T))
            _R1 = noise_stft[:, f].dot(np.conj(noise_stft[:, f].T))
            _Rxx = _R0 - _R1
        else:
            # covariance matrix estimated directly from single speaker
            _Rxx = target_stft[:, f].dot(np.conj(target_stft[:, f].T))

        # eigendecomposition
        [_d, _v] = np.linalg.eig(_Rxx)

        # index of leading eigenvector
        idx = np.argsort(_d)[::-1][0]

        # collect leading eigenvector and eigenvalue
        eigen_val.append(_d[idx])
        eigen_vec.append(_v[:, idx])

    # rescale eigenvectors by eigenvalues for each frequency
    for vec, val in zip(eigen_vec, eigen_val):
        if val != 0.0:
            # the part is modified from the MVDR implementation https://github.com/Enny1991/beamformers
            # vec = vec * val / np.abs(val)
            vec = vec / vec[0]  # normalized to the first channel
            h.append(vec)
        else:
            h.append(np.ones_like(vec))

    # return steering vector
    return np.vstack(h)


def apply_beamforming_weights(signals, weights):
    """
    Fastest way to apply beamforming weights in frequency domain.
    :param signals: nd_array (freq_bins (a), n_mics (b))
    :param weights: nd_array (n_mics (b), freq_bins (a), time_frames (c))
    :return: nd_array (freq_bins (a), time_frames (c)): filtered stft
    """
    return np.einsum("ab,bac->ac", np.conj(weights), signals)


def mvdr_weights(mixture_stft, h):
    C, F, T = mixture_stft.shape  # (channels, freq_bins, time)

    # covariance matrix

    R_y = np.einsum("a...c,b...c", mixture_stft, np.conj(mixture_stft)) / T
    R_y = condition_covariance(R_y, 1e-6)
    R_y /= np.trace(R_y, axis1=-2, axis2=-1)[..., None, None] + 1e-15
    # preallocate weights
    W = np.zeros((F, C), dtype="complex64")

    # compute weights for each frequency separately
    for i, r, _h in zip(range(F), R_y, h):
        # part = np.linalg.inv(r + np.eye(C, dtype='complex') * eps).dot(_h)
        part = solve(r, _h)
        _w = part / np.conj(_h).T.dot(part)

        W[i, :] = _w

    return W


def condition_covariance(x, gamma):
    """Code borrowed from https://github.com/fgnt/nn-gev/blob/master/fgnt/beamforming.py
    Please refer to the repo and to the paper (https://ieeexplore.ieee.org/document/7471664) for more information.
    see https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3)"""
    scale = gamma * np.trace(x, axis1=-2, axis2=-1)[..., None, None] / x.shape[-1]
    n = len(x.shape) - 2
    scaled_eye = np.eye(x.shape[-1], dtype=x.dtype)[(None,) * n] * scale
    return (x + scaled_eye) / (1 + gamma)


#### https://github.com/Enny1991/beamformers end ####


class OracleBeamformer(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()


if __name__ == "__main__":
    f = "/home/deepni/datasets/spatialReverbNoiseAlign/test_beamformer/0_mic.wav"
    f2 = "/home/deepni/datasets/spatialReverbNoiseAlign/test_beamformer/0_target.wav"
    fr = "/home/deepni/datasets/spatialReverbNoiseAlign/test_beamformer/0_reverb.wav"
    fn = "/home/deepni/datasets/spatialReverbNoiseAlign/test_beamformer/0_noise.wav"
    data, fs = audioread(f)  # T,C
    tgt, fs = audioread(f2)  # T,
    reverb, fs = audioread(fr)  # T,C
    noise, fs = audioread(fn)  # T,C
    data = data.T  # M,T
    reverb = reverb.T  # M,T
    noise = noise.T  # M,T

    enh = oracleMVDR(data, noise)
    # enh = mvdr(data, noise, target=reverb)
    print(enh.shape)
    audiowrite("out_n.wav", enh, fs)
