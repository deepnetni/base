import glob
import logging
import multiprocessing as mp
import os
import pickle
import random
import sys

from itertools import repeat
from pathlib import Path
from typing import Dict
from numba.core.datamodel import register

import numpy as np
import rir_generator as rir
import scipy.signal as sps
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from .synthesizer import DatasetDict
from rirGenerator import RIRGenerator, RIRDict

logger = logging.getLogger(__name__)


def apply_rir(audio: np.ndarray, rir: np.ndarray, N):
    """
    Input:
        - audio, [T] or [1,T]
        - rir, [C, T]
    Return: [C, T]
    """
    # 1, T
    audio = audio[None, :] if audio.ndim < 2 else audio
    reverb = sps.fftconvolve(audio, rir, "full", axes=-1)  # N1+N2-1

    # ! synchronize reverberant with anechoic
    lag = np.min(np.where(np.abs(rir) >= 0.5 * np.max(np.abs(rir)))[-1])
    reverb = reverb[:, lag : lag + N]

    # ! enforce enegy conservation
    reverb *= np.sqrt(np.mean(audio**2) / np.mean(reverb**2))

    return reverb  # C,T


def rms_channel(audio, dim=-1, keepdims=False, db=False):
    """
    audio: C,T or T,
    return: C, or ,
    """
    audio = np.asarray(audio)
    rms_value = np.sqrt(np.mean(audio**2, axis=dim, keepdims=keepdims))
    if db:
        return 20 * np.log10(rms_value + np.finfo(float).eps)
    else:
        return rms_value


class SpatialGenerater:
    """
    Input: conf_path, a yaml file refer to `templated/SpatialGenerator.yaml`

    API:
        - generate_rirs: a multiprocessing api to generate rirs, save to `save_p`;
    """

    def __init__(self, conf_path: str) -> None:
        with open(conf_path) as f:
            self.conf = yaml.load(f, Loader=yaml.FullLoader)

        # self.gene_rir = RIRGenerator(**self.conf["rir_configs"])
        self.dur_mc = self.conf["synth_duration"] * self.conf["synth_sampling_rate"]

        self.dsets_clean = DatasetDict(
            self.conf["datasets_clean"],
            sample_rate=self.conf["synth_sampling_rate"],
            resample_type=self.conf["synth_resampling_type"],
        )
        self.dsets_noise = DatasetDict(
            self.conf["datasets_noise"],
            sample_rate=self.conf["synth_sampling_rate"],
            resample_type=self.conf["synth_resampling_type"],
        )
        self.dset_rir = RIRDict(self.conf["datasets_rir"])

    def _rms(self, audio, db=False):
        audio = np.asarray(audio)
        rms_value = np.sqrt(np.mean(audio**2))
        if db:
            return 20 * np.log10(rms_value + np.finfo(float).eps)
        else:
            return rms_value

    def _normalize(
        self, audio, target_level=-25, rms_ix_start=0, rms_ix_end=None, return_scalar=False
    ):
        """Function to normalize"""
        rms_value = self._rms(audio[rms_ix_start:rms_ix_end])
        scalar = 10 ** (target_level / 20) / (rms_value + np.finfo(float).eps)
        audio = audio * scalar

        if return_scalar:
            return audio, scalar
        else:
            return audio

    def _clipping_solver(self, audio, threshold=0.99):
        """
        This function helps scale down the audio to solve the clipping issue
        returnes the scale that can be used to scale clean speech
        """
        amplitude_max = np.max(np.abs(audio))
        scale = 1.0
        if amplitude_max <= threshold:
            return audio, scale
        else:
            scale = threshold / amplitude_max
            scaled_audio = audio * scale
            return scaled_audio, scale

    def _mix_signals(self, x_clean, x_noise, snr, rms_clean=None, rms_noise=None, eps=1e-12):
        """mix signals with given snr.
        x_clean: C,T or T,
        x_noise: C,T or T,
        """
        assert (
            x_clean.shape[-1] == x_noise.shape[-1]
        ), f"len(x_clean): {x_clean.shape[-1]}, len(x_noise): {x_noise.shape[-1]}"

        if rms_clean is None:
            rms_clean = rms_channel(x_clean)
        if rms_noise is None:
            rms_noise = rms_channel(x_noise)
        clean_is_empty = rms_clean < eps

        rms_clean = rms_clean.mean()
        rms_noise = rms_noise.mean()
        if clean_is_empty and rms_clean is None:
            scalar = 1.0
        else:
            scalar = rms_clean / (rms_noise + eps) / (10 ** (snr / 20))

        noise = scalar * x_noise

        noisy = x_clean + noise

        # TODO check overflow
        noisy, clip_scaler = self._clipping_solver(noisy)
        return noisy, noise * clip_scaler, clip_scaler

    @staticmethod
    def _worker_rir(conf, idx, save_p):
        fname = os.path.join(save_p, f"rir_m{conf['array']['mics']}_{idx}.pkl")
        # meta = self.gene_rir.sample()
        gene = RIRGenerator(**conf)
        meta = gene.sample()
        # meta = gene()

        with open(fname, "wb") as fp:
            pickle.dump(meta, fp)

        # plt.figure()
        # plt.plot(meta["h"][0, :])
        # plt.savefig(os.path.join(save_p, f"{idx}.svg"), bbox_inches="tight")
        # plt.close()

    def generate_rirs(self, n=1, save_p="../rirs"):
        mp.freeze_support()
        p = mp.Pool(processes=30)
        os.makedirs(save_p) if not os.path.exists(save_p) else None
        p.starmap(
            SpatialGenerater._worker_rir,
            tqdm(
                zip(
                    repeat(self.conf["rir_configs"]),
                    range(n),
                    repeat(save_p),
                ),
                ncols=80,
                total=n,
            ),
        )

    def apply_rir_bug(self, audio, rir):
        """
        Input:
            - audio, [T]
            - rir, [C,T]
        Return:
            wav_tgt: C x T
        """
        nC, nL = rir.shape
        wav_rir = np.zeros([nC, audio.shape[0] + nL - 1])

        for i in range(nC):
            wav_rir[i] = sps.oaconvolve(audio, rir[i, :])

        # C x L
        wav_rir = wav_rir[:, : audio.shape[0]]
        return wav_rir

    def _get_main_channel(self, meta: Dict):
        """
        Compute the closest microphone according to the position of source and microphone array.
        """
        array = meta["mic_array"]  # mics, pos
        spk_pos = meta["spk_pos"]

        dist = (np.array(spk_pos) - np.array(array)) ** 2
        dist = dist.sum(-1)
        ch = np.argmin(dist)
        return ch

    def _generate_multi_speech(self):
        """
        mix noise and speech signals
        """
        dur_mc = self.conf["synth_duration"] * self.conf["synth_sampling_rate"]

        if dur_mc > 0:
            # +6000 for apply rir
            data, meta = self.dsets_clean.sample(duration=dur_mc + 6000)
            x_target = data["audio"] - data["audio"].mean()
        else:
            raise ValueError("Unsupported clean duration value!")

        x_mic = x_target.copy()
        x_noise = None

        # apply rir transform
        hrir, meta_rir = self.dset_rir.sample()
        # ch_ref = SpatialGenerater._get_main_channel(meta_rir)

        # x_mic, [C,T]
        x_mic = apply_rir(x_mic, hrir, dur_mc)
        x_reverb = x_mic.copy()
        x_target = x_target[: x_mic.shape[-1]]

        if random.random() < self.conf["synth_prop_noisy"]:
            noise_data, noise_meta = self.dsets_noise.sample(duration=dur_mc, thres=-45)
            x_noise = noise_data["audio"]  # get noise T,C

            # if random.random() < self.conf["synth_nearend_prop_add_gaussian_ne_noise"]:
            #     # replace the gaussian noise with our self record sample.
            #     bgnoise_data, bgnoise_meta = self.self_datasets.sample(
            #         duration=dur_nearend
            #     )
            #     bgnoise = bgnoise_data["audio"]
            #     x_noise += bgnoise  # adding self record noise
            #     # noise_db = rms(x_noise, db=True)
            #     # # make gaussian noise in range [noise_db-10, noise_db]
            #     # std = 10 ** ((noise_db - random.random() * 10) / 20)
            #     # gaussian_noise = std * np.random.randn(len(x_noise))
            #     # x_noise += gaussian_noise.astype(np.float32)

            snr_interval = self.conf["synth_snr_interval"]
            snr = random.uniform(min(snr_interval), max(snr_interval))

            x_noise = x_noise.T  # C,T
            x_mic, x_noise, clip_scaler = self._mix_signals(
                x_mic, x_noise, snr, rms_clean=rms_channel(x_target), rms_noise=rms_channel(x_noise)
            )
            x_target *= clip_scaler
            x_reverb *= clip_scaler

        if x_noise is None:
            x_noise = np.zeros_like(x_mic)

        assert x_mic.shape[-1] == len(x_target) == x_noise.shape[-1]

        # normalize volume according to the reference channel.
        mic_level = self.conf.get("synth_normalize_volume", None)
        assert mic_level is None, "TODO check"
        if mic_level is not None:
            ch_ref = self._get_main_channel(meta_rir)
            _, norm_scalar = self._normalize(
                x_mic[ch_ref], target_level=mic_level, return_scalar=True
            )
            x_mic *= norm_scalar
            x_target *= norm_scalar
            x_noise *= norm_scalar

        # C,T; T,;
        output = dict(target=x_target, mic=x_mic, noise=x_noise, meta=meta_rir, reverb=x_reverb)
        return output

    def generate(self):
        """
        Returns a dict with
            - target (desired signal), T
            - mic, C,T (target signal affected by noise and some gain changes)
        """
        data = self._generate_multi_speech()

        return data


if __name__ == "__main__":
    gene = SpatialGenerater("../template/spatialGenerator.yaml")
    gene.generate_rirs(50000, "/home/deepni/datasets/mc/rirs")
    gene.generate()

    # gene = RIRGenerator.from_yaml("../template/spatialGenerator.yaml")
    # gene = RIRGenerator.from_yaml("../template/vad_librispeech.yaml")
    # out = gene.sample()
    # print(out["h"].shape)
