import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from .rirGenerator import RIRDict
from .spatialGenerator import apply_rir
from .synthesizer import DatasetDict, audioread


class Aligner(object):
    def __init__(self, dirname: str) -> None:
        self.dir = Path(dirname)

    def convert_format(self, text_line):
        """
        return: ["st dur word", "..."]
        """
        text_line = text_line.replace('"', "").strip().split()
        assert len(text_line) == 3
        file_name, word_labels, timesteps = text_line

        word_list = word_labels.split(",")
        timestep_list = timesteps.split(",")
        assert len(word_list) == len(timestep_list)
        new_format = []

        st = 0.0
        for word, ed in zip(word_list, timestep_list):
            if word == "":
                word = "SIL"
                st = ed
                continue

            new_format.append("{} {} {}".format(st, str(round(float(ed) - float(st), 3)), word))
            st = ed

        return new_format, file_name

    def format(self):
        """
        return [(fname, data), (...)]
            - `data` is ["start, duration, word", "..."]
            - `fname` is subdir/xx/flac_file where `subdir` refer to the subdirectory of `dirname`.
                    e.g., train-clean-100/248/130697/248-130697-000.flac
        """
        tlist = list(self.dir.rglob("**/[!.]*.txt"))
        elist = []

        for f in tlist:  # .../xx.txt
            with open(f, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    new_format, file_name = self.convert_format(line)
                    # print(new_format, file_name)

                    fname = str(Path(f).parent.relative_to(self.dir.parent) / file_name) + ".flac"

                    elist.append((fname, new_format))

        return elist


class NoisyVADGenerator(object):
    """Generate noisy audio and corresponding VAD labels."""

    def __init__(self, conf_yaml: str) -> None:
        with open(conf_yaml) as f:
            self.conf = yaml.load(f, Loader=yaml.FullLoader)

        self.libri = Path(self.conf["LibriSpeech"]["dirname"])
        self.fs = self.conf["synth_sampling_rate"]
        self.flist = list(self.libri.rglob("**/[!.]*.flac"))
        # {'fname': 'start duration word', ...}
        self.aligner = Aligner(self.conf["LibriSpeech"]["alignment"])
        # [(fname, labels), (...)]
        self.align_l = self.prepare()

        self.dsets_noise = DatasetDict(
            self.conf["datasets_noise"],
            sample_rate=self.conf["synth_sampling_rate"],
            resample_type=self.conf["synth_resampling_type"],
        )
        self.dset_rir = RIRDict(self.conf["datasets_rir"])

    def prepare(self):
        real = []
        align_l = self.aligner.format()
        for fname, labels in align_l:
            fpath = self.libri.parent / fname
            real.append((str(fpath), labels)) if fpath.exists() else None

        return real

    def __iter__(self):
        self.pick_idx = 0
        return self

    def __next__(self):
        if self.pick_idx < len(self.align_l):
            fname, labels = self.align_l[self.pick_idx]

            data, fs = audioread(fname, sr=self.fs, resample_type="fft")
            vad = np.zeros_like(data)

            for l in labels:
                st, dur, w = l.split()
                st, dur = int(fs * float(st)), int(fs * float(dur))
                vad[st : st + dur] = 0.9

            # dout = np.stack([data, vad], axis=-1)
            return dict(file=fname, data=data, vad=vad)
        else:
            raise StopIteration

    def _rms(self, audio, db=False):
        audio = np.asarray(audio)
        rms_value = np.sqrt(np.mean(audio**2))
        if db:
            return 20 * np.log10(rms_value + np.finfo(float).eps)
        else:
            return rms_value

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

    def _mix_signals(
        self, x_clean, x_noise, snr, rms_clean=None, rms_noise=None, eps=1e-12
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """mix signals with given snr.
        x_clean: C,T or T,
        x_noise: C,T or T,
        """
        assert (
            x_clean.shape[-1] == x_noise.shape[-1]
        ), f"len(x_clean): {len(x_clean)}, len(x_noise): {len(x_noise)}"

        if rms_clean is None:
            rms_clean = self._rms(x_clean)
        if rms_noise is None:
            rms_noise = self._rms(x_noise)
        clean_is_empty = rms_clean < eps

        if clean_is_empty and rms_clean is None:
            scalar = 1.0
        else:
            scalar = rms_clean / (rms_noise + eps) / (10 ** (snr / 20))

        noise = scalar * x_noise

        noisy = x_clean + noise

        # check overflow
        noisy, clip_scaler = self._clipping_solver(noisy)
        return noisy, noise * clip_scaler, clip_scaler

    def _normalize(
        self,
        audio,
        target_level=-25,
        rms_ix_start=0,
        rms_ix_end=None,
        return_scalar=False,
    ):
        """Function to normalize"""
        rms_value = self._rms(audio[rms_ix_start:rms_ix_end])
        scalar = 10 ** (target_level / 20) / (rms_value + np.finfo(float).eps)
        audio = audio * scalar

        if return_scalar:
            return audio, scalar
        else:
            return audio

    def sample_vad_pair(self):
        fname, labels = random.sample(self.align_l, 1)[0]
        data, fs = audioread(fname, sr=self.fs, resample_type=self.conf["synth_resampling_type"])
        vad = np.zeros_like(data)

        for l in labels:
            st, dur, w = l.split()
            st, dur = int(fs * float(st)), int(fs * float(dur))
            vad[st : st + dur] = 0.9

        # dout = np.stack([data, vad], axis=-1)
        return dict(file=fname, audio=data, vad=vad)

    def sample_from(self, duration) -> Dict:
        """duration: in points"""
        remaining_samples = round(duration)
        utterances, vads, clip_fnames = [], [], []
        while remaining_samples > 0:
            meta = self.sample_vad_pair()
            clip_fnames.append(meta["file"])
            x, vad = meta["audio"], meta["vad"]
            if len(x) >= remaining_samples:
                offset = random.randint(0, len(x) - remaining_samples)
                x = x[offset : offset + remaining_samples]
                vad = vad[offset : offset + remaining_samples]
            utterances.append(x)
            vads.append(vad)
            remaining_samples -= len(x)

        return dict(
            audio=np.concatenate(utterances),
            vad=np.concatenate(vads),
            files=clip_fnames,
        )

    def sample(self):
        dur_mc = self.conf["synth_duration"] * self.fs

        if dur_mc > 0:
            meta = self.sample_from(duration=dur_mc)
            # x_target: np.ndarray
            x_target = meta["audio"]
            x_vad = meta["vad"]
        else:
            raise ValueError("Unsupported clean duration value!")

        x_mic = x_target.copy()
        x_noise = None

        # adding reverberation
        if random.random() < self.conf["synth_prop_reverb"]:
            h, meta_h = self.dset_rir.sample()
            x_mic = apply_rir(x_mic, h, dur_mc)  # C,T
            x_mic: np.ndarray
            x_mic = x_mic.squeeze()
        else:
            meta_h = None

        if random.random() < self.conf["synth_prop_noisy"]:
            noise_data, noise_meta = self.dsets_noise.sample(duration=dur_mc)
            x_noise = noise_data["audio"]  # get noise

            snr_interval = self.conf["synth_snr_interval"]
            snr = random.uniform(min(snr_interval), max(snr_interval))

            # here `x_mic` is a copy of x_target, original code
            # maybe apply gain change to the `x_mic`;
            # therefore, using rms(target) as the reference engergy
            # to calculate the SNR.
            x_mic, x_noise, clip_scaler = self._mix_signals(
                x_mic,
                x_noise,
                snr,
                rms_clean=self._rms(x_target),
                rms_noise=self._rms(x_noise),
            )
            x_target = x_target * clip_scaler

        # normalize volume according to the reference channel.
        mic_level = self.conf.get("synth_normalize_volume", None)
        if mic_level is not None:
            _, norm_scalar = self._normalize(x_mic, target_level=mic_level, return_scalar=True)
            x_mic *= norm_scalar
            x_target *= norm_scalar

        # mic is noisy, target is clean
        output = dict(mic=x_mic, vad=x_vad, target=x_target, fs=self.fs, rir=meta_h)
        return output

    def generate(self):
        """
        Return: a dict containing {"mic": noisy data, "vad", target": clean speech}
        """
        meta = self.sample()
        return meta


if __name__ == "__main__":
    dset = NoisyVADGenerator("../template/vad_librispeech.yaml")
    meta = dset.sample()
    print(meta.keys())
