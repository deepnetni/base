import os
import scipy
import librosa

import numpy as np
import onnxruntime as ort
from enum import Enum


__all__ = ["SigMOS", "Version"]


class Version(Enum):
    V1 = "v1"  # 15.10.2023


class SigMOS:
    """
    MOS Estimator for the P.804 standard.
    See https://arxiv.org/pdf/2309.07385.pdf
    """

    def __init__(self, model_dir=os.path.dirname(__file__), model_version=Version.V1):
        assert model_version in [v for v in Version]

        model_path_history = {
            Version.V1: os.path.join(
                model_dir, "model-sigmos_1697718653_41d092e8-epo-200.onnx"
            )
        }
        self.sampling_rate = 48_000
        self.resample_type = "fft"
        self.model_version = model_version

        # STFT params
        self.dft_size = 960
        self.frame_size = 480
        self.window_length = 960
        self.window = np.sqrt(np.hanning(int(self.window_length) + 1)[:-1]).astype(
            np.float32
        )

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path_history[model_version], options)

    def stft(self, signal):
        if signal.ndim > 1:  # B,T
            last_frame = signal.shape[-1] % self.frame_size
            if last_frame == 0:
                last_frame = self.frame_size

            padded_signal = np.pad(
                signal,
                (
                    (0, 0),
                    (
                        self.window_length - self.frame_size,
                        self.window_length - last_frame,
                    ),
                ),
            )

            frames = librosa.util.frame(
                padded_signal,
                frame_length=len(self.window),
                hop_length=self.frame_size,
                axis=-1,
            )  # B, frame_length, frame_idx
            frames = frames.transpose(0, -1, -2)
            spec = scipy.fft.rfft(frames * self.window, n=self.dft_size)
            return spec.astype(np.complex64)
        else:
            last_frame = len(signal) % self.frame_size
            if last_frame == 0:
                last_frame = self.frame_size

            padded_signal = np.pad(
                signal,
                (
                    (
                        self.window_length - self.frame_size,
                        self.window_length - last_frame,
                    ),
                ),
            )
            # print(padded_signal.shape, signal.shape)
            frames = librosa.util.frame(
                padded_signal,
                frame_length=len(self.window),
                hop_length=self.frame_size,
                axis=0,
            )  # frame_idx, frame_length
            spec = scipy.fft.rfft(frames * self.window, n=self.dft_size)
            return spec.astype(np.complex64)

    @staticmethod
    def compressed_mag_complex(x: np.ndarray, compress_factor=0.3):
        """
        input shape T,F, or B,T,F
        """
        if x.ndim == 2:  # T,F
            x = x.view(np.float32).reshape(x.shape + (2,)).swapaxes(-1, -2)  # T,2,F
            x2 = np.maximum((x * x).sum(axis=-2, keepdims=True), 1e-12)
            if compress_factor == 1:
                mag = np.sqrt(x2)
            else:
                x = np.power(x2, (compress_factor - 1) / 2) * x
                mag = np.power(x2, compress_factor / 2)

            features = np.concatenate((mag, x), axis=-2)  # T,3(r,i,m),F
            features = np.transpose(features, (1, 0, 2))  # 3,T,F
            return np.expand_dims(features, 0)
        else:  # B,T,F
            assert x.ndim == 3
            # view.reshape (b,t,f->b,t,f,2(r,i)) -> swap b,t,2,f
            x = x.view(np.float32).reshape(x.shape + (2,)).swapaxes(-1, -2)
            x2 = np.maximum((x * x).sum(axis=-2, keepdims=True), 1e-12)
            if compress_factor == 1:
                mag = np.sqrt(x2)
            else:
                x = np.power(x2, (compress_factor - 1) / 2) * x
                mag = np.power(x2, compress_factor / 2)

            features = np.concatenate((mag, x), axis=-2)
            features = np.transpose(features, (0, 2, 1, 3))
            return features

    def run(self, audio: np.ndarray, sr=None):
        if sr is not None and sr != self.sampling_rate:
            assert audio.ndim == 1
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.sampling_rate,
                res_type=self.resample_type,
            )
            print(f"Audio file resampled from {sr} to {self.sampling_rate}!")

        features = self.stft(audio)  # BTF
        features = self.compressed_mag_complex(features)

        onnx_inputs = {inp.name: features for inp in self.session.get_inputs()}
        # for inp in self.session.get_inputs():
        #     print(inp)
        # for inp in self.session.get_outputs():
        #     print(inp)

        if audio.ndim > 1:
            out = self.session.run(None, onnx_inputs)[0]  # 1,B,score
            out = np.array(out)  # B, score

            result = {
                "MOS_COL": out[:, 0],
                "MOS_DISC": out[:, 1],
                "MOS_LOUD": out[:, 2],
                "MOS_NOISE": out[:, 3],
                "MOS_REVERB": out[:, 4],
                "MOS_SIG": out[:, 5],
                "MOS_OVRL": out[:, 6],
            }
        else:
            output = self.session.run(None, onnx_inputs)[0][0]  # 1,1,score

            result = {
                "MOS_COL": float(output[0]),
                "MOS_DISC": float(output[1]),
                "MOS_LOUD": float(output[2]),
                "MOS_NOISE": float(output[3]),
                "MOS_REVERB": float(output[4]),
                "MOS_SIG": float(output[5]),
                "MOS_OVRL": float(output[6]),
            }
        return result


if __name__ == "__main__":
    """
    Sample code to run the SigMOS estimator.
    V1 (current model) is an alpha version and should be used in accordance.
    """
    import time

    model_dir = r"."
    sigmos_estimator = SigMOS(model_dir=model_dir)

    # input data must have sr=48kHz, otherwise please specify the sr (it will be resampled to 48kHz internally)
    sampling_rate = 48_000
    seed_v = 3
    np.random.seed(seed_v)
    st = time.time()
    for _ in range(8):
        dummy_data = np.random.rand(5 * sampling_rate)
        dummy_result = sigmos_estimator.run(dummy_data, sr=sampling_rate)
    ed = time.time()
    print(round(ed - st, 2))

    np.random.seed(seed_v)
    st = time.time()
    dummy_data = np.random.rand(8, 5 * sampling_rate)
    dummy_result = sigmos_estimator.run(dummy_data, sr=sampling_rate)
    ed = time.time()
    print(round(ed - st, 2))
    print(dummy_result["MOS_OVRL"], type(dummy_result["MOS_OVRL"]))
