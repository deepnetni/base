import numpy as np
from typing import List
from scipy.fftpack import fft


class MVDR(object):
    """
    a circular microphone array.
    """

    def __init__(
        self,
        mic_angle: List = [0, 60, 120, 180, 270, 330],
        mic_diameter=0.1,
        nfft: int = 512,
        nhop: int = 256,
        fs: int = 16000,
    ):
        self.mic_angle = np.array(mic_angle)
        self.mic_diameter = mic_diameter
        self.nfft = nfft
        self.nhop = nhop
        self.fs = fs

    def _get_sterring_vector(self, look_direction):
        n_mic = len(self.mic_angle)
        frequency_vector = np.linspace(0, self.fs, self.nfft)
        steering_vector = np.ones((len(frequency_vector), n_mic), dtype=np.complex64)
        look_direction = look_direction * (-1)  # ! why -1

        for i, f in enumerate(frequency_vector):
            for j, angle in enumerate(self.mic_angle):
                steering_vector[i, j] = np.complex(1, 2)

        pass


if __name__ == "__main__":
    pass
