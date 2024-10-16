# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:58:00 2019

@author: a-kojima
"""
import numpy as np
import numpy.matlib as npm
import soundfile as sf
from scipy import signal as sg
from scipy.fftpack import fft, ifft
from utils.audiolib import audioread, audiowrite


def stab(mat, theta, num_channels):
    d = np.power(np.array(10, dtype=np.complex64), np.arange(-num_channels, 0, dtype=np.float))
    result_mat = mat
    for i in range(1, num_channels + 1):
        if np.linalg.cond(mat) > theta:
            return result_mat
        result_mat = result_mat + d[i - 1] * np.eye(num_channels, dtype=np.complex64)
    return result_mat


def get_3dim_spectrum(wav_name, channel_vec, start_point, stop_point, frame, shift, fftl):
    """
    dump_wav : channel_size * speech_size (2dim)
    """
    samples, _ = sf.read(
        wav_name.replace("{}", str(channel_vec[0])),
        start=start_point,
        stop=stop_point,
        dtype="float32",
    )
    if len(samples) == 0:
        return None, None
    dump_wav = np.zeros((len(channel_vec), len(samples)), dtype=np.float16)
    dump_wav[0, :] = samples.T
    for ii in range(0, len(channel_vec) - 1):
        samples, _ = sf.read(
            wav_name.replace("{}", str(channel_vec[ii + 1])),
            start=start_point,
            stop=stop_point,
            dtype="float32",
        )
        dump_wav[ii + 1, :] = samples.T

    dump_wav = dump_wav / np.max(np.abs(dump_wav)) * 0.7
    window = sg.windows.hann(fftl + 1, sym=False)[:-1]
    multi_window = npm.repmat(window, len(channel_vec), 1)
    st = 0
    ed = frame
    number_of_frame = int((len(samples) - frame) / shift)
    spectrums = np.zeros((len(channel_vec), number_of_frame, int(fftl / 2) + 1), dtype=np.complex64)
    for ii in range(0, number_of_frame):
        multi_signal_spectrum = fft(dump_wav[:, st:ed], n=fftl, axis=1)[
            :, 0 : int(fftl / 2) + 1
        ]  # channel * number_of_bin
        spectrums[:, ii, :] = multi_signal_spectrum
        st = st + shift
        ed = ed + shift
    return spectrums, len(samples)


def get_3dim_spectrum_from_data(wav_data, frame, shift, fftl):
    """
    dump_wav : channel_size * speech_size (2dim)
    """
    len_sample, len_channel_vec = np.shape(wav_data)
    dump_wav = wav_data.T
    dump_wav = dump_wav / np.max(np.abs(dump_wav)) * 0.7
    window = sg.windows.hann(fftl + 1, sym=False)[:-1]
    multi_window = npm.repmat(window, len_channel_vec, 1)
    st = 0
    ed = frame
    number_of_frame = int((len_sample - frame) / shift)
    spectrums = np.zeros((len_channel_vec, number_of_frame, int(fftl / 2) + 1), dtype=np.complex64)
    for ii in range(0, number_of_frame):
        multi_signal_spectrum = fft(dump_wav[:, st:ed], n=fftl, axis=1)[
            :, 0 : int(fftl / 2) + 1
        ]  # channel * number_of_bin
        spectrums[:, ii, :] = multi_signal_spectrum
        st = st + shift
        ed = ed + shift
    return spectrums, len_sample


def my_det(matrix_):
    sign, lodget = np.linalg.slogdet(matrix_)
    return np.exp(lodget)


def spec2wav(spectrogram, sampling_frequency, fftl, frame_len, shift_len):
    n_of_frame, fft_half = np.shape(spectrogram)
    hanning = sg.windows.hann(fftl + 1, sym=False)[:-1]
    cut_data = np.zeros(fftl, dtype=np.complex64)
    result = np.zeros(sampling_frequency * 60 * 5, dtype=np.float32)
    start_point = 0
    end_point = start_point + frame_len
    for ii in range(0, n_of_frame):
        half_spec = spectrogram[ii, :]
        cut_data[0 : int(fftl / 2) + 1] = half_spec.T
        cut_data[int(fftl / 2) + 1 :] = np.flip(np.conjugate(half_spec[1 : int(fftl / 2)]), axis=0)
        cut_data2 = np.real(ifft(cut_data, n=fftl))
        result[start_point:end_point] = result[start_point:end_point] + np.real(
            cut_data2 * hanning.T
        )
        start_point = start_point + shift_len
        end_point = end_point + shift_len
    return result[0 : end_point - shift_len]


def multispec2wav(multi_spectrogram, beamformer, fftl, shift, multi_window, true_dur):
    channel, number_of_frame, fft_size = np.shape(multi_spectrogram)
    cut_data = np.zeros((channel, fftl), dtype=np.complex64)
    result = np.zeros((channel, true_dur), dtype=np.float32)
    start_p = 0
    end_p = start_p + fftl
    for ii in range(0, number_of_frame):
        cut_spec = multi_spectrogram[:, ii, :] * beamformer
        cut_data[:, 0:fft_size] = cut_spec
        cut_data[:, fft_size:] = np.transpose(np.flip(cut_spec[:, 1 : fft_size - 1], axis=1).T)
        cut_data2 = np.real(ifft(cut_data, n=fftl, axis=1))
        result[:, start_p:end_p] = result[:, start_p:end_p] + (cut_data2 * multi_window)
        start_p = start_p + shift
        end_p = end_p + shift
    return np.sum(result[:, 0 : end_p - shift], axis=0)


def check_beamformer(freq_beamformer, theta_cov):
    freq_beamformer = np.real(freq_beamformer)
    if len(freq_beamformer[freq_beamformer >= theta_cov]) != 0:
        return np.ones(np.shape(freq_beamformer), dtype=np.complex64) * (1 + 1j)
    return freq_beamformer


class minimum_variance_distortioless_response(object):
    def __init__(
        self,
        mic_angle_vector,
        mic_diameter,
        sampling_frequency=16000,
        fft_length=512,
        fft_shift=256,
        sound_speed=343,
    ):
        self.mic_angle_vector = mic_angle_vector
        self.mic_diameter = mic_diameter
        self.sampling_frequency = sampling_frequency
        self.fft_length = fft_length
        self.fft_shift = fft_shift
        self.sound_speed = sound_speed

    def get_sterring_vector(self, look_direction):
        number_of_mic = len(self.mic_angle_vector)
        frequency_vector = np.linspace(0, self.sampling_frequency, self.fft_length)
        steering_vector = np.ones((len(frequency_vector), number_of_mic), dtype=np.complex64)
        look_direction = look_direction * (-1)
        for f, frequency in enumerate(frequency_vector):
            for m, mic_angle in enumerate(self.mic_angle_vector):
                steering_vector[f, m] = complex(
                    np.exp(
                        (-1j)
                        * ((2 * np.pi * frequency) / self.sound_speed)
                        * (self.mic_diameter / 2)
                        * np.cos(np.deg2rad(look_direction) - np.deg2rad(mic_angle))
                    )
                )
        steering_vector = np.conjugate(steering_vector).T
        normalize_steering_vector = self.normalize(steering_vector)
        return normalize_steering_vector[0 : int(self.fft_length / 2) + 1, :]

    def normalize(self, steering_vector):
        for ii in range(0, self.fft_length):
            weight = np.matmul(np.conjugate(steering_vector[:, ii]).T, steering_vector[:, ii])
            steering_vector[:, ii] = steering_vector[:, ii] / weight
        return steering_vector

    def get_spatial_correlation_matrix(
        self, multi_signal, use_number_of_frames_init=10, use_number_of_frames_final=10
    ):
        # init
        number_of_mic = len(self.mic_angle_vector)
        frequency_grid = np.linspace(0, self.sampling_frequency, self.fft_length)
        frequency_grid = frequency_grid[0 : int(self.fft_length / 2) + 1]
        start_index = 0
        end_index = start_index + self.fft_length
        speech_length, number_of_channels = np.shape(multi_signal)
        R_mean = np.zeros((number_of_mic, number_of_mic, len(frequency_grid)), dtype=np.complex64)
        used_number_of_frames = 0

        # forward
        for _ in range(0, use_number_of_frames_init):
            multi_signal_cut = multi_signal[start_index:end_index, :]
            complex_signal = fft(multi_signal_cut, n=self.fft_length, axis=0)
            for f in range(0, len(frequency_grid)):
                R_mean[:, :, f] = R_mean[:, :, f] + np.multiply.outer(
                    complex_signal[f, :], np.conj(complex_signal[f, :]).T
                )
            used_number_of_frames = used_number_of_frames + 1
            start_index = start_index + self.fft_shift
            end_index = end_index + self.fft_shift
            if speech_length <= start_index or speech_length <= end_index:
                used_number_of_frames = used_number_of_frames - 1
                break

        # backward
        end_index = speech_length
        start_index = end_index - self.fft_length
        for _ in range(0, use_number_of_frames_final):
            multi_signal_cut = multi_signal[start_index:end_index, :]
            complex_signal = fft(multi_signal_cut, n=self.fft_length, axis=0)
            for f in range(0, len(frequency_grid)):
                R_mean[:, :, f] = R_mean[:, :, f] + np.multiply.outer(
                    complex_signal[f, :], np.conj(complex_signal[f, :]).T
                )
            used_number_of_frames = used_number_of_frames + 1
            start_index = start_index - self.fft_shift
            end_index = end_index - self.fft_shift
            if start_index < 1 or end_index < 1:
                used_number_of_frames = used_number_of_frames - 1
                break

        return R_mean / used_number_of_frames

    def get_mvdr_beamformer(self, steering_vector, R):
        number_of_mic = len(self.mic_angle_vector)
        frequency_grid = np.linspace(0, self.sampling_frequency, self.fft_length)
        frequency_grid = frequency_grid[0 : int(self.fft_length / 2) + 1]
        beamformer = np.ones((number_of_mic, len(frequency_grid)), dtype=np.complex64)
        for f in range(0, len(frequency_grid)):
            R_cut = np.reshape(R[:, :, f], [number_of_mic, number_of_mic])
            inv_R = np.linalg.pinv(R_cut)
            a = np.matmul(np.conjugate(steering_vector[:, f]).T, inv_R)
            b = np.matmul(a, steering_vector[:, f])
            b = np.reshape(b, [1, 1])
            beamformer[:, f] = (
                np.matmul(inv_R, steering_vector[:, f]) / b
            )  # number_of_mic *1   = number_of_mic *1 vector/scalar
        return beamformer

    def apply_beamformer(self, beamformer, complex_spectrum):
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)
        enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            enhanced_spectrum[:, f] = np.matmul(
                np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f]
            )
        return spec2wav(
            enhanced_spectrum,
            self.sampling_frequency,
            self.fft_length,
            self.fft_length,
            self.fft_shift,
        )


def chime3_conf():
    return dict(
        mic_diameter=0.1,
        mic_angles=[0, 45, 90, 225, 270, 315],
        look_direction=0,
        nframe=512,
        nhop=256,
        fs=16000,
    )


def run_mvdr(
    mix,
    mic_diameter=0.1,
    mic_angles=[0, 60, 120, 180, 270, 330],
    look_direction=0,
    nframe=512,
    nhop=256,
    fs=16000,
):
    """
    mix: T,M
    mic_diameter
    mic_angles: the angle of each mics
    nframe: frame length
    nhop: shift length
    fs: sampling rate
    """
    complex_spectrum, _ = get_3dim_spectrum_from_data(mix, nframe, nhop, nframe)
    mvdr_beamformer = minimum_variance_distortioless_response(
        np.array(mic_angles),
        mic_diameter,
        sampling_frequency=fs,
        fft_length=nframe,
        fft_shift=nhop,
    )
    steering_vector = mvdr_beamformer.get_sterring_vector(look_direction)
    spatial_correlation_matrix = mvdr_beamformer.get_spatial_correlation_matrix(mix)
    beamformer = mvdr_beamformer.get_mvdr_beamformer(steering_vector, spatial_correlation_matrix)
    enhanced_speech = mvdr_beamformer.apply_beamformer(beamformer, complex_spectrum)
    return enhanced_speech


if __name__ == "__main__":
    f = "/home/deepni/disk/self_test/18000_mic.wav"
    f2 = "/home/deepni/disk/self_test/18000_out.wav"
    data, fs = audioread(f)

    conf = dict(
        mic_diameter=0.2,
        mic_angles=np.arange(0, 360, 360 // 8),
        look_direction=0,
        nframe=512,
        nhop=256,
        fs=16000,
    )
    enh = run_mvdr(data, **conf)
    audiowrite(f2, enh, fs)
