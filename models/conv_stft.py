from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window


class STFT(nn.Module):
    def __init__(
        self,
        nframe: int = 512,
        nhop: int = 128,
        win: Union[str, np.ndarray] = "hann",
        nfft: Optional[int] = None,
    ):
        super().__init__()

        self.nframe = nframe
        self.nhop = nhop

        if nfft is None:
            # * rounding up to an exponential multiple of 2
            self.nfft = int(2 ** np.ceil(np.log2(nframe)))
        else:
            self.nfft = nfft

        kernel, window = self.init_conv_stft_kernels(win, False)
        inv_kernel, _ = self.init_conv_stft_kernels(win, True)

        self.register_buffer("weight", kernel)
        self.register_buffer("inv_weight", inv_kernel)
        self.register_buffer("window", window)
        self.register_buffer("enframe", torch.eye(nframe)[:, None, :])

    def init_conv_stft_kernels(self, win=Union[str, np.ndarray], inverse=False):
        if isinstance(win, str):
            if win == "hann sqrt":
                window = get_window(
                    "hann", nframe, fftbins=True
                )  # fftbins=True, win is not symmetric

                window = np.sqrt(window)
            else:
                window = get_window(win, self.nframe, fftbins=True)
        elif isinstance(win, np.ndarray):
            window = win

        N = self.nfft

        # * the fourier_baisis is nframe, N//2 + 1
        # [[ W_N^0x0, W_N^0x1, ..., W_N^0x(N-1) ]
        #  [ W_N^1x0, W_N^1x1, ..., W_N^1x(N-1) ]
        #  [ W_N^2x0, W_N^2x1, ..., W_N^2x(N-1) ]]
        fourier_basis = np.fft.rfft(np.eye(N))[: self.nframe]
        # print(fourier_basis.shape)  # 400, 257
        # * (nframe, nfft // 2 + 1)
        kernel_r, kernel_i = np.real(fourier_basis), np.imag(fourier_basis)

        # * reshape to (2 x (nfft // 2 + 1), nframe)
        kernel = np.concatenate([kernel_r, kernel_i], axis=1).T
        # print(kernel.shape)         # (514, 400)

        if inverse:
            # * A dot pinv(A) = I
            kernel = np.linalg.pinv(kernel).T

        kernel = kernel * window
        # * kernel is (out_channel, inp_channel, kernel_size)
        kernel = kernel[:, None, :]  # (2 x (nfft // 2 + 1), 1, nframe)

        return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(
            window[None, :, None].astype(np.float32)
        )

    def transform(self, x: torch.Tensor):
        """
        x shape should be: [ B, 1, T ] or [ B, T ]
        return: B,2,T,F
        """
        if x.dim() == 2:
            # * expand shape to (:, 1, :)
            x = torch.unsqueeze(x, dim=1)

        x = F.pad(x, (self.nframe // 2, self.nframe // 2))
        # * self.weight shape is [ 2 x (nfft//2 + 1), 1, nframe ]
        out_complex = F.conv1d(x, self.weight, stride=self.nhop)

        dim = self.nfft // 2 + 1
        real = out_complex[:, :dim, :]
        imag = out_complex[:, dim:, :]

        spec = torch.stack([real, imag], dim=1)

        return spec.transpose(-1, -2)

    def inverse(self, spec: torch.Tensor):
        """
        spec: B,2,T,F
        """
        r, i = spec[:, 0, ...], spec[:, 1, ...]  # B,T,F
        inputs = torch.cat([r, i], dim=-1).transpose(-1, -2)  # B,2F,T

        outputs = F.conv_transpose1d(inputs, self.inv_weight, stride=self.nhop)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.nhop)
        outputs = outputs / (coff + 1e-8)

        outputs_ = outputs[..., (self.nframe // 2) : -(self.nframe // 2)]
        return outputs_

    def forward(self, x):
        spec = self.transform(x)
        wav = self.inverse(spec)
        return wav


def verify_w_librosa():
    import librosa

    nframe = 512
    nhop = 128

    inp = torch.randn(1, 16000)
    net = STFT(nframe, nhop, "hann")
    xk = net.transform(inp)
    print("xk", xk.shape)
    out = net.inverse(xk)
    print("xk_", out.shape)
    print(torch.sum((inp - out) ** 2))

    np_inputs = inp.numpy().reshape(-1)
    librosa_stft = librosa.stft(  # B,F,T
        np_inputs,
        win_length=nframe,
        n_fft=nframe,
        hop_length=nhop,
        window="hann",
        center=True,
    )

    librosa_istft = librosa.istft(
        librosa_stft,
        hop_length=nhop,
        win_length=nframe,
        n_fft=nframe,
        window="hann",
        center=True,
    )

    librosa_stft = librosa_stft[None, ...]
    xkk = np.stack([librosa_stft.real, librosa_stft.imag], axis=1)
    xkk = xkk.transpose(-1, -2)
    print(xkk.shape, xk.numpy().shape, np.sum((xk.numpy() - xkk) ** 2))


def verify_self():
    inp = torch.randn(1, 16000)
    net = STFT(512, 320, "hann")
    xk = net.transform(inp)
    print(xk.shape)
    out = net.inverse(xk)
    print(out.shape)
    print(torch.sum((inp - out) ** 2))

    out = net(inp)
    print(torch.sum((inp - out) ** 2))


if __name__ == "__main__":
    verify_self()
    # verify_w_librosa()
