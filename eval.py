import torch
import torch.nn as nn
from vae import VAE
from AE import AE
from stft import STFT
from utils.audiolib import audioread, audiowrite
from matplotlib import pyplot as plt
import librosa


def spec(xk):
    # xk shape is  b,2,t,f
    r, i = xk[:, 0, ...], xk[:, 1, ...]
    mag = (r**2 + i**2) ** 0.5
    xk = 10 * torch.log10(mag**2 + 1e-10)
    return xk.permute(0, 2, 1)


if __name__ == "__main__":
    net = AE()
    stft = STFT(filter_length=512, hop_length=320)
    mic, fs = audioread(r"\\192.168.110.31\dataset\dns_to_liang\1_nearend.wav")
    # mic, fs = audioread(r"D:\dset_test\a.wav")
    if fs != 16000:
        mic = librosa.resample(mic, orig_sr=fs, target_sr=16000)

    ckpt = torch.load(r"D:\pcharm\ae\checkpoints\best.pth")
    net.load_state_dict(ckpt)
    net.eval()
    mic = torch.from_numpy(mic).float()[None, :]
    xk = stft.transform(mic)  # b 2 t f

    with torch.no_grad():
        gen, inp = net(xk)
    inp_spec = spec(inp.detach())
    gen_spec = spec(gen.detach())
    plt.subplot(211)
    plt.imshow(inp_spec[0], origin="lower", aspect="auto", cmap="jet")
    plt.subplot(212)
    plt.imshow(gen_spec[0], origin="lower", aspect="auto", cmap="jet")
    plt.show()

    wav = stft.inverse(gen)[0]
    print(wav.shape)
    audiowrite(r"D:\dset_test\1_gen.wav", wav.detach().numpy())
