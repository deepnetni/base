import numpy as np
import soundfile as sf
import torch
import os

from einops import rearrange
from models.conv_stft import STFT
from models.VADModel import CRNN_VAD_new, pack_frames_vad, CRNN_VAD_new_origin

if __name__ == "__main__":
    nframe = 512
    nhop = 256
    stft = STFT(nframe=nframe, nhop=nhop, win="hann sqrt")
    inpf = "D:\\share\\1108_noisy_recording.wav"
    # inpf = "D:\\share\\test.wav"
    # inpf = "D:\\share\\input.wav"
    # inpf = "D:\\share\\windnoise_wqy_20230516.wav"
    # inpf = "D:\\share\\MachineGun.wav"
    # inpf = "D:\\share\\DestroyerEngine.wav"
    fname, suffix = os.path.splitext(inpf)
    # outf = f"{fname}_out{suffix}"
    outf = f"{fname}_out_vad_4{suffix}"

    print(outf)
    if os.path.exists(outf):
        os.remove(outf)

    # ckpt = "trained_vad\\crnn_vad_128\\checkpoints\\epoch_0100.pth"
    # ckpt = "trained_vad\\crnn_vad\\checkpoints\\epoch_0063.pth"
    # model = CRNN_VAD_new(nframe // 2 + 1)
    ckpt = "trained_vad\\crnn_vad_origin\\checkpoints\\epoch_0193.pth"
    model = CRNN_VAD_new_origin(nframe // 2 + 1)

    model.load_state_dict(torch.load(ckpt))
    model.eval()

    data, sr = sf.read(inpf, dtype="float32")
    # mixture = torch.from_numpy(data).cuda()
    mixture = torch.from_numpy(data)
    mixture = mixture.unsqueeze(0)
    mixture = stft.transform(mixture) # b,c,t,f

    with torch.no_grad():
        vadp = model(mixture)

    vad_wav = pack_frames_vad(vadp, nframe, nhop)
    # enhanced = enhanced.detach().cpu().squeeze().numpy()
    vad_out = vad_wav.detach().squeeze().numpy()
    vad_out = np.stack([data[:vad_out.shape[0]], vad_out], axis=-1)
    sf.write(outf, vad_out, 16000)