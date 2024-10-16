import torch
import torch.nn as nn
from torch.autograd import Variable
from models.aia_net import (
    AIA_Transformer,
    AIA_Transformer_merge,
    AIA_DCN_Transformer_merge,
    AHAM,
    AHAM_ori,
    AIA_Transformer_new,
)
from models.conv_stft import STFT
import einops
from einops import rearrange
from einops.layers.torch import Rearrange
from utils.register import tables


class SPConvTranspose2d(nn.Module):  # sub-pixel convolution
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):  # dilated dense block
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, "pad{}".format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0))
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(self, "norm{}".format(i + 1), nn.LayerNorm(input_size))
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class dense_encoder(nn.Module):
    def __init__(self, width=64):
        super(dense_encoder, self).__init__()
        self.in_channels = 8
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1)
        )  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(257)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(257, 4, self.width)  # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
        )  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(129)
        self.enc_prelu1 = nn.PReLU(self.width)

        self.enc_conv2 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2)
        )  # [b, 64, nframes, 256]
        self.enc_norm2 = nn.LayerNorm(64)
        self.enc_prelu2 = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        out = self.enc_dense1(out)  # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F // 2]

        x2 = self.enc_prelu2(self.enc_norm2(self.enc_conv2(x)))  # [b, 64, T, F // 4]
        return x2


class dense_encoder_mag(nn.Module):
    def __init__(self, width=64):
        super(dense_encoder_mag, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1)
        )  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(257)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(257, 4, self.width)  # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=(0, 1),
        )  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(129)
        self.enc_prelu1 = nn.PReLU(self.width)

        self.enc_conv2 = nn.Conv2d(
            in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2)
        )  # [b, 64, nframes, 256]
        self.enc_norm2 = nn.LayerNorm(64)
        self.enc_prelu2 = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        out = self.enc_dense1(out)  # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F // 2]

        x2 = self.enc_prelu2(self.enc_norm2(self.enc_conv2(x)))  # [b, 64, T, F // 4]
        return x2


class dense_decoder(nn.Module):
    def __init__(self, width=64):
        super(dense_decoder, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.0)
        self.width = width
        self.dec_dense1 = DenseBlock(64, 4, self.width)

        self.dec_conv1 = SPConvTranspose2d(
            in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2
        )
        self.dec_norm1 = nn.LayerNorm(128)
        self.dec_prelu1 = nn.PReLU(self.width)

        self.dec_conv2 = SPConvTranspose2d(
            in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2
        )
        self.dec_norm2 = nn.LayerNorm(257)
        self.dec_prelu2 = nn.PReLU(self.width)
        #
        self.out_conv = nn.Conv2d(
            in_channels=self.width,
            out_channels=self.out_channels,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad(out))))

        # out = self.dec_conv2(self.pad(out))

        out = self.dec_prelu2(self.dec_norm2(self.pad1(self.dec_conv2(self.pad(out)))))
        #
        out = self.out_conv(out)
        # out.squeeze(dim=1)
        return out


class dense_decoder_masking(nn.Module):
    def __init__(self, width=64):
        super(dense_decoder_masking, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.0)
        self.width = width
        self.dec_dense1 = DenseBlock(64, 4, self.width)

        self.dec_conv1 = SPConvTranspose2d(
            in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2
        )
        self.dec_norm1 = nn.LayerNorm(128)
        self.dec_prelu1 = nn.PReLU(self.width)

        self.dec_conv2 = SPConvTranspose2d(
            in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2
        )
        self.dec_norm2 = nn.LayerNorm(257)
        self.dec_prelu2 = nn.PReLU(self.width)

        self.mask1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1)),
        )
        self.mask2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1)), nn.Tanh()
        )
        self.maskconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))

        self.out_conv = nn.Conv2d(
            in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1)
        )

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad(out))))

        out = self.dec_prelu2(self.dec_norm2(self.pad1(self.dec_conv2(self.pad(out)))))

        out = self.out_conv(out)
        out.squeeze(dim=1)
        out = self.mask1(out) * self.mask2(out)
        out = self.maskconv(out)  # mask
        return out


@tables.register("models", "seu_speech")
class dual_aia_trans_merge_crm(nn.Module):
    def __init__(self):
        super(dual_aia_trans_merge_crm, self).__init__()
        self.en_ri = dense_encoder()
        self.en_mag = dense_encoder_mag()
        self.aia_trans_merge = AIA_Transformer_merge(128, 64, num_layers=2)
        self.aham = AHAM_ori(input_channel=64)
        self.aham_mag = AHAM_ori(input_channel=64)
        self.stft = STFT(512, 256)

        # self.simam = simam_module()
        # self.simam_mag = simam_module()

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()
        self.de_mag_mask = dense_decoder_masking()

    def forward(self, d):
        """
        d: B,T,C
        """
        nB = d.size(0)
        d = rearrange(d, "b t c-> (b c) t")
        xk = self.stft.transform(d)
        x = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)  # c = 2

        batch_size, _, seq_len, _ = x.shape
        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]
        noisy_spec = torch.stack([noisy_real, noisy_imag], 1)
        x_mag_ori, x_phase_ori = torch.norm(noisy_spec, dim=1), torch.atan2(
            noisy_spec[:, -1, :, :], noisy_spec[:, 0, :, :]
        )
        x_mag = x_mag_ori.unsqueeze(dim=1)
        # ri/mag components enconde+ aia_transformer_merge
        x_ri = self.en_ri(x)  # BCTF
        x_mag_en = self.en_mag(x_mag)
        x_last_mag, x_outputlist_mag, x_last_ri, x_outputlist_ri = self.aia_trans_merge(
            x_mag_en, x_ri
        )  # BCTF, #BCTFG

        x_ri = self.aham(x_outputlist_ri)  # BCT
        x_mag_en = self.aham_mag(x_outputlist_mag)  # BCTF

        # x_ri = self.simam(x_ri)
        # x_mag_en = self.simam_mag(x_mag_en)
        x_mag_mask = self.de_mag_mask(x_mag_en)
        x_mag_mask = x_mag_mask.squeeze(dim=1)

        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim=1)
        x_imag = x_imag.squeeze(dim=1)
        # magnitude and ri components interaction

        x_mag_out = x_mag_mask * x_mag_ori
        # x_r_out,x_i_out = (x_mag_out * torch.cos(x_phase_ori) + x_real), (x_mag_out * torch.sin(x_phase_ori)+ x_imag)

        ##### recons by DCCRN
        mask_phase = torch.atan2(x_imag, x_real)

        est_phase = x_phase_ori + mask_phase

        x_r_out = x_mag_out * torch.cos(est_phase)
        x_i_out = x_mag_out * torch.sin(est_phase)

        # x_com_out = torch.stack((x_r_out,x_i_out),dim=1)

        xk = torch.stack([x_r_out, x_i_out], dim=1)
        wav = self.stft.inverse(xk)
        return wav


if __name__ == "__main__":
    from thop import profile
    import warnings

    model = dual_aia_trans_merge_crm()
    model.eval()

    input_test = torch.FloatTensor(1, 16000, 4)  # B,T,M

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This API is being deprecated")
        flops, params = profile(model, inputs=(input_test,), verbose=False)
    print(f"FLOPs={flops / 1e9}, params={params/1e6:.2f}")
