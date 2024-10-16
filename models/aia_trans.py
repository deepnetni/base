import os
import sys
from pathlib import Path
from einops import rearrange
from einops.layers.torch import Rearrange

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from models.aia_net import (
    AHAM,
    AHAM_ori,
    AIA_DCN_Transformer_merge,
    AIA_Transformer,
    AIA_Transformer_cau,
    AIA_Transformer_merge,
    AIA_Transformer_new,
)
from models.multiframe import DF
from models.conv_stft import STFT
from thop import profile
from models.multiframe import DF
from models.McNet import MCNetSpectrum
from models.SpatialFeatures import SpatialFeats
from models.MCAE import MCAE_4


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "lambda=%f)" % self.e_lambda
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )

        return x * self.activaton(y)


class dual_aia_complex_trans(nn.Module):
    def __init__(self):
        super(dual_aia_complex_trans, self).__init__()
        self.en_ri = dense_encoder()
        self.en_mag = dense_encoder_mag()
        self.dual_trans = AIA_Transformer(64, 64, num_layers=4)
        self.aham = AHAM(input_channel=64)
        self.dual_trans_mag = AIA_Transformer(64, 64, num_layers=4)
        self.aham_mag = AHAM(input_channel=64)

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()
        self.de_mag_mask = dense_decoder_masking()

    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        x_mag_ori = torch.norm(x, dim=1)
        x_mag = x_mag_ori.unsqueeze(dim=1)
        x_ri = self.en_ri(x)  # BCTF
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF
        x_mag_en = self.en_mag(x_mag)
        x_last_mag, x_outputlist_mag = self.dual_trans_mag(x_mag_en)  # BCTF, #BCTFG
        x_mag_en = self.aham_mag(x_outputlist_mag)  # BCTF
        x_mag_mask = self.de_mag_mask(x_mag_en)
        x_mag = x_mag_mask * x_mag
        x_mag = x_mag.squeeze(dim=1)
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim=1)
        x_imag = x_imag.squeeze(dim=1)
        x_com = torch.stack((x_real, x_imag), dim=1)
        pre_mag, pre_phase = torch.norm(x_com, dim=1), torch.atan2(
            x_com[:, -1, :, :], x_com[:, 0, :, :]
        )
        x_mag_out = (x_mag + pre_mag) / 2
        x_r_out, x_i_out = x_mag_out * torch.cos(pre_phase), x_mag_out * torch.sin(
            pre_phase
        )
        x_com_out = torch.stack((x_r_out, x_i_out), dim=1)

        return x_com_out


class dual_aia_trans_merge_crm_DCN(nn.Module):
    def __init__(self):
        super(dual_aia_trans_merge_crm_DCN, self).__init__()
        self.en_ri = dense_encoder()
        self.en_mag = dense_encoder_mag()
        self.aia_trans_merge = AIA_DCN_Transformer_merge(128, 64, num_layers=4)
        self.aham = AHAM_ori(input_channel=64)
        self.aham_mag = AHAM_ori(input_channel=64)

        # self.simam = simam_module()
        # self.simam_mag = simam_module()

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()
        self.de_mag_mask = dense_decoder_masking()

    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 8, :, :]
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

        return x_r_out, x_i_out, x_real, x_imag, x_mag_out


class slim_dual_aia_trans_merge(nn.Module):
    def __init__(self):
        super(slim_dual_aia_trans_merge, self).__init__()
        self.en_ri = dense_encoder()

        self.aia_trans = AIA_Transformer(128, 64, num_layers=4)
        self.aham = AHAM_ori(input_channel=64)

        self.de = dense_decoder()

    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]

        x_ri = self.en_ri(x)  # BCTF

        x_last_ri, x_outputlist_ri = self.aia_trans(x_ri)  # BCTF, #BCTFG

        x_ri = self.aham(x_outputlist_ri)  # BCT

        # real and imag decode
        x_mask = self.de(x_ri)

        mask_real = x_mask[:, 0, :, :]
        mask_imag = x_mask[:, 1, :, :]

        ####### reconstruct through DCCRN-E
        #### recons_DCCRN-E

        # spec_mags = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2 + 1e-8)
        # spec_phase = torch.atan2(noisy_imag + 1e-8, noisy_real)
        #
        # mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        # real_phase = mask_real / (mask_mags + 1e-8)
        # imag_phase = mask_imag / (mask_mags + 1e-8)
        # mask_phase = torch.atan2(
        #     imag_phase + 1e-8,
        #     real_phase
        # )
        # # mask_mags = torch.tanh(mask_mags)
        # est_mags = mask_mags * spec_mags
        # est_phase = spec_phase + mask_phase
        # enh_real = est_mags * torch.cos(est_phase)
        # enh_imag = est_mags * torch.sin(est_phase)

        ##### recons DCCRN_C
        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        return enh_real, enh_imag


class dual_aia_trans_merge_crm(nn.Module):
    def __init__(self):
        super(dual_aia_trans_merge_crm, self).__init__()
        self.en_ri = dense_encoder()
        self.en_mag = dense_encoder_mag()
        self.aia_trans_merge = AIA_Transformer_merge(128, 64, num_layers=2)
        self.aham = AHAM_ori(input_channel=64)
        self.aham_mag = AHAM_ori(input_channel=64)

        # self.simam = simam_module()
        # self.simam_mag = simam_module()

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()
        self.de_mag_mask = dense_decoder_masking()

    def forward(self, x):
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

        return x_r_out, x_i_out


class aia_complex_trans_ri(nn.Module):
    def __init__(self):
        super(aia_complex_trans_ri, self).__init__()
        self.en_ri = dense_encoder()

        self.dual_trans = AIA_Transformer(96, 96, num_layers=4)
        self.aham = AHAM(input_channel=96)

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()

    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape

        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]

        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x)  # BCTF
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim=1)
        x_imag = x_imag.squeeze(dim=1)

        # x_com=torch.stack((x_real, x_imag), dim=1)

        enh_real = noisy_real * x_real - noisy_imag * x_imag
        enh_imag = noisy_real * x_imag + noisy_imag * x_real

        return enh_real, enh_imag


class aia_complex_trans_ri_new(nn.Module):
    def __init__(self):
        super(aia_complex_trans_ri_new, self).__init__()
        self.en_ri = dense_encoder()

        self.dual_trans = AIA_Transformer_new(96, 96, num_layers=4)
        self.aham = AHAM(input_channel=96)

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()

    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape

        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]

        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x)  # BCTF
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim=1)
        x_imag = x_imag.squeeze(dim=1)

        # x_com=torch.stack((x_real, x_imag), dim=1)

        enh_real = noisy_real * x_real - noisy_imag * x_imag
        enh_imag = noisy_real * x_imag + noisy_imag * x_real

        return enh_real, enh_imag


class dense_encoder_pwc(nn.Module):
    """
    Input: B,C,T,F

    Arugments:
      - in_chanels: C of input;
      - feature_size: F of input
    """

    def __init__(
        self, in_channels: int, feature_size: int, out_channels, depth: int = 4
    ):
        super(dense_encoder_pwc, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pre_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
            ),  # [b, 64, nframes, 512]
            nn.LayerNorm(feature_size),
            nn.PReLU(out_channels),
        )
        self.enc_dense = DenseBlockPWC(
            depth=depth, in_channels=out_channels, input_size=feature_size
        )

        self.post_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # //2
                padding=(0, 1),
            ),
            nn.LayerNorm(feature_size // 2 + 1),
            nn.PReLU(out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # no padding
            ),
            nn.LayerNorm(feature_size // 4),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.enc_dense(x)
        x = self.post_conv(x)

        return x


class dense_encoder(nn.Module):
    """
    Input: B,C,T,F

    Arugments:
      - in_chanels: C of input;
      - feature_size: F of input
    """

    def __init__(
        self, in_channels: int, feature_size: int, out_channels, depth: int = 4
    ):
        super(dense_encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pre_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
            ),  # [b, 64, nframes, 512]
            nn.LayerNorm(feature_size),
            nn.PReLU(out_channels),
        )
        self.enc_dense = DenseBlock(
            depth=depth, in_channels=out_channels, input_size=feature_size
        )

        self.post_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # //2
                padding=(0, 1),
            ),
            nn.LayerNorm(feature_size // 2 + 1),
            nn.PReLU(out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # no padding
            ),
            nn.LayerNorm(feature_size // 4),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.enc_dense(x)
        x = self.post_conv(x)

        return x


class dense_encoder_mag(nn.Module):
    def __init__(self, width=64):
        super(dense_encoder_mag, self).__init__()
        self.in_channels = 6
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1)
        )  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(257)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(4, self.width, 257)  # [b, 64, nframes, 512]
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
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=(1, 3),
            stride=(1, 2),
        )  # [b, 64, nframes, 256]
        self.enc_norm2 = nn.LayerNorm(64)
        self.enc_prelu2 = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        out = self.enc_dense1(out)  # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F // 2]

        x2 = self.enc_prelu2(self.enc_norm2(self.enc_conv2(x)))  # [b, 64, T, F // 4]
        return x2


class dense_decoder_pwc(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, feature_size: int, depth: int = 4
    ):
        super(dense_decoder_pwc, self).__init__()
        self.in_channels = in_channels
        self.dec_dense = DenseBlockPWC(
            depth=depth, in_channels=in_channels, input_size=feature_size
        )

        self.dec_conv1 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),  # padding F
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),
            nn.LayerNorm(feature_size * 2),
            nn.PReLU(in_channels),
        )

        feature_size = feature_size * 2

        self.dec_conv2 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),
            nn.ConstantPad2d((1, 0, 0, 0), value=0.0),
            nn.LayerNorm(feature_size * 2 + 1),
            nn.PReLU(in_channels),
        )
        #
        self.out_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

    def forward(self, x):
        out = self.dec_dense(x)
        out = self.dec_conv1(out)

        out = self.dec_conv2(out)
        out = self.out_conv(out)
        return out


class dense_decoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, feature_size: int, depth: int = 4
    ):
        super(dense_decoder, self).__init__()
        self.out_channels = 1
        self.in_channels = in_channels
        self.dec_dense = DenseBlock(
            depth=depth, in_channels=in_channels, input_size=feature_size
        )

        self.dec_conv1 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),  # padding F
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),
            nn.LayerNorm(feature_size * 2),
            nn.PReLU(in_channels),
        )

        feature_size = feature_size * 2

        self.dec_conv2 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),
            nn.ConstantPad2d((1, 0, 0, 0), value=0.0),
            nn.LayerNorm(feature_size * 2 + 1),
            nn.PReLU(in_channels),
        )
        #
        self.out_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

    def forward(self, x):
        out = self.dec_dense(x)
        out = self.dec_conv1(out)

        out = self.dec_conv2(out)
        out = self.out_conv(out)
        return out


class dense_decoder_masking(nn.Module):
    def __init__(self, width=96):
        super(dense_decoder_masking, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.0)
        self.width = width
        self.dec_dense1 = DenseBlock(4, self.width, 64)

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


class SPConvTranspose2d(nn.Module):  # sub-pixel convolution
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        # batch_size, nchannels, H, W = out.shape
        out = rearrange(out, "b (r c) t f-> b c t (f r)", r=self.r)
        # out = out.view((batch_size, self.r, nchannels // self.r, H, W))  # b,r1,r2,h,w
        # out = out.permute(0, 2, 3, 4, 1)  # b,r2,h,w,r1
        # out = out.contiguous().view(
        #     (batch_size, nchannels // self.r, H, -1)
        # )  # b, r2, h, w*r
        return out


class DenseBlockPWC(nn.Module):  # dilated dense block
    def __init__(self, depth=4, in_channels=64, input_size: int = 257):
        super(DenseBlockPWC, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        # self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Sequential(
                    nn.Conv2d(
                        self.in_channels * (i + 1),
                        self.in_channels,
                        kernel_size=self.kernel_size,
                        dilation=(dil, 1),
                    ),
                    nn.LayerNorm(input_size),
                    nn.PReLU(in_channels),
                    nn.Conv2d(in_channels, in_channels, (1, 3), (1, 1), (0, 1)),
                    nn.LayerNorm(input_size),
                    nn.PReLU(in_channels),
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


class DenseBlock(nn.Module):  # dilated dense block
    def __init__(self, depth=4, in_channels=64, input_size: int = 257):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        # self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
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


class DF_dense_decoder(nn.Module):
    def __init__(self, in_channel: int, feature_size: int, out_channel: int):
        super(DF_dense_decoder, self).__init__()
        self.dec_dense1 = DenseBlock(4, in_channel, feature_size)

        self.dec_conv1 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            SPConvTranspose2d(
                in_channels=in_channel, out_channels=in_channel, kernel_size=(1, 3), r=2
            ),
            nn.LayerNorm(feature_size * 2),
            nn.PReLU(in_channel),
        )

        feature_size = feature_size * 2

        self.dec_conv2 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            SPConvTranspose2d(
                in_channels=in_channel, out_channels=in_channel, kernel_size=(1, 3), r=2
            ),  # B,C,T,2F
            nn.ConstantPad2d((1, 0, 0, 0), value=0.0),  # BCT,2F+1
            nn.LayerNorm(feature_size * 2 + 1),
            nn.PReLU(in_channel),
        )
        #
        self.out_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=(1, 5),
            padding=(0, 2),
        )

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_conv1(out)

        # out = self.dec_conv2(self.pad(out))

        out = self.dec_conv2(out)
        #
        out = self.out_conv(out)
        # out.squeeze(dim=1)
        return out


class DfOutputReshapeMF(nn.Module):
    """Coefficients output reshape for multiframe/MultiFrameModule

    Requires input of shape B, C, T, F, 2.
    """

    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs):
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        new_shape = list(coefs.shape)
        new_shape[-1] = -1
        new_shape.append(2)
        coefs = coefs.view(new_shape)
        coefs = coefs.permute(0, 3, 1, 2, 4)
        return coefs


class DF_aia_complex_trans_ri(nn.Module):
    def __init__(self):
        super(DF_aia_complex_trans_ri, self).__init__()
        self.en_ri = dense_encoder()

        self.dual_trans = AIA_Transformer(96, 96, num_layers=4)
        self.aham = AHAM(input_channel=96)

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()

        self.DF_de = DF_dense_decoder()

        self.df_order = 5
        self.df_bins = 481

        self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape

        noisy_real = x[:, 0, :, :]
        noisy_imag = x[:, 1, :, :]

        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x)  # BCTF
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim=1)
        x_imag = x_imag.squeeze(dim=1)

        df_coefs = self.DF_de(x_ri)

        df_coefs = df_coefs.permute(0, 2, 3, 1)

        df_coefs = self.df_out_transform(df_coefs).contiguous()

        # x_com=torch.stack((x_real, x_imag), dim=1)

        enh_real = noisy_real * x_real - noisy_imag * x_imag
        enh_imag = noisy_real * x_imag + noisy_imag * x_real

        enhanced_D = torch.stack([enh_real, enh_imag], 3)

        enhanced_D = enhanced_D.unsqueeze(1)

        DF_spec = self.df_op(enhanced_D, df_coefs)

        DF_spec = DF_spec.squeeze(1)

        DF_real = DF_spec[:, :, :, 0]
        DF_imag = DF_spec[:, :, :, 1]

        return DF_real, DF_imag


class RNN_FC(nn.Module):
    """
    feature_size: F of b,c,f,t
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = False,
        act: bool = True,
    ):
        super().__init__()
        # Sequence layer
        self.sequence_model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.sequence_model.flatten_parameters()

        # Fully connected layer

        if bidirectional:
            self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc_output_layer = nn.Linear(hidden_size, output_size)

        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, Feature]
        Returns:
            [B, T, Feature]
        """
        x, _ = self.sequence_model(x)
        x = self.act(self.fc_output_layer(x))

        return x


class Fusion_F(nn.Module):
    """
    input: B,C,T,F

    Arguments
    ---------
    feature_size, the `F` size of B,C,T,F
    """

    def __init__(self, inp_channels, mid_channels: int, feature_size: int, r=2) -> None:
        super().__init__()
        assert feature_size % r == 0, f"{feature_size}%{r}"

        self.pre = nn.Sequential(
            nn.Conv2d(inp_channels, mid_channels, (1, 1), (1, 1)),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(mid_channels),
        )

        self.layer_global = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=(1, feature_size), stride=(1, feature_size)
            ),  # B,C,T,1
            # B,C,T,1
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels // r,
                kernel_size=1,
                stride=1,
                padding=0,
                # groups=feature_size,
            ),  # b,t,f,1
            Rearrange("b c t f->b t f c"),
            nn.LayerNorm(mid_channels),
            Rearrange("b t f c->b c t f"),
            nn.PReLU(mid_channels),
            nn.Conv2d(
                in_channels=mid_channels // r,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                # groups=feature_size // r,
            ),
        )

        self.layer_local = nn.Sequential(
            Rearrange("b c t f->b f t c"),
            nn.Conv2d(
                in_channels=feature_size,
                out_channels=feature_size // r,
                kernel_size=1,
                stride=1,
                padding=0,
                # groups=feature_size,
            ),
            nn.LayerNorm(mid_channels),
            nn.PReLU(feature_size),
            nn.Conv2d(
                in_channels=feature_size // r,
                out_channels=feature_size,
                kernel_size=1,
                stride=1,
                padding=0,
                # groups=feature_size // r,
            ),
            Rearrange("b f t c->b c t f"),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x: BCTF
        y: BCTF
        """
        # x_ = x + y
        x_ = torch.concat([x, y], dim=1)
        x_ = self.pre(x_)
        x_2 = self.layer_global(x_) + self.layer_local(x_)
        w = x_2.sigmoid()
        return x * w + (1 - w) * y


class Fusion(nn.Module):
    """
    input: B,C,T,F

    Arguments
    ---------
    feature_size, the `F` size of B,C,T,F
    """

    def __init__(self, inp_channels: int, feature_size: int, r=2) -> None:
        super().__init__()
        assert feature_size % r == 0, f"{feature_size}%{r}"

        self.layer_global = nn.Sequential(
            # B,C,T,1
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels // r,
                kernel_size=1,
                stride=1,
                padding=0,
                # groups=feature_size,
            ),  # b,t,f,1
            Rearrange("b c t f->b t f c"),
            nn.LayerNorm(inp_channels),
            Rearrange("b t f c->b c t f"),
            nn.PReLU(inp_channels),
            nn.Conv2d(
                in_channels=inp_channels // r,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                # groups=feature_size // r,
            ),
        )

        self.layer_local = nn.Sequential(
            Rearrange("b c t f->b f t c"),
            nn.Conv2d(
                in_channels=feature_size,
                out_channels=feature_size // r,
                kernel_size=1,
                stride=1,
                padding=0,
                # groups=feature_size,
            ),
            nn.LayerNorm(inp_channels),
            nn.PReLU(feature_size),
            nn.Conv2d(
                in_channels=feature_size // r,
                out_channels=feature_size,
                kernel_size=1,
                stride=1,
                padding=0,
                # groups=feature_size // r,
            ),
            Rearrange("b f t c->b c t f"),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x_ = x + y
        x_2 = self.layer_global(y) + self.layer_local(x)
        w = x_2.sigmoid()
        return x * w + (1 - w) * y


class DF_AIA_TRANS_AE_F(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_size: int,
        mid_channels: int,
        ref_channel: int = 4,
    ):
        super().__init__()
        self.stft = STFT(512, 256)
        self.en_ri = dense_encoder(
            in_channels=in_channels * 2,
            out_channels=mid_channels,
            feature_size=feature_size,
            depth=4,
        )  # B, mid_c, T, F // 4

        mic_array = [
            [-0.1, 0.095, 0],
            [0, 0.095, 0],
            [0.1, 0.095, 0],
            [-0.1, -0.095, 0],
            [0, -0.095, 0],
            [0.1, -0.095, 0],
        ]
        self.spf_alg = SpatialFeats(mic_array)
        spf_in_channel = 93  # 15 * 5 + 18
        self.en_spf = nn.Sequential(
            dense_encoder(
                in_channels=spf_in_channel,
                out_channels=mid_channels,
                feature_size=feature_size,
                depth=4,
            ),  # B, mid_c, T, F // 4
            nn.Conv2d(mid_channels, mid_channels, (1, 1), (1, 1)),
            nn.Tanh(),
        )

        # self.z_pre = nn.Sequential(
        #     nn.Conv2d(192, mid_channels, (1, 1), (1, 1)),
        #     nn.LayerNorm(feature_size // 4 + 1),
        #     nn.PReLU(mid_channels),
        #     nn.ConstantPad2d((1, 0, 0, 0), value=0.0),
        #     nn.Conv2d(mid_channels, mid_channels, (1, 3), (1, 1)),
        #     nn.Tanh(),
        # )

        self.z_pre = nn.Sequential(
            Rearrange("b c t f m -> b (m c) t f"),
            nn.Conv2d(96, mid_channels, (1, 1), (1, 1)),
            nn.BatchNorm2d(mid_channels),
            nn.ConstantPad2d((1, 0, 0, 0), value=0.0),
            nn.Conv2d(mid_channels, mid_channels, (1, 3), (1, 1)),
            nn.PReLU(mid_channels),
        )

        self.fusion = Fusion_F(2 * mid_channels, mid_channels, 64, r=1)

        self.dual_trans = AIA_Transformer_cau(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)
        self.ref_channel = ref_channel
        self.in_channels = in_channels

        self.de1 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )
        self.de2 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )

        self.df_order = 5
        self.df_bins = feature_size
        self.DF_de = DF_dense_decoder(
            mid_channels, feature_size // 4, 2 * self.df_order
        )

        self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x, c):
        """
        x: B,T,C
        z: B,C(6x32,192),T,F
        c: B,C(40),T,F,M(6)
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)
        x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)

        # noisy_real = x[:, self.ref_channel, :, :]
        # noisy_imag = x[:, self.ref_channel + self.in_channels, :, :]

        x_spf = self.en_spf(self.spf_alg(x_))
        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x)  # BCTF, BCT1
        # x_ri = x_ri * x_spf + self.z_pre(c)
        # x_ri = (x_ri + self.z_pre(c)) * x_spf
        x_ri = x_ri * x_spf
        x_ri = self.fusion(x_ri, self.z_pre(c))

        # * FTLSTM
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # real and imag decode
        x_real = self.de1(x_ri)  # B,1,T,F
        x_imag = self.de2(x_ri)  # B,1,T,F
        # x_real, x_imag = x_.chunk(2, dim=1)
        x_real = x_real.squeeze(dim=1)  # B,T,F
        x_imag = x_imag.squeeze(dim=1)
        # enh_real = noisy_real * x_real - noisy_imag * x_imag
        # enh_imag = noisy_real * x_imag + noisy_imag * x_real

        enhanced_D = torch.stack([x_real, x_imag], 3)  # B,T,F,2
        enhanced_D = enhanced_D.unsqueeze(1)  # B,1,T,F,2

        # DF coeffs decoder
        df_coefs = self.DF_de(x_ri)  # BCTF
        df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        DF_spec = self.df_op(enhanced_D, df_coefs)  # B,1,T,F,2
        DF_spec = DF_spec.squeeze(1)

        DF_real = DF_spec[:, :, :, 0]
        DF_imag = DF_spec[:, :, :, 1]

        feat = torch.stack([DF_real, DF_imag], dim=1)  # B,2,T,F

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class DF_AIA_TRANS_AELOSS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_size: int,
        mid_channels: int,
        ref_channel: int = 4,
    ):
        super().__init__()
        self.stft = STFT(512, 256)
        self.en_ri = dense_encoder(
            in_channels=in_channels * 2,
            out_channels=mid_channels,
            feature_size=feature_size,
            depth=4,
        )  # B, mid_c, T, F // 4

        mic_array = [
            [-0.1, 0.095, 0],
            [0, 0.095, 0],
            [0.1, 0.095, 0],
            [-0.1, -0.095, 0],
            [0, -0.095, 0],
            [0.1, -0.095, 0],
        ]
        self.ae = MCAE_4(nframe=512, nhop=256, nfft=512, in_channels=6, mid_channel=32)
        self.spf_alg = SpatialFeats(mic_array)
        spf_in_channel = 93  # 15 * 5 + 18
        self.en_spf = nn.Sequential(
            dense_encoder(
                in_channels=spf_in_channel,
                out_channels=mid_channels,
                feature_size=feature_size,
                depth=4,
            ),  # B, mid_c, T, F // 4
            nn.Conv2d(mid_channels, mid_channels, (1, 1), (1, 1)),
            nn.Tanh(),
        )

        # self.z_pre = nn.Sequential(
        #     nn.Conv2d(192, mid_channels, (1, 1), (1, 1)),
        #     nn.LayerNorm(feature_size // 4 + 1),
        #     nn.PReLU(mid_channels),
        #     nn.ConstantPad2d((1, 0, 0, 0), value=0.0),
        #     nn.Conv2d(mid_channels, mid_channels, (1, 3), (1, 1)),
        #     nn.Tanh(),
        # )

        self.z_pre = nn.Sequential(
            Rearrange("b c t m -> b (c m) t 1"),
            nn.Conv2d(32 * 6, mid_channels, (1, 1), (1, 1)),
            Rearrange("b c t 1 -> b t 1 c"),
            nn.LayerNorm(mid_channels),
            Rearrange("b t 1 c -> b c t 1"),
            nn.PReLU(mid_channels),
        )

        self.fusion = Fusion(mid_channels, 64, r=1)

        self.dual_trans = AIA_Transformer_cau(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)
        self.ref_channel = ref_channel
        self.in_channels = in_channels

        self.de1 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )
        self.de2 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )

        self.df_order = 5
        self.df_bins = feature_size
        self.DF_de = DF_dense_decoder(
            mid_channels, feature_size // 4, 2 * self.df_order
        )

        self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        z: B,C(6x32,192),T,F
        c: B,C(40),T,M(6)
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)
        x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)

        c, l = self.ae(x)
        cpcloss, corr = self.ae.loss(c, l)

        # noisy_real = x[:, self.ref_channel, :, :]
        # noisy_imag = x[:, self.ref_channel + self.in_channels, :, :]

        x_spf = self.en_spf(self.spf_alg(x_))
        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x)  # BCTF, BCT1
        x_ri = x_ri * x_spf + self.z_pre(c)
        x_ri = (x_ri + self.z_pre(c)) * x_spf
        x_ri = x_ri * x_spf
        x_ri = self.fusion(x_ri, self.z_pre(c))
        # x_ri = self.z_pre(c)

        # * FTLSTM
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # real and imag decode
        x_real = self.de1(x_ri)  # B,1,T,F
        x_imag = self.de2(x_ri)  # B,1,T,F
        # x_real, x_imag = x_.chunk(2, dim=1)
        x_real = x_real.squeeze(dim=1)  # B,T,F
        x_imag = x_imag.squeeze(dim=1)
        # enh_real = noisy_real * x_real - noisy_imag * x_imag
        # enh_imag = noisy_real * x_imag + noisy_imag * x_real

        enhanced_D = torch.stack([x_real, x_imag], 3)  # B,T,F,2
        enhanced_D = enhanced_D.unsqueeze(1)  # B,1,T,F,2

        # DF coeffs decoder
        df_coefs = self.DF_de(x_ri)  # BCTF
        df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        DF_spec = self.df_op(enhanced_D, df_coefs)  # B,1,T,F,2
        DF_spec = DF_spec.squeeze(1)

        DF_real = DF_spec[:, :, :, 0]
        DF_imag = DF_spec[:, :, :, 1]

        feat = torch.stack([DF_real, DF_imag], dim=1)  # B,2,T,F

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav, cpcloss, corr


class DF_AIA_TRANS_AE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_size: int,
        mid_channels: int,
        ref_channel: int = 4,
    ):
        super().__init__()
        self.stft = STFT(512, 256)
        self.en_ri = dense_encoder(
            in_channels=in_channels * 2,
            out_channels=mid_channels,
            feature_size=feature_size,
            depth=4,
        )  # B, mid_c, T, F // 4

        mic_array = [
            [-0.1, 0.095, 0],
            [0, 0.095, 0],
            [0.1, 0.095, 0],
            [-0.1, -0.095, 0],
            [0, -0.095, 0],
            [0.1, -0.095, 0],
        ]
        self.spf_alg = SpatialFeats(mic_array)
        spf_in_channel = 93  # 15 * 5 + 18
        self.en_spf = nn.Sequential(
            dense_encoder(
                in_channels=spf_in_channel,
                out_channels=mid_channels,
                feature_size=feature_size,
                depth=4,
            ),  # B, mid_c, T, F // 4
            nn.Conv2d(mid_channels, mid_channels, (1, 1), (1, 1)),
            nn.Tanh(),
        )

        # self.z_pre = nn.Sequential(
        #     nn.Conv2d(192, mid_channels, (1, 1), (1, 1)),
        #     nn.LayerNorm(feature_size // 4 + 1),
        #     nn.PReLU(mid_channels),
        #     nn.ConstantPad2d((1, 0, 0, 0), value=0.0),
        #     nn.Conv2d(mid_channels, mid_channels, (1, 3), (1, 1)),
        #     nn.Tanh(),
        # )

        self.z_pre = nn.Sequential(
            Rearrange("b c t m -> b (c m) t 1"),
            nn.Conv2d(240, mid_channels, (1, 1), (1, 1)),
            Rearrange("b c t 1 -> b t 1 c"),
            nn.LayerNorm(mid_channels),
            Rearrange("b t 1 c -> b c t 1"),
            nn.PReLU(mid_channels),
        )

        self.fusion = Fusion(mid_channels, 64, r=1)

        self.dual_trans = AIA_Transformer_cau(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)
        self.ref_channel = ref_channel
        self.in_channels = in_channels

        self.de1 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )
        self.de2 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )

        self.df_order = 5
        self.df_bins = feature_size
        self.DF_de = DF_dense_decoder(
            mid_channels, feature_size // 4, 2 * self.df_order
        )

        self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x, c):
        """
        x: B,T,C
        z: B,C(6x32,192),T,F
        c: B,C(40),T,M(6)
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)
        x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)

        # noisy_real = x[:, self.ref_channel, :, :]
        # noisy_imag = x[:, self.ref_channel + self.in_channels, :, :]

        x_spf = self.en_spf(self.spf_alg(x_))
        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x)  # BCTF, BCT1
        # x_ri = x_ri * x_spf + self.z_pre(c)
        # x_ri = (x_ri + self.z_pre(c)) * x_spf
        x_ri = x_ri * x_spf
        x_ri = self.fusion(x_ri, self.z_pre(c))

        # * FTLSTM
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # real and imag decode
        x_real = self.de1(x_ri)  # B,1,T,F
        x_imag = self.de2(x_ri)  # B,1,T,F
        # x_real, x_imag = x_.chunk(2, dim=1)
        x_real = x_real.squeeze(dim=1)  # B,T,F
        x_imag = x_imag.squeeze(dim=1)
        # enh_real = noisy_real * x_real - noisy_imag * x_imag
        # enh_imag = noisy_real * x_imag + noisy_imag * x_real

        enhanced_D = torch.stack([x_real, x_imag], 3)  # B,T,F,2
        enhanced_D = enhanced_D.unsqueeze(1)  # B,1,T,F,2

        # DF coeffs decoder
        df_coefs = self.DF_de(x_ri)  # BCTF
        df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        DF_spec = self.df_op(enhanced_D, df_coefs)  # B,1,T,F,2
        DF_spec = DF_spec.squeeze(1)

        DF_real = DF_spec[:, :, :, 0]
        DF_imag = DF_spec[:, :, :, 1]

        feat = torch.stack([DF_real, DF_imag], dim=1)  # B,2,T,F

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class AIA_TRANS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_size: int,
        mid_channels: int,
        ref_channel: int = 4,
    ):
        super().__init__()
        self.stft = STFT(512, 256)
        self.en_ri = dense_encoder(
            in_channels=in_channels * 2,
            out_channels=mid_channels,
            feature_size=feature_size,
            depth=4,
        )  # B, mid_c, T, F // 4

        self.dual_trans = AIA_Transformer_cau(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)
        self.ref_channel = ref_channel
        self.in_channels = in_channels

        self.de1 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )
        self.de2 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )

        self.df_order = 5
        self.df_bins = feature_size
        self.DF_de = DF_dense_decoder(
            mid_channels, feature_size // 4, 2 * self.df_order
        )

        self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)
        # x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)

        # noisy_real = x[:, self.ref_channel, :, :]
        # noisy_imag = x[:, self.ref_channel + self.in_channels, :, :]

        # x_spf = self.en_spf(self.spf_alg(x_))
        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x)  # BCTF
        # x_ri = x_ri * x_spf

        # * FTLSTM
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # real and imag decode
        x_real = self.de1(x_ri)  # B,1,T,F
        x_imag = self.de2(x_ri)  # B,1,T,F
        # x_real, x_imag = x_.chunk(2, dim=1)
        x_real = x_real.squeeze(dim=1)  # B,T,F
        x_imag = x_imag.squeeze(dim=1)
        # enh_real = noisy_real * x_real - noisy_imag * x_imag
        # enh_imag = noisy_real * x_imag + noisy_imag * x_real

        enhanced_D = torch.stack([x_real, x_imag], 3)  # B,T,F,2
        enhanced_D = enhanced_D.unsqueeze(1)  # B,1,T,F,2

        # DF coeffs decoder
        df_coefs = self.DF_de(x_ri)  # BCTF
        df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        DF_spec = self.df_op(enhanced_D, df_coefs)  # B,1,T,F,2
        DF_spec = DF_spec.squeeze(1)

        DF_real = DF_spec[:, :, :, 0]
        DF_imag = DF_spec[:, :, :, 1]

        feat = torch.stack([DF_real, DF_imag], dim=1)  # B,2,T,F

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class DF_AIA_TRANS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_size: int,
        mid_channels: int,
        ref_channel: int = 4,
    ):
        super().__init__()
        self.stft = STFT(512, 256)
        self.en_ri = dense_encoder(
            in_channels=in_channels * 2,
            out_channels=mid_channels,
            feature_size=feature_size,
            depth=4,
        )  # B, mid_c, T, F // 4

        mic_array = [
            [-0.1, 0.095, 0],
            [0, 0.095, 0],
            [0.1, 0.095, 0],
            [-0.1, -0.095, 0],
            [0, -0.095, 0],
            [0.1, -0.095, 0],
        ]
        self.spf_alg = SpatialFeats(mic_array)
        spf_in_channel = 93  # 15 * 5 + 18
        self.en_spf = nn.Sequential(
            dense_encoder(
                in_channels=spf_in_channel,
                out_channels=mid_channels,
                feature_size=feature_size,
                depth=4,
            ),  # B, mid_c, T, F // 4
            nn.Conv2d(mid_channels, mid_channels, (1, 1), (1, 1)),
            nn.Tanh(),
        )

        self.dual_trans = AIA_Transformer_cau(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)
        self.ref_channel = ref_channel
        self.in_channels = in_channels

        self.de1 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )
        self.de2 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )

        self.df_order = 5
        self.df_bins = feature_size
        self.DF_de = DF_dense_decoder(
            mid_channels, feature_size // 4, 2 * self.df_order
        )

        self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)
        x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)

        # noisy_real = x[:, self.ref_channel, :, :]
        # noisy_imag = x[:, self.ref_channel + self.in_channels, :, :]

        x_spf = self.en_spf(self.spf_alg(x_))
        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x)  # BCTF
        x_ri = x_ri * x_spf

        # * FTLSTM
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # real and imag decode
        x_real = self.de1(x_ri)  # B,1,T,F
        x_imag = self.de2(x_ri)  # B,1,T,F
        # x_real, x_imag = x_.chunk(2, dim=1)
        x_real = x_real.squeeze(dim=1)  # B,T,F
        x_imag = x_imag.squeeze(dim=1)
        # enh_real = noisy_real * x_real - noisy_imag * x_imag
        # enh_imag = noisy_real * x_imag + noisy_imag * x_real

        enhanced_D = torch.stack([x_real, x_imag], 3)  # B,T,F,2
        enhanced_D = enhanced_D.unsqueeze(1)  # B,1,T,F,2

        # DF coeffs decoder
        df_coefs = self.DF_de(x_ri)  # BCTF
        df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        DF_spec = self.df_op(enhanced_D, df_coefs)  # B,1,T,F,2
        DF_spec = DF_spec.squeeze(1)

        DF_real = DF_spec[:, :, :, 0]
        DF_imag = DF_spec[:, :, :, 1]

        feat = torch.stack([DF_real, DF_imag], dim=1)  # B,2,T,F

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class DF_AIA_TRANS_densepwc(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_size: int,
        mid_channels: int,
        ref_channel: int = 4,
    ):
        super().__init__()
        self.stft = STFT(512, 256)
        self.en_ri = dense_encoder_pwc(
            in_channels=in_channels * 2,
            out_channels=mid_channels,
            feature_size=feature_size,
            depth=4,
        )  # B, mid_c, T, F // 4

        mic_array = [
            [-0.1, 0.095, 0],
            [0, 0.095, 0],
            [0.1, 0.095, 0],
            [-0.1, -0.095, 0],
            [0, -0.095, 0],
            [0.1, -0.095, 0],
        ]
        self.spf_alg = SpatialFeats(mic_array)
        spf_in_channel = 93  # 15 * 5 + 18
        self.en_spf = nn.Sequential(
            dense_encoder_pwc(
                in_channels=spf_in_channel,
                out_channels=mid_channels,
                feature_size=feature_size,
                depth=4,
            ),  # B, mid_c, T, F // 4
            nn.Conv2d(mid_channels, mid_channels, (1, 1), (1, 1)),
            nn.Tanh(),
        )

        self.dual_trans = AIA_Transformer_cau(mid_channels, mid_channels, num_layers=4)
        self.aham = AHAM(input_channel=mid_channels)
        self.ref_channel = ref_channel
        self.in_channels = in_channels

        self.de1 = dense_decoder_pwc(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )
        self.de2 = dense_decoder_pwc(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )

        self.df_order = 5
        self.df_bins = feature_size
        self.DF_de = DF_dense_decoder(
            mid_channels, feature_size // 4, 2 * self.df_order
        )

        self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, x):
        """
        x: B,T,C
        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)
        x_ = rearrange(xk, "(b m) c t f->b (m c) t f", b=nB)

        # noisy_real = x[:, self.ref_channel, :, :]
        # noisy_imag = x[:, self.ref_channel + self.in_channels, :, :]

        x_spf = self.en_spf(self.spf_alg(x_))
        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x)  # BCTF
        x_ri = x_ri * x_spf

        # * FTLSTM
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # real and imag decode
        x_real = self.de1(x_ri)  # B,1,T,F
        x_imag = self.de2(x_ri)  # B,1,T,F
        # x_real, x_imag = x_.chunk(2, dim=1)
        x_real = x_real.squeeze(dim=1)  # B,T,F
        x_imag = x_imag.squeeze(dim=1)
        # enh_real = noisy_real * x_real - noisy_imag * x_imag
        # enh_imag = noisy_real * x_imag + noisy_imag * x_real

        enhanced_D = torch.stack([x_real, x_imag], 3)  # B,T,F,2
        enhanced_D = enhanced_D.unsqueeze(1)  # B,1,T,F,2

        # DF coeffs decoder
        df_coefs = self.DF_de(x_ri)  # BCTF
        df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
        df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

        DF_spec = self.df_op(enhanced_D, df_coefs)  # B,1,T,F,2
        DF_spec = DF_spec.squeeze(1)

        DF_real = DF_spec[:, :, :, 0]
        DF_imag = DF_spec[:, :, :, 1]

        feat = torch.stack([DF_real, DF_imag], dim=1)  # B,2,T,F

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


class dual_aia_trans_chime(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_size: int,
        mid_channels: int,
        ref_channel: int = 4,
    ):
        super().__init__()

        self.stft = STFT(512, 256)
        self.en_ri = dense_encoder(
            in_channels=in_channels * 2,
            out_channels=mid_channels,
            feature_size=feature_size,
            depth=4,
        )  # B, mid_c, T, F // 4

        self.en_mag = dense_encoder_mag(mid_channels)
        self.aia_trans_merge = AIA_Transformer_merge(
            mid_channels * 2, mid_channels, num_layers=4
        )
        self.aham = AHAM_ori(input_channel=mid_channels)
        self.aham_mag = AHAM_ori(input_channel=mid_channels)

        # self.simam = simam_module()
        # self.simam_mag = simam_module()

        self.de1 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )
        self.de2 = dense_decoder(
            in_channels=mid_channels,
            out_channels=1,
            feature_size=feature_size // 4,
            depth=4,
        )
        self.de_mag_mask = dense_decoder_masking(mid_channels)

    def forward(self, x):
        """
        Args:
            x: B,T,C

        Returns:

        """
        nB = x.size(0)
        x = rearrange(x, "b t c-> (b c) t")
        xk = self.stft.transform(x)
        # x shape is B,12,T,F
        x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)

        spec_r, spec_i = x.chunk(2, dim=1)
        x_mag = (spec_r**2 + spec_i**2) ** 0.5

        # noisy_real = x[:, 4, :, :]
        # noisy_imag = x[:, 10, :, :]
        # noisy_spec = torch.stack([noisy_real, noisy_imag], 1)
        # x_mag_ori, x_phase_ori = torch.norm(noisy_spec, dim=1), torch.atan2(
        #     noisy_spec[:, -1, :, :], noisy_spec[:, 0, :, :]
        # )
        # x_mag = x_mag_ori.unsqueeze(dim=1)

        # ri/mag components enconde+ aia_transformer_merge
        x_ri = self.en_ri(x)  # BCTF
        x_mag_en = self.en_mag(x_mag)
        x_last_mag, x_outputlist_mag, x_last_ri, x_outputlist_ri = self.aia_trans_merge(
            x_mag_en, x_ri
        )  # BCTF, #BCTFG

        x_ri = self.aham(x_outputlist_ri)  # BCT
        # x_mag_en = self.aham_mag(x_outputlist_mag)  # BCTF

        # x_ri = self.simam(x_ri)
        # x_mag_en = self.simam_mag(x_mag_en)
        # x_mag_mask = self.de_mag_mask(x_mag_en)
        # x_mag_mask = x_mag_mask.squeeze(dim=1)

        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim=1)
        x_imag = x_imag.squeeze(dim=1)

        # x_real = x_real * x_mag_mask
        # x_imag = x_imag * x_mag_mask
        # magnitude and ri components interaction

        # x_mag_out = x_mag_mask * x_mag_ori

        ##### recons by DCCRN
        # mask_phase = torch.atan2(x_imag, x_real)

        # est_phase = x_phase_ori + mask_phase

        # x_r_out = x_mag_out * torch.cos(est_phase)
        # x_i_out = x_mag_out * torch.sin(est_phase)

        # x_com_out = torch.stack((x_r_out,x_i_out),dim=1)

        feat = torch.stack([x_real, x_imag], dim=1)

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


# class McNet_DF_AIA_TRANS(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         feature_size: int,
#         mid_channels: int,
#         ref_channel: int = 4,
#     ):
#         super().__init__()
#         self.pre_layer = MCNetSpectrum(
#             in_channels=in_channels,
#             ref_channel=ref_channel,
#             sub_freqs=(3, 2),
#             past_ahead=(5, 0),
#         )  # C,F,C'
#         self.ref_channel = ref_channel
#         self.in_channels = in_channels

#         self.stft = STFT(512, 256)
#         self.en_ri = dense_encoder(
#             # in_channels=in_channels * 2,
#             in_channels=2,
#             out_channels=mid_channels,
#             feature_size=feature_size,
#             depth=4,
#         )  # B, mid_c, T, F // 4

#         self.dual_trans = AIA_Transformer_cau(mid_channels, mid_channels, num_layers=4)
#         self.aham = AHAM(input_channel=mid_channels)

#         self.de1 = dense_decoder(
#             in_channels=mid_channels,
#             out_channels=1,
#             feature_size=feature_size // 4,
#             depth=4,
#         )
#         self.de2 = dense_decoder(
#             in_channels=mid_channels,
#             out_channels=1,
#             feature_size=feature_size // 4,
#             depth=4,
#         )

#         self.df_order = 5
#         self.df_bins = feature_size
#         self.DF_de = DF_dense_decoder(
#             mid_channels, feature_size // 4, 2 * self.df_order
#         )

#         self.df_op = DF(num_freqs=self.df_bins, frame_size=self.df_order, lookahead=0)

#         self.df_out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

#     def forward(self, x):
#         """
#         x: B,T,C
#         """
#         nB = x.size(0)
#         x = rearrange(x, "b t c-> (b c) t")
#         xk = self.stft.transform(x)
#         # x = rearrange(xk, "(b m) c t f->b (c m) t f", b=nB)
#         x = rearrange(xk, "(b m) c t f->b t f (m c)", b=nB)

#         x = self.pre_layer(x)  # B,2,T,F

#         noisy_real = x[:, 0, :, :]
#         noisy_imag = x[:, 1, :, :]
#         # noisy_real, noisy_imag = x.chunk(2, dim=1)

#         # ri components enconde+ aia_transformer
#         x_ri = self.en_ri(x)  # BCTF
#         x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
#         x_ri = self.aham(x_outputlist)  # BCTF

#         # real and imag decode
#         x_real = self.de1(x_ri)
#         x_imag = self.de2(x_ri)
#         # x_real, x_imag = x_.chunk(2, dim=1)
#         x_real = x_real.squeeze(dim=1)
#         x_imag = x_imag.squeeze(dim=1)

#         enh_real = noisy_real * x_real - noisy_imag * x_imag
#         enh_imag = noisy_real * x_imag + noisy_imag * x_real

#         # enhanced_D = torch.stack([x_real, x_imag], 3)  # B,T,F,2
#         enhanced_D = torch.stack([enh_real, enh_imag], 3)  # B,T,F,2
#         enhanced_D = enhanced_D.unsqueeze(1)  # B,1,T,F,2

#         # DF coeffs decoder
#         df_coefs = self.DF_de(x_ri)  # BCTF
#         df_coefs = df_coefs.permute(0, 2, 3, 1)  # B,T,F,10
#         df_coefs = self.df_out_transform(df_coefs).contiguous()  # B,5,T,F,2

#         DF_spec = self.df_op(enhanced_D, df_coefs)  # B,1,T,F,2
#         DF_spec = DF_spec.squeeze(1)

#         DF_real = DF_spec[:, :, :, 0]
#         DF_imag = DF_spec[:, :, :, 1]

#         feat = torch.stack([DF_real, DF_imag], dim=1)  # B,2,T,F

#         out_wav = self.stft.inverse(feat)  # B, 1, T
#         out_wav = torch.squeeze(out_wav, 1)
#         out_wav = torch.clamp(out_wav, -1, 1)

#         return out_wav


if __name__ == "__main__":
    model = DF_AIA_TRANS(in_channels=6, feature_size=257, mid_channels=96)  # C,F,C'
    # model = McNet_DF_AIA_TRANS(
    #     in_channels=6, feature_size=257, mid_channels=96
    # )  # C,F,C'
    # model = dual_aia_trans_chime(
    #     in_channels=6, feature_size=257, mid_channels=96
    # )  # C,F,C'
    model.eval()
    # x = torch.FloatTensor(4, 2, 10, 481)
    #
    # real, imag = model(x)
    # print(str(real.shape))
    input_test = torch.FloatTensor(1, 16000, 6)  # B,T,M
    flops, params = profile(model, inputs=(input_test,))
    print(f"FLOPs={flops / 1e9}, params={params/1e6:.2f}")

    out = model(input_test)
    print(out.shape)

    #
    # params_of_network = 0
    # for param in model.parameters():
    #     params_of_network += param.numel()
    #
    # print(f"\tNetwork: {params_of_network / 1e6} million.")
    # output = model(x)
