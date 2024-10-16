import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

import einops
from einops import rearrange
from einops.layers.torch import Rearrange
from itertools import combinations
from typing import List


# from conv_stft import STFT


class SpatialFeats(nn.Module):
    """
    Input: B,M,T,F complex type or B,2M,T,F real type
    Output: B,(5P+A),T,F, where P for mic combs (N*(N-1)/2), A for angles
    """

    def __init__(
        self,
        mic_pos: List[List],
        feats: List[str] = ["ILD", "IPD", "GCC", "DF"],
        sr: int = 16000,
    ) -> None:
        super(SpatialFeats, self).__init__()
        self.n_mic = len(mic_pos)
        self.mic_pos = np.array(mic_pos)
        self._init_mic_pos()
        # combs is a list of [(left, right), (...)] tuples
        self.combs = np.array(list(combinations(range(self.n_mic), 2)))
        # self.stft = STFT(nframe, nhop, nfft=nframe)
        self.eps = 1e-8
        self.snd_velocity = 340.0
        self.sr = sr
        self.feats = feats

    def _init_mic_pos(self):
        self.array_center = np.array([0, 0, 0])  # [3]
        self.mic_distance = np.zeros([self.n_mic, self.n_mic])
        self.mic_distance2center = np.zeros([self.n_mic])
        for i in range(self.n_mic):
            self.mic_distance[i] = [
                np.linalg.norm(self.mic_pos[i] - self.mic_pos[j])
                for j in range(self.n_mic)
            ]  # [M,M]
            self.mic_distance2center[i] = np.linalg.norm(
                self.mic_pos[i] - self.array_center
            )  # [M,1]
        self.mic_alphas = np.array(  # [M,1]
            [
                np.arctan2(self.mic_pos[i][1], self.mic_pos[i][0])
                for i in range(self.n_mic)
            ]
        )

    def _transform(self, x):
        """format input
        Input: [B,2M,T,F] real or [B,M,T,F] complex
        Output: [B,M,T,F] complex
        """
        if not torch.is_complex(x):
            x = rearrange(x, "b (m c) t f -> b m t f c", c=2).contiguous()
            x = torch.view_as_complex(x)
        return x

    def _get_delay_to_center(self, mic_idx, azm, ele=None):
        """Ignore the pitch angle (elevation)"""
        dist2center = self.mic_distance2center[mic_idx]
        delay = -dist2center * torch.cos(azm - self.mic_alphas[mic_idx])
        if ele is not None:
            delay *= torch.cos(ele)
        return delay

    def _zone_TPD_number(self, nbin: int, loc_info=[-180, 180], A=10, sort=False):
        """Compute the Target Phase Differences (TPD) value at each sampling angle `loc_info/A`
        Args:
            - nbin: one-side spectrum length.
            - loc_info: List, sampling interval, in angle.
            - A: int, number of sampling angles, default 10.
            - sort: bool, whether to sort the TPD values.
        Returns: [B, P, C, F], where P is the number of mic combs
        """
        # loc_info = torch.repeat_interleave(loc_info * torch.pi / 180)
        loc_info_ = np.array(loc_info).astype(np.float32) * torch.pi / 180

        dis_bs = []
        # NOTE: Sampling without considering the pitch angle (elevation),
        # azimuth is the angle of planar rwave
        ele_grid = torch.zeros(1)  # [1,]
        azm_grid = torch.linspace(*loc_info_, steps=A)  # [A,]
        # mesh_azm: [A,1], ele_grid: [A, 1]
        # mesh_azm, mesh_ele = torch.meshgrid(azm_grid, ele_grid)
        mesh_azm = azm_grid.reshape(-1, 1).repeat(1, len(ele_grid))
        mesh_ele = ele_grid.repeat(len(azm_grid), 1)

        for comb in self.combs:
            delay2center_left = self._get_delay_to_center(comb[0], mesh_azm, mesh_ele)
            delay2center_right = self._get_delay_to_center(comb[1], mesh_azm, mesh_ele)
            delay_diff = delay2center_left - delay2center_right
            dis_bs.append(delay_diff)  # [A,1]
        distances = torch.stack(dis_bs, 0)  # [P, A, 1]
        deltas = distances / self.snd_velocity * self.sr  # [P, A, 1] TDOA
        deltas = deltas.reshape(len(self.combs), -1)  # [P, A]
        # NOTE: sort will change the relative position of each angle in each A vectors.
        if sort:
            deltas, _ = torch.sort(deltas, dim=-1, descending=False)

        f_ndx = torch.linspace(start=0, end=nbin - 1, steps=nbin)  # [F]
        f_ndx = f_ndx[None, None, ...]  # [1,1,F]

        # NOTE delta_w = 2pi f delta_t
        # [P,A,1] * [1,1,F]
        tpd = torch.exp(
            -1j * deltas.unsqueeze(-1) * torch.pi * f_ndx / (nbin - 1)
        )  # [P, A, F]
        # tpd = tpd / len(self.combs)

        return tpd

    def _get_zone_DF(self, ipd: Tensor, zone_tpd: Tensor):
        """
        Args:
            - ipd, [B,P,T,F]
            - zone_tpd, [P,A,F], P for mic_pairs, A for number of sampling zones

        Return: B,A,T,F
        """
        zone_tpd = zone_tpd.to(ipd.device)
        cpx_ipd = torch.exp(1j * ipd)  # 1,P,A,F
        # bftp x bfpa -> bfta -> batf
        direction_feature = torch.einsum(
            "paf,bptf->batf", (zone_tpd.conj(), cpx_ipd)
        ) / len(self.combs)
        direction_feature = direction_feature.real  # B,A,T,F
        return direction_feature

    def compute_DF(self, x, angle=[-180, 180], A=18):
        """Directional Features
        Given the sampling zone `angle` and sampling points `A`, compute the cosine distance
            between the target phase difference (TDP) of each sampling points `A`
            and the final IDP get by the pair mics.

        Args:
            - x, [B,M,T,F] complex or [B,2M,T,F] real
            - angle, List, sampling zone;
            - A, int, the number of zone;

        Return: [B,A,T,F]
        """
        nbin = x.size(-1)
        x = self._transform(x)
        x = x[:, self.combs, ...]  # B,P,2,T,F
        # pha1 for left, pha2 for right
        pha1, pha2 = (
            x[:, :, 0, ...].angle(),
            x[:, :, 1, ...].angle(),
        )  # complex [B,P,T,F]
        zone_tpds = self._zone_TPD_number(nbin, angle, A)  # [P,A(18),F]
        raw_ipd = pha2 - pha1
        df = self._get_zone_DF(raw_ipd, zone_tpds)

        return df

    def compute_GCC(self, x):
        """Generalized Cross-Correlation
        Input:
            - [B,M,T,F] complex format;
            - [B,2M,T,F] real format, where real, imag are close together;
        Return: [B,2P,T,F], where P is the number of `self.combs`, 2 for real, imag;
        """
        # x = torch.view_as_complex(x)
        x = self._transform(x)

        x = x[:, self.combs, ...]  # B,P,2,T,F
        x1, x2 = x[:, :, 0, ...], x[:, :, 1, ...]  # complex
        cc = x1 * x2.conj()
        gcc = cc / (x1.abs() * x2.abs() + self.eps)
        gcc = torch.cat((gcc.real, gcc.imag), dim=1)

        return gcc

    def compute_ILD(self, x):
        """Interaural Level Difference
        Input: [B,M,T,F] complex or [B,2M,T,F] real
        Output: [B,P,T,F]
        """
        x = self._transform(x)
        x = x[:, self.combs, ...]  # B,P,2,T,F
        # x2 for right, x1 for left
        x1, x2 = x[:, :, 0, ...], x[:, :, 1, ...]  # complex
        ild = 10 * torch.log10((x2.abs() ** 2 + self.eps) / (x1.abs() ** 2 + self.eps))

        return ild

    def compute_IPD(self, x, ret_type: List[str] = ["cos", "sin"]):
        """Inter-channel phase difference
        Args:
            - ret_type, "cos", "sin", "raw"
        Input: [B,M,T,F] complex or [B,2M,T,F] real
        Output: [B,Pxlen(ret_type),T,F]
        """
        x = self._transform(x)
        x = x[:, self.combs, ...]  # B,P,2,T,F
        # pha1 for left, pha2 for right
        pha1, pha2 = x[:, :, 0, ...].angle(), x[:, :, 1, ...].angle()  # complex

        sin_ipd = torch.sin(pha2 - pha1)
        cos_ipd = torch.cos(pha2 - pha1)
        raw_ipd = torch.fmod(pha2 - pha1, torch.pi * 2)

        ipd = eval(ret_type[0] + "_ipd")  # str to variable
        for k in ret_type[1:]:
            ipd = torch.concat([ipd, eval(k + "_ipd")], dim=1)

        return ipd

    def compute_SRP_PHAT(self, x):
        """Steered Response Power
        x: [BS, M, F, T]
        gcc: [BS, P, F, T]
        return: SRP_PHAT nband x [BS, num_doa, T]
        """
        pass

    def forward(self, x, angle=[-180, 180], A=18, ipd_type=["cos", "sin"]):
        """format input
        Input: [B,2M,T,F] real or [B,M,T,F] complex
        Return: [B,(5P+A),T,F], where P for mic combs

        P = N-1+...+1 = N*(N-1)/2, where N is the mic channel
        """
        out = []
        if "GCC" in self.feats:
            gcc_ = self.compute_GCC(x)  # B,2P,T,F
            out.append(gcc_)
        if "DF" in self.feats:
            df_ = self.compute_DF(x, angle=angle, A=A)
            out.append(df_)  # B,A,T,F
        if "IPD" in self.feats:
            ipd_ = self.compute_IPD(x, ret_type=ipd_type)
            out.append(ipd_)  # B,2P,T,F
        if "ILD" in self.feats:
            ild_ = self.compute_ILD(x)
            out.append(ild_)  # B,P,T,F

        return torch.concat(out, dim=1)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.get_space_feas import FeatureComputerSpecific

    mic_array = [
        [-0.04, 0, 0],
        [0, 0.04, 0],
        [0.04, 0, 0],
        [0, -0.04, 0],
    ]

    opt = SpatialFeats(mic_array)
    inp = torch.randn(1, 6, 10, 257, 2)
    inp = torch.view_as_complex(inp)
    out = opt.compute_GCC(inp)
    out = opt.compute_IPD(inp)
    out = opt.compute_ILD(inp)
    print(out.shape)
    print(out[0, 1, 0, :10])

    out = opt.compute_DF(inp, A=18)

    out = opt(inp)
    print("#", out.shape)

    # loc_feature_conf = {
    #     "sr": 16000,
    #     "feature": ["DF1D", "GCC", "ILD", "IPD"],  # 使用DF特征
    #     "IPD_type": ["cos", "sin"],
    #     "SRP_PHAT_num": 18,
    #     "DFsort": False,
    #     "nFFT": 512,
    #     "frame_size": 512,
    #     "frame_hop": 256,
    #     "mic_pairs": "full",
    #     "mic_arch": [
    #         [-0.04, 0, 0],
    #         [0, 0.04, 0],
    #         [0.04, 0, 0],
    #         [0, -0.04, 0],
    #     ],  # 麦克风的三维坐标
    #     "spec": "all",  # all / mean / reference / [0, -1]
    #     "ref_mic_idx": 0,
    #     "zone_agg": None,  # None, RNN-TAA, DPRNN-TAA, TAA, TAC
    #     "sample_method": "number",  # number (#A: range // A), fixed (#range // A: A)
    #     "A": 18,  # DF划分区域个数
    #     "loc_info": [[-180, 180]],  # 待检测的区域
    #     "df_dim": 8,
    # }
    # fc = FeatureComputerSpecific(loc_feature_conf)
    # feature_dict = fc(inp)
    # for k, v in feature_dict.items():
    #     print(k, v.shape)
    #     v = v.permute(0, 1, 3, 2)
    #     print(v[0, 1, 0, :10])
