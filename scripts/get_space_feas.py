"""
@author: Rongzhi Gu (lorrygu), Yi Luo (oulyluo), Jingjie Fan (alkaidfan)
@copyright: Tencent AI Lab
modules for source localization
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import math
from itertools import combinations


def band_split(num_bins=257, sr=16000):
    """分带操作

    Parameters
    ----------
    num_bins : int, optional
        FFT点数(取一半), by default 257
    sr : int, optional
        采样率, by default 16000

    Returns
    -------
    list, int
        返回分带结果和频带数目
    """
    bandwidth_1000 = int(np.floor(1000 / (sr / 2.0) * num_bins))
    band_width = [bandwidth_1000] * 7
    band_width.append(num_bins - np.sum(band_width))
    nband = len(band_width)
    return band_width, nband


class cLN(nn.Module):
    def __init__(self, dimension):
        super(cLN, self).__init__()

        self.eps = torch.finfo(torch.float32).eps
        self.gain = nn.Parameter(torch.ones(1, dimension, 1))
        self.bias = nn.Parameter(torch.zeros(1, dimension, 1))

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        # entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        # entry_cnt = torch.from_numpy(entry_cnt).type(input.type())

        entry_cnt = torch.arange(
            channel,
            channel * (time_step + 1),
            channel,
            dtype=input.dtype,
            device=input.device,
        )  # lorrygu-221012: cuda devices alignment here
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(
            2
        )  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(
            x.type()
        )


class ResRNN(nn.Module):
    """
    the RNN block in Band and Sequence modeling module
    """

    def __init__(self, input_size, hidden_size, causal, rnn_type="LSTM", res=True):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.causal = causal
        self.bidirectional = not causal

        self.eps = torch.finfo(torch.float32).eps

        if self.causal:
            # self.LN = cLN(input_size)
            self.LN = nn.BatchNorm1d(input_size, self.eps)
        else:
            self.LN = nn.GroupNorm(1, self.input_size, self.eps)

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.proj = nn.Linear(hidden_size * (int(self.bidirectional) + 1), input_size)
        self.res = res

    def forward(self, input):
        # input shape: B*K, N, T (across T) / (B*T, N, K) (across K)
        B, N, T = input.shape

        rnn_output, _ = self.rnn(
            self.LN(input).transpose(1, 2).contiguous()
        )  # B*K, T, hidden_size / B*T, K, hidden_size
        proj_output = self.proj(rnn_output.contiguous().view(B * T, -1)).view(
            B, T, N
        )  # B*K*T, hidden_size -> B*K, T, N / B*T* K, hidden_size -> (B*T, N, K)
        proj_output = proj_output.transpose(1, 2).contiguous()  # B*K, N, T / B*T, K, N

        output = proj_output + input if self.res else proj_output
        return output


class BSNet(nn.Module):
    def __init__(self, in_channel, nband=7, num_layer=1, causal=False, bi_comm=True):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband

        self.band_rnn = []
        for _ in range(num_layer):
            self.band_rnn.append(ResRNN(self.feature_dim, self.feature_dim * 2, causal))
        self.band_rnn = nn.Sequential(*self.band_rnn)
        self.band_comm = ResRNN(self.feature_dim, self.feature_dim * 2, not bi_comm)

    def forward(self, input):
        # input shape: B, nband*N, T
        B, N, T = input.shape

        band_output = self.band_rnn(
            input.view(B * self.nband, self.feature_dim, -1)
        ).view(B, self.nband, -1, T)

        # band comm
        band_output = (
            band_output.permute(0, 3, 2, 1).contiguous().view(B * T, -1, self.nband)
        )
        output = (
            self.band_comm(band_output)
            .view(B, T, -1, self.nband)
            .permute(0, 3, 2, 1)
            .contiguous()
        )  # [B, K, N, T]

        return output.view(B, N, T)


class MicrophoneArray(object):
    def __init__(self, universal_conf, mic_conf):
        """初始化麦克风阵列结构和信息

        Parameters
        ----------
        universal_conf : dict
            通用配置, 包括sr, nFFT等
        mic_conf : dict
            具体的麦克风阵列配置
        """
        self.init_mic_pos(mic_conf)
        self.sr = universal_conf["sr"]
        self.nFFT = universal_conf["nFFT"]
        self.snd_velocity = 340.0

        if mic_conf["mic_pairs"] == "full":
            self.mic_pairs = np.array(list(combinations(range(self.n_mic), 2)))
        else:
            self.mic_pairs = np.array(mic_conf["mic_pairs"])
        self.pair_left = [t[0] for t in self.mic_pairs]
        self.pair_right = [t[1] for t in self.mic_pairs]
        self.n_mic_pairs = self.mic_pairs.shape[0]
        self.num_bins = self.nFFT // 2 + 1
        self.f_idx = torch.linspace(start=0, end=self.num_bins - 1, steps=self.num_bins)

    def init_mic_pos(self, mic_conf):
        mic_arch = mic_conf["mic_arch"]
        if mic_arch is None:
            self.mic_position = np.array([[0, 0, 0]])
        else:
            self.mic_position = np.array(mic_arch)
        self.array_center = np.array([0, 0, 0])  # [3]

        self.n_mic = self.mic_position.shape[0]
        self.mic_distance = np.zeros([self.n_mic, self.n_mic])
        self.mic_distance2center = np.zeros([self.n_mic])

        for i in range(self.n_mic):
            self.mic_distance[i] = [
                np.linalg.norm(self.mic_position[i] - self.mic_position[j])
                for j in range(self.n_mic)
            ]
            self.mic_distance2center[i] = np.linalg.norm(
                self.mic_position[i] - self.array_center
            )

        self.mic_alphas = np.array(
            [
                math.atan2(self.mic_position[i][1], self.mic_position[i][0])
                for i in range(self.n_mic)
            ]
        )

    def get_delay_to_center(self, mic_idx, azm, ele=None):
        dist2center = self.mic_distance2center[mic_idx]
        delay = -dist2center * torch.cos(azm - self.mic_alphas[mic_idx])
        if ele is not None:
            delay *= torch.cos(ele)
        return delay

    @staticmethod
    def _expand_dim_to_target_tensor(src_tensor, target_tensor):
        target_dim = target_tensor.ndim
        src_dim = src_tensor.ndim

        for _ in range(target_dim - src_dim):
            src_tensor = src_tensor.unsqueeze(0)
        return src_tensor

    def get_zone_TPD_number(self, loc_info, A=10, sort=False):
        """计算每个采样角度上的TPD值

        Parameters
        ----------
        loc_info : array shape=(B, 2)
            采样区间, in radian
        C : int, optional
            采样的角度个数, by default 10
        sort : bool, optional
            是否对TPD值进行排序, by default False

        Returns
        -------
        tensor
            shape = (B, P, C, F)
        """
        bs = loc_info.shape[0]
        dis = []
        for b in range(bs):
            dis_bs = []
            # NOTE: 暂时不考虑俯仰角的采样
            ele_grid = torch.zeros(1)
            azm_grid = torch.linspace(loc_info[b, 0], loc_info[b, 1], steps=A)
            mesh_azm, mesh_ele = torch.meshgrid(azm_grid, ele_grid)
            for pair in self.mic_pairs:
                delay2center_left = self.get_delay_to_center(
                    pair[0], mesh_azm, mesh_ele
                )
                delay2center_right = self.get_delay_to_center(
                    pair[1], mesh_azm, mesh_ele
                )
                delay_diff = delay2center_left - delay2center_right
                dis_bs.append(delay_diff)
            dis.append(torch.stack(dis_bs, 0))  # [P, C, 1]

        distances = torch.stack(dis, dim=0).to(loc_info.device)  # [B, P, C, 1]
        deltas = distances / self.snd_velocity * self.sr  # [B, P, C, 1] TDOA的值
        deltas = deltas.reshape(bs, self.n_mic_pairs, -1)  # [B, P, C]
        # NOTE: 排序会改变每个C维向量中，各个角度的相对位置关系
        if sort:
            deltas, _ = torch.sort(deltas, dim=-1, descending=False)
        self.f_idx = torch.linspace(
            start=0, end=self.num_bins - 1, steps=self.num_bins
        )  # * repeated code
        f_idx = (
            self.f_idx.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(deltas.device)
        )  # [1,1,1,F]

        tpd = torch.exp(
            -1j * deltas.unsqueeze(-1) * math.pi * f_idx / (self.num_bins - 1)
        )  # [B, P, C, F]
        tpd = tpd / (self.n_mic_pairs)

        return tpd


class FeatureComputerSpecific(nn.Module):
    def __init__(self, universal_conf):
        """初始化特征计算器的信息, 特定于指定的麦克风阵列

        Parameters
        ----------
        universal_conf : dict
            所使用的特征的配置, xxx.yaml/ssl/loc_info
        """
        super(FeatureComputerSpecific, self).__init__()
        self.conf = universal_conf
        self.frame_size = self.conf["frame_size"]
        self.frame_hop = self.conf["frame_hop"]
        self.nFFT = self.conf["nFFT"]
        self.sr = self.conf["sr"]
        self.input_feature = self.conf["feature"]
        self.df_dim = self.conf["df_dim"]
        self.num_bins = self.nFFT // 2 + 1
        self.epsilon = 1e-8

        self.band_width, self.nband = band_split(self.num_bins, self.sr)
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.nFFT,
            win_length=self.frame_size,
            hop_length=self.frame_hop,
            center=True,
            power=None,
            window_fn=torch.hann_window,
        )

        assert "mic_arch" in universal_conf.keys(), "loc_feature中应该包含阵列信息"
        mic_conf_keys = ["mic_arch", "mic_pairs"]
        mic_conf = {key: universal_conf[key] for key in mic_conf_keys}
        self.array = MicrophoneArray(universal_conf, mic_conf)

        # 频谱中使用的通道数
        self.spec_num = 0
        if "spec" in self.input_feature:
            if self.conf["spec"] == "all":
                self.spec_num = self.array.n_mic
            elif self.conf["spec"] == "mean":
                self.spec_num = 1
            elif self.conf["spec"] == "reference":
                self.spec_num = 1
            elif isinstance(self.conf["spec"], list):
                self.spec_num = len(self.conf["spec"])
            else:
                raise NotImplementedError(
                    f"{self.conf['spec']} is not implemented for spec"
                )

        # shape=[B, P, F, T]的特征数量, 如cosIPD,sinIPD, ILD等
        # FIXME 把这些特征reshape成[B, P*F, T]的形式，合理吗?
        self.sf_num = 0
        if "IPD" in self.input_feature:
            z = len(self.conf["IPD_type"]) if "IPD_type" in self.conf else 1
            self.sf_num += z * self.array.n_mic_pairs
        if "ILD" in self.input_feature:
            self.sf_num += self.array.n_mic_pairs
        if "GCC" in self.input_feature:
            self.sf_num += 2 * self.array.n_mic_pairs

        # shape=nband x [B, N, T]的特征数量, 如SRP_PHAT等
        self.phat_num = 0
        if "SRP_PHAT" in self.input_feature:
            self.srp_phat_num = self.conf["SRP_PHAT_num"]
            self.phat_num += self.srp_phat_num

        # shape= [B, N, F, T]的特征数量, 如DF1D等
        self.df_num = 0
        if "DF1D" in self.input_feature:
            # self.A = self.conf['A']
            # self.df_num += self.A
            sample_method = self.conf["sample_method"]
            zone_agg = self.conf["zone_agg"]
            self.zone_agg = zone_agg
            assert sample_method == "number", "暂时只实现了指定数目的角度采样"
            if zone_agg is None or zone_agg == "":
                self.zone_att_layer = None
            elif zone_agg in ["TAA", "TAC"]:
                # transform-and-average/concatenate
                # self.zone_att_layer = nn.Sequential(
                #     nn.Conv1d(self.num_bins, 256, 1),
                #     nn.Tanh(),
                #     nn.Conv1d(256, self.num_bins, 1),
                #     nn.Tanh(),
                # )
                self.zone_att_layer = nn.ModuleList([])
                for i in range(self.nband):
                    self.zone_att_layer.append(
                        nn.Sequential(
                            nn.Conv1d(self.band_width[i], self.band_width[i], 1),
                            nn.Tanh(),
                            nn.Conv1d(self.band_width[i], self.df_dim, 1),
                            nn.Tanh(),
                        )
                    )
            elif zone_agg.startswith("RNN"):
                # input: [BS*T, F, V=A*E], output: [BS*T, F, V (1)]
                bidirectional = False
                self.zone_att_layer = ResRNN(
                    self.num_bins, self.num_bins, not bidirectional, res=False
                )
            self.A = self.conf["A"]
            if zone_agg is None or zone_agg == "" or zone_agg == "TAC":
                self.df_num = self.A
            elif zone_agg == "TAA" or zone_agg == "RNN-L":
                self.df_num = 1
            elif zone_agg == "RNN-L2":
                self.df_num = 2

    def forward(self, mixture):
        """计算输入信号的频谱, spatiai_feature, direacition_feature等特征

        Parameters
        ----------
        mixture : tensor
            麦克风的接收信号
        array : object, optional
            ad-hoc时输入的阵列实例, by default None

        Returns
        -------
            各种特征
        """
        array = self.array
        mixture = mixture[:, : array.n_mic]
        mix_stft = self.stft(mixture)  # [B, M, F, T]
        # mix_stft = mixture.permute(0, 1, 3, 2)
        magnitude, phase = mix_stft.abs(), mix_stft.angle()
        feature_dict = {}

        if "IPD" in self.input_feature:
            raw_ipd, cos_ipd, sin_ipd = self.compute_IPD(array, phase)
            if "IPD_type" not in self.conf:
                ipd = cos_ipd  # default
            else:
                ipd = eval(self.conf["IPD_type"][0] + "_ipd")

                for ipd_type in self.conf["IPD_type"][1:]:
                    ipd = torch.cat((ipd, eval(ipd_type + "_ipd")), 1)
            feature_dict["IPD"] = ipd

        if "ILD" in self.input_feature:
            ild = self.compute_ILD(array, magnitude)
            feature_dict["ILD"] = ild

        if "GCC" in self.input_feature:
            gcc = self.compute_GCC(array, mix_stft)
            feature_dict["GCC"] = gcc

        if "SRP_PHAT" in self.input_feature:
            srp_phat = self.compute_SRP_PHAT(array, mix_stft, num_doa=self.srp_phat_num)
            feature_dict["SRP_PHAT"] = srp_phat

        if "DF1D" in self.input_feature:
            batchsize = mix_stft.shape[0]
            self.update_loc_info(self.conf["loc_info"], batchsize)
            raw_ipd = phase[:, array.pair_right] - phase[:, array.pair_left]
            tpd = array.get_zone_TPD_number(
                self.loc_info, self.A, sort=self.conf["DFsort"]
            )
            direction_feature = self.get_zone_DF(raw_ipd, tpd)
            feature_dict["DF1D"] = direction_feature

        if "spec" in self.input_feature:
            if self.conf["spec"] == "mean":
                mix_stft = torch.mean(mix_stft, dim=1, keepdim=True)
            elif self.conf["spec"] == "reference":
                self.spec_num = 1
                mix_stft = mix_stft[:, self.conf["ref_mic_idx"]].unsqueeze(1)
            elif isinstance(self.conf["spec"], list):
                mix_stft = mix_stft[:, self.conf["spec"]]
            feature_dict["spec"] = mix_stft

        return feature_dict

    def compute_IPD(self, array, phase):
        """phase [B, M, F, T], return IPD [B, P, F, T]"""
        sin_ipd = torch.sin(phase[:, array.pair_right] - phase[:, array.pair_left])
        cos_ipd = torch.cos(phase[:, array.pair_right] - phase[:, array.pair_left])
        raw_ipd = torch.fmod(
            phase[:, array.pair_right] - phase[:, array.pair_left], math.pi * 2
        )
        return raw_ipd, cos_ipd, sin_ipd

    def compute_ILD(self, array, magnitude):
        """magnitude [B, M, F, T], return ILD [B, P, F, T]"""
        ild = 10 * torch.log10(
            (magnitude[:, array.pair_right] ** 2 + self.epsilon)
            / (magnitude[:, array.pair_left] ** 2 + self.epsilon)
        )
        return ild

    def compute_GCC(self, array, mix_stft):
        """
        mix_stft: [B, M, F, T], return: gcc, [B, 2P, F, T]
        """
        cc = mix_stft[:, array.pair_left] * mix_stft[:, array.pair_right].conj()
        gcc = cc / (
            mix_stft[:, array.pair_left].abs() * mix_stft[:, array.pair_right].abs()
            + self.epsilon
        )
        gcc = torch.cat((gcc.real, gcc.imag), dim=1)
        return gcc

    def compute_SRP_PHAT(self, array, mix_stft, num_doa=180):
        """
        Steered Response Power
        mix_stft: [BS, M, F, T]
        gcc: [BS, P, F, T]
        return: SRP_PHAT nband x [BS, num_doa, T]
        """
        cc = mix_stft[:, array.pair_left] * mix_stft[:, array.pair_right].conj()
        gcc = cc / (
            mix_stft[:, array.pair_left].abs() * mix_stft[:, array.pair_right].abs()
            + self.epsilon
        )
        srp_phats = [list() for _ in range(self.nband)]

        for p in range(array.n_mic_pairs):
            this_pair = array.mic_pairs[p]
            mic_spacing = array.mic_distance[this_pair[0], this_pair[1]]
            max_tdoa = mic_spacing / array.snd_velocity

            est_tdoa = torch.linspace(-max_tdoa, max_tdoa, num_doa)  # [num_doa]
            sep_freq = torch.linspace(0, array.sr // 2, self.num_bins)  # [F]
            band_idx = 0
            for b in range(self.nband):
                subband_freq = sep_freq[band_idx : band_idx + self.band_width[b]]
                subband_gcc = gcc[:, p, band_idx : band_idx + self.band_width[b]]
                exp_part = torch.outer(
                    1j * 2 * math.pi * est_tdoa, subband_freq
                )  # [num_doa, F_k]
                gcc_phat = torch.einsum(
                    "vf,bft->bvt", (torch.exp(exp_part.to(gcc.device)), subband_gcc)
                ).real
                gcc_phat[gcc_phat < 0] = 0
                srp_phats[b].append(gcc_phat)
                band_idx += self.band_width[b]
        for i in range(self.nband):
            srp_phats[i] = torch.stack(srp_phats[i], 1).sum(1)
        return srp_phats

    def update_loc_info(self, loc_info, B):
        if isinstance(loc_info, list):
            info = loc_info[0]
            info = torch.tensor(info, dtype=torch.float32) * math.pi / 180
            info = info.unsqueeze(0)
            info = torch.repeat_interleave(info, B, dim=0)
        self.loc_info = info

    def get_zone_DF(self, raw_ipd, tpd):
        """
        raw_ipd: [B, P, F, T]
        tpd: [B, P, C, F]
        df: [B, C, F, T]
        """
        tpd = tpd.to(raw_ipd.device)
        cpx_ipd = torch.exp(1j * raw_ipd)
        direction_feature = torch.einsum("bpcf,bpft->bcft", (tpd.conj(), cpx_ipd))
        direction_feature = direction_feature.real  # [B, C, F, T]
        # return direction_feature
        b, c, f, t = direction_feature.shape

        def average(tensor):
            # [B*T, F, C] => [B, F, T]
            return tensor.mean(-1).reshape(b, -1, f).transpose(-1, -2)

        def lastone(tensor):
            # [B*T, F, C] => [B, 1, F, T]
            return tensor[..., -1].reshape(b, -1, f).transpose(-1, -2).unsqueeze(1)

        def lasttwo(tensor):
            # [B*T, F, C] => [B, 2, F, T]
            return (
                tensor[..., -2:].reshape(b, -1, f, 2).permute((0, 3, 2, 1))
            )  # [B, 2, F, T]

        if self.zone_agg is None or self.zone_agg == "":
            return direction_feature
        elif self.zone_agg in ["TAA", "TAC"]:
            ztt_df = []
            all_df = direction_feature.reshape(b * c, f, -1).contiguous()
            band_idx = 0
            for i in range(self.nband):
                subband_df = all_df[
                    :, band_idx : band_idx + self.band_width[i]
                ].contiguous()
                subband_df = self.zone_att_layer[i](subband_df)
                subband_df = subband_df.reshape(b, c, -1, t)
                if self.zone_agg == "TAA":
                    subband_df = torch.sum(subband_df, dim=1).unsqueeze(
                        1
                    )  # [B, 1, F, T]
                ztt_df.append(subband_df)
            return ztt_df
        elif self.zone_agg.startswith("RNN"):
            all_df = direction_feature.permute((0, 3, 2, 1)).reshape(-1, f, c)
            if self.zone_agg.endswith("L2"):
                rnn_in = torch.cat((all_df, all_df[..., 0].unsqueeze(-1)), dim=-1)
                rnn_out = self.zone_att_layer(rnn_in)
                rnn_out = lasttwo(rnn_out)
            else:
                rnn_out = self.zone_att_layer(all_df)
                rnn_out = lastone(rnn_out)
            return rnn_out


class FeatureComputerAdhoc(nn.Module):
    def __init__(self, universal_conf):
        """初始化特征计算器的信息, 特定于ad-hoc的阵列

        Parameters
        ----------
        universal_conf : dict
            所使用的特征的配置, xxx.yaml/ssl/loc_info
        """
        super(FeatureComputerAdhoc, self).__init__()
        self.conf = universal_conf
        self.frame_size = self.conf["frame_size"]
        self.frame_hop = self.conf["frame_hop"]
        self.nFFT = self.conf["nFFT"]
        self.sr = self.conf["sr"]
        self.input_feature = self.conf["feature"]
        self.num_bins = self.nFFT // 2 + 1
        self.epsilon = 1e-8

        self.band_width, self.nband = band_split(self.num_bins, self.sr)
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.nFFT,
            win_length=self.frame_size,
            hop_length=self.frame_hop,
            center=False,
            power=None,
            window_fn=torch.hann_window,
        )

        assert "mic_arch" not in universal_conf.keys(), "loc_feature中不能包含阵列信息"

        # 频谱中使用的通道数
        if self.conf["spec"] == "all":
            raise NotImplementedError("ad-hoc阵列默认不使用全部的通道数")
        elif self.conf["spec"] == "mean":
            self.spec_num = 1
        elif self.conf["spec"] == "reference":
            self.spec_num = 1
        elif isinstance(self.conf["spec"], list):
            self.spec_num = len(self.conf["spec"])
        else:
            raise NotImplementedError(
                f"{self.conf['spec']} is not implemented for spec"
            )

        # shape=[B, P->z, F, T]的特征数量, 如cosIPD,sinIPD, ILD等
        self.sf_num = 0
        if "IPD" in self.input_feature:
            z = len(self.conf["IPD_type"]) if "IPD_type" in self.conf else 1
            self.sf_num += z
            self.ipd_rnn = self.init_ResRNN(z)
        if "ILD" in self.input_feature:
            self.sf_num += 1
            self.ild_rnn = self.init_ResRNN(z=1)
        if "GCC" in self.input_feature:
            self.sf_num += 2
            self.gcc_rnn = self.init_ResRNN(z=2)

        # shape=nband x [B, N, T]的特征数量, 如SRP_PHAT等
        self.phat_num = 0
        if "SRP_PHAT" in self.input_feature:
            self.srp_phat_num = self.conf["SRP_PHAT_num"]
            self.phat_num += self.srp_phat_num

        # shape= [B, N, F, T]的特征数量, 如DF1D等
        self.df_num = 0
        if "DF1D" in self.input_feature:
            sample_method = self.conf["sample_method"]
            zone_agg = self.conf["zone_agg"]
            self.zone_agg = zone_agg
            assert sample_method == "number", "暂时只实现了指定数目的角度采样"
            if zone_agg is None or zone_agg == "":
                self.zone_att_layer = None
            elif zone_agg in ["TAA", "TAC"]:
                # transform-and-average/concatenate
                self.zone_att_layer = nn.Sequential(
                    nn.Conv1d(self.num_bins, 256, 1),
                    nn.Tanh(),
                    nn.Conv1d(256, self.num_bins, 1),
                    nn.Tanh(),
                )
            elif zone_agg.startswith("RNN"):
                # input: [BS*T, F, V=A*E], output: [BS*T, F, V (1)]
                bidirectional = False
                self.zone_att_layer = ResRNN(
                    self.num_bins, self.num_bins, not bidirectional, res=False
                )
            self.A = self.conf["A"]
            if zone_agg is None or zone_agg == "" or zone_agg == "TAC":
                self.df_num = self.A
            elif zone_agg == "TAA" or zone_agg == "RNN-L":
                self.df_num = 1
            elif zone_agg == "RNN-L2":
                self.df_num = 2

    def forward(self, mixture, array):
        """计算输入信号的频谱, spatiai_feature, direacition_feature等特征

        Parameters
        ----------
        mixture : tensor
            麦克风的接收信号
        array : object
            ad-hoc时输入的阵列实例

        Returns
        -------
            各种特征
        """
        mixture = mixture[:, : array.n_mic]
        mix_stft = self.stft(mixture)  # [B, M, F, T]
        magnitude, phase = mix_stft.abs(), mix_stft.angle()
        feature_dict = {}

        if "IPD" in self.input_feature:
            raw_ipd, cos_ipd, sin_ipd = self.compute_IPD(array, phase)
            if "IPD_type" not in self.conf:
                ipd = cos_ipd  # default
            else:
                ipd = eval(self.conf["IPD_type"][0] + "_ipd")
                for ipd_type in self.conf["IPD_type"][1:]:
                    ipd = torch.stack((ipd, eval(ipd_type + "_ipd")), 1)
                # [B, 2, P, F, T]
            ipd = self.aggregate(ipd, "ipd")  # nband x [B, 2, F, T]
            feature_dict["IPD"] = ipd

        if "ILD" in self.input_feature:
            ild = self.compute_ILD(array, magnitude)  # [B, P, F, T]
            ild = self.aggregate(ild, "ild")  # nband x [B, 1, F, T]
            feature_dict["ILD"] = ild

        if "GCC" in self.input_feature:
            gcc = self.compute_GCC(array, mix_stft)  # [B, 2, P, F, T]
            gcc = self.aggregate(gcc, "gcc")  # nband x [B, 2, F, T]
            feature_dict["GCC"] = gcc

        if "SRP_PHAT" in self.input_feature:
            srp_phat = self.compute_SRP_PHAT(array, mix_stft, num_doa=self.srp_phat_num)
            # nband x [B, srp_phat_num, T]
            feature_dict["SRP_PHAT"] = srp_phat

        if "DF1D" in self.input_feature:
            batchsize = mix_stft.shape[0]
            self.update_loc_info(self.conf["loc_info"], batchsize)
            raw_ipd = phase[:, array.pair_right] - phase[:, array.pair_left]
            tpd = array.get_zone_TPD_number(
                self.loc_info, self.A, sort=self.conf["DFsort"]
            )
            direction_feature = self.get_zone_DF(raw_ipd, tpd)  # [B, df_num, F, T]
            feature_dict["DF1D"] = direction_feature

        if self.conf["spec"] == "mean":
            mix_stft = torch.mean(mix_stft, dim=1, keepdim=True)
        elif self.conf["spec"] == "reference":
            self.spec_num = 1
            mix_stft = mix_stft[:, self.conf["ref_mic_idx"]].unsqueeze(1)
        elif isinstance(self.conf["spec"], list):
            mix_stft = mix_stft[:, self.conf["spec"]]

        return mix_stft, feature_dict

    def init_ResRNN(self, z=1):
        resrnn_module = nn.ModuleList([])
        for i in range(self.nband):
            resrnn_module.append(
                ResRNN(
                    self.band_width[i] * z,
                    self.band_width[i] * z,
                    causal=True,  # 单向RNN的效果比双向的好,且减少计算量
                    res=False,
                )
            )
        return resrnn_module

    def compute_IPD(self, array, phase):
        """phase [B, M, F, T], return IPD [B, P, F, T]"""
        sin_ipd = torch.sin(phase[:, array.pair_right] - phase[:, array.pair_left])
        cos_ipd = torch.cos(phase[:, array.pair_right] - phase[:, array.pair_left])
        raw_ipd = torch.fmod(
            phase[:, array.pair_right] - phase[:, array.pair_left], math.pi * 2
        )
        return raw_ipd, cos_ipd, sin_ipd

    def compute_ILD(self, array, magnitude):
        """magnitude [B, M, F, T], return ILD [B, P, F, T]"""
        ild = 10 * torch.log10(
            (magnitude[:, array.pair_right] ** 2 + self.epsilon)
            / (magnitude[:, array.pair_left] ** 2 + self.epsilon)
        )
        return ild

    def compute_GCC(self, array, mix_stft):
        """
        mix_stft: [B, M, F, T], return: gcc, [B, 2, P, F, T]
        """
        cc = mix_stft[:, array.pair_left] * mix_stft[:, array.pair_right].conj()
        gcc = cc / (
            mix_stft[:, array.pair_left].abs() * mix_stft[:, array.pair_right].abs()
            + self.epsilon
        )
        gcc = torch.stack((gcc.real, gcc.imag), dim=1)
        return gcc

    def split_feature(self, feature):
        """对特征做分带操作

        Parameters
        ----------
        feature : tensor
            feature in shape [B, P, F, T]

        Returns
        -------
        list
            band_split feature in nband x [B, P, F_k, T]
        """
        subband_feature = []
        band_idx = 0
        for i in range(self.nband):
            subband_feature.append(
                feature[:, :, band_idx : band_idx + self.band_width[i]].contiguous()
            )
            band_idx += self.band_width[i]
        return subband_feature

    def aggregate(self, feature, name):
        """处理计算好的spatial feature

        Parameters
        ----------
        feature : tensor
            feature in shape [B, P, F, T] or [B, n, P, F, T]
        name : string
            特征的名字

        Returns
        -------
        list
            nband x [B, n, F, T]
        """
        if feature.ndim == 4:  # ILD
            b, p, f, t = feature.shape
            # nband x [B, P, n*F_k, T]
            subband_feature = self.split_feature(feature)
            n = 1
        else:
            b, n, p, f, t = feature.shape  # IPD / GCC
            # nband x [B, P, n*F_k, T]
            subband_feature = self.split_feature(feature[:, 0])
            for _n in range(1, n):
                subband_featurei = self.split_feature(feature[:, _n])
                for i in range(self.nband):
                    subband_feature[i] = torch.cat(
                        (subband_feature[i], subband_featurei[i]), -2
                    )

        transformed_feature = []
        for i in range(self.nband):
            this_feature = (
                subband_feature[i].permute((0, 3, 2, 1)).reshape(b * t, -1, p)
            )  # [B*T, n*F_k, P]
            this_feature = getattr(self, f"{name}_rnn")[i](this_feature)
            this_out = (
                this_feature[..., -1]
                .reshape(b, -1, this_feature.shape[1])
                .transpose(-1, -2)
            )
            # [B, n, F_k, T]
            this_out = this_out.unsqueeze(1)
            if n != 1:
                this_out = this_out.reshape(b, n, -1, t)
            transformed_feature.append(this_out)

        return transformed_feature

    def compute_SRP_PHAT(self, array, mix_stft, num_doa=180):
        """
        Steered Response Power
        mix_stft: [BS, M, F, T]
        gcc: [BS, P, F, T]
        return: SRP_PHAT nband x [BS, num_doa, T]
        """
        cc = mix_stft[:, array.pair_left] * mix_stft[:, array.pair_right].conj()
        gcc = cc / (
            mix_stft[:, array.pair_left].abs() * mix_stft[:, array.pair_right].abs()
            + self.epsilon
        )
        srp_phats = [list() for _ in range(self.nband)]

        for p in range(array.n_mic_pairs):
            this_pair = array.mic_pairs[p]
            mic_spacing = array.mic_distance[this_pair[0], this_pair[1]]
            max_tdoa = mic_spacing / array.snd_velocity

            est_tdoa = torch.linspace(-max_tdoa, max_tdoa, num_doa)  # [num_doa]
            sep_freq = torch.linspace(0, array.sr // 2, self.num_bins)  # [F]
            band_idx = 0
            for b in range(self.nband):
                subband_freq = sep_freq[band_idx : band_idx + self.band_width[b]]
                subband_gcc = gcc[:, p, band_idx : band_idx + self.band_width[b]]
                exp_part = torch.outer(
                    1j * 2 * math.pi * est_tdoa, subband_freq
                )  # [num_doa, F_k]
                gcc_phat = torch.einsum(
                    "vf,bft->bvt", (torch.exp(exp_part.to(gcc.device)), subband_gcc)
                ).real
                gcc_phat[gcc_phat < 0] = 0
                srp_phats[b].append(gcc_phat)
                band_idx += self.band_width[b]
        for i in range(self.nband):
            srp_phats[i] = torch.stack(srp_phats[i], 1).sum(1)
        return srp_phats

    def update_loc_info(self, loc_info, B):
        if isinstance(loc_info, list):
            info = loc_info[0]
            info = torch.tensor(info, dtype=torch.float32) * math.pi / 180
            info = info.unsqueeze(0)
            info = torch.repeat_interleave(info, B, dim=0)
        self.loc_info = info

    def get_zone_DF(self, raw_ipd, tpd):
        """
        raw_ipd: [B, P, F, T]
        tpd: [B, P, C, F]
        df: [B, C, F, T]
        """
        tpd = tpd.to(raw_ipd.device)
        cpx_ipd = torch.exp(1j * raw_ipd)
        direction_feature = torch.einsum("bpcf,bpft->bcft", (tpd.conj(), cpx_ipd))
        direction_feature = direction_feature.real  # [B, C, F, T]
        b, c, f, t = direction_feature.shape

        def average(tensor):
            # [B*T, F, C] => [B, F, T]
            return tensor.mean(-1).reshape(b, -1, f).transpose(-1, -2)

        def lastone(tensor):
            # [B*T, F, C] => [B, 1, F, T]
            return tensor[..., -1].reshape(b, -1, f).transpose(-1, -2).unsqueeze(1)

        def lasttwo(tensor):
            # [B*T, F, C] => [B, 2, F, T]
            return (
                tensor[..., -2:].reshape(b, -1, f, 2).permute((0, 3, 2, 1))
            )  # [B, 2, F, T]

        if self.zone_agg is None or self.zone_agg == "":
            return direction_feature
        elif self.zone_agg in ["TAA", "TAC"]:
            all_df = direction_feature.reshape(b * c, f, -1)
            ztt_df = self.zone_att_layer(all_df)
            ztt_df = ztt_df.reshape(b, c, f, -1)  # [B, C, F, T]
            if self.zone_agg == "TAA":
                ztt_df = torch.sum(ztt_df, dim=1).unsqueeze(1)  # [B, 1, F, T]
            return ztt_df
        elif self.zone_agg.startswith("RNN"):
            all_df = direction_feature.permute((0, 3, 2, 1)).reshape(-1, f, c)
            if self.zone_agg.endswith("L2"):
                rnn_in = torch.cat((all_df, all_df[..., 0].unsqueeze(-1)), dim=-1)
                rnn_out = self.zone_att_layer(rnn_in)
                rnn_out = lasttwo(rnn_out)
            else:
                rnn_out = self.zone_att_layer(all_df)
                rnn_out = lastone(rnn_out)
            return rnn_out


if __name__ == "__main__":
    # loc_feature_conf = {
    #     "sr": 16000,
    #     "feature": ["spec","IPD", "ILD", "GCC","spec"],
    #     "IPD_type": ["cos", "sin"],
    #     "SRP_PHAT_num": 18,
    #     "DFsort": False,
    #     "nFFT": 512,
    #     "frame_size": 512,
    #     "frame_hop": 256,
    #     "mic_pairs": "full",
    #     "mic_arch": [[-0.04, 0, 0], [0, 0.04, 0], [0.04, 0, 0], [0, -0.04, 0]],
    #     "spec": "reference",  # all / mean / reference / [0, -1]
    #     "ref_mic_idx": 0,
    #     "zone_agg": "",  # None, RNN-TAA, DPRNN-TAA, TAA, TAC
    #     "sample_method": "number",  # number (#A: range // A), fixed (#range // A: A)
    #     "A": 18,
    #     "df_dim": 8,
    #     "loc_info": [[-180, 180]],
    # }

    loc_feature_conf = {
        "sr": 16000,
        "feature": ["DF1D", "GCC", "ILD", "IPD"],  # 使用DF特征
        "IPD_type": ["cos", "sin"],
        "SRP_PHAT_num": 18,
        "DFsort": False,
        "nFFT": 512,
        "frame_size": 512,
        "frame_hop": 256,
        "mic_pairs": "full",
        "mic_arch": [
            [-0.04, 0, 0],
            [0, 0.04, 0],
            [0.04, 0, 0],
            [0, -0.04, 0],
        ],  # 麦克风的三维坐标
        "spec": "all",  # all / mean / reference / [0, -1]
        "ref_mic_idx": 0,
        "zone_agg": None,  # None, RNN-TAA, DPRNN-TAA, TAA, TAC
        "sample_method": "number",  # number (#A: range // A), fixed (#range // A: A)
        "A": 18,  # DF划分区域个数
        "loc_info": [[-180, 180]],  # 待检测的区域
        "df_dim": 8,
    }

    mic_conf1 = {
        "mic_pairs": "full",
        "mic_arch": [
            [-0.025, 0, 0],
            [-0.0176776695, 0.0176776695, 0],
            [0, 0.025, 0],
            [0.0176776695, 0.0176776695, 0],
            [0.025, 0, 0],
            [0.0176776695, -0.0176776695, 0],
            [0, -0.025, 0],
            [-0.0176776695, -0.0176776695, 0],
        ],
    }

    mic_conf2 = {
        "mic_pairs": "full",
        "mic_arch": [
            [-0.025, 0, 0],
            [0.025, 0, 0],
        ],
    }

    mic_conf3 = {
        "mic_pairs": "full",
        "mic_arch": [
            [-0.025, 0, 0],
            [0.025, 0, 0],
            [0.025, 0, 0],
        ],
    }

    array = MicrophoneArray(loc_feature_conf, mic_conf3)

    x = torch.rand(1, 4, 16000)
    fc = FeatureComputerSpecific(loc_feature_conf)
    feature_dict = fc(x)
    # print("debug", feature_dict)
    for k, v in feature_dict.items():
        print(k, v.shape)
