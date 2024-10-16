import numpy
import torch
from l3das.metrics import task1_metric
from utils.audiolib import audioread, audiowrite

# enh = "/home/deepni/datasets/l3das/L3DAS22_Task1_dev/data/1272-128104-0000_A.wav"
enh = "/home/deepni/github/base/trained_mcse_l3das_win/pred_mcse_vtest_100/labels/1272-128104-0000.wav"
src = "/home/deepni/datasets/l3das/L3DAS22_Task1_dev/labels/1272-128104-0000.wav"

enh_d, fs = audioread(enh)
src_d, _ = audioread(src)

N = len(enh_d)
src_d = src_d[:N]
audiowrite("enh.wav", enh_d, fs)
audiowrite("src.wav", src_d, fs)
# src_d = torch.from_numpy(src_d)
# enh_d = torch.from_numpy(enh_d)
# enh_d = enh_d[..., 0]
print(enh_d.shape, src_d.shape)
task1_metric = task1_metric(src_d, enh_d)
print(task1_metric)
