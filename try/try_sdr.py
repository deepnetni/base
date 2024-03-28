#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
import numpy as np


def sdr_(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1))
    den = np.sum(np.square(references - estimates), axis=(1))
    num += delta
    den += delta
    return 10 * np.log10(num / den)


a = torch.randn(2, 1000)
b = torch.randn(2, 1000)

out = sdr(preds=a, target=b)
print(out.shape, out.mean())

s = 0
for i, j in zip(a.numpy(), b.numpy()):
    s += sdr_(j, i)

print(s)
# a = torch.concat([torch.randn(2, 1000), torch.zeros(2, 1000)], dim=-1)
# b = torch.concat([torch.randn(2, 1000), torch.zeros(2, 1000)], dim=-1)
# out = sdr(preds=a, target=b)
# print(out.shape, out.mean())
