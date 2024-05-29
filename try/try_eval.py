#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np


a = "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2"

b = eval(a)
print(b)


x = torch.tensor(3.0, requires_grad=True)
x2 = x.clone()

y = 3 * x + x2 * 5
y.backward()
print(x.grad, x2.grad)


a = torch.randn(3, 4, 5, 6)

b = a[:, :2]
print(b.shape)


a = torch.arange(12).reshape(1, 3, 4)
m, k = a.max(-1)
print(a.shape, a, k.shape, m)


a = torch.tensor([0.1, 0.1, 0.1, 0.7])

b = F.gumbel_softmax(a, 2, hard=True)
print(b, b.device.type)


sz = 100
p = 0.5
m = 10
rng = np.random.default_rng(None)
# print(rng.random())
num = int(p * sz / 10.0 + rng.random())
lengths = np.full(num, m)
min_len = min(lengths)
if sz - min_len <= num:
    min_len = sz - num - 1

mask_idc = rng.choice(sz - min_len, num, replace=False)
print(mask_idc)
mask_idc = np.asarray(
    [mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])]
)
mask_idc = np.unique(mask_idc[mask_idc < sz])

print(mask_idc, mask_idc.shape)
