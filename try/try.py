import torch
import torch.nn as nn
import numpy as np


a = torch.arange(8).reshape(2, 4, 1).float() * 100
l = nn.LayerNorm([1])

print(a)

out = l(a)
print(out)


a = torch.arange(12).reshape(1, 3, 4)
b = [3, 2, 1]
b = [torch.arange(i) for i in b]
print(c.shape)
