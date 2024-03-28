from itertools import pairwise
import torch
import torch.nn as nn


inp = torch.arange(24).reshape(1, 2, 3, 4).float()

print(inp)


l = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 1))
out = l(inp)
print(out)
print("##", out.shape)


l = nn.AdaptiveAvgPool2d((None, 1))
out = l(inp)
print(out)
print("#", out.shape)
