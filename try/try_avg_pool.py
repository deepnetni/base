import torch
import torch.nn as nn


inp = torch.arange(24).reshape(1, 2, 3, 4)


print(inp)


l = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 1))
out = l(inp)
print(out, out.shape)
