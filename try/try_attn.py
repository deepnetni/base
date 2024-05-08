import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange


l = nn.MultiheadAttention(embed_dim=20, num_heads=5, kdim=20, vdim=10, batch_first=True)
inp = torch.randn(2, 30, 20)
# k = torch.randn(2, 30, 5)
# v = torch.randn(2, 30, 10)
out, w = l(inp, inp, inp)
print(out.shape, w)
