import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange


a = torch.randn(3, 3)

b = nn.Softmax()(a)
print(b, b.sum(0), b.sum(1))
print(torch.diag(b))

c = torch.tensor(0)
print(c)
