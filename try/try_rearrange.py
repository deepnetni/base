import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange


a = torch.randn(1, 10, 5, 4)
b = rearrange(a, "b (m c) t f->b (c m) t f", m=2)
print(b.shape)
