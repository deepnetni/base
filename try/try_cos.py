import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange


a = torch.randn(2, 4)
b = torch.randn(2, 4)


sim = a.unsqueeze(1) @ b.unsqueeze(-1)
c = torch.norm(a, dim=-1) * torch.norm(b, dim=-1)
sim = sim.squeeze() / c
print(sim.shape, c.shape, sim.shape)
print(sim)

sim = F.cosine_similarity(a, b, dim=-1)
print(sim.shape)
print(sim)


c = torch.randn(4, 10)
sim = F.cosine_similarity(c.unsqueeze(1), c.unsqueeze(0), dim=2)
print(sim.shape, sim)
