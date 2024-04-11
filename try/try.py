import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

a = torch.randn(1, 1, 1, 2, 3)
a = a.squeeze(0, 1, 2)
print(a.shape)

a = torch.arange(10).reshape(-1, 1)
print(a.repeat(1, 2))
