import torch
import numpy as np


a = torch.arange(24).reshape(2, 4, 3)

print(a[..., :1, 0])
