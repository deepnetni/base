import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


a = torch.arange(12).reshape(1, 2, 6)
print(a[..., :0])
