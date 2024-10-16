import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.optim import lr_scheduler


class t(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return x


class t2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.fc = nn.Linear(2, 4)

    def forward(self, x):
        return x


net = t()
net2 = t2()


opt = torch.optim.Adam(params=net.parameters(), lr=5e-4)
lr = lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)


lv = opt.state_dict()["param_groups"][0]["lr"]
print(lv)
lr.step()
lv = opt.state_dict()["param_groups"][0]["lr"]
print(lv)


opt.add_param_group({"params": net2.parameters(), "lr": 3e-4})

lv = opt.state_dict()["param_groups"][1]["lr"]
print(lv)
lr.step()
lv = opt.state_dict()["param_groups"][1]["lr"]
print(lv)
