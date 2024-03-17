#!/usr/bin/env python3
from pathlib import Path
import re
import torch

dir = Path("/home/deepnetni/trunk/CHiME3/data/audio/16kHz/isolated/train")

for f in dir.iterdir():
    if not f.is_dir():
        continue

    # if re.match(".*simu$", str(f)):
    if f.match("*simu"):
        a = list(f.rglob(r"*CH1.wav"))

t = str(a[100])
print(t, type(t))
p = ("(CAF|PED|STR|BUS).CH1.wav", "ORG.wav")
print(re.sub(*p, t))

a = torch.arange(12).reshape(3, 2, 2)

print(a)


c = torch.concat([a, a[:, :-1, ...]], dim=1)
print(c)
