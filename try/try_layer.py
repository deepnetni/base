import torch
import torch.nn as nn


a = torch.randn(2, 3, 4)
c = torch.randn(2, 3, 6)

l = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 6))
for n, p in l.named_parameters():
    print(n, p.shape, p)
    if n == "1.weight":
        p.requires_grad = False
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, l.parameters()))
opt.zero_grad()

b = l(a)
print(b.shape)

loss = nn.functional.mse_loss(b, c)
print(loss)

loss.backward()
opt.step()
print("+++++++++++++++++")
for n, p in l.named_parameters():
    print(n, p.shape, p)
