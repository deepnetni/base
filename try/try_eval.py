#!/usr/bin/env python3
import torch


a = "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2"

b = eval(a)
print(b)


x = torch.tensor(3.0, requires_grad=True)
x2 = x.clone()

y = 3 * x + x2 * 5
y.backward()
print(x.grad, x2.grad)


a = torch.randn(3, 4, 5, 6)

b = a[:, :2]
print(b.shape)
