import torch

x = torch.FloatTensor([[1.0, 2.0]])  # 1x2
w1 = torch.FloatTensor([[2.0], [1.0]])  # 2x1
w2 = torch.FloatTensor([[1.0, 2.0]])  # 2x1
w1.requires_grad = True
w2.requires_grad = True

d_ = torch.matmul(w1, w2)
# d = d_
# f = torch.matmul(d, w2)
f = d_.sum()
f.backward()

print(w1.grad, w2.grad)
