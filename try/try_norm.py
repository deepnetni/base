import torch


a = torch.arange(60).reshape(1, 3, 4, 5).float()


mu = torch.mean(a, dim=(-3, -2, -1), keepdim=True)
# b = a.sum(-1) / 4
# c = torch.sum((a - mu.unsqueeze(-1)) ** 2, dim=-1) / 4
var = torch.var(a, dim=(-3, -2, -1), keepdim=True, unbiased=False)


b = (a - mu) / torch.sqrt(var + 1e-5)
print(b)

l = torch.nn.LayerNorm([3, 4, 5], elementwise_affine=False)
out = l(a)
print(out)
for n, p in l.named_parameters():
    print(n, p.shape)

l = torch.nn.LayerNorm([4, 5])
out = l(a)
# print(out)
for n, p in l.named_parameters():
    print(n, p.shape)
    print(p)
