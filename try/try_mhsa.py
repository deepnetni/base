import torch
import torch.nn as nn


if __name__ == "__main__":
    # inp = torch.randn(2, 10, 8)
    # l = nn.MultiheadAttention(embed_dim=8, num_heads=4, batch_first=True)
    inp = torch.randn(10, 2, 8)
    l = nn.MultiheadAttention(embed_dim=8, num_heads=4)

    mask = torch.ones(10, 10).triu(1).bool()
    print(mask)
    out, w = l(inp, inp, inp, attn_mask=mask)
    print(w)
    # out, w = l(inp, inp, inp, is_causal=True)
