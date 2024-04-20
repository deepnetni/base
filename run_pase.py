import torch
from models.pase.models.frontend import wf_builder

if __name__ == "__main__":
    pase = wf_builder("config/frontend/PASE+.cfg").eval()
    pase.load_pretrained("./pretrained/pase_e199.ckpt", load_last=True, verbose=True)
    pase.cuda()

    inp = torch.randn(2, 1, 16000).cuda()
    out = pase(inp)
    print(out.shape)
    out = out.flatten(1)
    print(out.shape)
