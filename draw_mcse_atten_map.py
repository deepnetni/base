import argparse

import torch
import numpy as np
from matplotlib import pyplot as plt

from models.mcse_skip_exp_attn import *
from utils.trunk import CHiMe3
from utils.logger import *
from utils.ini_opts import read_ini
from utils.audiolib import audiowrite

from models.conv_stft import STFT
import yaml
from utils.mp_decoder import mpStarMap


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="", type=str)
    parser.add_argument("--ckpt", help="", type=str)
    parser.add_argument("--out", help="", type=str)
    parser.add_argument("--conf", help="config file")

    args = parser.parse_args()
    return args


# @mpStarMap(2)
def draw_in_one(idx, args):
    dset = CHiMe3(cfg["chime"]["valid_dset"], subdir="dev")

    d, d_cln = dset[idx]  # T,C
    d = d[16000 : 16000 * 7]
    d.unsqueeze_(0)
    d_cln = d_cln[16000 : 16000 * 5]
    d_cln.unsqueeze_(0)
    xk_cln = stft.transform(d_cln)
    # d = d.cuda()
    with torch.no_grad():
        out, attn_w, xk, wf, wt, xsp = net(d)

    # xk: M,2,T,F
    # attn_w: F,T,T
    # wf: T,F,F
    # wt: F,T,T causal
    print(attn_w.shape, xk.shape, wf.shape, wt.shape, xsp.shape)
    fs = 16000
    xlabel = np.arange(0, fs // 2 + 1, 1000 if fs <= 16000 else 3000)  # 1000, 2000, ..., Frequency
    xticks = (257 - 1) * xlabel * 2 // fs

    ##############
    # spec FxTxT #
    ##############
    # plt.imshow(spec, origin="lower", aspect="auto", cmap="winter")
    xk = xk[4, ...]  # 2,t,f
    mag = np.sqrt((xk**2).sum(0))  # T,F
    spec = 10 * np.log10(mag**2 + 1e-10).transpose(-1, -2)  # F,T
    plt.subplot(411)
    plt.imshow(spec, origin="lower", aspect="auto", cmap="viridis")
    plt.axis("off")
    # plt.savefig(os.path.join(args.out, f"{idx}_noisy.svg"), bbox_inches="tight")

    xk_cln = xk_cln[0, ...]
    mag = np.sqrt((xk_cln**2).sum(0))  # T,F
    spec = 10 * np.log10(mag**2 + 1e-10).transpose(-1, -2)  # F,T
    plt.subplot(412)
    plt.imshow(spec, origin="lower", aspect="auto", cmap="viridis")
    plt.axis("off")
    # plt.savefig(os.path.join(args.out, f"{idx}_clean.svg"), bbox_inches="tight")

    ##############
    # attn FxTxT #
    ##############
    # attn_w = attn_w.mean(0)[-1]  # T, pick the global result
    # attn_w = attn_w[-1]  # T, pick the global result
    # attn_w = attn_w.unsqueeze(0).repeat(10, 1)  # 10xT
    attn_w = attn_w[:, -1, :]  # F,T, pick the global result
    plt.subplot(413)
    plt.imshow(
        attn_w.detach().numpy(),
        origin="lower",
        aspect="equal",
        cmap="viridis",
        # interpolation="nearest",
    )
    plt.axis("off")
    # plt.savefig(os.path.join(args.out, f"{args.name}_{idx}_attn.svg"), bbox_inches="tight")

    ############
    # wf TxFxF #
    ############
    # fig, ax = plt.subplots()
    wf = wf[-1]  # FxF
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xlabel)
    plt.subplot(427)
    plt.xticks(ticks=xticks, labels=xlabel)
    plt.yticks(ticks=xticks, labels=xlabel)
    plt.imshow(wf, origin="lower", aspect="equal", cmap="viridis")
    # plt.savefig(os.path.join(args.out, args.name + f"_{idx}_wf.svg"), bbox_inches="tight")

    ############
    # wt FxTxT #
    ############
    wt = wt[:, -1, :]  # FxT
    plt.subplot(428)
    plt.imshow(wt, origin="lower", aspect="equal", cmap="viridis", interpolation="nearest")
    # plt.savefig(os.path.join(args.out, args.name + f"_{idx}_wt.svg"), bbox_inches="tight")

    plt.savefig(os.path.join(args.out, args.name + f"_{idx}.svg"), bbox_inches="tight")


def draw(idx, args, dur=5):
    dset = CHiMe3(cfg["chime"]["valid_dset"], subdir="dev")

    d, d_cln = dset[idx]  # T,C
    d = d[16000 : 16000 * dur]
    audiowrite(os.path.join(args.out, "noisy.wav"), d, 16000)
    d.unsqueeze_(0)

    d_cln = d_cln[16000 : 16000 * dur]
    audiowrite(os.path.join(args.out, "clean.wav"), d_cln, 16000)
    d_cln.unsqueeze_(0)
    xk_cln = stft.transform(d_cln)
    # d = d.cuda()
    with torch.no_grad():
        out, attn_w, xk, wf, wt, xsp = net(d)

    # xk: M,2,T,F
    # attn_w: F,T,T
    # wf: T,F,F
    # wt: F,T,T causal
    # xsp: B,C,T,F
    print(attn_w.shape, xk.shape, wf.shape, wt.shape, xsp.shape)
    fs = 16000
    xlabel = np.arange(0, fs // 2 + 1, 1600 if fs <= 16000 else 3000)  # 1000, 2000, ..., Frequency
    xticks = (257 - 1) * xlabel * 2 // fs

    ##############
    # spec FxTxT #
    ##############
    # plt.imshow(spec, origin="lower", aspect="auto", cmap="winter")
    xk = xk[4, ...]  # 2,t,f
    mag = np.sqrt((xk**2).sum(0))  # T,F
    spec = 10 * np.log10(mag**2 + 1e-10).transpose(-1, -2)  # F,T
    plt.imshow(spec, origin="lower", aspect="auto", cmap="viridis")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(args.out, f"{idx}_noisy.svg"), bbox_inches="tight")

    xk_cln = xk_cln[0, ...]
    mag = np.sqrt((xk_cln**2).sum(0))  # T,F
    spec = 10 * np.log10(mag**2 + 1e-10).transpose(-1, -2)  # F,T
    plt.imshow(spec, origin="lower", aspect="auto", cmap="viridis")
    plt.xticks([])
    plt.yticks([])
    # plt.axis("off")
    plt.savefig(os.path.join(args.out, f"{idx}_clean.svg"), bbox_inches="tight")

    ##############
    # attn FxTxT #
    ##############
    # attn_w = attn_w.mean(0)[-1]  # T, pick the global result
    # attn_w = attn_w.unsqueeze(0).repeat(10, 1)  # 10xT
    attn_w = attn_w[:, -1, :]  # F,T, pick the global result
    plt.figure()
    plt.imshow(
        attn_w.detach().numpy(),
        origin="lower",
        aspect="equal",
        cmap="viridis",
        interpolation="nearest",
    )
    # plt.xticks([])
    plt.yticks([])
    # plt.ylabel("Frequency / Hz")
    # plt.xlabel("Frame Index")
    # plt.axis("off")
    plt.savefig(os.path.join(args.out, f"{args.name}_{idx}_attn.svg"), bbox_inches="tight")
    plt.close()

    ############
    # wf TxFxF #
    ############
    # fig, ax = plt.subplots()
    plt.figure()
    wf = wf[-1]  # FxF
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xlabel)
    plt.xticks(ticks=xticks, labels=xlabel)
    plt.yticks(ticks=xticks, labels=xlabel)
    # plt.xlabel("Frequency / Hz")
    # plt.ylabel("Frequency / Hz")
    plt.imshow(wf, origin="lower", aspect="equal", cmap="viridis")
    plt.savefig(os.path.join(args.out, args.name + f"_{idx}_wf.svg"), bbox_inches="tight")
    plt.close()

    ############
    # wt FxTxT #
    ############
    plt.figure()
    wt = wt[:, -1, :]  # FxT
    plt.imshow(wt, origin="lower", aspect="equal", cmap="viridis", interpolation="nearest")
    plt.savefig(os.path.join(args.out, args.name + f"_{idx}_wt.svg"), bbox_inches="tight")
    # scale = torch.arange(1, attn_w.shape[-1] + 1).reshape(-1, 1)
    # attn_w = attn_w.mean(0) * scale  # TxT
    # attn_w = attn_w[:, -1, ...]  # F,T
    # plt.colorbar()

    ###############
    # wlstm CxTxT #
    ###############
    # plt.figure()
    # wlstm = wlstm.mean(0)[-1, :].unsqueeze(0).repeat(10, 1)  # FxT
    # plt.axis("off")
    # plt.imshow(wlstm, origin="lower", aspect="auto", cmap="viridis", interpolation="nearest")
    # plt.savefig(os.path.join(args.out, args.name + f"_{idx}_wlstm.svg"), bbox_inches="tight")

    ############
    # xsp B,C,T,F #
    ############
    # xsp = xsp.squeeze(0)
    # fig, ax = plt.subplots(25, 1, constrained_layout=True)
    # for i, axi in enumerate(ax.flat):
    #     xsp_ = xsp[i, ...].permute(1, 0)  # FxT
    #     # plt.xticks([])
    #     # plt.yticks([])
    #     axi.set_xticks([])
    #     axi.set_yticks([])
    #     axi.imshow(xsp_, origin="lower", aspect="auto", cmap="viridis")
    # for i in range(20):
    #     plt.figure()
    #     xsp_ = xsp[i, ...].permute(1, 0)  # FxT
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.imshow(xsp_, origin="lower", aspect="auto", cmap="viridis")
    #     plt.savefig(os.path.join(args.out, args.name + f"_{idx}_xsp_{i}.svg"), bbox_inches="tight")


if __name__ == "__main__":
    args = parse()
    cp = CPrint()
    cfg_fname = "config/yconf_mcse_chime.yaml" if args.conf is None else args.conf
    print("##", cfg_fname)

    if os.path.splitext(cfg_fname)[-1] == ".ini":
        cfg = read_ini(cfg_fname)
    elif os.path.splitext(cfg_fname)[-1] == ".yaml":
        with open(cfg_fname, "r") as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        raise RuntimeError("File not supported.")

    os.makedirs(args.out) if not os.path.exists(args.out) else None
    stft = STFT(512, 256)

    # md_conf = {"in_channels": 6, "feature_size": 257, "mid_channels": 72}
    md_conf = cfg["chime"]["model_conf"]
    g_mics = md_conf["in_channels"]
    cfg["config"]["name"] = cfg["config"]["name"] if args.name is None else args.name
    md_name = cfg["config"]["name"]
    cp.r(f"current: {md_name}")
    model = tables.models.get(md_name)
    assert model is not None
    net = model(**md_conf)

    assert args.ckpt is not None
    assert args.out is not None
    net.load_state_dict(torch.load(args.ckpt))
    # net.cuda()
    net.eval()

    # for i in range(10):
    #     draw_in_one(i, args=args)
    draw(0, args)
