"""Sample N images from a RAE-flow checkpoint and save a square grid PNG for eyeballing.
Single-GPU, no DDP. Reuses the model/decoder/sampler from sample_rae_flow.py."""
import os, sys, math, argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/work/c30636/DiT")
from sample_rae_flow import (load_model, velocity_ode, velocity_ode_ag, RAE,
                             ENC_NAME, DEC_CFG, DEC_PT, STATS, LATENT)

# 16 recognizable ImageNet classes for a nice grid
CLASSES = [207, 281, 291, 388, 323, 130, 88, 933,
           949, 985, 980, 417, 279, 90, 437, 992]


def main(a):
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    model = load_model(a.ckpt, a.arch, a.num_classes, dev)
    bad = load_model(a.ag_ckpt, a.arch, a.num_classes, dev) if a.ag_ckpt else None
    rae = RAE(encoder_name=ENC_NAME, resolution=256, decoder_config_path=DEC_CFG,
              decoder_patch_size=16, pretrained_decoder_path=DEC_PT,
              noise_tau=0.0, normalization_stat_path=STATS).to(dev).eval()

    n = a.n
    y = torch.tensor(CLASSES[:n], device=dev)
    z = torch.randn(n, *LATENT, device=dev)
    with torch.no_grad():
        if bad is not None:
            x0 = velocity_ode_ag(model, bad, z, y, a.num_steps, a.ag_scale, dev)
        else:
            x0 = velocity_ode(model, z, y, a.num_steps, a.cfg_scale, a.num_classes, dev)
        imgs = rae.decode(x0).clamp(0, 1)                       # [n,3,256,256]
    arr = imgs.mul(255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

    g = int(math.ceil(n ** 0.5)); s = 256; pad = 4
    canvas = Image.new("RGB", (g * s + (g + 1) * pad, g * s + (g + 1) * pad), (255, 255, 255))
    for i, im in enumerate(arr):
        r, c = divmod(i, g)
        canvas.paste(Image.fromarray(im), (pad + c * (s + pad), pad + r * (s + pad)))
    canvas.save(a.out)
    print("saved", a.out, "classes", CLASSES[:n])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", default="dit", choices=["dit", "dit_rope", "dit_rope_ddt"])
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--cfg-scale", type=float, default=1.8)
    p.add_argument("--ag-ckpt", default=None)
    p.add_argument("--ag-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/work/c30636/DiT/outputs/grid_400k.png")
    main(p.parse_args())
