# Quick CPU side-by-side sample comparison of two DiT checkpoints.
# Uses identical noise + class labels + sampling steps for both, so the
# only difference is the model weights (REPA vs baseline).
import os, time, argparse
from collections import OrderedDict
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid

from models import DiT_models
from models_repa import DiT_REPA_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from utils_config import parse_args

_p = argparse.ArgumentParser()
_p.add_argument("--repa-ckpt", default="results/001-DiT-XL-2-repa/checkpoints/0050000.pt")
_p.add_argument("--baseline-ckpt", default="results/003-DiT-XL-2/checkpoints/0050000.pt")
_p.add_argument("--outputs-dir", default="outputs")
_a = parse_args(_p)

DEV = "cpu"
N = 16                       # images per model (4x4 grid)
STEPS = 50                   # sampling steps (reduced for CPU)
CFG = 4.0
SEED = 0
CKPTS = [
    ("repa_xl2_50k",     _a.repa_ckpt,     True),
    ("baseline_xl2_50k", _a.baseline_ckpt, False),
]
# a fixed set of recognizable ImageNet classes
CLASSES = [207, 360, 387, 388, 933, 980, 250, 270, 279, 291, 88, 11, 130, 323, 562, 417]


def strip(sd):
    return OrderedDict((k.replace("_orig_mod.", ""), v) for k, v in sd.items())


def main():
    print("threads:", torch.get_num_threads(), flush=True)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEV).eval()
    diffusion = create_diffusion(str(STEPS))
    latent = 32

    # fixed noise + labels, identical across models
    torch.manual_seed(SEED)
    z0 = torch.randn(N, 4, latent, latent)
    y0 = torch.tensor(CLASSES[:N])

    for name, path, is_repa in CKPTS:
        t0 = time.time()
        if is_repa:
            model = DiT_REPA_models["DiT-XL/2"](input_size=latent, num_classes=1000,
                                                z_dim=768, proj_dim=2048, align_depth=8)
        else:
            model = DiT_models["DiT-XL/2"](input_size=latent, num_classes=1000)
        model = model.to(DEV).eval()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        miss, unexp = model.load_state_dict(strip(ck["ema"]), strict=False)
        print(f"[{name}] loaded ema (missing={len(miss)} unexpected={len(unexp)})", flush=True)

        # classifier-free guidance setup
        z = torch.cat([z0, z0], 0)
        y = torch.cat([y0, torch.tensor([1000] * N)], 0)
        model_kwargs = dict(y=y, cfg_scale=CFG)

        with torch.no_grad():
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False,
                model_kwargs=model_kwargs, progress=True, device=DEV,
            )
            samples, _ = samples.chunk(2, dim=0)
            imgs = vae.decode(samples / 0.18215).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        grid = make_grid(imgs, nrow=4)
        arr = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        out = os.path.join(_a.outputs_dir, f"cmp_{name}.png")
        Image.fromarray(arr).save(out)
        print(f"[{name}] saved {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
