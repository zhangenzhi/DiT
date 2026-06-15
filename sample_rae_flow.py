"""
Sample from a Path-A RAE-flow DiT checkpoint (our DiT trained in the DINOv3-K7
RAE latent with velocity flow + shift), decode via the official RAE ViT-XL
decoder, save a .npz (arr_0 uint8 NHWC) for fid_from_stats.py.

Velocity ODE (euler) with the SAME logit-normal/SD3 time shift used in training,
plus standard CFG. Single-node DDP.
"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from models import DiT
from models_rope import DiT_RoPE, DiT_RoPE_DDT
from download import find_model
from utils_config import parse_args
from rae_utils import LATENT, add_rae_root_arg, load_rae


SHIFT = math.sqrt(math.prod(LATENT) / 4096.0)   # = 8.0


def build_net(arch, num_classes, device, base_model_depth=None):
    if arch == "dit_rope_ddt":
        return DiT_RoPE_DDT(input_size=16, patch_size=1, in_channels=1024,
                            enc_hidden=1152, dec_hidden=2048, enc_depth=28, dec_depth=2,
                            enc_heads=16, dec_heads=16, num_classes=num_classes,
                            learn_sigma=False, base_model_depth=base_model_depth).to(device)
    Net = DiT_RoPE if arch == "dit_rope" else DiT
    return Net(input_size=16, patch_size=1, in_channels=1024, hidden_size=1152,
               depth=28, num_heads=16, num_classes=num_classes, learn_sigma=False).to(device)


def load_model(ckpt, arch, num_classes, device, base_model_depth=None):
    model = build_net(arch, num_classes, device, base_model_depth)
    sd = find_model(ckpt)  # ['ema'] if present
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("repa_projector")}  # drop unused REPA head
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert not miss and not unexp, f"key mismatch miss={miss[:3]} unexp={unexp[:3]}"
    return model.eval()


@torch.no_grad()
def velocity_ode(model, z, y, num_steps, cfg, num_classes, device):
    t_grid = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    t_grid = SHIFT * t_grid / (1 + (SHIFT - 1) * t_grid)
    x = z.double()
    use_cfg = cfg > 1.0
    ynull = torch.full_like(y, num_classes)
    for tc, tn in zip(t_grid[:-1], t_grid[1:]):
        if use_cfg:
            xin = torch.cat([x, x], 0).float()
            yin = torch.cat([y, ynull], 0)
            tin = torch.full((xin.shape[0],), float(tc), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(xin, tin, yin).double()
            vc, vu = v.chunk(2, 0)
            v = vu + cfg * (vc - vu)
        else:
            tin = torch.full((x.shape[0],), float(tc), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(x.float(), tin, y).double()
        x = x + (tn - tc) * v
    return x.float()


@torch.no_grad()
def velocity_ode_ag(model_g, model_b, z, y, num_steps, w, device):
    """Autoguidance (Karras 2024): guide the good model with an undertrained copy
    of itself. Both passes are class-conditional (no unconditional). v = vb + w*(vg-vb)."""
    t_grid = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    t_grid = SHIFT * t_grid / (1 + (SHIFT - 1) * t_grid)
    x = z.double()
    for tc, tn in zip(t_grid[:-1], t_grid[1:]):
        tin = torch.full((x.shape[0],), float(tc), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vg = model_g(x.float(), tin, y).double()
            vb = model_b(x.float(), tin, y).double()
        v = vb + w * (vg - vb)
        x = x + (tn - tc) * v
    return x.float()


@torch.no_grad()
def velocity_ode_ig(model, z, y, num_steps, ig_scale, device, ig_low=0.0, ig_high=1.0):
    """RAEv2 Internal Guidance: the model returns (full, base early-exit). Guide the
    full prediction with its own weak early-exit head: v = base + ig_scale*(full-base).
    One forward, no second model and no null class (unlike AG/CFG). ig_low/high gate
    guidance to a window of the (pre-shift) time fraction; (0,1) = always on (RAEv2)."""
    lin = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    t_grid = SHIFT * lin / (1 + (SHIFT - 1) * lin)
    x = z.double()
    for i, (tc, tn) in enumerate(zip(t_grid[:-1], t_grid[1:])):
        tfrac = float(lin[i])
        tin = torch.full((x.shape[0],), float(tc), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            full, base = model(x.float(), tin, y)
        full = full.double(); base = base.double()
        v = base + ig_scale * (full - base) if ig_low <= tfrac <= ig_high else full
        x = x + (tn - tc) * v
    return x.float()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    dist.init_process_group("nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    dev = rank % torch.cuda.device_count(); torch.cuda.set_device(dev)
    device = torch.device("cuda", dev)
    torch.manual_seed(args.global_seed * world + rank)

    use_ig = args.ig_base_depth > 0
    model = load_model(args.ckpt, args.arch, args.num_classes, device,
                       base_model_depth=args.ig_base_depth or None)
    model_bad = None
    if args.ag_ckpt and not use_ig:
        model_bad = load_model(args.ag_ckpt, args.arch, args.num_classes, device)
    if rank == 0:
        if use_ig:
            print(f"INTERNAL GUIDANCE: ckpt={os.path.basename(args.ckpt)} base_depth={args.ig_base_depth} "
                  f"ig_scale={args.ig_scale} interval=({args.ig_low},{args.ig_high}) "
                  f"steps={args.num_steps} shift={SHIFT:.2f}", flush=True)
        elif model_bad is not None:
            print(f"AUTOGUIDANCE: good={os.path.basename(args.ckpt)} weak={os.path.basename(args.ag_ckpt)} "
                  f"w={args.ag_scale} steps={args.num_steps} shift={SHIFT:.2f}", flush=True)
        else:
            print(f"loaded model; cfg={args.cfg_scale} steps={args.num_steps} shift={SHIFT:.2f}", flush=True)

    rae = load_rae(args.rae_root, device).eval()

    ckpt_s = os.path.basename(args.ckpt).replace(".pt", "")
    if use_ig:
        tag = f"ig{args.ig_scale}-d{args.ig_base_depth}"
        if (args.ig_low, args.ig_high) != (0.0, 1.0):
            tag += f"-int{args.ig_low}_{args.ig_high}"
    elif model_bad is not None:
        bad_s = os.path.basename(args.ag_ckpt).replace(".pt", "")
        tag = f"ag{args.ag_scale}-w{bad_s}"
    else:
        tag = f"cfg{args.cfg_scale}"
    folder = f"{args.sample_dir}/rae-flow-{ckpt_s}-{tag}-seed{args.global_seed}"
    if rank == 0:
        os.makedirs(folder, exist_ok=True)
    dist.barrier()

    n = args.per_proc_batch_size
    total = int(math.ceil(args.num_fid_samples / (n * world)) * n * world)
    per = total // world; iters = per // n
    pbar = tqdm(range(iters)) if rank == 0 else range(iters)
    done = 0
    for _ in pbar:
        labels = torch.randint(0, args.num_classes, (n,), device=device)
        z = torch.randn(n, *LATENT, device=device)
        with torch.no_grad():
            if use_ig:
                x0 = velocity_ode_ig(model, z, labels, args.num_steps, args.ig_scale, device,
                                     ig_low=args.ig_low, ig_high=args.ig_high)
            elif model_bad is not None:
                x0 = velocity_ode_ag(model, model_bad, z, labels, args.num_steps, args.ag_scale, device)
            else:
                x0 = velocity_ode(model, z, labels, args.num_steps, args.cfg_scale, args.num_classes, device)
            imgs = rae.decode(x0).clamp(0, 1)
        arr = imgs.mul(255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        for i, im in enumerate(arr):
            Image.fromarray(im).save(f"{folder}/{i*world+rank+done:06d}.png")
        done += n * world
    dist.barrier()
    if rank == 0:
        imgs = [np.asarray(Image.open(f"{folder}/{i:06d}.png")) for i in range(args.num_fid_samples)]
        np.savez(f"{folder}.npz", arr_0=np.stack(imgs))
        print(f"saved {folder}.npz [{args.num_fid_samples}]", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    add_rae_root_arg(p)
    p.add_argument("--sample-dir", default="outputs/rae_flow_samples")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--per-proc-batch-size", type=int, default=32)
    p.add_argument("--num-fid-samples", type=int, default=10000)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--cfg-scale", type=float, default=1.8)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--arch", default="dit", choices=["dit", "dit_rope", "dit_rope_ddt"])
    p.add_argument("--ag-ckpt", default=None,
                   help="weak/undertrained checkpoint for autoguidance (Karras). If set, CFG is ignored.")
    p.add_argument("--ag-scale", type=float, default=2.0, help="autoguidance weight w in vb+w*(vg-vb)")
    p.add_argument("--ig-base-depth", type=int, default=0,
                   help="Internal Guidance: encoder depth of the early-exit base head (0=off; must match training, RAEv2=8). dit_rope_ddt only. Overrides AG/CFG.")
    p.add_argument("--ig-scale", type=float, default=2.0, help="IG weight in base+ig_scale*(full-base)")
    p.add_argument("--ig-low", type=float, default=0.0, help="IG interval lower bound on pre-shift time fraction (RAEv2 default 0)")
    p.add_argument("--ig-high", type=float, default=1.0, help="IG interval upper bound (RAEv2 default 1 = always on)")
    args = parse_args(p)
    main(args)
