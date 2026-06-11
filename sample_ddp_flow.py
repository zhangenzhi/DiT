"""
Sample from a FLOW-MATCHING (velocity) DiT-REPA checkpoint trained by
train_feat_repa_flow.py, using the SiT-style euler / euler-maruyama samplers.
Saves PNGs and a .npz (arr_0 uint8 NHWC) that fid_from_stats.py can score
against the official ADM reference statistics.

Single-node DDP. Mirror of sample_ddp.py but for the velocity objective.
"""
import torch
import torch.distributed as dist
from models_repa import DiT_REPA_models
from models import DiT_models
from diffusers.models import AutoencoderKL
from download import find_model
from flow_samplers import euler_sampler, euler_maruyama_sampler
from PIL import Image
import numpy as np
import math
import os
import argparse
from tqdm import tqdm
from utils_config import parse_args


def create_npz_from_sample_folder(sample_dir, num):
    samples = []
    for i in tqdm(range(num), desc="Building .npz"):
        samples.append(np.asarray(Image.open(f"{sample_dir}/{i:06d}.png").convert("RGB")).astype(np.uint8))
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz to {npz_path} [shape={samples.shape}].")
    return npz_path


class _VelWrap(torch.nn.Module):
    """Make the (eval-mode) model return a tuple so the SiT samplers' [0] works."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, t, y=None):
        out = self.m(x, t, y)
        return (out,) if torch.is_tensor(out) else out


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    latent_size = args.image_size // 8
    if args.repa:
        model = DiT_REPA_models[args.model](
            input_size=latent_size, num_classes=args.num_classes, learn_sigma=False,
            z_dim=args.z_dim, proj_dim=args.proj_dim, align_depth=args.align_depth,
        ).to(device)
    else:
        model = DiT_models[args.model](
            input_size=latent_size, num_classes=args.num_classes, learn_sigma=False,
        ).to(device)
    state_dict = find_model(args.ckpt)  # picks ['ema'] if present
    # Checkpoints are saved from a torch.compile'd model -> keys carry an
    # "_orig_mod." prefix. Strip it so an uncompiled model loads correctly
    # (otherwise strict=False would silently load nothing -> random weights).
    state_dict = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                  for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    n_loaded = len(state_dict) - len(unexpected)
    print(f"Loaded {n_loaded}/{len(state_dict)} tensors (missing={len(missing)}, unexpected={len(unexpected)})")
    assert len(missing) == 0 and len(unexpected) == 0, \
        f"checkpoint/model key mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
    model.eval()
    sample_model = _VelWrap(model)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "")
    folder_name = f"{model_string_name}-{ckpt_string_name}-flow-{args.mode}-" \
                  f"cfg-{args.cfg_scale}-gh-{args.guidance_high}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // n)
    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)
    total = 0
    sampler = euler_maruyama_sampler if args.mode == "sde" else euler_sampler
    for _ in pbar:
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        samples = sampler(
            model=sample_model, latents=z, y=y, num_steps=args.num_steps,
            cfg_scale=args.cfg_scale, guidance_low=args.guidance_low,
            guidance_high=args.guidance_high, path_type="linear",
        ).to(torch.float32)
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(255. * (samples + 1) / 2., 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="DiT-B/2")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    p.add_argument("--sample-dir", type=str, default="outputs/flow_samples")
    p.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--per-proc-batch-size", type=int, default=64)
    p.add_argument("--num-fid-samples", type=int, default=50000)
    p.add_argument("--global-seed", type=int, default=0)
    # flow sampling recipe (SiT/REPA defaults)
    p.add_argument("--mode", type=str, default="sde", choices=["sde", "ode"])
    p.add_argument("--num-steps", type=int, default=250)
    p.add_argument("--cfg-scale", type=float, default=1.8)
    p.add_argument("--guidance-low", type=float, default=0.0)
    p.add_argument("--guidance-high", type=float, default=0.7)
    # repa arch
    p.add_argument("--repa", action="store_true")
    p.add_argument("--z-dim", type=int, default=768)
    p.add_argument("--proj-dim", type=int, default=2048)
    p.add_argument("--align-depth", type=int, default=8)
    args = parse_args(p)
    main(args)
