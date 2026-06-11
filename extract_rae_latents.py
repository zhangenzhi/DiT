"""
Offline-precompute RAE (DINOv3-L K7) latents for ImageNet, sharded by class.
Each rank handles a disjoint subset of classes -> no cross-rank coordination.
Output: <out>/<classname>.npy  shape [N_c, 1024, 16, 16] fp16 (normalized latents,
exactly rae.encode()'s output). Labels follow sorted-class order (== ImageFolder).

Run: torchrun --nproc_per_node=4 extract_rae_latents.py --data-path <imagenet/train> --out <dir>
"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from utils_config import parse_args
from rae_utils import add_rae_root_arg, load_rae

EXTS = {".jpg", ".jpeg", ".png", ".JPEG"}


def center_crop(img, size):
    while min(*img.size) >= 2 * size:
        img = img.resize(tuple(x // 2 for x in img.size), resample=4)
    scale = size / min(*img.size)
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), resample=2)
    a = np.array(img); ch = (a.shape[0] - size) // 2; cw = (a.shape[1] - size) // 2
    return Image.fromarray(a[ch:ch + size, cw:cw + size])


def main(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    dev = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(dev)
    os.makedirs(args.out, exist_ok=True)

    rae = load_rae(args.rae_root, dev, decoder=False).eval()  # decoder not needed for encode
    for p in rae.parameters():
        p.requires_grad_(False)

    classes = sorted(d.name for d in os.scandir(args.data_path) if d.is_dir())
    if args.max_classes:
        classes = classes[:args.max_classes]
    my_classes = classes[rank::world]
    if rank == 0:
        print(f"{len(classes)} classes, {world} ranks; rank0 gets {len(my_classes)}", flush=True)

    for ci, c in enumerate(my_classes):
        outp = os.path.join(args.out, f"{c}.npy")
        if os.path.exists(outp):
            continue
        cdir = os.path.join(args.data_path, c)
        files = sorted(f for f in os.listdir(cdir) if os.path.splitext(f)[1] in EXTS)
        lat = []
        for i in range(0, len(files), args.batch_size):
            batch = files[i:i + args.batch_size]
            imgs = []
            for f in batch:
                try:
                    im = center_crop(Image.open(os.path.join(cdir, f)).convert("RGB"), 256)
                    if args.flip:
                        im = im.transpose(Image.FLIP_LEFT_RIGHT)
                    imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
                except Exception as e:
                    print(f"skip {f}: {e}", flush=True)
            if not imgs:
                continue
            x = torch.stack(imgs).to(dev)
            with torch.no_grad():
                z = rae.encode(x)  # [b,1024,16,16] normalized
            lat.append(z.to(torch.float16).cpu().numpy())
        arr = np.concatenate(lat, 0)
        np.save(outp, arr)
        if rank == 0 or ci % 20 == 0:
            print(f"[rank{rank}] {c}: {arr.shape} -> {outp}", flush=True)
    dist.barrier()
    if rank == 0:
        print("ALL DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True, help="ImageNet train/ directory (ImageFolder layout)")
    p.add_argument("--out", required=True, help="Output directory for per-class latent .npy files")
    add_rae_root_arg(p)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-classes", type=int, default=0, help="limit #classes (0=all); for smoke")
    p.add_argument("--flip", action="store_true", help="horizontally flip every image before encoding (for flip-aug latents)")
    args = parse_args(p)
    main(args)
