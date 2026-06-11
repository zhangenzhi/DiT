# Compute FID on the first N contiguous PNGs already written by generate.py,
# WITHOUT waiting for the full 50k run to finish. Reuses fid_from_stats helpers.
import argparse, os
import numpy as np
from PIL import Image
import torch
from fid_from_stats import features_from_images, stats, frechet_distance
from utils_config import parse_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="folder of 000000.png ... samples")
    p.add_argument("--n", type=int, required=True, help="use the first N contiguous PNGs")
    p.add_argument("--ref-npz", required=True,
                   help="ADM reference stats npz (VIRTUAL_imagenet256_labeled.npz)")
    p.add_argument("--batch-size", type=int, default=128)
    args = parse_args(p)

    # verify contiguity and load
    imgs = []
    for i in range(args.n):
        fp = os.path.join(args.dir, f"{i:06d}.png")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"missing {fp} (only {i} contiguous PNGs available)")
        imgs.append(np.asarray(Image.open(fp).convert("RGB"), dtype=np.uint8))
    imgs = np.stack(imgs)
    print(f"loaded {imgs.shape[0]} PNGs, shape {imgs.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = features_from_images(imgs, device, args.batch_size)
    mu_g, sigma_g = stats(feats)
    ref = np.load(args.ref_npz)
    fid = frechet_distance(mu_g, sigma_g, ref["mu"], ref["sigma"])
    print(f"PARTIAL FID-{imgs.shape[0]} (official SiT-XL/2+REPA, vs ADM ref) = {fid:.4f}")


if __name__ == "__main__":
    main()
