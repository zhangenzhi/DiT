# Compute FID-50k against the official ADM/guided-diffusion reference statistics
# (VIRTUAL_imagenet256_labeled.npz, which stores precomputed 2048-d Inception
# mu/sigma). Features are extracted with the same pytorch-fid InceptionV3 port
# that torchmetrics uses, which matches the TF Inception pool3 within FID tolerance.
#
# Usage:
#   python fid_from_stats.py --samples-npz <samples.npz> [--ref-npz <ref.npz>]
#   python fid_from_stats.py --self-check        # validate extractor vs stored stats
import argparse
import numpy as np
import torch
from scipy import linalg
from torchmetrics.image.fid import NoTrainInceptionV3
from utils_config import parse_args

def build_extractor(device):
    fe = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
    return fe


@torch.no_grad()
def features_from_images(imgs_uint8_nhwc, device, batch_size=128):
    """imgs: (N, H, W, 3) uint8 -> (N, 2048) float64 features."""
    fe = build_extractor(device)
    feats = []
    n = imgs_uint8_nhwc.shape[0]
    for i in range(0, n, batch_size):
        chunk = imgs_uint8_nhwc[i:i + batch_size]
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous().to(device)  # NCHW uint8
        f = fe(x)
        feats.append(f.double().cpu())
    return torch.cat(feats, 0).numpy()


def stats(feats):
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Standard Frechet distance (same formula as pytorch-fid / ADM evaluator)."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-npz", type=str, default=None,
                   help="npz with arr_0 (N,256,256,3) uint8 generated samples")
    p.add_argument("--ref-npz", type=str, default=None,
                   help="ADM reference stats npz (VIRTUAL_imagenet256_labeled.npz)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--self-check", action="store_true",
                   help="Extract features from the reference npz's own arr_0 and FID against stored mu/sigma")
    args = parse_args(p)

    if not args.ref_npz:
        p.error("--ref-npz is required (or set ref-npz in the YAML config)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = np.load(args.ref_npz)
    mu_ref, sigma_ref = ref["mu"], ref["sigma"]

    if args.self_check:
        imgs = ref["arr_0"]
        print(f"[self-check] extracting features from {imgs.shape[0]} reference images...")
        feats = features_from_images(imgs, device, args.batch_size)
        mu_g, sigma_g = stats(feats)
        fid = frechet_distance(mu_g, sigma_g, mu_ref, sigma_ref)
        print(f"[self-check] FID(our features of ref arr_0  vs  stored mu/sigma) = {fid:.4f}")
        print("  (small => extractor is compatible with the stored stats; the residual is the "
              "10k-subset vs full-reference sampling gap.)")
        return

    assert args.samples_npz, "Provide --samples-npz or use --self-check"
    samp = np.load(args.samples_npz)["arr_0"]
    print(f"Extracting features from {samp.shape[0]} samples ({args.samples_npz})...")
    feats = features_from_images(samp, device, args.batch_size)
    mu_g, sigma_g = stats(feats)
    fid = frechet_distance(mu_g, sigma_g, mu_ref, sigma_ref)
    print(f"FID-{samp.shape[0]} (vs official ADM reference {args.ref_npz}) = {fid:.4f}")


if __name__ == "__main__":
    main()
