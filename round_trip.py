"""Latent round-trip sanity: verify our offline-extracted latents are correct.
For a few images of one class:
  (A) original (center-crop 256)
  (B) fresh rae.encode -> rae.decode   (known-good official path)
  (C) our stored latent -> rae.decode  (the 626GB we trained on)
Numerically compares stored latent vs fresh encode. If C~=B~=A, extraction is fine."""
import os, sys, numpy as np, torch
from PIL import Image
import torchvision.transforms as T
sys.path.insert(0, "/work/c30636/DiT")
from sample_rae_flow import RAE, ENC_NAME, DEC_CFG, DEC_PT, STATS

CLASS = "n01440764"
IMGDIR = f"/work/c30778/dataset/imagenet/train/{CLASS}"
LAT = f"/work/c30636/dataset/rae_dinov3l_k7_latents/train/{CLASS}.npy"
N = 4
OUT = "/work/c30636/DiT/outputs/round_trip.png"


def center_crop(img, size):
    while min(img.size) >= 2 * size:
        img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)
    scale = size / min(img.size)
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), resample=Image.BICUBIC)
    a = np.array(img)
    h, w = a.shape[:2]
    top = (h - size) // 2; left = (w - size) // 2
    return Image.fromarray(a[top:top + size, left:left + size])


def main():
    dev = torch.device("cuda")
    rae = RAE(encoder_name=ENC_NAME, resolution=256, decoder_config_path=DEC_CFG,
              decoder_patch_size=16, pretrained_decoder_path=DEC_PT,
              noise_tau=0.0, normalization_stat_path=STATS).to(dev).eval()

    files = sorted(f for f in os.listdir(IMGDIR) if f.lower().endswith((".jpeg", ".jpg", ".png")))[:N]
    origs = [center_crop(Image.open(os.path.join(IMGDIR, f)).convert("RGB"), 256) for f in files]
    x = torch.stack([T.ToTensor()(o) for o in origs]).to(dev)   # [N,3,256,256] in [0,1]

    with torch.no_grad():
        z_fresh = rae.encode(x).float()                          # fresh encode (normalized)
        img_fresh = rae.decode(z_fresh).clamp(0, 1)
        z_ours = torch.from_numpy(np.load(LAT, mmap_mode="r")[:N].astype(np.float32)).to(dev)
        img_ours = rae.decode(z_ours).clamp(0, 1)

    d_abs = (z_ours - z_fresh).abs()
    print(f"=== latent numerical compare (ours vs fresh encode), class {CLASS}, N={N} ===")
    print(f"fresh  z: mean={z_fresh.mean():.4f} std={z_fresh.std():.4f} min={z_fresh.min():.3f} max={z_fresh.max():.3f}")
    print(f"ours   z: mean={z_ours.mean():.4f} std={z_ours.std():.4f} min={z_ours.min():.3f} max={z_ours.max():.3f}")
    print(f"|ours-fresh|: mean={d_abs.mean():.5f}  max={d_abs.max():.4f}  rel={d_abs.mean()/z_fresh.abs().mean():.4f}")
    cos = torch.nn.functional.cosine_similarity(z_ours.flatten(1), z_fresh.flatten(1)).mean()
    print(f"cosine(ours, fresh) per-sample mean = {cos:.5f}   (1.0 = identical)")

    # grid: rows = [orig, fresh-RT, ours-RT], cols = N
    s, pad = 256, 4
    canvas = Image.new("RGB", (N * s + (N + 1) * pad, 3 * s + 4 * pad), (255, 255, 255))
    rows = [torch.stack([T.ToTensor()(o) for o in origs]).to(dev), img_fresh, img_ours]
    for r, imgs in enumerate(rows):
        arr = imgs.mul(255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        for c, im in enumerate(arr):
            canvas.paste(Image.fromarray(im), (pad + c * (s + pad), pad + r * (s + pad)))
    canvas.save(OUT)
    print(f"saved {OUT}  (row1=original, row2=fresh encode->decode, row3=OUR latent->decode)")


if __name__ == "__main__":
    main()
