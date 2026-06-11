# Summary plot: the latent-space decomposition on DiT-XL/2-scale models.
# All FID-10k vs official ADM reference stats (VIRTUAL_imagenet256_labeled.npz), cfg=1.8.
import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p, cfg_only="cfg=1.8"):
    pts = []
    try:
        for ln in open(p):
            if "FID" not in ln or (cfg_only and cfg_only not in ln):
                continue
            m = re.search(r"(\d{7})", ln)        # the 7-digit step token (0050000, ...)
            if not m:
                continue
            step = int(m.group(1))
            fid = float(ln.strip().split("=")[-1])
            pts.append((step, fid))
    except FileNotFoundError:
        pass
    return sorted(set(pts))

O = "/work/c30636/DiT/outputs"
ddpm = load(f"{O}/fid_ddpm_xl.txt")          # SD-VAE DDPM-REPA XL
sdflow = load(f"{O}/fid_flow_xl_curve.txt")  # SD-VAE flow-REPA XL
rae = load(f"{O}/fid_rae_flow.txt")          # RAE-flow (no REPA), our DiT

plt.figure(figsize=(8, 5.5))
for pts, lab, c, m in [
    (ddpm, "SD-VAE DDPM-REPA (XL)", "tab:gray", "s"),
    (sdflow, "SD-VAE flow-REPA (XL)", "tab:blue", "o"),
    (rae, "RAE-flow / DINOv3 latent (no REPA)", "tab:red", "o"),
]:
    if not pts:
        continue
    xs = [s / 1000 for s, _ in pts]; ys = [f for _, f in pts]
    plt.plot(xs, ys, m + "-", color=c, label=f"{lab}")
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}", (x, y), fontsize=7, color=c)

plt.axhline(1.44, ls="--", color="purple", alpha=0.6, label="official SiT-XL/2+REPA (1.44, 50k)")
plt.axhline(1.17, ls="--", color="green", alpha=0.6, label="official RAEv2 (1.17, 50k)")
plt.title("Latent space is the lever: SD-VAE → RAE (DiT-XL scale, official ADM stats, 10k, cfg1.8)")
plt.xlabel("training step (k)"); plt.ylabel("FID (official reference, 10k samples)")
plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
out = f"{O}/rae_flow_summary.png"
plt.tight_layout(); plt.savefig(out, dpi=120)
print("saved", out)
for nm, pts in [("DDPM-XL", ddpm), ("SD-VAE flow-XL", sdflow), ("RAE-flow(noREPA)", rae)]:
    print(nm, "->", " ".join(f"{s//1000}k:{f:.1f}" for s, f in pts))
