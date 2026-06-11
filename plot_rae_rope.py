# RAE-flow backbone A/B: vanilla DiT (LayerNorm+GELU+sincos) vs DiT_RoPE
# (RMSNorm+SwiGLU+2D RoPE). Same RAE DINOv3-L K7 latent, same flow+shift objective,
# same param count (677M), same ruler: FID-10k vs official ADM stats, cfg=1.8.
import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    pts = []
    try:
        for ln in open(p):
            if "FID" not in ln or "cfg=1.8" not in ln:
                continue
            m = re.search(r"(\d{7})", ln)
            if not m:
                continue
            pts.append((int(m.group(1)), float(ln.strip().split("=")[-1])))
    except FileNotFoundError:
        pass
    return sorted(set(pts))

import argparse
from utils_config import parse_args
_p = argparse.ArgumentParser()
_p.add_argument("--outputs-dir", default="outputs",
                help="Directory holding the FID result .txt files; plots are saved here too")
O = parse_args(_p).outputs_dir
base = load(f"{O}/fid_rae_flow.txt")   # vanilla DiT, no-REPA (exp 008)
rope = load(f"{O}/fid_rae_rope.txt")   # DiT_RoPE (exp 011)

plt.figure(figsize=(8, 5.5))
for pts, lab, c, m in [
    (base, "vanilla DiT (LayerNorm+GELU+sincos)", "tab:blue", "o"),
    (rope, "DiT_RoPE (RMSNorm+SwiGLU+2D RoPE)", "tab:red", "s"),
]:
    if not pts:
        continue
    xs = [s / 1000 for s, _ in pts]; ys = [f for _, f in pts]
    plt.plot(xs, ys, m + "-", color=c, label=lab)
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.2f}", (x, y), fontsize=7, color=c)

plt.axhline(7.95, ls=":", color="tab:blue", alpha=0.5)
plt.axhline(1.17, ls="--", color="green", alpha=0.5, label="official RAEv2 (1.17, 50k)")
plt.title("RAE latent, same objective/params: backbone A/B\n(official ADM stats, FID-10k, cfg1.8)")
plt.xlabel("training step (k)"); plt.ylabel("FID-10k (official reference)")
plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
out = f"{O}/rae_rope_vs_vanilla.png"
plt.tight_layout(); plt.savefig(out, dpi=120)
print("saved", out)
print("vanilla:", " ".join(f"{s//1000}k:{f:.2f}" for s, f in base))
print("  RoPE :", " ".join(f"{s//1000}k:{f:.2f}" for s, f in rope))
