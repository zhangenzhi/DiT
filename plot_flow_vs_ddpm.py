# Plot the same-ruler (official ADM stats, 10k samples, cfg=1.8) FID curves:
# DDPM-REPA vs Flow-REPA on DiT-B/2.
import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path, want_gh=None):
    pts = []
    for ln in open(path):
        p = ln.strip().split("\t")
        if len(p) < 3:
            continue
        step = int(re.sub(r"\D", "", p[1]))
        fid = float(p[-1].split("=")[-1])
        cfg = next((x for x in p if x.startswith("cfg=")), "")
        if cfg != "cfg=1.8":
            continue
        pts.append((step, fid))
    return sorted(set(pts))

import argparse
from utils_config import parse_args
_p = argparse.ArgumentParser()
_p.add_argument("--outputs-dir", default="outputs",
                help="Directory holding the FID result .txt files; plots are saved here too")
O = parse_args(_p).outputs_dir

ddpm = load(f"{O}/fid_ddpm_b2.txt")
flow = load(f"{O}/fid_flow_b2_curve.txt")

plt.figure(figsize=(7, 5))
for pts, lab, c in [(ddpm, "DDPM-REPA (eps+sigma)", "tab:blue"),
                    (flow, "Flow-REPA (velocity)", "tab:red")]:
    xs = [s / 1000 for s, _ in pts]; ys = [f for _, f in pts]
    plt.plot(xs, ys, "o-", color=c, label=f"{lab}")
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}", (x, y), fontsize=8, color=c)
plt.title("DiT-B/2 REPA: Flow vs DDPM (official ADM stats, 10k, cfg=1.8)")
plt.xlabel("training step (k)"); plt.ylabel("FID (official ref, 10k samples)")
plt.grid(True, alpha=0.3); plt.legend()
out = f"{O}/flow_vs_ddpm_b2.png"
plt.tight_layout(); plt.savefig(out, dpi=120)
print("saved", out)
