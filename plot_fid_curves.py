# Plot FID-vs-step curves (REPA vs baseline) for B/2 and XL/2 from the
# per-model results files written by the fid-curve jobs. Robust to partial data.
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
from utils_config import parse_args
_p = argparse.ArgumentParser()
_p.add_argument("--outputs-dir", default="outputs",
                help="Directory holding the FID result .txt files; plots are saved here too")
O = parse_args(_p).outputs_dir
SERIES = {
    "b2_repa": ("DiT-B/2",  "REPA",     "tab:red"),
    "b2_base": ("DiT-B/2",  "baseline", "tab:blue"),
    "xl_repa": ("DiT-XL/2", "REPA",     "tab:red"),
    "xl_base": ("DiT-XL/2", "baseline", "tab:blue"),
}

def load(tag):
    p = f"{O}/fid_curve_{tag}.txt"
    pts = []
    if os.path.exists(p):
        for ln in open(p):
            parts = ln.strip().split("\t")
            if len(parts) != 2:
                continue
            m = re.search(r"(\d+)\.pt", parts[0])
            try:
                if m:
                    pts.append((int(m.group(1)), float(parts[1])))
            except ValueError:
                pass
    return sorted(set(pts))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, model in zip(axes, ["DiT-B/2", "DiT-XL/2"]):
    for tag, (m, label, color) in SERIES.items():
        if m != model:
            continue
        pts = load(tag)
        if not pts:
            continue
        xs = [s/1000 for s, _ in pts]
        ys = [f for _, f in pts]
        ax.plot(xs, ys, "o-", color=color, label=f"{label} (n={len(pts)})")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), fontsize=7, color=color)
    ax.set_title(f"{model}: FID vs training step")
    ax.set_xlabel("step (k)"); ax.set_ylabel("FID (10k samples, cfg=1.5)")
    ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout()
out = f"{O}/fid_curves.png"
plt.savefig(out, dpi=110)
print("saved", out)

# also dump a quick text table
for tag in SERIES:
    pts = load(tag)
    print(tag, "->", " ".join(f"{s//1000}k:{f:.1f}" for s, f in pts) or "(no data yet)")
