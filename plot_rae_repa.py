# RAE-flow: REPA(DINOv3 target) vs no-REPA, same ruler (official ADM stats, 10k, cfg1.8).
import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    pts = []
    try:
        for ln in open(p):
            f = ln.strip().split("\t")
            if len(f) < 2 or "FID" not in ln:
                continue
            step = int(re.sub(r"\D", "", f[0]))
            fid = float(ln.strip().split("=")[-1])
            pts.append((step, fid))
    except FileNotFoundError:
        pass
    return sorted(set(pts))

norepa = load("/work/c30636/DiT/outputs/fid_rae_flow.txt")
repa = load("/work/c30636/DiT/outputs/fid_rae_repa.txt")  # restarted REPA run (with nan-guard)

plt.figure(figsize=(7.5, 5))
for pts, lab, c in [(norepa, "RAE-flow (no REPA)", "tab:blue"),
                    (repa, "RAE-flow + REPA (DINOv3)", "tab:red")]:
    if not pts:
        continue
    xs = [s / 1000 for s, _ in pts]; ys = [f for _, f in pts]
    plt.plot(xs, ys, "o-", color=c, label=f"{lab} (n={len(pts)})")
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}", (x, y), fontsize=8, color=c)
# reference lines
plt.axhline(1.17, ls="--", color="green", alpha=0.6, label="official RAEv2 (1.17, 50k)")
plt.title("RAE latent: REPA vs no-REPA (our DiT+flow, official ADM stats, 10k, cfg1.8)")
plt.xlabel("training step (k)"); plt.ylabel("FID (official ref, 10k)")
plt.grid(True, alpha=0.3); plt.legend()
out = "/work/c30636/DiT/outputs/rae_repa_vs_norepa.png"
plt.tight_layout(); plt.savefig(out, dpi=120)
print("saved", out)
print("no-REPA:", " ".join(f"{s//1000}k:{f:.1f}" for s, f in norepa))
print("  +REPA:", " ".join(f"{s//1000}k:{f:.1f}" for s, f in repa))
