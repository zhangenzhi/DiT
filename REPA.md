# REPA on DiT — REPA vs. baseline comparison

Adds **REPA** (Representation Alignment, Yu et al., ICLR 2025) to this DiT
codebase and a clean baseline to compare against. REPA adds an auxiliary loss
that aligns an intermediate DiT hidden state with frozen **DINOv2-B/14** patch
features.

Branch: `repa` (forked from `compile`, the plain baseline). The only difference
between the two runs below is the REPA loss.

## Files

| File | Purpose |
|------|---------|
| `models_repa.py` | `DiT_REPA`: DiT + 3-layer MLP projector; `forward` returns `(out, zs)` in train mode, `out` only in eval. |
| `extract_features_repa.py` | Extracts SD-VAE latents **and** DINOv2-B/14 patch features `[256, 768]` (fp16). |
| `train_feat_repa.py` | Baseline training loop (`train_feat_bf16_cp_bz_scale.py`) + REPA loss. |
| `sample_ddp.py`, `evaluate_fid_ddp.py` | Add `--repa` to load `DiT_REPA` checkpoints. |
| `scripts/extract-feat-repa.sh`, `scripts/repa-b2-n1.sh`, `scripts/baseline-b2-n1.sh` | PBS jobs. |

## How REPA integrates

`diffusion.training_losses(model, x, t, kw)` calls the model once and expects a
single tensor. We pass a closure that captures the projected hidden state `zs`
while returning only the diffusion prediction:

```python
captured = {}
def model_fn(x_t, ts, **kw):
    out, zs = model(x_t, ts, **kw); captured["zs"] = zs; return out
loss = diffusion.training_losses(model_fn, x, t, kw)["loss"].mean()
loss = loss + repa_lambda * (-cos_sim(normalize(captured["zs"]), normalize(dino)).mean())
```

Token grids match: DiT-B/2 on a 32×32 latent = 16×16 = 256 tokens; DINOv2-B/14
at 224px = 16×16 = 256 tokens. The latent-flip augmentation flips the DINOv2
token grid jointly so the alignment target stays consistent.

## Workflow

**1. Extract latents + DINOv2 features** (one-off; ~0.5 TB for full ImageNet):

```bash
qsub scripts/extract-feat-repa.sh
# -> /work/c30636/dataset/dit_feat_repa/train/<class>/<img>.npy        (latent)
# -> /work/c30636/dataset/dit_feat_repa_dino/train/<class>/<img>_dino.npy (DINOv2)
```

**2. Train both runs on the SAME latents** (REPA is the only delta):

```bash
qsub scripts/repa-b2-n1.sh       # train_feat_repa.py --repa-lambda 0.5 --align-depth 8
qsub scripts/baseline-b2-n1.sh   # train_feat_bf16_cp_bz_scale.py (no REPA)
```

**3. Evaluate FID at matched steps:**

```bash
# REPA checkpoints need --repa
torchrun --nproc_per_node=4 evaluate_fid_ddp.py --model DiT-B/2 --repa \
  --real-data-dir /work/c30778/dataset/imagenet/val \
  --ckpt-dir ./results/<repa-exp>/checkpoints

torchrun --nproc_per_node=4 evaluate_fid_ddp.py --model DiT-B/2 \
  --real-data-dir /work/c30778/dataset/imagenet/val \
  --ckpt-dir ./results/<baseline-exp>/checkpoints
```

## Key hyperparameters (REPA defaults)

- `--repa-lambda 0.5`, `--align-depth 8`, `--z-dim 768`, `--proj-dim 2048`
- Encoder: `vit_base_patch14_dinov2.lvd142m` (timm), ImageNet normalization, 224px input.

## Results — REPA vs baseline @ 50k steps (cfg=1.5, 10k samples, vae=ema, ImageNet val)

| Scale       | REPA FID | baseline FID | Δ (relative)   |
|-------------|----------|--------------|----------------|
| DiT-B/2     | 54.92    | 64.36        | −9.44 (−15%)   |
| DiT-XL/2    | 30.29    | 45.90        | −15.6 (−34%)   |

REPA's gain grows with model scale (−15% at B/2 → −34% at XL/2), consistent with
the paper's finding that representation alignment helps larger DiT/SiT more.
FID-vs-step curves (every ~50k) launched to quantify the speedup factor.

---

# RAE-flow (Path A) — EMA decay ablation

Separate line of work: our DiT trained **in the RAE (DINOv3-L K7) latent space**
with a flow-matching objective (`train_rae_flow.py`, `--arch dit_rope`, Muon,
GBS 1024, flip aug, 100k steps on 16× GB200). Not REPA — recorded here for now.

The EMA decay was hardcoded at `0.9999` (~10k-step averaging horizon). At high LR
this keeps the early-training chaos in the average, so EMA checkpoints are
**useless for early evaluation**. RAEv2 uses `0.9995` (~2k-step horizon). We added
`--ema-decay` (default `0.9995`) and ran two otherwise-identical 100k lines.

## Results — EMA 0.9999 vs 0.9995 (cfg 1.8, ImageNet-256, ADM ref stats)

| step | EMA 0.9999 (old) | EMA 0.9995 (new) | metric  |
|------|------------------|------------------|---------|
| 10k  | 308.24           | **12.24**        | FID-10k |
| 20k  | 125.04           | **8.68**         | FID-10k |
| 100k | 3.93             | **3.71**         | FID-50k |

Two findings:

1. **Early-checkpoint EMA artifact is eliminated.** At 10k the EMA model goes from
   unusable (308) to good (12); at 20k from 125 to 8.7. The short horizon lets EMA
   track the real weights once LR is still high.
2. **No cost at convergence — slightly better.** Final FID-50k 3.71 vs 3.93: the
   short horizon also averages less noise during the 2e-5 LR tail.

So `0.9995` strictly dominates: it makes mid-training FID meaningful (matters for
early-stopping / checkpoint selection) **and** lowers the final FID. Now the
default. Eval pipeline: `scripts/fid-rae-flow.sbatch` (per-job isolated sample
dir, ADM `VIRTUAL_imagenet256_labeled.npz` reference).
