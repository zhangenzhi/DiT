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
