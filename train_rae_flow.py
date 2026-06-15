"""
Path A: train OUR plain DiT in the RAE (DINOv3-L K7) latent space with a
flow-matching velocity objective + the dimension-dependent noise-schedule shift.

- Latent: official RAE encoder (DINOv3-L, K7 multi-layer-sum) -> [B,1024,16,16],
  normalized by the official stats. Computed on-the-fly (frozen, no_grad).
- Model: our DiT (models.DiT) with input_size=16, patch_size=1, in_channels=1024,
  learn_sigma=False (outputs velocity). NO classic REPA (RAEv2's released recipe omits it).
- Transport: linear interpolant, velocity target, logit-normal(0,1) time, then the
  SD3/RAEv2 timestep shift  t <- s*t/(1+(s-1)*t),  s = sqrt(prod(latent)/4096) = 8.
- Eval/decode handled separately (sample_rae_flow.py) via the official RAE decoder.

NOTE: this is RAE-style (latent swap), NOT full RAEv2 (no DDT head / internal guidance).
"""
import os, sys, math, argparse, logging
from copy import deepcopy
from collections import OrderedDict
from glob import glob
from time import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
# Muon's Newton-Schulz (gram_newton_schulz) is torch.compile'd and recompiles per
# distinct matrix shape. The DDT/IG arch has many more 2D param shapes than dit_rope
# (1152 enc + 2048 dec + base head), exceeding the default recompile_limit of 8.
torch._dynamo.config.recompile_limit = 256
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import DiT                          # our DiT
from models_repa import DiT_REPA                # REPA variant
from models_rope import DiT_RoPE, DiT_RoPE_REPA, DiT_RoPE_DDT  # RoPE variants
from utils_config import parse_args
from rae_utils import add_rae_root_arg, load_rae


@torch.no_grad()
def update_ema(ema, model, decay=0.9999):
    ep = OrderedDict(ema.named_parameters()); mp = OrderedDict(model.named_parameters())
    for n, p in mp.items():
        ep[n].mul_(decay).add_(p.data, alpha=1 - decay)


def logger_setup(d):
    if dist.get_rank() == 0:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S',
                            handlers=[logging.StreamHandler(), logging.FileHandler(f"{d}/log.txt")])
        return logging.getLogger(__name__)
    lg = logging.getLogger(__name__); lg.addHandler(logging.NullHandler()); return lg


def center_crop(img, size):
    # DiT-style center crop to a square then resize to `size`
    while min(*img.size) >= 2 * size:
        img = img.resize(tuple(x // 2 for x in img.size), resample=4)
    scale = size / min(*img.size)
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), resample=2)
    a = np.array(img); ch = (a.shape[0] - size) // 2; cw = (a.shape[1] - size) // 2
    from PIL import Image
    return Image.fromarray(a[ch:ch + size, cw:cw + size])


class RaeLatentDataset(torch.utils.data.Dataset):
    """Loads precomputed per-class RAE latents (<dir>/<class>.npy [N,1024,16,16] fp16).
    Yields (latent_float_tensor, label). Label = sorted-class index (== ImageFolder)."""
    def __init__(self, latent_dir, flip_dir=None):
        self.classes = sorted(f[:-4] for f in os.listdir(latent_dir) if f.endswith(".npy"))
        self.arrs, self.index = [], []  # index: list of (class_i, row)
        for ci, c in enumerate(self.classes):
            a = np.load(os.path.join(latent_dir, f"{c}.npy"), mmap_mode="r")
            self.arrs.append(a)
            self.index.extend((ci, r) for r in range(a.shape[0]))
        self.flip_arrs = None
        if flip_dir:
            # latents of horizontally-flipped images (same files/order); p=0.5 swap = flip aug
            self.flip_arrs = [np.load(os.path.join(flip_dir, f"{c}.npy"), mmap_mode="r")
                              for c in self.classes]
            for ci, c in enumerate(self.classes):
                assert self.flip_arrs[ci].shape == self.arrs[ci].shape, f"flip mismatch {c}"

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ci, r = self.index[i]
        src = self.arrs
        if self.flip_arrs is not None and np.random.rand() < 0.5:
            src = self.flip_arrs
        z = torch.from_numpy(np.asarray(src[ci][r]).astype(np.float32))
        return z, ci


def main(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0)); device = local
    torch.cuda.set_device(device)
    torch.manual_seed(args.global_seed * world + rank)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        # max(existing index)+1 so deleted dirs never recycle a number (avoids clobbering kept ckpts)
        existing = glob(f"{args.results_dir}/[0-9][0-9][0-9]-*")
        idx = 1 + max([int(os.path.basename(d)[:3]) for d in existing], default=-1)
        exp = f"{args.results_dir}/{idx:03d}-DiT-XL-2-rae-flow"
        ckdir = f"{exp}/checkpoints"; os.makedirs(ckdir, exist_ok=True)
        logger = logger_setup(exp)
        logger.info(f"exp dir {exp}")
    else:
        logger = logger_setup(None)

    # --- RAE encoder only needed for ONLINE encoding; offline mode skips it ---
    rae = None
    if not args.latent_dir:
        rae = load_rae(args.rae_root, device)
        rae.eval()
        for p in rae.parameters():
            p.requires_grad_(False)

    latent_size = (1024, 16, 16)
    shift = math.sqrt(math.prod(latent_size) / 4096.0)   # = 8.0
    if rank == 0:
        logger.info(f"latent={latent_size} shift={shift:.3f} (logit-normal time, velocity)")

    # --- our DiT, adapted to the RAE latent (optionally with REPA) ---
    use_repa = args.repa_lambda > 0
    common = dict(input_size=16, patch_size=1, in_channels=1024, hidden_size=1152,
                  depth=28, num_heads=16, num_classes=args.num_classes, learn_sigma=False)
    repa_kw = dict(z_dim=1024, proj_dim=2048, align_depth=args.align_depth)
    use_ig = args.ig_base_depth > 0
    if args.arch == "dit_rope_ddt":
        model = DiT_RoPE_DDT(input_size=16, patch_size=1, in_channels=1024,
                             enc_hidden=args.enc_hidden, dec_hidden=2048, enc_depth=28, dec_depth=2,
                             enc_heads=args.enc_heads, dec_heads=16, num_classes=args.num_classes,
                             learn_sigma=False,
                             base_model_depth=args.ig_base_depth or None).to(device)
        if rank == 0:
            logger.info(f"ARCH: DiT_RoPE_DDT (two-stream: 28x{args.enc_hidden} enc + 2x2048 dec)"
                        + (f" + IG base@depth{args.ig_base_depth} coeff={args.ig_base_coeff}" if use_ig else ", no IG"))
    elif args.arch == "dit_rope":
        if use_repa:
            model = DiT_RoPE_REPA(**common, **repa_kw).to(device)
            if rank == 0:
                logger.info(f"ARCH: DiT_RoPE_REPA (RMSNorm+SwiGLU+RoPE) REPA ON target=DINOv3 latent "
                            f"align_depth={args.align_depth} lambda={args.repa_lambda}")
        else:
            model = DiT_RoPE(**common).to(device)
            if rank == 0:
                logger.info("ARCH: DiT_RoPE (RMSNorm + SwiGLU + 2D RoPE), no REPA")
    elif use_repa:
        # REPA target = the clean DINOv3-K7 latent itself (z_dim = 1024 channels).
        model = DiT_REPA(**common, **repa_kw).to(device)
        if rank == 0:
            logger.info(f"REPA ON (vanilla DiT): target=DINOv3 latent, align_depth={args.align_depth}, lambda={args.repa_lambda}")
    else:
        model = DiT(**common).to(device)
    # cuDNN emits channels_last (non-contiguous) grads for the 1x1-conv PatchEmbed
    # weights, violating DDP's grad-layout contract. Benign at 1152 but at 1440 the
    # bucket packing shifts and DDP all-reduce reads scrambled memory -> grad norm
    # explodes (~1e5) -> NaN. Force every grad contiguous before DDP reduces it
    # (hooks registered pre-wrap fire before DDP's reduction hook; no-op for the
    # already-contiguous majority, compile-safe since hooks run outside the graph).
    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(lambda g: g.contiguous())
    if os.environ.get("DISABLE_COMPILE") != "1":
        model = torch.compile(model)
    ema = deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad_(False)
    model = DDP(model, device_ids=[local])
    if rank == 0:
        logger.info(f"DiT params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    if args.latent_dir:
        ds = RaeLatentDataset(args.latent_dir, flip_dir=args.latent_flip_dir or None)
        logger.info(f"OFFLINE latents from {args.latent_dir} flip_aug={'ON ('+args.latent_flip_dir+')' if args.latent_flip_dir else 'off'}")
    else:
        tfm = transforms.Compose([
            transforms.Lambda(lambda im: center_crop(im.convert("RGB"), 256)),
            transforms.ToTensor(),  # -> [0,1]; RAE.encode handles DINOv3 normalization
        ])
        ds = ImageFolder(args.data_path, transform=tfm)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(ds, batch_size=args.global_batch_size // world, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f"dataset {len(ds):,} imgs")

    if args.optimizer == "muon":
        # RAEv2's gmuon: Muon (Newton-Schulz) for 2D params, AdamW for the rest.
        from gram_newton_schulz import Muon as GMuon

        class MuonAdamW:
            """Tiny composite mirroring RAEv2 utils.optim_utils.MuonAdamW."""
            def __init__(self, muon, adamw):
                self._muon, self._adamw = muon, adamw
                self.param_groups = muon.param_groups + adamw.param_groups
            def zero_grad(self, set_to_none=True):
                self._muon.zero_grad(set_to_none=set_to_none); self._adamw.zero_grad(set_to_none=set_to_none)
            @torch.no_grad()
            def step(self):
                self._muon.step(); self._adamw.step()
            def state_dict(self):
                return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}
            def load_state_dict(self, sd):
                self._muon.load_state_dict(sd["muon"]); self._adamw.load_state_dict(sd["adamw"])
                self.param_groups = self._muon.param_groups + self._adamw.param_groups

        plist = list(model.parameters())
        p2d = [p for p in plist if p.ndim == 2]
        prest = [p for p in plist if p.ndim != 2]
        muon = GMuon(p2d, lr=args.learning_rate, momentum=0.95, nesterov=True, weight_decay=0.0,
                     ns_coefficients_preset="POLAR_EXPRESS_COEFFICIENTS", ns_use_kernels=False,
                     adjust_lr="rms_norm")
        adamw = torch.optim.AdamW(prest, lr=args.learning_rate, weight_decay=0)
        opt = MuonAdamW(muon, adamw)
        if rank == 0:
            logger.info(f"OPT: Muon(gmuon) {len(p2d)} 2D params + AdamW {len(prest)} rest, lr={args.learning_rate}")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    update_ema(ema, model.module, decay=0)
    resume_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.module.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        resume_step = int(ck.get("step", 0))
        logger.info(f"RESUMED from {args.resume} at step {resume_step}")
    model.train()

    def sample_t(n):
        # logit-normal(0,1) then SD3/RAEv2 shift
        t = torch.sigmoid(torch.randn(n, device=device))
        return shift * t / (1 + (shift - 1) * t)

    step = resume_step; run = 0.0; run_r = 0.0; logn = 0; n_skip = 0; t0 = time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for x, y in loader:
            y = y.to(device)
            if args.latent_dir:
                x0 = x.to(device).float()                   # precomputed normalized latent
            else:
                with torch.no_grad():
                    x0 = rae.encode(x.to(device)).float()   # online-encoded normalized latent
            noise = torch.randn_like(x0)
            t = sample_t(x0.shape[0])
            te = t.view(-1, 1, 1, 1)
            xt = (1 - te) * x0 + te * noise
            v_tgt = noise - x0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if use_repa:
                    v_pred, zs = model(xt, t, y)          # zs: [B, 256, 1024] projected hidden @align_depth
                    x0_tok = x0.flatten(2).transpose(1, 2)  # [B,256,1024] clean DINOv3 latent as tokens
                    # eps=1e-2 (not 1e-6): caps the 1/||z|| gradient of F.normalize so a
                    # near-zero-norm projected token cannot produce an exploding/inf grad
                    # (root cause of the REPA-run divergence at ~135k).
                    repa = -(F.normalize(zs.float(), dim=-1, eps=1e-2)
                             * F.normalize(x0_tok.float(), dim=-1, eps=1e-2)).sum(-1).mean()
                    v_mse = F.mse_loss(v_pred.float(), v_tgt)
                    loss = v_mse + args.repa_lambda * repa
                elif use_ig:
                    # IG: model returns (full, base early-exit); supervise both on
                    # the same velocity target so the base head is a valid weak denoiser.
                    v_full, v_base = model(xt, t, y)
                    v_mse = F.mse_loss(v_full.float(), v_tgt)
                    v_mse_base = F.mse_loss(v_base.float(), v_tgt)
                    repa = v_mse_base.detach()       # reuse the repa log slot for base-loss visibility
                    loss = v_mse + args.ig_base_coeff * v_mse_base
                else:
                    v_pred = model(xt, t, y)
                    v_mse = F.mse_loss(v_pred.float(), v_tgt)
                    repa = torch.zeros((), device=device)
                    loss = v_mse
            # Full RAEv2-style schedule (from step 0): hold base_lr for lr_hold_steps,
            # then linear decay to lr_final by lr_decay_end_steps, then hold lr_final.
            if args.lr_decay_end_steps > 0:
                if step < args.lr_hold_steps:
                    cur_lr = args.learning_rate
                elif step >= args.lr_decay_end_steps:
                    cur_lr = args.lr_final
                else:
                    prog = (step - args.lr_hold_steps) / max(1, args.lr_decay_end_steps - args.lr_hold_steps)
                    cur_lr = args.learning_rate + (args.lr_final - args.learning_rate) * prog
                for g in opt.param_groups:
                    g["lr"] = cur_lr
            # End-of-training LR anneal: linear decay base_lr -> lr_final over
            # lr_decay_steps (measured from resume_step), then hold at lr_final.
            elif args.lr_decay_steps > 0:
                prog = min(1.0, max(0.0, (step - resume_step) / args.lr_decay_steps))
                cur_lr = args.learning_rate + (args.lr_final - args.learning_rate) * prog
                for g in opt.param_groups:
                    g["lr"] = cur_lr
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # NaN/inf guard: a bf16 spike can make grads inf -> clipping can't fix it.
            # Skip the optimizer/EMA update on non-finite grads so one bad step
            # cannot corrupt the weights permanently.
            if torch.isfinite(gn):
                opt.step(); update_ema(ema, model.module, decay=args.ema_decay)
            else:
                n_skip += 1
                if rank == 0:
                    logger.info(f"(step={step+1:07d}) SKIP non-finite grad (gn={gn}); total skips={n_skip}")
            run += v_mse.item(); run_r += repa.item(); logn += 1; step += 1
            if step % args.log_every == 0:
                torch.cuda.synchronize()
                sps = logn / (time() - t0)
                a = torch.tensor([run / logn, run_r / logn], device=device); dist.all_reduce(a, op=dist.ReduceOp.SUM)
                a = (a / world).tolist()
                logger.info(f"(step={step:07d}) v_mse={a[0]:.4f} repa={a[1]:.4f} gn={gn:.2f} "
                            f"lr={opt.param_groups[0]['lr']:.1e} sps={sps:.2f}")
                run = 0.0; run_r = 0.0; logn = 0; t0 = time()
            if step % args.ckpt_every == 0 and step > 0 and rank == 0:
                cp = {"model": model.module.state_dict(), "ema": ema.state_dict(),
                      "opt": opt.state_dict(), "step": step, "args": args}
                # atomic: a crash mid-write must never leave a truncated .pt behind
                torch.save(cp, f"{ckdir}/{step:07d}.pt.tmp")
                os.replace(f"{ckdir}/{step:07d}.pt.tmp", f"{ckdir}/{step:07d}.pt")
                logger.info(f"saved {ckdir}/{step:07d}.pt")
            if step % args.ckpt_every == 0:
                dist.barrier()
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    logger.info("done"); dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True, help="ImageNet train/ directory (ImageFolder layout)")
    add_rae_root_arg(p)
    p.add_argument("--latent-dir", default=None, help="If set, train on precomputed RAE latents (skip online DINOv3 encode)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=10000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lr-final", type=float, default=1e-5, help="target LR at end of anneal/decay")
    p.add_argument("--lr-decay-steps", type=int, default=0, help="linear-decay base_lr->lr_final over this many steps from resume_step (0=constant LR)")
    p.add_argument("--lr-hold-steps", type=int, default=0, help="RAEv2 schedule: hold base_lr this many steps before decaying")
    p.add_argument("--lr-decay-end-steps", type=int, default=0, help="RAEv2 schedule: step at which LR reaches lr_final (0=disabled)")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"], help="muon = RAEv2 gmuon (Muon 2D + AdamW rest)")
    p.add_argument("--ema-decay", type=float, default=0.9995, help="EMA decay (RAEv2 uses 0.9995; old runs used 0.9999)")
    p.add_argument("--latent-flip-dir", default=None, help="dir of flipped-image latents; if set, p=0.5 flip augmentation")
    p.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0=unlimited); for smoke tests")
    p.add_argument("--repa-lambda", type=float, default=0.0, help="REPA aux-loss weight (0=off). Target=clean DINOv3 latent.")
    p.add_argument("--ig-base-depth", type=int, default=0, help="Internal Guidance: encoder depth for the early-exit base head (0=off; RAEv2 uses 8). dit_rope_ddt only.")
    p.add_argument("--ig-base-coeff", type=float, default=1.0, help="weight of the IG base-head velocity loss (RAEv2: 1.0)")
    p.add_argument("--enc-hidden", type=int, default=1152, help="DDT encoder width (RAEv2 imagenet uses 1440). dit_rope_ddt only.")
    p.add_argument("--enc-heads", type=int, default=16, help="DDT encoder heads (RAEv2: 20 at width 1440, head_dim 72). dit_rope_ddt only.")
    p.add_argument("--align-depth", type=int, default=8)
    p.add_argument("--resume", default=None, help="checkpoint .pt to resume model/ema/opt/step from")
    p.add_argument("--arch", default="dit", choices=["dit", "dit_rope", "dit_rope_ddt"],
                   help="dit (vanilla) | dit_rope (RMSNorm+SwiGLU+RoPE) | dit_rope_ddt (two-stream wide decoder). REPA: dit/dit_rope only.")
    args = parse_args(p)
    main(args)
