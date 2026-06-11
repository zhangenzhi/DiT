# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
REPA training for DiT with a FLOW-MATCHING / VELOCITY objective (SiT-style),
instead of the DDPM epsilon+learned-sigma objective.

This is `train_feat_repa.py` with exactly two changes (per the experiment plan):
  1. Generative objective: linear stochastic interpolant + velocity prediction
     (`x_t = (1-t)x + t*noise`, target `v = noise - x`, loss = MSE on v), with
     continuous t ~ U[0,1].  The model is built with `learn_sigma=False` so the
     final layer outputs `in_channels` (the velocity), not `2*in_channels`.
  2. Learning rate: CONSTANT 1e-4 (no 0.6x linear scaling, no warmup, no cosine).

Everything else is kept identical to train_feat_repa.py on purpose: bf16 +
torch.compile + DDP, global batch 512 by default, latent-space horizontal flip
p=0.5, REPA alignment (normalized negative cosine, depth 8, lambda 0.5),
gradient clip 1.0, EMA. The REPA projector / alignment code is unchanged.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from torch.cuda.amp import autocast

from models_repa import DiT_REPA_models
from download import resume_from_checkpoint
from utils_config import parse_args

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                       Paired (latent, DINOv2) Dataset                         #
#################################################################################

class LatentDinoDataset(Dataset):
    """
    Pairs each VAE latent (.npy) with its precomputed DINOv2 feature (_dino.npy).
    Identical to train_feat_repa.py (latent-space horizontal flip kept, p=0.5).
    """
    def __init__(self, features_path, dino_path, flip_p=0.5, grid=16):
        self.features_path = features_path
        self.dino_path = dino_path
        self.flip_p = flip_p
        self.grid = grid

        classes = sorted(d.name for d in os.scandir(features_path) if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.samples = []  # (latent_path, dino_path, label)
        for c in classes:
            cls_dir = os.path.join(features_path, c)
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.endswith(".npy"):
                    continue
                name = fname[:-4]
                lat_path = os.path.join(cls_dir, fname)
                dino_p = os.path.join(dino_path, c, name + "_dino.npy")
                self.samples.append((lat_path, dino_p, self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        lat_path, dino_p, label = self.samples[index]
        latent = torch.from_numpy(np.load(lat_path))               # [4, 32, 32] fp32
        dino = torch.from_numpy(np.load(dino_p)).float()           # [256, 768]

        if self.flip_p > 0 and torch.rand(1).item() < self.flip_p:
            latent = latent.flip(-1)                               # flip width
            g = self.grid
            dino = dino.view(g, g, -1).flip(1).reshape(g * g, -1)  # flip token-grid width

        return latent, dino, label


#################################################################################
#                       Flow-matching (linear interpolant)                      #
#################################################################################

def flow_velocity_target(x, noise, t):
    """
    Linear stochastic interpolant (SiT default, path_type='linear'):
        x_t = (1 - t) * x + t * noise
        v   = d/dt [x_t] = noise - x        (since d_alpha=-1, d_sigma=+1)
    t: (B,) in [0,1]. Returns (x_t, v_target).
    """
    t_exp = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t_exp) * x + t_exp * noise
    v_target = noise - x
    return x_t, v_target


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    device = local_rank
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank} (local={local_rank}), seed={seed}, world_size={world_size}.")

    # Experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-repa-flow"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model (learn_sigma=False -> final layer outputs the velocity, in_channels)
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_REPA_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=False,
        z_dim=args.z_dim,
        proj_dim=args.proj_dim,
        align_depth=args.align_depth,
    )
    model = model.to(device)
    if rank == 0:
        logger.info(f"FLOW(v) + REPA: align_depth={args.align_depth}, z_dim={args.z_dim}, "
                    f"proj_dim={args.proj_dim}, lambda={args.repa_lambda}, learn_sigma=False")
        logger.info("Compiling model with torch.compile...")
    model = torch.compile(model, mode="default")
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model, device_ids=[local_rank])
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    dataset = LatentDinoDataset(
        features_path=args.features_path,
        dino_path=args.dino_path,
        flip_p=0.5,
        grid=latent_size // 2,  # patch_size=2 -> token grid side
    )
    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False, sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} latent images ({args.features_path})")

    # --- CHANGE 2: constant LR 1e-4 (no 0.6x scaling, no warmup, no cosine) ---
    base_lr = args.learning_rate
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0)
    steps_per_epoch = len(dataset) // args.global_batch_size
    # Keep a (constant) scheduler object so resume_from_checkpoint stays happy.
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    logger.info(f"Constant LR: {base_lr:.2e}, steps/epoch: {steps_per_epoch}")

    update_ema(ema, model.module, decay=0)  # init EMA with synced weights
    model.train()
    ema.eval()
    if rank == 0:
        logger.info("Training with BF16 mixed precision (flow-matching velocity objective).")

    start_epoch = 0
    train_steps = 0
    log_steps = 0
    running_loss = running_diff = running_repa = 0

    if args.resume:
        start_epoch, train_steps = resume_from_checkpoint(
            args=args, model=model, ema=ema, opt=opt, device=device,
            logger=logger, steps_per_epoch=steps_per_epoch, scheduler=scheduler,
        )

    start_time = time()
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, z_dino, y in loader:
            x = x.to(device)
            z_dino = z_dino.to(device)          # [B, T, z_dim] target features
            y = y.to(device)
            with autocast(enabled=True, dtype=torch.bfloat16):
                # --- CHANGE 1: flow-matching velocity objective ---
                noise = torch.randn_like(x)
                t = torch.rand(x.shape[0], device=device)          # continuous t ~ U[0,1]
                x_t, v_target = flow_velocity_target(x, noise, t)
                v_pred, zs = model(x_t, t, y=y)                    # model returns (velocity, zs)
                diff_loss = F.mse_loss(v_pred.float(), v_target.float())

                # REPA alignment loss (unchanged): maximize per-token cosine similarity.
                zs_n = F.normalize(zs.float(), dim=-1, eps=1e-6)
                tgt_n = F.normalize(z_dino.float(), dim=-1, eps=1e-6)
                repa_loss = -(zs_n * tgt_n).sum(dim=-1).mean()

                loss = diff_loss + args.repa_lambda * repa_loss
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            update_ema(ema, model.module, decay=0.9995)

            running_loss += loss.item()
            running_diff += diff_loss.item()
            running_repa += repa_loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg = torch.tensor([running_loss, running_diff, running_repa], device=device) / log_steps
                dist.all_reduce(avg, op=dist.ReduceOp.SUM)
                avg = (avg / dist.get_world_size()).tolist()
                avg_loss, avg_diff, avg_repa = avg
                current_lr = opt.param_groups[0]["lr"]
                logger.info(f"(Step={train_steps:07d}) Loss: {avg_loss:.4f} (v_mse: {avg_diff:.4f}, "
                            f"repa: {avg_repa:.4f}), GNorm: {grad_norm:.2f} , LR: {current_lr:.2e}, "
                            f"Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = running_diff = running_repa = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "train_steps": train_steps,
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=str, required=True)
    parser.add_argument("--dino-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_REPA_models.keys()), default="DiT-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Constant LR (no schedule)")
    parser.add_argument("--resume", type=str, default=None)
    # --- REPA params ---
    parser.add_argument("--repa-lambda", type=float, default=0.5)
    parser.add_argument("--align-depth", type=int, default=8)
    parser.add_argument("--z-dim", type=int, default=768)
    parser.add_argument("--proj-dim", type=int, default=2048)
    args = parse_args(parser)
    main(args)
