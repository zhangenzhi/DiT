# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
REPA training script for DiT using PyTorch DDP.

Identical to train_feat_bf16_cp_bz_scale.py (the plain `compile` baseline:
bf16 + torch.compile + DDP, linear-scaled LR with warmup + cosine decay), with
one addition: a representation-alignment (REPA) auxiliary loss that aligns an
intermediate DiT hidden state with frozen DINOv2 patch features.

    total_loss = diffusion_loss + repa_lambda * (-mean cos_sim(proj(h), dino))

Set --repa-lambda 0 to recover the exact baseline (the projector is then unused
but still present). For the apples-to-apples baseline, run the original
train_feat_bf16_cp_bz_scale.py on the same latents instead.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
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
import math  # Added for cosine calculation
from torch.cuda.amp import autocast

from models_repa import DiT_REPA_models
from diffusion import create_diffusion
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
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
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

      <features_path>/<class>/<name>.npy        -> latent  [4, 32, 32]
      <dino_path>/<class>/<name>_dino.npy       -> dino    [256, 768] (fp16)

    Class -> label mapping follows torchvision's ImageFolder/DatasetFolder
    convention (sorted class directory names), so labels match a baseline run
    over the same latent tree.

    When flip=True a horizontal flip is applied jointly to the latent (width
    axis) and the DINOv2 token grid (width axis of the 16x16 grid), keeping the
    alignment target consistent with the (flipped) input.
    """
    def __init__(self, features_path, dino_path, flip_p=0.5, grid=16):
        self.features_path = features_path
        self.dino_path = dino_path
        self.flip_p = flip_p
        self.grid = grid

        classes = sorted(
            d.name for d in os.scandir(features_path) if d.is_dir()
        )
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.samples = []  # (latent_path, dino_path, label)
        for c in classes:
            cls_dir = os.path.join(features_path, c)
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.endswith(".npy"):
                    continue
                name = fname[:-4]  # strip .npy
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
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new REPA DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    device = local_rank
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank} (local={local_rank}), seed={seed}, world_size={world_size}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-repa"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_REPA_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        z_dim=args.z_dim,
        proj_dim=args.proj_dim,
        align_depth=args.align_depth,
    )
    model = model.to(device)
    if rank == 0:
        logger.info(f"REPA: align_depth={args.align_depth}, z_dim={args.z_dim}, "
                    f"proj_dim={args.proj_dim}, lambda={args.repa_lambda}")
        logger.info("Compiling model with torch.compile...")
    model = torch.compile(model, mode="default")
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model, device_ids=[local_rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data (Moved before optimizer to calculate total steps):
    dataset = LatentDinoDataset(
        features_path=args.features_path,
        dino_path=args.dino_path,
        flip_p=0.5,
        grid=latent_size // 2,  # patch_size=2 -> token grid side
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} latent images ({args.features_path})")

    # Calculate LR and Steps:
    # 1. Linear Scaling Rule: lr = base_lr * (global_batch_size / 256)
    base_lr = 0.6 * 1e-4 * (args.global_batch_size / 256)

    # 2. Setup Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0)

    # 3. Setup Scheduler (Warmup + Cosine Decay)
    steps_per_epoch = len(dataset) // args.global_batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    logger.info(f"Base LR: {base_lr:.2e}, Total Steps: {total_steps}, Warmup Steps: {warmup_steps}")

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: 0 -> 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay: 1 -> 0
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    if rank == 0:
        logger.info("Training with BF16 mixed precision.")
        if not torch.cuda.is_bf16_supported():
            logger.warning("Warning: BF16 requested but not supported by this hardware. Performance may degrade or error.")

    # Variables for monitoring/logging purposes:
    start_epoch = 0
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_diff = 0
    running_repa = 0

    # Resume logic
    if args.resume:
        start_epoch, train_steps = resume_from_checkpoint(
            args=args,
            model=model,
            ema=ema,
            opt=opt,
            device=device,
            logger=logger,
            steps_per_epoch=steps_per_epoch,
            scheduler=scheduler
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
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            with autocast(enabled=True, dtype=torch.bfloat16):
                # Run the model once; capture the REPA projection while handing
                # training_losses only the diffusion prediction tensor.
                captured = {}

                def model_fn(x_t, ts, **kw):
                    out, zs = model(x_t, ts, **kw)
                    captured["zs"] = zs
                    return out

                loss_dict = diffusion.training_losses(model_fn, x, t, model_kwargs)
                diff_loss = loss_dict["loss"].mean()

                # REPA alignment loss: maximize per-token cosine similarity.
                zs = captured["zs"]                              # [B, T, z_dim]
                # eps guards against exploding gradients from any near-zero-norm token
                zs_n = F.normalize(zs.float(), dim=-1, eps=1e-6)
                tgt_n = F.normalize(z_dino.float(), dim=-1, eps=1e-6)
                repa_loss = -(zs_n * tgt_n).sum(dim=-1).mean()

                loss = diff_loss + args.repa_lambda * repa_loss
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()  # Update LR per step
            update_ema(ema, model.module, decay=0.9995)

            # Log loss values:
            running_loss += loss.item()
            running_diff += diff_loss.item()
            running_repa += repa_loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg = torch.tensor([running_loss, running_diff, running_repa], device=device) / log_steps
                dist.all_reduce(avg, op=dist.ReduceOp.SUM)
                avg = (avg / dist.get_world_size()).tolist()
                avg_loss, avg_diff, avg_repa = avg

                # Get current LR
                current_lr = opt.param_groups[0]["lr"]

                logger.info(f"(Step={train_steps:07d}) Loss: {avg_loss:.4f} (diff: {avg_diff:.4f}, "
                            f"repa: {avg_repa:.4f}), GNorm: {grad_norm:.2f} , LR: {current_lr:.2e}, "
                            f"Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                running_diff = 0
                running_repa = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "train_steps": train_steps,
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=str, required=True, help="Path to the directory containing latent .npy files")
    parser.add_argument("--dino-path", type=str, required=True, help="Path to the directory containing *_dino.npy features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_REPA_models.keys()), default="DiT-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Number of epochs for learning rate warmup")
    parser.add_argument("--resume", type=str, default=None)
    # --- REPA params ---
    parser.add_argument("--repa-lambda", type=float, default=0.5, help="Weight of the REPA alignment loss")
    parser.add_argument("--align-depth", type=int, default=8, help="Align the hidden state after this many DiT blocks")
    parser.add_argument("--z-dim", type=int, default=768, help="Encoder feature dim (DINOv2-B=768)")
    parser.add_argument("--proj-dim", type=int, default=2048, help="Hidden dim of the REPA projector MLP")
    args = parse_args(parser)
    main(args)
