# train_feat_bf16_cp_bz.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
A minimal training script for DiT using PyTorch DDP.
Modified for 16-channel VAE latents (SD3/Flux).
"""
import torch
# Accelerate matrix multiplication
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import DatasetFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import math 
from torch.cuda.amp import autocast

from models import DiT_models
from diffusion import create_diffusion

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


def npy_loader(path):
    """
    Loader for .npy files.
    """
    return torch.from_numpy(np.load(path))


class LatentFlip(object):
    """
    Randomly flip the latent feature horizontally.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        # x shape: [C, H, W]
        if torch.rand(1) < self.p:
            return x.flip(-1)
        return x

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True) 
        experiment_index = len(glob(f"{args.results_dir}/*"))
        # 增加 vae-ch 标识到文件夹名，避免混淆
        model_string_name = args.model.replace("/", "-") 
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-ch{args.vae_channels}"
        checkpoint_dir = f"{experiment_dir}/checkpoints" 
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    # --- 关键修改：传递 in_channels 参数 ---
    # 注意：你需要确保你的 models/DiT_models.py 中的 DiT 构造函数接受 in_channels 参数
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=args.vae_channels  # <--- 这里适配 16 通道
    )
    
    model = model.to(device)
    if rank == 0:
        logger.info("Compiling model with torch.compile...")
    
    # torch.compile 可能会有些兼容性问题，如果报错可尝试注释掉
    model = torch.compile(model, mode="default")
    
    ema = deepcopy(model).to(device) 
    requires_grad(ema, False)
    
    model = DDP(model, device_ids=[rank])
    
    diffusion = create_diffusion(timestep_respacing="") 
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Model Input Channels: {args.vae_channels}")

    # Setup data:
    transform = transforms.Compose([
        LatentFlip(p=0.5)
    ])
    
    # 使用 DatasetFolder 加载 .npy
    dataset = DatasetFolder(
        root=args.features_path, 
        loader=npy_loader, 
        extensions=('.npy',), 
        transform=transform
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

    # Optimizer & Scheduler
    base_lr = 1e-4 * (args.global_batch_size / 256)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0)

    steps_per_epoch = len(dataset) // args.global_batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    logger.info(f"Base LR: {base_lr:.2e}, Total Steps: {total_steps}, Warmup Steps: {warmup_steps}")

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    if rank == 0:
        logger.info("Training with BF16 mixed precision.")
    
    # Training Loop
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            
            # 使用 BF16 进行前向传播
            with autocast(enabled=True, dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            update_ema(ema, model.module, decay=0.9995)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                
                current_lr = opt.param_groups[0]["lr"]
                logger.info(f"(Step={train_steps:07d}) Train Loss: {avg_loss:.4f}, GNorm: {grad_norm:.2f} , LR: {current_lr:.2e}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
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
    parser.add_argument("--features-path", type=str, required=True, help="Path to the directory containing .npy files")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--global-seed", type=int, default=0)
    
    # --- 新增参数 ---
    parser.add_argument("--vae-channels", type=int, default=16, help="Latent channels (4 for SD1.5/SDXL, 16 for SD3/Flux)")
    
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    args = parser.parse_args()
    main(args)
    
# torchrun --nnodes=1 --nproc_per_node=4 train_sd3feat.py --model DiT-B/2 --features-path /work/c30778/sd3_features/train