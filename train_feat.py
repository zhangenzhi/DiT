# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
Modified to train on pre-extracted features (Latents).
"""
import torch
# Optimization flags
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

# Import AMP for mixed precision
from torch.cuda.amp import autocast

from models import DiT_models
from diffusion import create_diffusion
from download import resume_from_checkpoint

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
        if name.startswith("_orig_mod."):
            name = name.replace("_orig_mod.", "")
            
        if name in ema_params:
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
#                        Feature Loading & Augmentation                         #
#################################################################################

def npy_loader(path):
    """
    Loader for .npy files.
    extract_features.py saved them as numpy arrays.
    We load them and convert to torch tensor.
    """
    return torch.from_numpy(np.load(path))

class LatentFlip(object):
    """
    Randomly flip the latent feature horizontally.
    In VAE latent space (N, C, H, W), flipping the width dimension (-1) 
    corresponds to a horizontal flip of the decoded image.
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
    Trains a new DiT model using pre-extracted features.
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
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Training on features from: {args.features_path}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    model = model.to(device)
    if rank == 0:
        logger.info("Compiling model with torch.compile...")
    model = torch.compile(model, mode="default")
    model = DDP(model, device_ids=[rank])
    
    diffusion = create_diffusion(timestep_respacing="")
    # VAE is NOT loaded here as we use pre-extracted features
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data
    # 使用 DatasetFolder 加载 .npy 文件
    # features_path 结构应该是 class_id/filename.npy
    transform = transforms.Compose([
        LatentFlip(p=0.5) # 对 Latent 进行水平翻转增强
    ])
    
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
    logger.info(f"Dataset contains {len(dataset):,} feature files ({args.features_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    start_epoch = 0

    # Resume logic
    steps_per_epoch = len(dataset) // args.global_batch_size
    if args.resume:
        start_epoch, train_steps = resume_from_checkpoint(
            args=args, 
            model=model, 
            ema=ema, 
            opt=opt, 
            device=device, 
            logger=logger,
            steps_per_epoch=steps_per_epoch
        )
    
    # --- BF16 Setup ---
    if args.bf16:
        if rank == 0:
            logger.info("Training with BF16 mixed precision.")
            if not torch.cuda.is_bf16_supported():
                logger.warning("Warning: BF16 requested but not supported by this hardware. Performance may degrade or error.")
    else:
        if rank == 0:
            logger.info("Training with FP32 (default).")

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            # x 是 pre-extracted latents: [B, 4, H/8, W/8]
            # y 是 labels: [B]
            x = x.to(device)
            y = y.to(device)
            
            # 注意: extract_features.py 中已经做了 scaling (x * 0.18215)
            # 所以这里不需要再做 vae.encode 和 mul_(0.18215)
            
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            
            # --- Training Step with BF16 Support ---
            opt.zero_grad()
            
            with autocast(enabled=args.bf16, dtype=torch.bfloat16):
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()

            loss.backward()
            opt.step()
            
            update_ema(ema, model.module)

            # Log loss values:
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
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "epoch": epoch
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
    # 改为 features-path
    parser.add_argument("--features-path", type=str, required=True, help="Path to the directory containing .npy files")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=20_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 training")
    
    args = parser.parse_args()
    main(args)

# 运行示例:
# torchrun --nnodes=1 --nproc_per_node=2 train_feat.py --model DiT-B/2 --features-path /work/c30778/dataset/dit_feat --bf16