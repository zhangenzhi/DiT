# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample from DiT and compute FID on-the-fly using torchmetrics.
Optimized for H100 with bfloat16 precision.
"""
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms, datasets
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import os
import math
from tqdm import tqdm

# 引入 torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance

def main(args):
    # --- H100 Speedups ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- DDP Setup ---
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, world_size={world_size}, device={device}.")
    else:
        rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Starting in single-process mode.")

    # --- Setup PyTorch ---
    torch.manual_seed(args.seed + rank)
    torch.set_grad_enabled(False)

    # --- Load DiT Model (Convert to BF16) ---
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    model = torch.compile(model, mode="default")
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    # H100 核心优化：转为 bfloat16
    model = model.to(dtype=torch.bfloat16) 
    
    diffusion = create_diffusion(str(args.num_sampling_steps))
    
    # --- Load VAE Model (Convert to BF16) ---
    # StabilityAI 的 VAE 在 bf16 下通常比 fp16 更稳定
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    vae = vae.to(dtype=torch.bfloat16)

    # --- Setup TorchMetrics FID ---
    # feature=2048 是标准 FID 设置 (InceptionV3 pool3 layer)
    # InceptionV3 内部通常运行在 float32，torchmetrics 会自动处理输入类型
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    
    # --- Setup Real Data Loader ---
    if args.real_data_dir is None:
        raise ValueError("Please provide --real-data-dir for FID calculation")

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.PILToTensor(), 
    ])
    
    dataset = datasets.ImageFolder(args.real_data_dir, transform=transform)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    real_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    real_iter = iter(real_loader)

    # --- Calculation Loop ---
    samples_per_gpu = args.num_samples // world_size
    batch_size = args.batch_size
    num_batches = math.ceil(samples_per_gpu / batch_size)
    
    if rank == 0:
        print(f"Generating {args.num_samples} samples (BF16) and computing FID against {args.real_data_dir}...")
        iterator = tqdm(range(num_batches), desc="FID Computing (BF16)")
    else:
        iterator = range(num_batches)

    total_generated = 0
    
    for _ in iterator:
        n = batch_size
        if total_generated + n > samples_per_gpu:
            n = samples_per_gpu - total_generated

        # A. 生成 Fake Images
        # 注意：这里初始噪声 z 保持 float32 或 bf16 均可，Autocast 会处理
        # 建议噪声保持标准分布精度，但在计算时转为 bf16
        z = torch.randn(n, 4, latent_size, latent_size, device=device, dtype=torch.bfloat16)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # CFG Setup
        z_combined = torch.cat([z, z], 0)
        y_null = torch.tensor([args.num_classes] * n, device=device)
        y_combined = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y_combined, cfg_scale=args.cfg_scale)

        # Autocast 上下文，确保所有计算在 H100 上以 BF16 进行
        with torch.autocast("cuda", dtype=torch.bfloat16):
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, 
                z_combined.shape, 
                z_combined, 
                clip_denoised=False, 
                model_kwargs=model_kwargs, 
                progress=False, 
                device=device
            )
            samples, _ = samples.chunk(2, dim=0)
            
            # VAE Decode (BF16)
            images = vae.decode(samples / 0.18215).sample

        # 后处理：转回 float32 并在 [0, 255] 范围内量化
        # FID metric 输入需要是 uint8
        images = (images / 2 + 0.5).clamp(0, 1)
        images = (images * 255).to(torch.uint8)

        # Update FID metric (Fake)
        # 此时 images 是 uint8，fid_metric 内部会将它们转为 float32 输入给 InceptionV3
        fid_metric.update(images, real=False)

        # B. 读取 Real Images
        try:
            real_batch, _ = next(real_iter)
        except StopIteration:
            real_iter = iter(real_loader)
            real_batch, _ = next(real_iter)
        
        real_batch = real_batch.to(device)
        if real_batch.shape[1] == 1:
            real_batch = real_batch.repeat(1, 3, 1, 1)
            
        fid_metric.update(real_batch, real=True)
        
        total_generated += n

    # --- Compute Final FID ---
    if rank == 0:
        print("Synchronizing and computing final FID score...")
    
    fid_score = fid_metric.compute()
    
    if rank == 0:
        print(f"\n{'='*20}")
        print(f"FID Score (BF16): {fid_score.item():.4f}")
        print(f"{'='*20}\n")
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32) # BF16 允许更大的 batch size
    parser.add_argument("--real-data-dir", type=str, required=True)

    args = parser.parse_args()
    main(args)

# torchrun --nnodes=1 --nproc_per_node=4 evaluate_fid_ddp.py --real-data-dir /work/c30778/dataset/imagenet/val --model DiT-B/2 --num-samples 10000 --ckpt ./results/039-DiT-B-2-MinSNR/checkpoints/0410000.pt