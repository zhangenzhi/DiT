# extract_features.py
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import os
import argparse

def main(args):
    # Setup PyTorch Distributed
    # 即使是单卡，这段代码也能兼容运行
    assert torch.cuda.is_available(), "Extraction requires CUDA."
    
    # 简单的分布式环境初始化
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
    else:
        rank = 0
        world_size = 1
        gpu = 0
        torch.cuda.set_device(gpu)

    device = torch.device(f"cuda:{gpu}")

    # Load VAE model
    # 使用 Stability AI 的官方 VAE
    print(f"Rank {rank}: Loading VAE model...")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()

    # Setup ImageNet Dataset
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # 这里的 root 需要指向包含 train/val 子文件夹的目录
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=sampler, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create output directory
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        print(f"Total dataset size: {len(dataset)}")
        print(f"Features will be saved to: {args.features_path}")

    # 等待目录创建完成
    if world_size > 1:
        dist.barrier()

    # Start Extraction
    print(f"Rank {rank}: Starting extraction...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            # VAE Encode
            # Map input images to latent space + normalize latents
            # DiT 论文使用的 scale factor 是 0.18215
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
            x = x.detach().cpu().numpy() # [B, 4, H/8, W/8]
            y = y.detach().cpu().numpy() # [B] labels

            # Save to disk
            # 我们需要构建与 ImageFolder 对应的路径结构
            # 计算当前 batch 在整个 dataset 中的真实索引
            # 注意：DistributedSampler 会对索引进行重排，这里为了保证文件名正确，
            # 我们直接利用 Loader 迭代出的数据，并根据 dataset 的结构手动创建对应的目录。
            
            # 为了简化逻辑，DiT 官方 train.py 实际上是遍历 features 目录。
            # 所以我们需要镜像 ImageNet 的目录结构 (train/n01440764/...)。
            
            # 获取当前 batch 中每个样本的文件路径
            # 这是一个 hack，利用 sampler 的索引回溯 dataset
            start_idx = i * args.batch_size
            # 注意：DistributedSampler 的逻辑比较复杂，直接获取路径最稳妥的方式是修改 Dataset
            # 但为了不修改库代码，我们这里使用简化的 "扁平化+标签" 存储，或者你可以
            # 依赖 dataset.samples (如果 sampler 没有 shuffle，这在 inference 模式下是成立的)
            
            # --- 关键修正 ---
            # 由于 DistributedSampler 可能会 shuffle (虽然我们设了 shuffle=False)，
            # 但最安全的方式是直接让 Loader 返回路径。
            # 鉴于不能修改 torchvision，我们这里假设 loader 是顺序的（DistributedSampler 默认 shuffle=False 是按 rank 切分的顺序）。
            
            # 实际上，dataset.samples[index] 可以获取路径。
            # 我们需要计算当前进程处理的 indices。
            indices = list(sampler)[i * args.batch_size : (i + 1) * args.batch_size]
            
            for b in range(x.shape[0]):
                idx = indices[b]
                path, class_idx = dataset.samples[idx]
                
                # 构建输出路径
                # 原始: /data/imagenet/train/n01440764/n01440764_10026.JPEG
                # 目标: /data/features/train/n01440764/n01440764_10026.npy
                
                rel_path = os.path.relpath(path, args.data_path) # train/n01440764/n01440764_10026.JPEG
                rel_path_no_ext = os.path.splitext(rel_path)[0]  # train/n01440764/n01440764_10026
                save_path = os.path.join(args.features_path, rel_path_no_ext + ".npy")
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, x[b]) # 保存单个 npy 文件
                
            if i % 100 == 0:
                print(f"Rank {rank}: Processed batch {i}/{len(loader)}")

    print(f"Rank {rank}: Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to ImageNet train directory")
    parser.add_argument("--features-path", type=str, required=True, help="Path to save features")
    parser.add_argument("--vae-path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()
    main(args)
    
# torchrun --nproc_per_node=2 extract_features.py --data-path /work/c30778/dataset/imagenet --features-path /work/c30778/dataset/dit_feat