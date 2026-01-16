import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import os
import argparse

# --- 自定义 Dataset 以安全获取路径 ---
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        # 返回 (image_stack, label, path)
        return original_tuple + (path,)

# --- 核心：多视图增强 Transform ---
class MultiViewTransform:
    def __init__(self, image_size, num_augs=0, min_scale=0.5):
        """
        生成 1 张 Center Crop + num_augs 张 Random Crop
        """
        self.num_augs = num_augs
        
        # 1. 确定性 Center Crop (用于验证或基准)
        # 使用 torchvision 标准实现替代原本的手写 numpy 实现，速度更快且支持 GPU 加速
        self.center_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 2. 随机增强 Crop (用于泛化)
        if num_augs > 0:
            self.aug_transform = transforms.Compose([
                # RandomResizedCrop 是 ImageNet 训练的标配
                transforms.RandomResizedCrop(image_size, scale=(min_scale, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __call__(self, img):
        # 总是包含一张 Center Crop
        crops = [self.center_transform(img)]
        
        # 添加 N 张增强图
        for _ in range(self.num_augs):
            crops.append(self.aug_transform(img))
            
        # Stack 起来: [N+1, 3, H, W]
        return torch.stack(crops)

def main(args):
    # Setup PyTorch Distributed
    assert torch.cuda.is_available(), "Extraction requires CUDA."
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    else:
        rank = 0
        world_size = 1
        gpu = 0
        torch.cuda.set_device(gpu)

    device = torch.device(f"cuda:{gpu}")

    # Load VAE model
    if rank == 0:
        print(f"Loading VAE model from {args.vae_path}...")
    
    # 开启 xformers 或 memory efficient attention 以节省显存（如果库支持）
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
        vae.enable_tiling() # 防止高分辨率下 OOM，虽然 256x256 不需要，但作为保险
    except Exception as e:
        print(f"Warning: Failed to load VAE or enable tiling: {e}")
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
        
    vae.eval()

    # Setup Transform with Augmentation
    # 这里的 image_size 应该是 VAE 输入尺寸 (如 256)
    transform = MultiViewTransform(
        image_size=args.image_size, 
        num_augs=args.num_augs,     # 新增参数
        min_scale=args.min_scale    # 新增参数
    )
    
    dataset = ImageFolderWithPaths(args.data_path, transform=transform)
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        print(f"Total dataset size (images): {len(dataset)}")
        print(f"Augmentations per image: {args.num_augs}")
        print(f"Total latents to generate: {len(dataset) * (args.num_augs + 1)}")
        print(f"Features will be saved to: {args.features_path}")

    if world_size > 1:
        dist.barrier()

    print(f"Rank {rank}: Starting extraction...")
    
    with torch.no_grad():
        # loader 返回: x shape [B, N_views, 3, H, W], y, paths
        for i, (x, y, paths) in enumerate(loader):
            # x: [B, N+1, 3, H, W]
            B, N_views, C, H, W = x.shape
            
            # Flatten batch and views to [B * (N+1), 3, H, W] for efficient VAE encoding
            x = x.view(B * N_views, C, H, W).to(device)
            
            # VAE Encode
            # 使用 sample() 注入随机性，这对于 Diffusion 训练是必须的 (KL Regularization)
            dist_moments = vae.encode(x).latent_dist
            latents = dist_moments.sample().mul_(0.18215)
            
            # 拷回 CPU 并转为 Numpy
            latents = latents.detach().cpu().numpy() # [B*N_views, 4, H/8, W/8]
            
            # Reshape back to handle saving logic: [B, N_views, 4, h, w]
            latents = latents.reshape(B, N_views, *latents.shape[1:])
            
            # 保存逻辑
            for b in range(B):
                path = paths[b]
                rel_path = os.path.relpath(path, args.data_path)
                rel_path_no_ext = os.path.splitext(rel_path)[0]
                base_save_path = os.path.join(args.features_path, rel_path_no_ext)
                
                # 确保目录存在
                os.makedirs(os.path.dirname(base_save_path), exist_ok=True)
                
                for v in range(N_views):
                    latent_data = latents[b, v] # [4, h, w]
                    
                    if v == 0:
                        # Index 0 is always Center Crop (Canonical)
                        # 保存为 "image.npy"
                        save_name = f"{base_save_path}.npy"
                    else:
                        # Index > 0 are Augmentations
                        # 保存为 "image_aug_0.npy", "image_aug_1.npy"...
                        save_name = f"{base_save_path}_aug_{v-1}.npy"
                    
                    np.save(save_name, latent_data)
                
            if i % 50 == 0 and rank == 0:
                print(f"Processed batch {i}/{len(loader)}")

    if world_size > 1:
        dist.barrier()
    print(f"Rank {rank}: Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, required=True)
    parser.add_argument("--vae-path", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--image-size", type=int, default=256)
    
    # 注意：这里的 batch-size 指的是图片的数量。
    # 实际送入 VAE 的 Batch Size 会变成 batch_size * (num_augs + 1)。
    # 如果显存不够，请调小这个值。
    parser.add_argument("--batch-size", type=int, default=1024) 
    
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    
    # 新增增强参数
    parser.add_argument("--num-augs", type=int, default=9, help="每张图片生成的额外增强副本数量 (不含原图)")
    parser.add_argument("--min-scale", type=float, default=0.6, help="RandomResizedCrop 的最小比例")
    
    args = parser.parse_args()
    main(args)