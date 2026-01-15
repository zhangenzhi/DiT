# extract_features_v2.py
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import os
import argparse

# --- VAE 配置参数表 ---
# SD3 和 Flux 使用 16 通道，且 Scale/Shift 系数不同
VAE_CONSTANTS = {
    "sd1.5": {"scale": 0.18215, "shift": 0.0},
    "sdxl":  {"scale": 0.13025, "shift": 0.0}, # 注意：部分 SDXL VAE 变体可能沿用 0.18215，请根据具体模型确认
    "sd3":   {"scale": 1.5305,  "shift": 0.0609},
    "flux":  {"scale": 0.3611,  "shift": 0.0},   # Flux 通常不需要 Shift
    "custom": {"scale": 1.0,    "shift": 0.0}    # 如果不确定，通过命令行覆盖
}

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def main(args):
    # Setup PyTorch Distributed
    assert torch.cuda.is_available()
    
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

    # --- 1. 获取 VAE 参数 ---
    if args.vae_type in VAE_CONSTANTS:
        config = VAE_CONSTANTS[args.vae_type]
        scale_factor = config["scale"]
        shift_factor = config["shift"]
    else:
        # 允许手动覆盖
        scale_factor = args.scale_factor
        shift_factor = args.shift_factor
        
    if rank == 0:
        print(f"Loading VAE: {args.vae_path} (Type: {args.vae_type})")
        print(f"Using Scale Factor: {scale_factor}, Shift Factor: {shift_factor}")
        print(f"Precision: {args.precision}")

    # --- 2. 加载模型并转换精度 ---
    # 大部分 16ch VAE 在 bf16 下运行良好且更快
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae").to(device, dtype=dtype)
    except:
        # 兼容不带 subfolder 的情况
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, dtype=dtype)
        
    vae.eval()

    # Setup data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
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
        print(f"Total dataset size: {len(dataset)}")

    if world_size > 1:
        dist.barrier()

    print(f"Rank {rank}: Starting extraction...")
    
    with torch.no_grad():
        for i, (x, y, paths) in enumerate(loader):
            # 将输入图像转为对应精度
            x = x.to(device, dtype=dtype)
            
            # --- 3. 核心 Encode 逻辑 ---
            posterior = vae.encode(x).latent_dist
            
            # 离线提取通常建议使用 Mode (均值)，避免将随机噪声固化在文件中
            if args.sample:
                latents = posterior.sample()
            else:
                latents = posterior.mode()
            
            # --- 4. 应用 Shift 和 Scale ---
            # Formula: (x - shift) * scale
            if shift_factor != 0:
                latents = latents - shift_factor
            
            latents = latents * scale_factor
            
            # 转回 float32 保存，或者是 float16 节省空间（根据需求）
            # 这里保持 float32 兼容性更好
            x_npy = latents.float().cpu().numpy()
            
            for b in range(x_npy.shape[0]):
                path = paths[b]
                rel_path = os.path.relpath(path, args.data_path)
                rel_path_no_ext = os.path.splitext(rel_path)[0]
                save_path = os.path.join(args.features_path, rel_path_no_ext + ".npy")
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, x_npy[b])
                
            if i % 50 == 0:
                print(f"Rank {rank}: Processed batch {i}/{len(loader)}")

    print(f"Rank {rank}: Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, required=True)
    
    # 模型选择
    parser.add_argument("--vae-path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae-type", type=str, default="sd3", choices=["sd1.5", "sdxl", "sd3", "flux", "custom"], help="Choose VAE config for scale/shift factors")
    
    # 高级参数
    parser.add_argument("--sample", action="store_true", help="Use random sampling instead of mode (mean). Default is False (deterministic).")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16"], help="Inference precision")
    
    # 手动覆盖参数 (当 vae-type=custom 时使用)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--shift-factor", type=float, default=0.0)

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
    
# torchrun --nproc_per_node=4 extract_features_v2.py \
#   --data-path /work/c30778/dataset/imagenet \
#   --features-path /work/c30778/dataset/sd3_features \
#   --vae-path "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --vae-type sd3 \
#   --image-size 256 \
#   --precision bf16