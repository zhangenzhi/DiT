# extract_features_repa.py
#
# Like extract_features.py, but in addition to the SD-VAE latent it also
# extracts frozen DINOv2-B/14 patch features for each image, for use as the
# REPA alignment target.
#
#   <features-path>/<class>/<img>.npy       -> VAE latent  [4, 32, 32]   (fp32)
#   <dino-path>/<class>/<img>_dino.npy      -> DINOv2 feat [256, 768]    (fp16)
#
# The VAE latent is byte-compatible with the plain baseline extraction; the
# DINOv2 features live in a separate directory tree so the baseline's
# DatasetFolder loader stays clean.
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import os
import argparse
import timm


# --- 自定义 Dataset 以安全获取路径 ---
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)


def center_crop_arr(pil_image, image_size):
    """Center cropping implementation from ADM."""
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


# DINOv2 ImageNet normalization (applied to a [0,1] tensor).
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main(args):
    # Setup PyTorch Distributed
    assert torch.cuda.is_available(), "Extraction requires CUDA."

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
    if rank == 0:
        print(f"Loading VAE model from {args.vae_path}...")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()

    # Load frozen DINOv2 encoder (REPA target).
    if rank == 0:
        print(f"Loading DINOv2 encoder {args.dino_model} (img_size={args.dino_size})...")
    encoder = timm.create_model(
        args.dino_model, pretrained=True, num_classes=0, img_size=args.dino_size
    ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    dino_mean = DINO_MEAN.to(device)
    dino_std = DINO_STD.to(device)

    # Setup data: VAE wants images normalized to [-1, 1].
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),  # -> [0, 1]
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
        os.makedirs(args.dino_path, exist_ok=True)
        print(f"Total dataset size: {len(dataset)}")
        print(f"Latents  -> {args.features_path}")
        print(f"DINO feat-> {args.dino_path}")

    if world_size > 1:
        dist.barrier()

    print(f"Rank {rank}: Starting extraction...")

    with torch.no_grad():
        for i, (img01, y, paths) in enumerate(loader):
            img01 = img01.to(device)  # [B, 3, H, W] in [0, 1]

            # VAE Encode: normalize to [-1, 1].
            vae_in = img01 * 2.0 - 1.0
            latents = vae.encode(vae_in).latent_dist.sample().mul_(0.18215)
            latents = latents.detach().cpu().numpy()  # [B, 4, H/8, W/8]

            # DINOv2 Encode: resize to dino_size, ImageNet-normalize.
            dino_in = F.interpolate(
                img01, size=(args.dino_size, args.dino_size),
                mode="bicubic", align_corners=False
            ).clamp(0, 1)
            dino_in = (dino_in - dino_mean) / dino_std
            feats = encoder.forward_features(dino_in)  # [B, 1 + N, D]
            feats = feats[:, encoder.num_prefix_tokens:, :]  # [B, N, D] patch tokens
            feats = feats.detach().cpu().to(torch.float16).numpy()

            # Save
            for b in range(img01.shape[0]):
                path = paths[b]
                rel_path = os.path.relpath(path, args.data_path)
                rel_path_no_ext = os.path.splitext(rel_path)[0]

                lat_path = os.path.join(args.features_path, rel_path_no_ext + ".npy")
                os.makedirs(os.path.dirname(lat_path), exist_ok=True)
                np.save(lat_path, latents[b])

                dino_save = os.path.join(args.dino_path, rel_path_no_ext + "_dino.npy")
                os.makedirs(os.path.dirname(dino_save), exist_ok=True)
                np.save(dino_save, feats[b])

            if i % 100 == 0:
                print(f"Rank {rank}: Processed batch {i}/{len(loader)}")

    if world_size > 1:
        dist.barrier()
    print(f"Rank {rank}: Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, required=True, help="Output dir for VAE latents")
    parser.add_argument("--dino-path", type=str, required=True, help="Output dir for DINOv2 features")
    parser.add_argument("--vae-path", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--dino-model", type=str, default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--dino-size", type=int, default=224, help="DINOv2 input size (224 -> 16x16=256 tokens)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
