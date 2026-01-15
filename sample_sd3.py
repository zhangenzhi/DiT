# sample_16ch.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Sample new images from a pre-trained DiT (Modified for 16-channel VAEs).
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import os

# --- VAE 配置表 (必须与 extract_features_v2.py 保持一致) ---
VAE_CONSTANTS = {
    "sd1.5": {"scale": 0.18215, "shift": 0.0,    "channels": 4},
    "sdxl":  {"scale": 0.13025, "shift": 0.0,    "channels": 4},
    "sd3":   {"scale": 1.5305,  "shift": 0.0609, "channels": 16},
    "flux":  {"scale": 0.3611,  "shift": 0.0,    "channels": 16},
    "custom": {"scale": 1.0,    "shift": 0.0,    "channels": 16} # 默认值，会被命令行覆盖
}

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. 解析 VAE 参数 ---
    if args.vae_type in VAE_CONSTANTS:
        config = VAE_CONSTANTS[args.vae_type]
        scale_factor = config["scale"]
        shift_factor = config["shift"]
        vae_channels = config["channels"]
    else:
        scale_factor = args.scale_factor
        shift_factor = args.shift_factor
        vae_channels = args.vae_channels
    
    print(f"Sampling Config -> Model: {args.model}, VAE: {args.vae_type}")
    print(f"Latent Channels: {vae_channels}, Scale: {scale_factor}, Shift: {shift_factor}")

    # Load DiT model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=vae_channels  # <--- 关键：初始化正确的通道数
    ).to(device)
    
    # model = torch.compile(model, mode="default") # 调试时建议先注释掉 compile
    
    # Load Checkpoint
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
        
    state_dict = find_model(ckpt_path)
    
    # 处理 ema 权重键名不匹配问题 (可选)
    if "ema" in state_dict:
        state_dict = state_dict["ema"]
        
    model.load_state_dict(state_dict)
    model.eval()
    
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # --- 2. 加载 VAE 模型 ---
    print(f"Loading VAE from: {args.vae_path}")
    try:
        # 尝试加载带 subfolder 的 (如 SD3/SDXL)
        vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae").to(device)
    except:
        # 尝试直接加载 (如 sd-vae-ft-mse)
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    # Labels to condition the model with:
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels)
    
    # --- 3. 创建噪声 (使用 vae_channels) ---
    z = torch.randn(n, vae_channels, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    print("Sampling...")
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    # --- 4. 逆向 Shift & Scale (Inverse Transform) ---
    # 训练时: (x - shift) * scale
    # 推理时: (z / scale) + shift
    samples = samples / scale_factor
    if shift_factor != 0:
        samples = samples + shift_factor

    # Decode
    with torch.no_grad():
        # 如果显存不够，可以将 vae 转为 bf16
        # vae = vae.to(dtype=torch.bfloat16)
        # samples = samples.to(dtype=torch.bfloat16)
        samples = vae.decode(samples).sample

    # Save and display images:
    save_image(samples, "sample_16ch.png", nrow=4, normalize=True, value_range=(-1, 1))
    print("Saved sample_16ch.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to your trained checkpoint")
    
    # --- 新增 VAE 参数 ---
    parser.add_argument("--vae-path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers", help="Path to HF repo or local folder")
    parser.add_argument("--vae-type", type=str, default="sd3", choices=["sd1.5", "sdxl", "sd3", "flux", "custom"])
    
    # 手动覆盖参数 (仅当 vae-type=custom 时需要)
    parser.add_argument("--vae-channels", type=int, default=16)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--shift-factor", type=float, default=0.0)

    args = parser.parse_args()
    main(args)

# python sample_sd3.py \
#   --model DiT-B/2 \
#   --ckpt ./results/013-DiT-B-2-ch16/checkpoints/0045000.pt \
#   --vae-path "stabilityai/stable-diffusion-3-medium-diffusers" \
#   --vae-type sd3 \
#   --image-size 256