# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained DiT models
"""
from torchvision.datasets.utils import download_url
import torch
import os


pretrained_models = {'DiT-XL-2-512x512.pt', 'DiT-XL-2-256x256.pt'}


def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        return download_model(model_name)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://dl.fbaipublicfiles.com/DiT/models/{model_name}'
        download_url(web_path, 'pretrained_models')
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model

def resume_from_checkpoint(args, model, ema, opt, device, logger, steps_per_epoch=None):
    import torch.distributed as dist
    start_epoch = 0
    train_steps = 0

    if not args.resume:
        return start_epoch, train_steps

    if dist.get_rank() == 0:
        logger.info(f"Resuming checkpoint from: {args.resume}")

    # [重要] weights_only=False 是必须的，因为你的 checkpoint 里包含 'args' (Pickle 对象)
    # 如果只保存了 state_dict，可以改为 True
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

    # 1. 加载模型权重 (处理 DDP 的 module 前缀)
    # 既然外面传进来的是 DDP(model)，我们需要访问 .module
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    # 2. 加载 EMA 和 优化器
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    
    if opt is not None and "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])

    # 3. 恢复训练进度 (Steps)
    if "train_steps" in checkpoint:
        train_steps = checkpoint["train_steps"]
    else:
        # Fallback: 尝试从文件名提取 (例如 050000.pt -> 50000)
        try:
            fname = os.path.basename(args.resume)
            train_steps = int(fname.split('.')[0])
        except ValueError:
            if dist.get_rank() == 0:
                logger.warning("Could not extract train_steps from checkpoint or filename. Starting steps from 0.")
            train_steps = 0

    # 4. 恢复训练进度 (Epoch)
    if "epoch" in checkpoint:
        # 如果保存的是 epoch 100 结束时的状态，那么下次应该从 101 开始
        start_epoch = checkpoint["epoch"] + 1
    elif steps_per_epoch is not None and steps_per_epoch > 0:
        # Fallback: 根据 steps 估算 epoch
        start_epoch = train_steps // steps_per_epoch
        if dist.get_rank() == 0:
            logger.info(f"Inferring start_epoch={start_epoch} from train_steps={train_steps}")

    if dist.get_rank() == 0:
        logger.info(f"Resumed successfully at Epoch {start_epoch}, Step {train_steps}")

    return start_epoch, train_steps

if __name__ == "__main__":
    # Download all DiT checkpoints
    for model in pretrained_models:
        download_model(model)
    print('Done.')
