#!/bin/bash
# Quick single-node 4-GPU test of the 1440 DDP grad-contiguity fix. Reads latents
# straight from Lustre (no staging), 30 steps, IG + enc 1440. Pass/fail = no NaN.
#SBATCH -J ddp_fix
#SBATCH -p 1n4gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH -t 0:30:00
#SBATCH -o /work/hps0/home/ea0104/DiT/outputs/slurm_logs/%x-%j.out

set -euo pipefail
source /work/hps0/home/ea0104/miniforge3/etc/profile.d/conda.sh
conda activate DiT
export DISABLE_COMPILE=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAE_ROOT=/work/hps0/home/ea0104/RAEv2
export DINOV3_REPO_DIR=/work/hps0/home/ea0104/dinov3
export DINOV3_CKPT_DIR=$RAE_ROOT/pretrained_models/encoders/dinov3
export PYTHONPATH=$RAE_ROOT/src:${PYTHONPATH:-}
export NCCL_DMABUF_ENABLE=0 NCCL_PROTO=Simple OMP_NUM_THREADS=8
cd /work/hps0/home/ea0104/DiT

numactl --membind=0,1 torchrun --standalone --nproc_per_node=4 \
  train_rae_flow.py --config configs/train_rae_flow.yaml \
  --latent-dir /work/hps0/home/ea0104/datasets/rae_dinov3l_k7_latents/train \
  --arch dit_rope_ddt --ig-base-depth 8 --enc-hidden 1440 --enc-heads 20 \
  --optimizer muon --learning-rate 2e-4 --global-batch-size 256 \
  --results-dir /tmp/w1440_ddptest --ckpt-every 999999 --max-steps 30 --log-every 5
echo "TEST DONE"
