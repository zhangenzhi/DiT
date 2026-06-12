#!/bin/bash
# RAE (DINOv3-L K7) latent extraction on the GB200 cluster: 1 node x 4 GPU.
# Submit normal + flip variants:
#   sbatch scripts/extract-rae-latents.sbatch
#   FLIP=1 sbatch --export=ALL,FLIP=1 scripts/extract-rae-latents.sbatch
# Resumable: existing per-class .npy files are skipped.
#SBATCH -J rae_extract
#SBATCH -p 1n4gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH -t 1-00:00:00
#SBATCH -o /work/hps0/home/ea0104/DiT/outputs/slurm_logs/%x-%j.out

set -euo pipefail

source /work/hps0/home/ea0104/miniforge3/etc/profile.d/conda.sh
conda activate DiT

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAE_ROOT=/work/hps0/home/ea0104/RAEv2
export DINOV3_REPO_DIR=/work/hps0/home/ea0104/dinov3
export DINOV3_CKPT_DIR=$RAE_ROOT/pretrained_models/encoders/dinov3
export PYTHONPATH=$RAE_ROOT/src:${PYTHONPATH:-}
export OMP_NUM_THREADS=8

cd /work/hps0/home/ea0104/DiT

DATA_PATH=/work/hps0/home/ea0104/datasets/imagenet-1k/train
if [ "${FLIP:-0}" = "1" ]; then
  OUT=/work/hps0/home/ea0104/datasets/rae_dinov3l_k7_latents_flip/train
  FLIP_ARG=--flip
else
  OUT=/work/hps0/home/ea0104/datasets/rae_dinov3l_k7_latents/train
  FLIP_ARG=
fi

echo "RAE extract: flip=${FLIP:-0} out=$OUT"
torchrun --standalone --nproc_per_node=4 extract_rae_latents.py \
  --data-path "$DATA_PATH" --out "$OUT" --batch-size 64 $FLIP_ARG
echo "EXTRACT DONE"
