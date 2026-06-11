#!/bin/bash
#SBATCH -J dit-train
#SBATCH -p 4n4gpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=1          # 每节点 1 个 torchrun,由它派生 4 个 GPU 进程
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144          # 整节点 Grace 核都给 dataloader
#SBATCH -t 4-00:00:00
#SBATCH -o /work/hps0/home/ea0104/DiT/results/slurm-%j.out

set -euo pipefail

ENV_TAR=/work/hps0/home/ea0104/DiT-env.tar.gz
CODE_DIR=/work/hps0/home/ea0104/DiT
DATA_PATH=/work/hps0/home/ea0104/datasets/imagenet-1k/train

# 训练脚本与超参:提交时可覆盖,如
#   TRAIN_SCRIPT=train_feat_bf16.py EXTRA_ARGS="--model DiT-L/2" sbatch scripts/train_multinode.sbatch
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train.py}
GLOBAL_BATCH=${GLOBAL_BATCH:-256}     # 必须能被总 GPU 数(默认 16)整除
EXTRA_ARGS=${EXTRA_ARGS:-}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT=29500

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=SYS        # 跨节点走 GPUDirect RDMA

mkdir -p "$CODE_DIR/results"

# ---- 1) 每节点把 conda 环境解压到本地 NVMe(避免 16 个 rank 同时从 Lustre import)----
srun --ntasks-per-node=1 bash -c '
  SCRATCH=${USER_SCRATCH_DIR:-/tmp/'$USER'-$SLURM_JOB_ID}
  mkdir -p "$SCRATCH/env"
  tar -xf '"$ENV_TAR"' -C "$SCRATCH/env"
  source "$SCRATCH/env/bin/activate"
  conda-unpack
  echo "$(hostname): env staged to $SCRATCH/env"
'

# ---- 2) 可选:数据也搬到本地 NVMe(首次拷贝约需数分钟,长训练值得)----
# srun --ntasks-per-node=1 bash -c '
#   SCRATCH=${USER_SCRATCH_DIR:-/tmp/'$USER'-$SLURM_JOB_ID}
#   cp -r '"$DATA_PATH"' "$SCRATCH/train"
# '
# DATA_PATH='$SCRATCH/train'   # 启用时把下面 --data-path 改为本地路径

# ---- 3) 每节点启动 torchrun ----
srun --ntasks-per-node=1 bash -c '
  SCRATCH=${USER_SCRATCH_DIR:-/tmp/'$USER'-$SLURM_JOB_ID}
  source "$SCRATCH/env/bin/activate"
  cd '"$CODE_DIR"'
  exec torchrun \
    --nnodes='"$SLURM_NNODES"' \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    '"$TRAIN_SCRIPT"' \
      --data-path '"$DATA_PATH"' \
      --results-dir '"$CODE_DIR"'/results \
      --global-batch-size '"$GLOBAL_BATCH"' \
      --num-workers 16 \
      '"$EXTRA_ARGS"'
'
