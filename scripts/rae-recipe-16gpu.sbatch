#!/bin/bash
# Best recipe on the GB200 cluster: DiT_RoPE (uniform 1152) + Muon + full LR schedule
# (hold 2e-4 for 25ep -> linear decay to 2e-5 by 50ep -> hold to 80ep) +
# global batch 1024 (16 GPU x 64) + flip augmentation (precomputed flipped latents).
# 80ep @ 1024 = 100k steps. Steps/epoch = 1250.
# Submit after both extraction jobs:
#   sbatch --dependency=afterok:<norm_id>:<flip_id> scripts/rae-recipe-16gpu.sbatch
#SBATCH -J rae_recipe
#SBATCH -p 4n4gpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH -t 2-00:00:00
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
export NCCL_DEBUG=WARN
# GB200 nodes: dmabuf MR registration unsupported (mlx5dv_reg_dmabuf_mr -> error 524)
export NCCL_DMABUF_ENABLE=0
# Platform-wide NCCL collective hang (15-75 min under sustained load, any node).
# A/B matrix on a reproducing node: default + NVLS-off both hang; PROTO=Simple
# and ALGO=Ring both pass. LL/LL128 protocols are the suspect path; Simple costs
# nothing for our large (33MB) gradient buckets.
export NCCL_PROTO=Simple

CODE_DIR=/work/hps0/home/ea0104/DiT
LATDIR=/work/hps0/home/ea0104/datasets/rae_dinov3l_k7_latents/train
FLIPDIR=/work/hps0/home/ea0104/datasets/rae_dinov3l_k7_latents_flip/train
# Separate runs MUST use separate results dirs: auto-resume picks the newest
# checkpoint under $RESULTS_DIR, so sharing a dir would cross-resume runs.
#   RESULTS_DIR=results/ema9995 sbatch --export=ALL,RESULTS_DIR=results/ema9995 ...
RESULTS_DIR=${RESULTS_DIR:-results}
EMA_DECAY=${EMA_DECAY:-0.9995}
ARCH=${ARCH:-dit_rope}
# Internal Guidance: set IG_BASE_DEPTH=8 (needs ARCH=dit_rope_ddt). 0 = off.
IG_BASE_DEPTH=${IG_BASE_DEPTH:-0}
IG_BASE_COEFF=${IG_BASE_COEFF:-1.0}
MAX_STEPS=${MAX_STEPS:-100000}
LOG_EVERY=${LOG_EVERY:-100}

N_LAT=$(ls "$LATDIR"/*.npy 2>/dev/null | wc -l)
N_FLIP=$(ls "$FLIPDIR"/*.npy 2>/dev/null | wc -l)
echo "RAE-RECIPE-16GPU: Muon + LRsched(31250/62500, 2e-4->2e-5) + GBS=1024 + flip"
echo "latents: normal=$N_LAT flip=$N_FLIP"
[ "$N_LAT" -ge 1000 ] || { echo "ERROR: latents incomplete ($N_LAT/1000)"; exit 1; }
[ "$N_FLIP" -ge 1000 ] || { echo "ERROR: flipped latents incomplete ($N_FLIP/1000)"; exit 1; }

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT=29516

# Stage latents to node-local NVMe so training never random-reads Lustre.
# 2x 626G sequential copy; per-class .npy files are large so plain cp is fine.
srun --ntasks-per-node=1 bash -c "
  set -e
  echo \"\$(hostname): staging latents to \$USER_SCRATCH_DIR\"
  mkdir -p \$USER_SCRATCH_DIR/lat \$USER_SCRATCH_DIR/lat_flip
  # membind=0,1: GB200 exposes GPU HBM as NUMA nodes; without binding, page
  # cache from the 1.25TB copy spills into HBM and stalls CUDA allocations.
  numactl --membind=0,1 cp $LATDIR/*.npy \$USER_SCRATCH_DIR/lat/ &
  numactl --membind=0,1 cp $FLIPDIR/*.npy \$USER_SCRATCH_DIR/lat_flip/ &
  wait
  echo \"\$(hostname): staged \$(ls \$USER_SCRATCH_DIR/lat | wc -l)+\$(ls \$USER_SCRATCH_DIR/lat_flip | wc -l) files\"
"

# Auto-resume + in-job retry. Each attempt resumes from the newest checkpoint
# (max loss per crash = ckpt interval, ~33 min). Note retries reuse the SAME
# node allocation — they protect against transient fabric/IB flakiness, NOT
# against a persistently bad node; keep known-bad nodes out via --exclude
# (currently ar17n[12-15], see memory/cluster notes).
latest_ckpt() {
  # `|| true`: no checkpoints yet must not kill the job under set -e/pipefail
  { ls "$CODE_DIR"/$RESULTS_DIR/0*-DiT-XL-2-rae-flow/checkpoints/[0-9]*.pt 2>/dev/null || true; } \
    | awk -F/ '{print $NF, $0}' | sort | tail -1 | cut -d' ' -f2
}

for ATTEMPT in $(seq 1 99); do
  CKPT=$(latest_ckpt)
  RESUME_ARG=""
  [ -n "$CKPT" ] && { RESUME_ARG="--resume $CKPT"; echo "attempt $ATTEMPT: resuming from $CKPT"; } \
                 || echo "attempt $ATTEMPT: fresh start"
  if srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c "
    source /work/hps0/home/ea0104/miniforge3/etc/profile.d/conda.sh
    conda activate DiT
    cd $CODE_DIR
    exec numactl --membind=0,1 torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 \
      --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
      ./train_rae_flow.py --config configs/train_rae_flow.yaml \
      --latent-dir \$USER_SCRATCH_DIR/lat --latent-flip-dir \$USER_SCRATCH_DIR/lat_flip --arch $ARCH \
      --ig-base-depth $IG_BASE_DEPTH --ig-base-coeff $IG_BASE_COEFF \
      --optimizer muon --learning-rate 2e-4 --lr-final 2e-5 \
      --lr-hold-steps 31250 --lr-decay-end-steps 62500 \
      --results-dir $RESULTS_DIR --ema-decay $EMA_DECAY --log-every $LOG_EVERY \
      --global-batch-size 1024 --ckpt-every 5000 --max-steps $MAX_STEPS $RESUME_ARG
  "; then
    echo "training finished (attempt $ATTEMPT)"; exit 0
  fi
  echo "attempt $ATTEMPT crashed; retrying from latest checkpoint"
done
echo "ERROR: all attempts exhausted"; exit 1
