#!/bin/bash
# FID-N for a Path-A RAE-flow checkpoint on the GB200 cluster (4 GPU).
# Samples via flow ODE + official RAE decoder, FID vs official ADM stats.
#   CKPT=<abs .pt> [NSAMP=10000] [CFG=1.8] [ARCH=dit_rope] sbatch --export=ALL,CKPT=... scripts/fid-rae-flow.sbatch
#SBATCH -J fid_rae
#SBATCH -p 1n4gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH -t 4:00:00
#SBATCH -o /work/hps0/home/ea0104/DiT/outputs/slurm_logs/%x-%j.out

set -euo pipefail

source /work/hps0/home/ea0104/miniforge3/etc/profile.d/conda.sh
conda activate DiT

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAE_ROOT=/work/hps0/home/ea0104/RAEv2
export DINOV3_REPO_DIR=/work/hps0/home/ea0104/dinov3
export DINOV3_CKPT_DIR=$RAE_ROOT/pretrained_models/encoders/dinov3
export PYTHONPATH=$RAE_ROOT/src:${PYTHONPATH:-}
export NCCL_DMABUF_ENABLE=0
export NCCL_PROTO=Simple

cd /work/hps0/home/ea0104/DiT
: "${CKPT:?set CKPT=<abs path .pt>}"
: "${NSAMP:=10000}"; : "${CFG:=1.8}"; : "${ARCH:=dit_rope}"
: "${OUTFILE:=/work/hps0/home/ea0104/DiT/outputs/fid_rae_flow.txt}"
REF=/work/hps0/home/ea0104/datasets/fid_refs/VIRTUAL_imagenet256_labeled.npz
[ -f "$REF" ] || { echo "ERROR: ADM ref npz missing: $REF"; exit 1; }

CKPT_S=$(basename "$CKPT" .pt)
# tag the run with its lineage (results dir) so main/ema9995 rows are distinguishable
LINEAGE=$(echo "$CKPT" | grep -oP 'results/\K[^/]+(?=/|$)' | head -1)
# Internal Guidance eval: set IG_BASE_DEPTH=8 (needs ARCH=dit_rope_ddt). 0 = use CFG.
: "${IG_BASE_DEPTH:=0}"; : "${IG_SCALE:=2.0}"
: "${ENC_HIDDEN:=1152}"; : "${ENC_HEADS:=16}"
SDIR="outputs/rae_flow_samples/${LINEAGE}-${SLURM_JOB_ID}"

if [ "$IG_BASE_DEPTH" -gt 0 ]; then
  GUID="ig${IG_SCALE}-d${IG_BASE_DEPTH}"; GUIDFLAGS="--ig-base-depth $IG_BASE_DEPTH --ig-scale $IG_SCALE"
else
  GUID="cfg${CFG}"; GUIDFLAGS="--cfg-scale $CFG"
fi
echo "FID-RAE-FLOW: CKPT=$CKPT_S LINEAGE=$LINEAGE NSAMP=$NSAMP GUID=$GUID ARCH=$ARCH enc=$ENC_HIDDEN"

numactl --membind=0,1 torchrun --standalone --nproc_per_node=4 \
  ./sample_rae_flow.py --ckpt "$CKPT" --arch "$ARCH" --num-fid-samples "$NSAMP" \
  --enc-hidden $ENC_HIDDEN --enc-heads $ENC_HEADS \
  $GUIDFLAGS --per-proc-batch-size 32 --num-steps 100 --sample-dir "$SDIR"

NPZ="${SDIR}/rae-flow-${CKPT_S}-${GUID}-seed0.npz"
[ -f "$NPZ" ] || { echo "ERROR no npz: $NPZ"; exit 1; }
FID=$(python fid_from_stats.py --samples-npz "$NPZ" --ref-npz "$REF" | awk '/^FID-/{print $NF}')
echo "RESULT(rae-flow): $LINEAGE/$CKPT_S $GUID -> FID${NSAMP}=$FID"
echo -e "${LINEAGE}/${CKPT_S}\t${GUID}\tFID${NSAMP}=${FID}" >> "$OUTFILE"
sort "$OUTFILE"
