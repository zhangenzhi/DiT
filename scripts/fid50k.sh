#!/bin/sh
#PBS -N fid50k
#PBS -q sg
#PBS -l select=1:ngpus=1
#PBS -l walltime=12:00:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
# Single-GPU: generate 50k samples for ONE checkpoint, then compute FID-50k against
# the official ADM reference stats (VIRTUAL_imagenet256_labeled.npz).
# vars via qsub -v: CKPT(abs .pt), MODEL, REPAFLAG(""/--repa), GLOW(0.0), GHIGH(1.0),
#                   OUTFILE(abs aggregate), CFG(1.5)
__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; else . "/home/c30746/miniconda3/etc/profile.d/conda.sh"; fi
unset __conda_setup
cd /work/c30636/DiT; conda activate DiT
export HF_HOME=/work/c30636/dataset/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${REPAFLAG:=}"; : "${GLOW:=0.0}"; : "${GHIGH:=1.0}"; : "${CFG:=1.5}"
: "${OUTFILE:=/work/c30636/DiT/outputs/fid50k.txt}"
SAMPLE_DIR=/work/c30636/DiT/outputs/samples50k
mkdir -p "$SAMPLE_DIR"
echo "FID50K (1 GPU): MODEL=$MODEL CKPT=$CKPT REPAFLAG=$REPAFLAG CFG=$CFG INTERVAL=[$GLOW,$GHIGH]"

# 1. Generate 50k samples (sample_ddp.py uses DDP internally -> torchrun with 1 proc)
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  ./sample_ddp.py --model "$MODEL" $REPAFLAG --vae ema --cfg-scale "$CFG" \
  --num-fid-samples 50000 --num-sampling-steps 250 --per-proc-batch-size 64 \
  --guidance-low "$GLOW" --guidance-high "$GHIGH" \
  --sample-dir "$SAMPLE_DIR" --ckpt "$CKPT"

# 2. Reconstruct the deterministic .npz path written by sample_ddp.py
MODEL_S=$(echo "$MODEL" | tr '/' '-')
CKPT_S=$(basename "$CKPT" .pt)
NPZ="${SAMPLE_DIR}/${MODEL_S}-${CKPT_S}-size-256-vae-ema-cfg-${CFG}-seed-0.npz"
echo "Expecting samples npz: $NPZ"
ls -la "$NPZ"

# 3. Compute FID-50k vs official ADM reference stats
FID=$(python fid_from_stats.py --samples-npz "$NPZ" | awk '/^FID-/{print $NF}')
echo "RESULT: $MODEL $CKPT_S cfg=$CFG interval=[$GLOW,$GHIGH] -> FID50k=$FID"
echo "${MODEL}\t${CKPT_S}\tcfg=${CFG}\tinterval=[${GLOW},${GHIGH}]\tFID50k=${FID}" >> "$OUTFILE"
cat "$OUTFILE"
