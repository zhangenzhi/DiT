#!/bin/sh
#PBS -N gi_sweep
#PBS -q sg
#PBS -l select=1:ngpus=4:mpiprocs=4
#PBS -l walltime=12:00:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
# Guidance-interval sweep on a SINGLE checkpoint.
# vars via qsub -v: CKPT (abs .pt), MODEL, REPAFLAG(""/--repa), OUTFILE(abs aggregate)
module load gcc ompi
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE); export MASTER_PORT=29500
export DETECTED_IFNAME=$(ip -o -4 addr show | awk '!/^[0-9]+: lo|docker|169\.254/ {print $2}' | head -n 1)
if [ -z "$DETECTED_IFNAME" ]; then export NCCL_SOCKET_IFNAME=^lo,docker0,usb0,virbr0; export GLOO_SOCKET_IFNAME=^lo,docker0,usb0,virbr0;
else export NCCL_SOCKET_IFNAME=$DETECTED_IFNAME; export GLOO_SOCKET_IFNAME=$DETECTED_IFNAME; fi
export NCCL_IB_HCA=mlx5; export NCCL_IB_DISABLE=0; export NCCL_P2P_DISABLE=0; export NCCL_DEBUG=WARN
__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; else . "/home/c30746/miniconda3/etc/profile.d/conda.sh"; fi
unset __conda_setup
cd /work/c30636/DiT; conda activate DiT
export HF_HOME=/work/c30636/dataset/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${REPAFLAG:=}"
: "${OUTFILE:=/work/c30636/DiT/outputs/gi_sweep.txt}"
echo "GI-SWEEP: MODEL=$MODEL CKPT=$CKPT REPAFLAG=$REPAFLAG OUTFILE=$OUTFILE"

# (low high) interval pairs, fractions of the 1000-step trajectory.
# 0.0 1.0 = plain cfg control; others drop guidance at the highest (and lowest) noise.
SET="0.0:1.0 0.0:0.8 0.0:0.7 0.0:0.6 0.1:0.7 0.1:0.6"

for pair in $SET; do
  LOW=${pair%%:*}; HIGH=${pair##*:}
  RESFILE="/work/c30636/DiT/outputs/gi_${LOW}_${HIGH}.txt"
  : > "$RESFILE"
  echo "=== interval [$LOW, $HIGH] ==="
  mpirun -np 1 --map-by ppr:1:node --bind-to none \
    -x HF_HOME -x HF_HUB_OFFLINE -x TRANSFORMERS_OFFLINE -x MASTER_ADDR -x MASTER_PORT \
    -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x NCCL_DEBUG \
    -x PATH \
    torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=$PBS_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ./evaluate_fid_ddp.py --model "$MODEL" $REPAFLAG --vae ema --cfg-scale 1.5 \
    --num-samples 10000 --real-data-dir /work/c30778/dataset/imagenet/val \
    --guidance-low "$LOW" --guidance-high "$HIGH" \
    --ckpt "$CKPT" --results-file "$RESFILE"
  FID=$(awk -F'\t' 'END{print $2}' "$RESFILE")
  echo "${LOW}\t${HIGH}\t${FID}" >> "$OUTFILE"
done
echo "DONE. Aggregate -> $OUTFILE"
cat "$OUTFILE"
