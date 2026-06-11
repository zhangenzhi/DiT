#!/bin/sh
#PBS -N fid_ddpm
#PBS -q sg
#PBS -l select=1:ngpus=4:mpiprocs=4
#PBS -l walltime=04:00:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
# Generate from a DDPM DiT-REPA checkpoint and FID vs official ADM stats,
# so it is on the SAME ruler as the flow eval (fid_from_stats / official VIRTUAL npz).
# vars: CKPT(abs .pt), MODEL(DiT-B/2), NSAMP(10000), CFG(1.5), OUTFILE(abs)
module load gcc ompi
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE); export MASTER_PORT=29502
export DETECTED_IFNAME=$(ip -o -4 addr show | awk '!/^[0-9]+: lo|docker|169\.254/ {print $2}' | head -n 1)
if [ -z "$DETECTED_IFNAME" ]; then export NCCL_SOCKET_IFNAME=^lo,docker0,usb0,virbr0; export GLOO_SOCKET_IFNAME=^lo,docker0,usb0,virbr0;
else export NCCL_SOCKET_IFNAME=$DETECTED_IFNAME; export GLOO_SOCKET_IFNAME=$DETECTED_IFNAME; fi
export NCCL_IB_HCA=mlx5; export NCCL_IB_DISABLE=0; export NCCL_P2P_DISABLE=0; export NCCL_DEBUG=WARN
__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; else . "/home/c30746/miniconda3/etc/profile.d/conda.sh"; fi
unset __conda_setup
cd /work/c30636/DiT; conda activate DiT
export HF_HOME=/work/c30636/dataset/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${MODEL:=DiT-B/2}"; : "${NSAMP:=10000}"; : "${CFG:=1.5}"
: "${OUTFILE:=/work/c30636/DiT/outputs/fid_ddpm_b2.txt}"
CKPT_S=$(basename "$CKPT" .pt)
echo "FID-DDPM: MODEL=$MODEL CKPT=$CKPT_S NSAMP=$NSAMP CFG=$CFG (DDPM 250 steps, official stats)"

mpirun -np 1 --map-by ppr:1:node --bind-to none \
  -x HF_HOME -x HF_HUB_OFFLINE -x TRANSFORMERS_OFFLINE -x MASTER_ADDR -x MASTER_PORT \
  -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x NCCL_DEBUG -x PATH \
  torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=$PBS_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  ./sample_ddp.py --model "$MODEL" --repa --ckpt "$CKPT" \
  --vae ema --cfg-scale "$CFG" --num-sampling-steps 250 \
  --per-proc-batch-size 64 --num-fid-samples "$NSAMP" \
  --sample-dir /work/c30636/DiT/outputs/ddpm_samples

MODEL_S=$(echo "$MODEL" | tr '/' '-')
NPZ="/work/c30636/DiT/outputs/ddpm_samples/${MODEL_S}-${CKPT_S}-size-256-vae-ema-cfg-${CFG}-seed-0.npz"
echo "npz: $NPZ"; ls -la "$NPZ"
FID=$(python fid_from_stats.py --samples-npz "$NPZ" | awk '/^FID-/{print $NF}')
echo "RESULT(ddpm): $MODEL $CKPT_S cfg=$CFG -> FID${NSAMP}=$FID"
echo "${MODEL}	${CKPT_S}	cfg=${CFG}	FID${NSAMP}=${FID}" >> "$OUTFILE"
cat "$OUTFILE"
