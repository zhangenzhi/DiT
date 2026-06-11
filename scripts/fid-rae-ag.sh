#!/bin/sh
#PBS -N fid_rae_ag
#PBS -q sg
#PBS -l select=1:ngpus=4:mpiprocs=4
#PBS -l walltime=04:00:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
# Autoguidance eval: good model GOOD guided by undertrained BAD with weight W.
# Sample -> RAE decode -> FID vs official ADM stats. vars: GOOD BAD W ARCH NSAMP
module load gcc ompi
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE); export MASTER_PORT=29509
export DETECTED_IFNAME=$(ip -o -4 addr show | awk '!/^[0-9]+: lo|docker|169\.254/ {print $2}' | head -n 1)
if [ -z "$DETECTED_IFNAME" ]; then export NCCL_SOCKET_IFNAME=^lo,docker0,usb0,virbr0; export GLOO_SOCKET_IFNAME=^lo,docker0,usb0,virbr0;
else export NCCL_SOCKET_IFNAME=$DETECTED_IFNAME; export GLOO_SOCKET_IFNAME=$DETECTED_IFNAME; fi
export NCCL_IB_HCA=mlx5; export NCCL_IB_DISABLE=0; export NCCL_P2P_DISABLE=0; export NCCL_DEBUG=WARN
__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; else . "/home/c30746/miniconda3/etc/profile.d/conda.sh"; fi
unset __conda_setup
conda activate DiT
export HF_HOME=/work/c30636/dataset/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DINOV3_REPO_DIR=/work/c30636/dinov3
export DINOV3_CKPT_DIR=/work/c30636/RAEv2/pretrained_models/encoders/dinov3
export PYTHONPATH=/work/c30636/RAEv2/src:$PYTHONPATH
cd /work/c30636/DiT
: "${NSAMP:=10000}"; : "${W:=2.0}"; : "${ARCH:=dit}"
: "${OUTFILE:=/work/c30636/DiT/outputs/fid_rae_ag.txt}"
G_S=$(basename "$GOOD" .pt); B_S=$(basename "$BAD" .pt)
echo "FID-RAE-AG: good=$G_S weak=$B_S W=$W NSAMP=$NSAMP"
mpirun -np 1 --map-by ppr:1:node --bind-to none \
  -x HF_HOME -x HF_HUB_OFFLINE -x TRANSFORMERS_OFFLINE -x DINOV3_REPO_DIR -x DINOV3_CKPT_DIR -x PYTHONPATH \
  -x MASTER_ADDR -x MASTER_PORT -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME \
  -x NCCL_IB_HCA -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x NCCL_DEBUG -x PATH \
  torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=$PBS_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  ./sample_rae_flow.py --ckpt "$GOOD" --ag-ckpt "$BAD" --ag-scale "$W" --arch "$ARCH" \
  --num-fid-samples "$NSAMP" --per-proc-batch-size 32 --num-steps 100
NPZ="/work/c30636/DiT/outputs/rae_flow_samples/rae-flow-${G_S}-ag${W}-w${B_S}-seed0.npz"
echo "npz: $NPZ"; [ -f "$NPZ" ] || { echo "ERROR no npz"; exit 1; }
FID=$(python fid_from_stats.py --samples-npz "$NPZ" | awk '/^FID-/{print $NF}')
echo "RESULT(rae-ag): good=$G_S weak=$B_S W=$W -> FID${NSAMP}=$FID"
echo "good=${G_S}	weak=${B_S}	W=${W}	FID${NSAMP}=${FID}" >> "$OUTFILE"
cat "$OUTFILE"
