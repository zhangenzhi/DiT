#!/bin/sh
#------ qsub option --------#
#PBS -N fid_eval
#PBS -q lg
#PBS -l select=1:ngpus=4:mpiprocs=4
#PBS -l walltime=06:00:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
#
# Parameterized FID eval. Pass via:  qsub -v CKPT=/path/0050000.pt,REPAFLAG=--repa,MODEL=DiT-XL/2 scripts/fid-eval.sh
#   CKPT      : checkpoint .pt to evaluate (required)
#   REPAFLAG  : "--repa" for REPA checkpoints, "" for baseline (default "")
#   MODEL     : DiT model (default DiT-B/2)

module load gcc ompi
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500
export DETECTED_IFNAME=$(ip -o -4 addr show | awk '!/^[0-9]+: lo|docker|169\.254/ {print $2}' | head -n 1)
if [ -z "$DETECTED_IFNAME" ]; then
    export GLOO_SOCKET_IFNAME=^lo,docker0,usb0,virbr0
    export NCCL_SOCKET_IFNAME=^lo,docker0,usb0,virbr0
else
    export GLOO_SOCKET_IFNAME=$DETECTED_IFNAME
    export NCCL_SOCKET_IFNAME=$DETECTED_IFNAME
fi
export NCCL_IB_HCA=mlx5
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=INFO

__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup";
else . "/home/c30746/miniconda3/etc/profile.d/conda.sh"; fi
unset __conda_setup

cd /work/c30636/DiT
conda activate DiT

# VAE 与 Inception 权重已缓存, 强制离线
export HF_HOME=/work/c30636/dataset/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

: "${REPAFLAG:=}"
: "${MODEL:=DiT-B/2}"
echo "FID eval: MODEL=$MODEL CKPT=$CKPT REPAFLAG=$REPAFLAG"

mpirun -np 1 \
    --map-by ppr:1:node \
    --bind-to none \
    -x HF_HOME -x HF_HUB_OFFLINE -x TRANSFORMERS_OFFLINE \
    -x MASTER_ADDR -x MASTER_PORT \
    -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME \
    -x NCCL_IB_HCA -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x NCCL_DEBUG \
    -x PATH -x CKPT -x REPAFLAG -x MODEL \
    torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$PBS_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ./evaluate_fid_ddp.py --model "$MODEL" $REPAFLAG \
    --vae ema --cfg-scale 1.5 --num-samples 10000 \
    --real-data-dir /work/c30778/dataset/imagenet/val \
    --ckpt "$CKPT"
