#!/bin/sh
#------ qsub option --------#
#PBS -N repa_flow_b2
#PBS -q sg
#PBS -l select=1:ngpus=4:mpiprocs=4
#PBS -l walltime=240:00:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/

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
export NCCL_DEBUG=WARN

__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"; else . "/home/c30746/miniconda3/etc/profile.d/conda.sh"; fi
unset __conda_setup

cd /work/c30636/DiT
conda activate DiT

# Flow-matching (velocity) + REPA, constant LR 1e-4. Override via -v MODEL=,GBS=
: "${MODEL:=DiT-B/2}"
: "${GBS:=512}"
echo "REPA-FLOW training: MODEL=$MODEL (flow/velocity, constant LR 1e-4, batch $GBS)"
mpirun -np 1 --map-by ppr:1:node --bind-to none \
    -x MASTER_ADDR -x MASTER_PORT -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME \
    -x NCCL_IB_HCA -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x NCCL_DEBUG -x PATH \
    torchrun --nnodes=1 --nproc_per_node=4 \
    --rdzv_id=$PBS_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ./train_feat_repa_flow.py --model "$MODEL" \
    --features-path /work/c30636/dataset/dit_feat_repa/train \
    --dino-path /work/c30636/dataset/dit_feat_repa_dino/train \
    --global-batch-size 512 --learning-rate 1e-4 --repa-lambda 0.5 --align-depth 8 \
    --ckpt-every 10000
