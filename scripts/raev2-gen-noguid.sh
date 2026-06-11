#!/bin/sh
#PBS -N raev2_noguid
#PBS -q sg
#PBS -l select=1:ngpus=4:mpiprocs=4
#PBS -l walltime=04:00:00
#PBS -W group_list=c30746
#PBS -j oe
#PBS -o /lustre1/work/c30636/DiT/outputs/pbs_logs/
# Generate from official RAEv2 dinov3l-k7 checkpoint, then FID vs official ADM stats.
# vars: NSAMP (default 10000)
module load gcc ompi
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE); export MASTER_PORT=29514
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
: "${NSAMP:=10000}"
cd /work/c30636/RAEv2
CFG=configs/stage2/sampling/imagenet-dinov3l-k7-noguid.yaml
OUT=/work/c30636/RAEv2/outputs/raev2_NOGUID_n${NSAMP}.npz
mkdir -p /work/c30636/RAEv2/outputs
echo "RAEv2 gen: NSAMP=$NSAMP cfg=$CFG"

mpirun -np 1 --map-by ppr:1:node --bind-to none \
  -x HF_HOME -x HF_HUB_OFFLINE -x TRANSFORMERS_OFFLINE -x DINOV3_REPO_DIR -x DINOV3_CKPT_DIR -x PYTHONPATH \
  -x MASTER_ADDR -x MASTER_PORT -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME \
  -x NCCL_IB_HCA -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x NCCL_DEBUG -x PATH \
  torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=$PBS_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  src/rae_gen.py --config "$CFG" --num-samples "$NSAMP" --batch-size 32 --out "$OUT"

echo "=== FID vs official ADM stats ==="
[ -f "$OUT" ] && python /work/c30636/DiT/fid_from_stats.py --samples-npz "$OUT" || echo "ERROR: no npz produced"
