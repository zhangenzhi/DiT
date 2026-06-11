#!/bin/bash
# Wait for the 650k annealed ckpt, then: AG FID-50k + AG sample grid. Report.
GOOD=/work/c30636/DiT/results/017-DiT-XL-2-rae-flow/checkpoints/0650000.pt
WEAK=/work/c30636/DiT/results/011-DiT-XL-2-rae-flow/checkpoints/0050000.pt
OUTF=/work/c30636/DiT/outputs/fid_rae_anneal_ag.txt
GRID=/work/c30636/DiT/outputs/grid_anneal650k_ag.png
# 1) wait for 0650000.pt
while [ ! -f "$GOOD" ]; do
  qstat 555614 >/dev/null 2>&1 || { [ -f "$GOOD" ] || { echo "WARN: 017 ended w/o 0650000.pt"; break; }; }
  sleep 120
done
echo "$(date +%H:%M) 0650000.pt ready"
cd /work/c30636/DiT
# 2) AG FID-50k (good=650k anneal, weak=011/50k, w=2.0) + 3) AG grid
jf=$(qsub -v "GOOD=$GOOD,BAD=$WEAK,W=2.0,ARCH=dit_rope,NSAMP=50000,OUTFILE=$OUTF" scripts/fid-rae-ag.sh)
jg=$(qsub -v "CKPT=$GOOD,ARCH=dit_rope,AG_CKPT=$WEAK,AG_SCALE=2.0,OUT=$GRID" scripts/grid.sh)
echo "submitted AG-FID50k=$jf  grid=$jg"
# 4) wait for FID-50k line + grid png
for i in $(seq 1 200); do
  grep -q "FID50000" "$OUTF" 2>/dev/null && [ -f "$GRID" ] && break
  sleep 120
done
echo "=== ANNEAL+AG RESULT ==="
cat "$OUTF" 2>/dev/null
echo "grid: $GRID  ($([ -f "$GRID" ] && echo ready || echo missing))"
