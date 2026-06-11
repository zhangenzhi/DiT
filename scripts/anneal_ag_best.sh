#!/bin/bash
# After anneal (017) finishes: eval cfg FID-10k on settled ckpts 600-650k, pick the
# LOWEST-FID one, then run AG FID-50k + grid on that best checkpoint.
CKD=/work/c30636/DiT/results/017-DiT-XL-2-rae-flow/checkpoints
WEAK=/work/c30636/DiT/results/011-DiT-XL-2-rae-flow/checkpoints/0050000.pt
ANN=/work/c30636/DiT/outputs/fid_rae_rope_anneal.txt
AGOUT=/work/c30636/DiT/outputs/fid_rae_anneal_ag.txt
GRID=/work/c30636/DiT/outputs/grid_anneal_best_ag.png
CANDS="0600000 0610000 0620000 0630000 0640000 0650000"
cd /work/c30636/DiT
# 1) wait for training to finish (650k)
while qstat 555614 >/dev/null 2>&1; do sleep 120; done
echo "$(date +%H:%M) anneal training done"
# 2) ensure cfg FID-10k for all candidates
for s in $CANDS; do
  [ -f "$CKD/$s.pt" ] || continue
  grep -q "^$s" "$ANN" 2>/dev/null && continue
  qsub -v "CKPT=$CKD/$s.pt,ARCH=dit_rope,NSAMP=10000,CFG=1.8,OUTFILE=$ANN" scripts/fid-rae-flow.sh
  echo "submitted cfg eval $s"
done
# 3) wait until all candidates evaluated
for i in $(seq 1 120); do
  have=0
  for s in $CANDS; do grep -q "^$s" "$ANN" 2>/dev/null && have=$((have+1)); done
  [ "$have" -ge 6 ] && break
  sleep 60
done
# 4) pick lowest-FID candidate (steps 0600000-0650000)
best=$(grep -E "^06[0-5]0000" "$ANN" | awk -F'=' '{print $NF, $0}' | sort -n | head -1 | awk '{print $2}' | cut -f1)
bestfid=$(grep "^$best" "$ANN" | head -1 | awk -F= '{print $NF}')
echo "BEST anneal ckpt = $best  (cfg FID-10k=$bestfid)"
# 5) AG FID-50k + grid on the best ckpt
jf=$(qsub -v "GOOD=$CKD/$best.pt,BAD=$WEAK,W=2.0,ARCH=dit_rope,NSAMP=50000,OUTFILE=$AGOUT" scripts/fid-rae-ag.sh)
jg=$(qsub -v "CKPT=$CKD/$best.pt,ARCH=dit_rope,AG_CKPT=$WEAK,AG_SCALE=2.0,OUT=$GRID" scripts/grid.sh)
echo "AG-FID50k=$jf grid=$jg on best=$best"
# 6) wait result
for i in $(seq 1 200); do
  grep -q "FID50000" "$AGOUT" 2>/dev/null && [ -f "$GRID" ] && break
  sleep 120
done
echo "=== DONE: best=$best cfg=$bestfid ==="
cat "$AGOUT"
echo "grid=$GRID"
