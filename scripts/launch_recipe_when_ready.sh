#!/bin/bash
# Wait for flip extraction (556182) to finish + all 1000 class files present,
# then submit the RAEv2-recipe training (16 GPU) and start its auto-eval loop.
FLIPDIR=/work/c30636/dataset/rae_dinov3l_k7_latents_flip/train
cd /work/c30636/DiT
# 1) wait for extraction job to leave the queue
while qstat 556182 >/dev/null 2>&1; do sleep 300; done
N=$(ls "$FLIPDIR"/*.npy 2>/dev/null | wc -l)
echo "$(date +%H:%M) extraction job gone; flip classes=$N"
if [ "$N" -lt 1000 ]; then
  echo "ERROR: extraction incomplete ($N/1000) — NOT submitting training"; exit 1
fi
# 2) submit training (retry on token limit every 10 min, up to 12h)
for i in $(seq 1 72); do
  JID=$(qsub scripts/rae-recipe-lg.sh 2>&1)
  case "$JID" in *sjms*) echo "submitted recipe: $JID"; break;; *) echo "$(date +%H:%M) qsub failed: $JID"; sleep 600;; esac
done
case "$JID" in *sjms*) ;; *) echo "GAVE UP submitting"; exit 1;; esac
JOB=${JID%%.*}
# 3) discover the new exp dir (contains 'OPT: Muon' in log) then auto-eval every 20k ckpt
D=""
for i in $(seq 1 120); do
  D=$(grep -l "OPT: Muon" results/0[12][0-9]-DiT-XL-2-rae-flow/log.txt 2>/dev/null | head -1)
  [ -n "$D" ] && break
  sleep 60
done
[ -z "$D" ] && { echo "no Muon run dir found"; exit 1; }
CKDIR=$(dirname "$D")/checkpoints
SUB=/work/c30636/DiT/.recipe_submitted
OUT=/work/c30636/DiT/outputs/fid_rae_recipe.txt
echo "recipe run dir: $(dirname $D)  job=$JOB"
touch "$SUB"
while true; do
  for pt in "$CKDIR"/*.pt; do
    [ -e "$pt" ] || continue
    s=$(basename "$pt" .pt); n=$((10#$s))
    [ $((n % 20000)) -eq 0 ] || continue
    grep -qx "$s" "$SUB" && continue
    qsub -v "CKPT=$pt,ARCH=dit_rope,NSAMP=10000,CFG=1.8,OUTFILE=$OUT" scripts/fid-rae-flow.sh
    echo "$(date +%H:%M) eval $s"; echo "$s" >> "$SUB"
  done
  qstat $JOB >/dev/null 2>&1 || { sleep 60;
    for pt in "$CKDIR"/*.pt; do s=$(basename "$pt" .pt); n=$((10#$s));
      [ $((n % 20000)) -eq 0 ] || continue; grep -qx "$s" "$SUB" && continue;
      qsub -v "CKPT=$pt,ARCH=dit_rope,NSAMP=10000,CFG=1.8,OUTFILE=$OUT" scripts/fid-rae-flow.sh; echo "$s" >> "$SUB"; done
    break; }
  sleep 300
done
echo "RECIPE AUTO-EVAL DONE"
