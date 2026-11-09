#!/bin/bash
# One monitoring tick on the Leonardo login node: read state via squeue/sacct, submit what is due, print one STATUS line.
# State lives in $ST on Leonardo, so the Mac side holds no connection between ticks.
set -u
W=/leonardo_scratch/large/userexternal/sleung00/WORK
ST=$W/driver_state; mkdir -p "$ST"
ACC=AIFAC_P02_1031
NCH=192; MAX_DA=8; MAX_QML_TRY=3
log(){ echo "$(date +%F_%T) $*" >> "$ST/driver.log"; echo "LOG $*"; }

Q=$(squeue -u "$USER" -r -h -o "%j|%K|%T")
nq=$(printf "%s\n" "$Q" | grep -c .)
nda=$(printf "%s\n" "$Q" | grep -c -E '^da_(boost|merge)\|')
qml_queued=$(printf "%s\n" "$Q" | awk -F'|' '$1=="qml_final"{print $2}')
parts=$(ls "$W"/qkd_qml/qml/data/final_parts/*.npz 2>/dev/null | wc -l)
chunks=$(ls "$W"/qkd/results/da_dataset_lim/chunk_*_of_0${NCH}.json 2>/dev/null | wc -l)
free=$((4 - nq)); abort=""

# --- DA first (it needs several 30-min rounds) ---
merge_state=none
if [ -f "$ST/merge_id" ]; then
  merge_state=$(sacct -j "$(cat "$ST/merge_id")" -X -n -o State | head -1 | tr -d ' ')
  case "$merge_state" in FAILED*|TIMEOUT*|CANCELLED*|OUT_OF_ME*|NODE_FAIL*) abort="merge $merge_state";; esac
elif [ "$chunks" -ge "$NCH" ] && [ "$nda" -eq 0 ] && [ "$free" -gt 0 ]; then
  id=$(cd "$W/qkd" && sbatch --parsable --account=$ACC -p boost_usr_prod --qos=boost_qos_dbg --time=00:30:00 hpc/leonardo/da_merge.sbatch)
  echo "$id" > "$ST/merge_id"; log "submitted da_merge $id"; free=$((free-1)); merge_state=SUBMITTED
elif [ "$chunks" -lt "$NCH" ] && [ "$nda" -eq 0 ] && [ "$free" -gt 0 ]; then
  n=$(cat "$ST/da_tries" 2>/dev/null || echo 0)
  if [ "$n" -ge "$MAX_DA" ]; then abort="da_boost resubmitted $n times, still $chunks/$NCH"
  else
    id=$(cd "$W/qkd" && sbatch --parsable --account=$ACC --export=ALL,NCHUNKS=$NCH,PRESET=lim hpc/leonardo/da_boost.sbatch)
    echo $((n+1)) > "$ST/da_tries"; log "submitted da_boost $id (try $((n+1)), chunks $chunks/$NCH)"; free=$((free-1))
  fi
fi

# --- QML: (variant, seed) indices whose 5 parts are missing and that are not queued ---
VARS=(dense dense dense dense sparse sparse sparse sparse sparse); SEEDS=(1 2 3 4 0 1 2 3 4)
todo=()
for i in 0 1 2 3 4 5 6 7 8; do
  k=$(ls "$W"/qkd_qml/qml/data/final_parts/${VARS[$i]}__*__s${SEEDS[$i]}.npz 2>/dev/null | wc -l)
  [ "$k" -ge 5 ] && continue
  printf "%s\n" "$qml_queued" | grep -qx "$i" && continue
  t=$(cat "$ST/qml_try_$i" 2>/dev/null || echo 0)
  if [ "$t" -ge "$MAX_QML_TRY" ]; then abort="${abort:+$abort; }qml task $i failed $t times"; continue; fi
  todo+=("$i")
done
if [ "${#todo[@]}" -gt 0 ] && [ "$free" -gt 0 ]; then
  sel=("${todo[@]:0:$free}"); arr=$(IFS=,; echo "${sel[*]}")
  id=$(cd "$W/qkd_qml" && sbatch --parsable --account=$ACC --qos=boost_qos_dbg --time=00:30:00 --array="${arr}%2" qml/leonardo/final_array.sbatch)
  for i in "${sel[@]}"; do echo $(( $(cat "$ST/qml_try_$i" 2>/dev/null || echo 0) + 1 )) > "$ST/qml_try_$i"; done
  log "submitted qml_final $id array $arr"
fi

done=0; [ "$parts" -ge 50 ] && [ "$merge_state" = COMPLETED ] && done=1
echo "STATUS $(date +%T) queue=$nq parts=$parts/50 chunks=$chunks/$NCH merge=$merge_state done=$done abort=${abort:-none}"
