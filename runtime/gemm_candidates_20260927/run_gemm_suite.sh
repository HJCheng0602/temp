#!/usr/bin/env bash
set -uo pipefail
task_dir=/home/admin/dsv41-vllm-20260917/gemm_candidates_20260927
cd "$task_dir" || exit 1
for mode in bf16 deepgemm marlin grouped dense-new; do
  date -u '+%FT%TZ'
  printf 'START %s\n' "$mode"
  if [ "$mode" = dense-new ]; then
    timeout 1200 bash run_gemm_candidates.sh "$mode" --batches 32 > "$mode.full.log" 2>&1
  else
    timeout 1200 bash run_gemm_candidates.sh "$mode" > "$mode.full.log" 2>&1
  fi
  printf 'EXIT %s %s\n' "$mode" "$?"
  tail -n 3 "$mode.full.log"
done
date -u '+%FT%TZ'
printf 'SUITE_DONE\n'
