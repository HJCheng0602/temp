#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/admin/dsv41-vllm-20260917
BASE=/home/admin/dsv41-existing-ops-20260917
OUTPUT="$ROOT/largebatch_20260924"
if [ -e "$OUTPUT" ]; then
  printf 'Refusing to overwrite run directory: %s\n' "$OUTPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT/reproduction"
cp "$ROOT/largebatch_pp2.py" "$ROOT/parallel_matrix.py" \
  "$ROOT/patch_sm120_pages.py" "$ROOT/check_sm120_pages.py" \
  "$ROOT/summarize_largebatch.py" "$ROOT/run_largebatch_20260924.sh" \
  "$ROOT/pip-freeze-serving.txt" "$OUTPUT/reproduction/"
nvidia-smi -q -x > "$OUTPUT/reproduction/gpu.xml"
nvidia-smi topo -m > "$OUTPUT/reproduction/topology.txt"
lscpu -J > "$OUTPUT/reproduction/cpu.json"
uname -a > "$OUTPUT/reproduction/kernel.txt"
cp /mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash/config.json \
  "$OUTPUT/reproduction/model-config.json"
export LD_LIBRARY_PATH="$BASE/env/lib:${LD_LIBRARY_PATH:-}"
run_status=0
"$ROOT/env/bin/python" -u "$ROOT/largebatch_pp2.py" \
  --contexts 2048,8192,32768 --batches 16,32,64,128,256 \
  --capacity-safety-ratio 0.98 --output-root "$OUTPUT" || run_status=$?
"$ROOT/env/bin/python" "$ROOT/summarize_largebatch.py" \
  --root "$OUTPUT" --output "$OUTPUT/SUMMARY.json"
printf '%s\n' "$run_status" > "$OUTPUT/driver-exit-code.txt"
tar -czf /mnt/fuse/dsv41-largebatch-20260924.tar.gz -C "$ROOT" largebatch_20260924
sha256sum /mnt/fuse/dsv41-largebatch-20260924.tar.gz
exit "$run_status"
