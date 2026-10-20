#!/bin/bash
set -eu
trial_dir=/home/admin/prefill_ar_timeline_20261013
runtime_dir=/home/admin/dsv41-vllm-20260917
export PYTHONPATH=/home/admin/prefill_allreduce_20261013/deps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "GPU compute processes are already active; refusing to interfere."
    exit 1
fi
nsys --version > "$trial_dir/nsys_version.txt"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$trial_dir/gpu_before.csv"
set +e
timeout --signal=TERM --kill-after=30s 2400s nsys profile \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output="$trial_dir/trace" \
    "$runtime_dir/env/bin/python" -u "$trial_dir/trace.py" --output "$trial_dir/results"
trial_status=$?
set -e
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$trial_dir/gpu_after.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$trial_dir/compute_after.csv"
echo "TIMELINE_EXIT=$trial_status"
if [ "$trial_status" -ne 0 ]; then
    exit "$trial_status"
fi
nsys export --type=sqlite --output="$trial_dir/trace.sqlite" "$trial_dir/trace.nsys-rep"
"$runtime_dir/env/bin/python" "$runtime_dir/prefill_report_20260929/export_trace.py" \
    "$trial_dir/trace.sqlite" "$trial_dir/dependency.jsonl.gz" --all-runtime
sha256sum "$trial_dir/trace.nsys-rep" "$trial_dir/dependency.jsonl.gz" > "$trial_dir/SHA256SUMS"
tar -czf /tmp/prefill_ar_timeline_20261013.tar.gz -C /home/admin prefill_ar_timeline_20261013
echo "TIMELINE_BUNDLE_READY=/tmp/prefill_ar_timeline_20261013.tar.gz"
