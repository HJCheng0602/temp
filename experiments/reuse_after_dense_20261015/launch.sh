#!/bin/bash
set -eu
trial_dir=/home/admin/reuse_after_dense_20261015
runtime_dir=/home/admin/dsv41-vllm-20260917
export PYTHONPATH=/home/admin/prefill_allreduce_20261013/deps
export LD_LIBRARY_PATH=/home/admin/dsv41-existing-ops-20260917/env/lib:$runtime_dir/env/lib/python3.12/site-packages/nvidia/cu13/lib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo 'GPU compute processes already active.'
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$trial_dir/gpu_before.csv"
set +e
timeout --signal=TERM --kill-after=30s 1800s nsys profile \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output="$trial_dir/trace" \
    "$runtime_dir/env/bin/python" -u "$trial_dir/collect.py" --output "$trial_dir/results"
trial_status=$?
set -e
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$trial_dir/gpu_after.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > "$trial_dir/compute_after.csv"
echo "REUSE_CAPTURE_EXIT=$trial_status"
if [ "$trial_status" -ne 0 ]; then exit "$trial_status"; fi
nsys export --type=sqlite --output="$trial_dir/trace.sqlite" "$trial_dir/trace.nsys-rep"
sha256sum "$trial_dir/trace.sqlite" "$trial_dir/trace.nsys-rep" > "$trial_dir/SHA256SUMS"
tar -czf /tmp/reuse_after_dense_20261015.tar.gz -C /home/admin reuse_after_dense_20261015
echo 'REUSE_BUNDLE_READY=/tmp/reuse_after_dense_20261015.tar.gz'
