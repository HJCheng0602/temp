#!/bin/bash
set -eu
trial_dir=/home/admin/prefill_dense_fix_20261015
runtime_dir=/home/admin/dsv41-vllm-20260917
export PYTHONPATH=/home/admin/prefill_allreduce_20261013/deps
export LD_LIBRARY_PATH=/home/admin/dsv41-existing-ops-20260917/env/lib:$runtime_dir/env/lib/python3.12/site-packages/nvidia/cu13/lib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "GPU compute processes already active."
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$trial_dir/gpu_before.csv"
set +e
timeout --signal=TERM --kill-after=30s 2400s "$runtime_dir/env/bin/python" -u "$trial_dir/run.py" --output "$trial_dir/results"
trial_status=$?
set -e
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$trial_dir/gpu_after.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > "$trial_dir/compute_after.csv"
echo "DENSE_FIX_EXIT=$trial_status"
exit "$trial_status"
