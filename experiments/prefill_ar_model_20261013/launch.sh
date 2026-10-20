#!/bin/bash
set -eu
trial_dir=/home/admin/prefill_ar_model_20261013
runtime_dir=/home/admin/dsv41-vllm-20260917
export PYTHONPATH=/home/admin/prefill_allreduce_20261013/deps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "GPU compute processes are already active; refusing to interfere."
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$trial_dir/gpu_before.csv"
set +e
timeout --signal=TERM --kill-after=30s 2400s "$runtime_dir/env/bin/python" -u "$trial_dir/run.py" --output "$trial_dir/results"
trial_status=$?
set -e
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$trial_dir/gpu_after.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$trial_dir/compute_after.csv"
echo "INTEGRATION_EXIT=$trial_status"
exit "$trial_status"
