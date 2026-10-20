#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/home/admin/dsv41-vllm-20260917/gemm_candidates_20260927
VLLM_ROOT=/home/admin/dsv41-vllm-20260917
OPS_ENV=/home/admin/dsv41-existing-ops-20260917/env
export CUDA_HOME="$VLLM_ROOT/env/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$VLLM_ROOT/env/bin:$CUDA_HOME/bin:$OPS_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$OPS_ENV/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export CXX="$OPS_ENV/bin/x86_64-conda-linux-gnu-c++"
export CC="$OPS_ENV/bin/x86_64-conda-linux-gnu-cc"
export NVCC_PREPEND_FLAGS="-ccbin=$CXX"
export MAX_JOBS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0 FLASHINFER_CUDA_ARCH_LIST=12.0a
export TORCH_CUDA_ARCH_LIST=12.0
cd "$TASK_ROOT"
mode=${1:?mode required}
shift
case "$mode" in
  dense-current)
    exec python compare_dense_candidates.py --output dense_current.json "$@"
    ;;
  dense-new)
    export PYTHONPATH="$TASK_ROOT/fi070:${PYTHONPATH:-}"
    exec python compare_dense_candidates.py --output dense_new.json "$@"
    ;;
  marlin)
    exec python compare_marlin_candidate.py --run --output marlin.json "$@"
    ;;
  deepgemm)
    exec python compare_deepgemm_candidate.py --run --output deepgemm.json "$@"
    ;;
  grouped)
    exec python compare_flashinfer_grouped.py --run --output grouped.json "$@"
    ;;
  bf16)
    exec python compare_bf16_candidates.py --cublaslt --layout production --output bf16.json "$@"
    ;;
  *)
    printf 'Unknown mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac
