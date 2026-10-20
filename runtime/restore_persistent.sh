#!/usr/bin/env bash
set -euo pipefail
PERSIST=/mnt/fuse/dsv41-tools-20260917
BASE=/home/admin/dsv41-existing-ops-20260917
ROOT=/home/admin/dsv41-vllm-20260917
mkdir -p "$PERSIST/existing-ops" "$PERSIST/vllm" "$PERSIST/cache"
for pair in "existing-ops:$BASE" "vllm:$ROOT"; do
  source_name=${pair%%:*}
  dest=${pair#*:}
  if [ ! -e "$dest" ]; then
    ln -s "$PERSIST/$source_name" "$dest"
  fi
  if [ "$(readlink -f "$dest")" != "$PERSIST/$source_name" ]; then
    printf 'Refusing to overwrite existing directory: %s\n' "$dest" >&2
    exit 1
  fi
done
mkdir -p /home/admin/.cache
for cache_name in flashinfer vllm torch triton; do
  mkdir -p "$PERSIST/cache/$cache_name"
  if [ ! -e "/home/admin/.cache/$cache_name" ]; then
    ln -s "$PERSIST/cache/$cache_name" "/home/admin/.cache/$cache_name"
  fi
done
mkdir -p "$BASE/bootstrap" "$BASE/logs" "$ROOT/logs"
export MAMBA_ROOT_PREFIX="$BASE/mamba"
export PIP_CONFIG_FILE=/dev/null
export PIP_EXTRA_INDEX_URL=
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export UV_INDEX_URL="$PIP_INDEX_URL"
export UV_EXTRA_INDEX_URL=
export UV_NO_CONFIG=1
export UV_HTTP_TIMEOUT=120
if [ ! -x "$BASE/bootstrap/bin/micromamba" ]; then
  curl --fail --location --continue-at - --retry 2 --max-time 900 \
    https://micro.mamba.pm/api/micromamba/linux-64/latest \
    --output "$BASE/bootstrap/micromamba.tar.bz2"
  python -c 'import sys,tarfile; t=tarfile.open(sys.argv[1]); t.extract("bin/micromamba", sys.argv[2])' \
    "$BASE/bootstrap/micromamba.tar.bz2" "$BASE/bootstrap"
fi
if [ ! -x "$BASE/env/bin/python" ]; then
  "$BASE/bootstrap/bin/micromamba" create -y -p "$BASE/env" \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
    --override-channels python=3.12 pip gxx_linux-64=13 cmake ninja
fi
"$BASE/env/bin/python" -m pip install uv
if [ ! -x "$ROOT/env/bin/python" ]; then
  "$BASE/env/bin/uv" venv --python "$BASE/env/bin/python" "$ROOT/env"
fi
"$BASE/env/bin/uv" pip install --python "$ROOT/env/bin/python" \
  'https://wheels.vllm.ai/8538017f4b1d01b543345f5dbe1993cb66c80fa9/vllm-0.3.1.dev7%2Bg8538017f4-cp38-abi3-manylinux_2_28_x86_64.whl' \
  torch==2.13.0 flashinfer-python==0.6.18.post1 \
  nvidia-cuda-nvcc==13.0.88 nvidia-cuda-cccl==13.0.85 \
  nvidia-nvvm==13.0.88 nvidia-cuda-crt==13.0.88 py-spy==0.4.2
CUDA_ROOT="$ROOT/env/lib/python3.12/site-packages/nvidia/cu13"
if [ ! -e "$CUDA_ROOT/lib/libcudart.so" ]; then
  ln -s libcudart.so.13 "$CUDA_ROOT/lib/libcudart.so"
fi
if [ ! -e "$CUDA_ROOT/lib/libcuda.so" ]; then
  ln -s /usr/lib64/libcuda.so.1 "$CUDA_ROOT/lib/libcuda.so"
fi
"$ROOT/env/bin/python" "$ROOT/patch_sm120_pages.py"
LD_LIBRARY_PATH="$BASE/env/lib" "$ROOT/env/bin/python" "$ROOT/check_sm120_pages.py"
"$BASE/env/bin/uv" pip freeze --python "$ROOT/env/bin/python" > "$ROOT/pip-freeze-serving.txt"
printf '\nPERSISTENT_ENV_READY\n'
