# PP2 TP4 EP4 Fast GEMM Trial

This is an opt-in experiment for the installed DeepSeek V4.1 Flash runtime on
eight RTX PRO 5000 72 GB GPUs. Installed vLLM files and checkpoint files are not
modified. The existing deferred PP receive fix is checked against its manifest
before each launch.

## Changes

- Dense MXFP8 GEMMs select FlashInfer b12x, cuDNN, or CUTLASS for the measured
  shapes and decode batches up to 256. Unrecognized shapes and larger token
  batches retain the original backend.
- BF16 `wo_a` uses cuBLASLt for batches 128 through 256, preserving the real
  input, weight, and output strides. Smaller batches retain `torch.bmm`.
- Mode `all` additionally selects the existing Marlin MXFP4 MoE backend.
  This keeps MXFP4 weights but changes MoE activations from FP8 to BF16.
  The existing activation clamp and routing configuration remain in the model.
- LM head and other operators retain their existing implementations when the
  candidate microbenchmarks did not establish a useful improvement.
- The router retains its FP32 output contract. The earlier BF16-output router
  microbenchmark is not a drop-in replacement for the installed `GateLinear`.

The import hooks are enabled only by `DSV41_FAST_ENABLED=1`, which the launcher
sets for `dense` and `all`. The shared projection function is cloned with a local
torch proxy; global `torch.bmm` and installed source files are not patched.
Each worker loads a previously compiled extension and creates cuBLASLt plans
before capture. Plans cannot be created inside a CUDA graph.

## Measurement

The test uses the existing 2K-token workload, a fixed cohort of 1, 2, 4, 8, 16,
32, 64, 128, or 256 requests, 96 generated tokens, and three repeats. PP remains
20 + 20 layers, TP4 and EP4, with FULL decode CUDA graphs.

The primary latency is stage 0 host forward submission through all sampled
tokens being available at the engine. Every measured token step must contain
one unsplit cohort across all eight ranks, with no padding and no preemption.
CUDA events are used only for each GPU's local duration, never subtracted
between GPUs. Client inter-token latency is reported separately. No full
profiler is enabled during the timing runs.

Token comparisons establish repeat consistency and agreement on this workload.
They do not replace a model quality evaluation, particularly when enabling the
Marlin W4A16 path.

## Reproduce On The Developer Machine

The scripts must reside in
`/home/admin/dsv41-vllm-20260917/fast_gemm_trial`. The existing environment,
workload, fixed-batch scheduler, and PP timing hooks are prerequisites.
Run one GPU experiment at a time, starting with all GPUs idle.

```bash
cd /home/admin/dsv41-vllm-20260917
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
env/bin/python fast_gemm_trial/build_validate.py
env/bin/python fast_gemm_trial/run_trial.py \
  --mode baseline --directory "$PWD/fast_gemm_trial/results/baseline_new"
env/bin/python fast_gemm_trial/run_trial.py \
  --mode all --baseline "$PWD/fast_gemm_trial/results/baseline_new" \
  --directory "$PWD/fast_gemm_trial/results/all_new"
env/bin/python fast_gemm_trial/summarize_trial.py \
  --baseline "$PWD/fast_gemm_trial/results/baseline_new" \
  --candidate "$PWD/fast_gemm_trial/results/all_new" \
  --outputdir "$PWD/fast_gemm_trial/comparison_new"
```

Mode `dense` provides an ablation with the original MoE backend. Trial output
directories must be new. The driver stops its own server process group on
completion or failure. Compilation is limited to two jobs; numerical checks
and timings in `build_validate.py` use one GPU and exit afterward.
