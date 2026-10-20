# Prefill TTFT Sweep

This isolated driver measures batch-one TTFT for exact prompt lengths from 2K
through 128K on PP2 TP4 EP4. It uses the installed vLLM environment, the
verified deferred PP receive fix, the 20+20 layer partition, eager execution,
and an 8192-token chunked-prefill budget.

Each length receives one warmup followed by three unprofiled measurements. The
client sends tokenizer-produced token IDs and verifies the server-reported
prompt token count. Per-rank CUDA-event records preserve every prefill chunk.
The driver stops only the server process group it starts.

Run in `/home/admin/dsv41-vllm-20260917` with all GPUs idle:

```bash
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
env/bin/python prefill_ttft_sweep/run_sweep.py \
  --directory "$PWD/prefill_ttft_sweep/results/baseline_8k"
```

`run_comparison.py --mode profile` records one warmed 128K request in one
profiler session. `--mode no_chunk` explicitly disables scheduler chunking,
sets the token budget to maximum prompt length plus one, and checks that each
measured request executes exactly one prefill forward on every rank. Both
modes reuse the baseline workload and stop their own server after completion
or failure. Each no-chunk length is warmed and measured three times before
advancing, so smaller successful cases survive a later failure.

```bash
env/bin/python prefill_ttft_sweep/run_comparison.py --mode profile \
  --directory "$PWD/prefill_ttft_sweep/results/profile_128k"
env/bin/python prefill_ttft_sweep/run_comparison.py --mode no_chunk \
  --directory "$PWD/prefill_ttft_sweep/results/no_chunk_128k"
env/bin/python prefill_ttft_sweep/analyze_chunk_trace.py \
  prefill_ttft_sweep/results/profile_128k/trace_staging \
  --single-request-chunks 16 --chunk-tokens 8192 --ranks 0,4 \
  --out prefill_ttft_sweep/results/profile_128k/chunk_trace_analysis.json
```

Trace analysis validates the ordered prefill annotations and cumulative context
length before assigning ordinal chunk labels. Kernels are linked through CPU
External IDs and CUDA runtime/driver correlations, not GPU timestamp
containment. Kernel duration sums, interval unions, and CUDA-event model
elapsed times have different meanings; none alone is a pure compute critical
path. Profiled TTFT is not substituted into the unprofiled comparison.
