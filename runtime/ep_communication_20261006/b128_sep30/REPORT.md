# EP Communication Attribution: B128

Old 2026-09-30 Marlin + FlashInfer PCIe IPC profile; offline CPU reanalysis.

Profile batches: [1, 128]; selected B128. Stable window: 9 of 15 replays.

| Stage | Operation | Calls / GPU / step | us / GPU / step | Stage kernel share |
|---:|---|---:|---:|---:|
| 0 | MoE output combine | 20 | 3319.157639 | 22.155% |
| 0 | Attention output | 20 | 1062.327278 | 7.091% |
| 0 | Embedding | 1 | 174.444556 | 1.164% |
| 1 | MoE output combine | 20 | 3092.328306 | 22.114% |
| 1 | Attention output | 20 | 1027.284278 | 7.346% |

Mean observed dependency span: 28.606376 ms.

| Operation | Consumer-first ms / step | Producer-first ms / step | Consumer-first path share |
|---|---:|---:|---:|
| MoE output combine | 1.935231 | 1.935399 | 6.765% |
| Attention output | 1.991264 | 1.991307 | 6.961% |
| Embedding | 0.048440 | 0.048440 | 0.169% |

Reuse strict trace/case/sidecar loaders and selected 9-replay window; build observed dependency graph with step_graph; classify IPC by preceding exact graph signatures; allocate dependency intervals consumer-first and producer-first.

- Old Sep30 baseline profile; no new GPU experiment.
- Collective tails include rendezvous and waiting and are not physical wire time.
- Raw stage rows are per-GPU averages within each stage, not wall time.
- Dependency attribution is an observed path proxy, not a marginal speedup bound.
- This report contains no B64 M32 claim and no extrapolation from B64.
