# Live SM120 decode and prefill source export

Exported inside developer container 78c7dcafe00c on 2026-10-20.
Every retained implementation file was read from the actual remote runtime;
no local source snapshot was copied over the running environment.

- vllm_overlay/vllm: all Python files changed or added relative to the installed
  vLLM wheel RECORD. vllm_overlay_manifest.json records original and current hashes.
- runtime: custom deployment scripts, PP receive/graph fixes, fixed-cohort
  scheduler, decode GEMM experiments, CED prefill, source-20 cache head,
  projection policy, and retained patch sources from the runtime root.
- experiments: current sibling prefill/collective/projection/profiling scripts.
- references: three actual FlashInfer SM120 kernel headers used by the runtime.
- evidence: synthetic token-ID workloads, IPC tuning cache, and the retained
  NCCL projection comparison accuracy records.

The original runtime root is /home/admin/dsv41-vllm-20260917, a symlink to
/mnt/fuse/dsv41-tools-20260917/vllm. vLLM was installed from a wheel and has no
source Git repository there. This export creates a separate delivery repository;
it does not overwrite the existing environment or alter inference behavior.

Base vLLM: 0.3.1.dev7+g8538017f4, commit
8538017f4b1d01b543345f5dbe1993cb66c80fa9. The October prefill overlay uses
FlashInfer 0.7.1. environment_versions.json records observed package versions,
not a newly tested installation lockfile. Weights, compiled binaries, and raw
traces are excluded. Historical launchers retain their original Linux paths.

To reconstruct, first install the exact base wheel and matching dependencies.
Review each original SHA-256 in vllm_overlay_manifest.json before applying its
replacement under the installed vllm package. Added files have before=null.
Place runtime files at the original runtime root and sibling experiments at
their recorded source paths (or adapt launcher paths). The incremental patch
installers may require earlier instrumented sources; do not bypass their guards.
source_manifest.json records the absolute original path and SHA-256 for every
copied file. verify_export.py checks the export; --live also checks originals.

Accuracy scope: the retained NCCL ordinary-projection comparison passed its
recorded gates. FlashInfer IPC comparisons had logits differences. Decoder
bounded replay is approximate relative to full-prefix decoder execution.
Marlin W4A16 changes activation precision. FullTile and swapAB attention were
research candidates, not implemented improvements in this export.
