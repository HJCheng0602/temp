# Four-GPU CED Prefill Experiment

## Objective

Implement a prefill worker group on four RTX PRO 5000 GPUs in one NUMA domain.
It executes backbone layers 0 through 19 and a small layer-20 global KV/index-K
head. It must not load decoder attention-output or MoE weights. A decode worker
group receives the complete continuation state and the final 128 encoder
boundary rows, runs Decoder SWA Bounded Replay, samples a real first token, and
can continue ordinary generation through all 40 layers.

This is an experimental text-only, batch-one implementation. The current task
includes measuring 2K, 4K, 8K, 16K, 32K, 64K, and 128K prompts. K means 1024.
Prefix caching and speculative decoding remain disabled for this comparison.

## Execution Contract

The P group owns TP4/EP4 over GPUs in one measured NUMA domain. Its source-20
head consumes the same normalized attention input as the complete model and
writes the existing packed main KV and index-K formats. It does not run source
attention, index queries, attention output projections, or decoder MoE.

The handoff includes:

- Encoder global main KV and index-K state.
- Encoder SWA tail, compression-group state, and Engram token history needed
  to continue generation on D without re-running the prompt.
- Decoder source-20 global main KV and index-K for the entire prompt.
- Final `min(prompt_length, 128)` boundary HC rows and matching `pre_mix`, token
  IDs, and absolute positions.
- Cache layout and logical position metadata, with destination page ownership
  established explicitly rather than assuming equal page IDs.

The D group keeps the full model for subsequent generated tokens. For its
initial replay it skips layers 0 through 19, reuses the transferred source-20
global state, and truncates every decoder SWA window to the replay interval.
Global attention continues to see the entire causal prefix. The final prompt
position is already included in this interval; it must not be duplicated.

Bounded replay intentionally differs from an exact full-prefix decoder pass.
Correctness must be checked against a reference using the same truncation,
alongside separately reported differences from full-forward logits.

## Capacity and Transport

The 2026-09-27 real PP2/TP4 logs report 36.14 GiB for encoder-stage model loading
and 35.52 GiB for decoder-stage model loading, with Engram tables in pinned host
memory. These measurements supersede earlier estimates that placed Engram on
GPU. P fits on four cards; D full-model TP4 needs a fresh capacity check and may
need CPU offload to coexist on the remaining four cards. Any such offload must
be explicit in the benchmark configuration and interpretation.

The prototype must measure its actual transport, including packing, device/host
copies, destination mapping, and readiness. A same-host transport result must
not be described as cross-machine RDMA performance.

## Measurement and Acceptance

Record request start, P completion, transfer readiness, replay completion, and
first sampled token. End-to-end TTFT includes all work after request submission;
P-only time is reported separately. Model loading and compilation happen before
measurement, and warmup requests are excluded. Measure at least three repeats
per prompt length after warmup and preserve the individual observations.

Required evidence before claiming completion:

1. Four-GPU P parameter manifest excludes decoder Q/O and MoE parameters.
2. Source-head caches agree with the original source-20 cache production path.
3. Cross-worker handoff retains full logical prefix length and valid SWA state.
4. D produces real finite logits, a first token, and several subsequent tokens.
5. Bounded replay agrees with a same-semantics reference within documented
   numerical tolerances; approximation to full forward is not mislabeled exact.
6. The requested seven prompt lengths have measured TTFT, or a concrete
   unresolved failure remains and the task is not marked complete.
7. Test processes are cleaned up and GPU state is checked after experiments.

## Current State

Implementation and remote validation are in progress. No CED TTFT results have
been produced yet.

The four-rank `smoke_b` load on 2026-09-28 verified 35.493 GiB of named GPU
parameters and 47.208 GiB of CPU parameters per rank. The layer-20 parameter
manifest contains exactly the attention norm plus four cache-head leaves; the
runtime rejects any additional decoder parameters on P. GPU parameter bytes
exclude workspaces, buffers, and KV allocation.

The first run identified a missing engine-core KV layout resolution in the
standalone harness, now supplied through vLLM's resolver. The next run completed
the encoder forward and failed in the added cache head because the index RMSNorm
uses `variance_epsilon`, not `eps`; this interface mismatch has been corrected.
The isolated CUDA head check then passed for 1, 128, and 2048 rows. Maximum
latent absolute error against its dense reference was 0.0009765625.

The first simultaneous P/D run (`pd_d`) hit the container host-memory limit
while loading D. The kernel recorded a memcg OOM at 07:23:14 on 2026-09-28.
Its eight workers were confirmed gone. The `pd_e` run caps pinned allocation
caching and power-of-two rounding at 1024 MiB, then frees unused host allocator
buffers after loading. P rank 0 subsequently reported 51,301,458,824 allocated
pinned bytes and 52,131,752 KiB RSS. D startup is still being validated.

The reference check now compares both first-token logits and three continuation
token IDs between the transferred state and a locally rebuilt bounded-replay
state. This is not a comparison against exact full-prefix decoder execution.
