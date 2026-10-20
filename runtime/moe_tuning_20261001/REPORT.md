# MoE Tuning Summary

Artifact root: `/home/admin/dsv41-vllm-20260917/moe_tuning_20261001`

Marlin sweep records: 80; FlashInfer records: 1; profiles: 2.

## Cold Marlin Timing

- M8: 16 cases, min 130.656 us, median 194.563 us
- M16: 16 cases, min 102.810 us, median 153.037 us
- M32: 16 cases, min 102.125 us, median 150.989 us
- M48: 16 cases, min 116.938 us, median 171.928 us
- M64: 16 cases, min 101.990 us, median 148.174 us
- M8 to M32 paired gain: 16 cases; min 3.4299553845553734%, median 23.87711308743075%

## Correctness

Marlin FP32-reference rows: 80; finite rows: 80.
Profile rows with local-assignment-aware MFU: 2.

## Service

{"paired_tpot": [{"baseline_p50_ms": 19.150975858792663, "batch": 64, "gain_pct": 8.565113546569805, "label": "b64_r1", "m32_p50_ms": 17.5106730312109}, {"baseline_p50_ms": 19.13915597833693, "batch": 64, "gain_pct": 8.474108717168352, "label": "b64_r2", "m32_p50_ms": 17.517283093184233}, {"baseline_p50_ms": 19.1485860850662, "batch": 64, "gain_pct": 8.7880335813027, "label": "b64_r3", "m32_p50_ms": 17.465801909565926}], "service_baseline": [{"batch": 64, "label": "b64_r1", "p50_ms": 19.150975858792663, "p95_ms": 19.75102419964969, "path": "/home/admin/dsv41-vllm-20260917/moe_tuning_20261001/service_baseline/summary.json"}, {"batch": 64, "label": "b64_r2", "p50_ms": 19.13915597833693, "p95_ms": 19.83773484826088, "path": "/home/admin/dsv41-vllm-20260917/moe_tuning_20261001/service_baseline/summary.json"}, {"batch": 64, "label": "b64_r3", "p50_ms": 19.1485860850662, "p95_ms": 19.919715076684952, "path": "/home/admin/dsv41-vllm-20260917/moe_tuning_20261001/service_baseline/summary.json"}], "service_m32": [{"batch": 64, "label": "b64_r1", "p50_ms": 17.5106730312109, "p95_ms": 18.61570212058723, "path": "/home/admin/dsv41-vllm-20260917/moe_tuning_20261001/service_m32/summary.json"}, {"batch": 64, "label": "b64_r2", "p50_ms": 17.517283093184233, "p95_ms": 18.475645035505295, "path": "/home/admin/dsv41-vllm-20260917/moe_tuning_20261001/service_m32/summary.json"}, {"batch": 64, "label": "b64_r3", "p50_ms": 17.465801909565926, "p95_ms": 18.5311496257782, "path": "/home/admin/dsv41-vllm-20260917/moe_tuning_20261001/service_m32/summary.json"}]}
Output identity: 4/4 labels exact.
