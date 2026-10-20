"""Single-GPU top-k checks with frozen scores, including selection-boundary ties."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stable', action='store_true')
    args = parser.parse_args()
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
    from vllm.v1.worker.workspace import init_workspace_manager
    if args.stable:
        import os
        from types import SimpleNamespace
        import stable_topk_diagnostic
        os.environ['DSV41_STABLE_TOPK_DIAGNOSTIC'] = '1'
        stable_topk_diagnostic.install(SimpleNamespace(ops=ops, SparseIndexerTopk=SparseIndexerTopk))
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    init_workspace_manager(torch.device('cuda:0'))
    torch.manual_seed(726)
    selector = SparseIndexerTopk(backend='auto')
    results = []
    for length in (512, 513, 1024, 2048):
        for kind in ('unique', 'ties', 'zero'):
            logits = torch.randn((1, 4096), device='cuda', dtype=torch.float32)
            if kind == 'ties':
                logits = logits.round().clamp_min(0)
            elif kind == 'zero':
                logits.zero_()
            lengths = torch.tensor([[length]], device='cuda', dtype=torch.int32)
            starts = torch.zeros(1, device='cuda', dtype=torch.int32)
            ends = lengths.reshape(-1)
            out = torch.full((1, 512), -1, device='cuda', dtype=torch.int32)
            for phase in ('decode', 'prefill'):
                def invoke():
                    out.fill_(-1)
                    if phase == 'decode':
                        selector(logits, lengths, 1, out, 512, length)
                    else:
                        ops.top_k_per_row_prefill(logits, starts, ends, out, 1,
                                                 logits.stride(0), logits.stride(1), 512)
                invoke()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    invoke()
                sequences = []
                for _ in range(12):
                    graph.replay()
                    sequences.append(out.cpu().reshape(-1).tolist())
                scores = logits[0, :length].cpu()
                best = float(scores.topk(512).values[-1])
                valid_sets = [set(row) for row in sequences]
                legal = all(len(row) == 512 and all(0 <= index < length for index in row)
                            and len(set(row)) == 512 for row in sequences)
                threshold_valid = legal and all(float(scores[row].min()) >= best for row in sequences)
                results.append(dict(length=length, kind=kind, phase=phase,
                                    backend=selector.resolve_backend(logits, 512, 1) if phase == 'decode' else 'top_k_per_row_prefill',
                                    legal=legal, threshold_valid=threshold_valid,
                                    raw_variants=len({tuple(row) for row in sequences}),
                                    set_variants=len({tuple(sorted(row)) for row in valid_sets}),
                                    threshold=best, threshold_ties=int((scores == best).sum()),
                                    first_indices=sequences[0][:16]))
                print(json.dumps(results[-1]), flush=True)
    args.output.write_text(json.dumps(results, indent=2))
    assert all(r['legal'] and r['threshold_valid'] for r in results)
    if args.stable:
        assert all(r['raw_variants'] == 1 and r['set_variants'] == 1 for r in results)
        logits = torch.arange(4096, device='cuda', dtype=torch.float32).repeat(2, 1)
        starts = torch.tensor([0, 1000], device='cuda', dtype=torch.int32)
        ends = starts + 513
        out = torch.empty((2, 512), device='cuda', dtype=torch.int32)
        ops.top_k_per_row_prefill(logits, starts, ends, out, 2, 4096, 1, 512)
        assert torch.equal(out[0], out[1])
        assert torch.equal(out[0].sort().values, torch.arange(1, 513, device='cuda', dtype=torch.int32))
        print('STABLE_OFFSET_CONTRACT_PASSED', flush=True)


if __name__ == '__main__':
    main()
