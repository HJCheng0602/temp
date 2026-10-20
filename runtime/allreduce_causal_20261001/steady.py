"""Graph length, steady kernel, and IPC algorithm controls on four local GPUs."""
import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import time

import torch
import torch.distributed as dist


def gather(value):
    out = [None] * 4
    dist.all_gather_object(out, value)
    return out


def clocks():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu',
        '--format=csv,noheader'], text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['nccl', 'ipc'], required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--variants', action='store_true')
    ap.add_argument('--dump', action='store_true')
    args = ap.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', timeout=timedelta(seconds=120))
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from flashinfer.comm import PcieIpcAllReduceWorkspace
    from flashinfer.comm.pcie_ipc_policy import IpcLaunchConfig
    nccl = PyNcclCommunicator(dist.group.WORLD, device=torch.device('cuda', rank)) if args.backend == 'nccl' else None
    workspace = None
    if args.backend == 'ipc':
        workspace = PcieIpcAllReduceWorkspace(group=dist.group.WORLD,
            max_numel=4096 * 5120, dtype=torch.bfloat16, tune_batches=(1, 8, 32))
        workspace.prepare([(b, 5120) for b in (1, 8, 32)], dtype=torch.bfloat16)
    report = {'backend': args.backend, 'profile': args.profile,
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'env': {k: v for k, v in os.environ.items() if k.startswith(('NCCL_', 'CUDA_'))},
              'clocks_before': clocks() if rank == 0 else None, 'cases': []}
    batches = [1] if args.variants or args.dump else [1, 8, 32]
    for batch in batches:
        index = torch.arange(batch * 5120, device='cuda').reshape(batch, 5120)
        x = ((index % 9) - 4 + rank).to(torch.bfloat16)
        expected = (4 * ((index % 9) - 4) + 6).to(torch.bfloat16)
        out = torch.empty_like(x)
        configs = [None]
        if args.variants:
            configs = [IpcLaunchConfig(blocks=blocks, threads=threads, variant=v)
                       for v in (0, 1, 2) for blocks in (1, 4, 16, 128) for threads in (64, 128, 256)]
        for config in configs:
            def operation():
                if nccl is not None:
                    nccl.all_reduce(x, out_tensor=out)
                else:
                    workspace.all_reduce(x, out=out, config=config)

            for _ in range(10):
                operation()
            torch.cuda.synchronize()
            selected = None if workspace is None else asdict(config or workspace.tuned_launch_config(x))
            lengths = [128] if args.profile or args.variants else [1, 8, 32, 128, 512]
            if args.dump:
                lengths = [4]
            for calls in lengths:
                graph = torch.cuda.CUDAGraph(keep_graph=True) if args.dump else torch.cuda.CUDAGraph()
                if args.dump:
                    graph.enable_debug_mode()
                dist.barrier()
                with torch.cuda.graph(graph):
                    for _ in range(calls):
                        operation()
                if args.dump:
                    graph.debug_dump(str(Path(args.output).with_suffix(f'.rank{rank}.dot')))
                for _ in range(8):
                    graph.replay()
                torch.cuda.synchronize()
                measurements = []
                for repeat in range(11):
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    host_start = time.monotonic_ns()
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    host_end = time.monotonic_ns()
                    measurements.append({'us_per_call': start.elapsed_time(end) * 1000 / calls,
                                         'host_start_ns': host_start, 'host_end_ns': host_end})
                exact = bool(torch.equal(out, expected))
                profile = []
                if args.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        dist.barrier()
                        graph.replay()
                        torch.cuda.synchronize()
                    for event in prof.events():
                        if event.device_type == torch.autograd.DeviceType.CUDA:
                            profile.append({'name': event.name, 'start_us': event.time_range.start,
                                            'duration_us': event.time_range.elapsed_us()})
                records = gather({'rank': rank, 'samples': measurements, 'exact': exact, 'kernels': profile})
                if rank == 0:
                    maxima = [max(r['samples'][i]['us_per_call'] for r in records) for i in range(11)]
                    skew = [(max(r['samples'][i]['host_start_ns'] for r in records) -
                             min(r['samples'][i]['host_start_ns'] for r in records)) / 1000 for i in range(11)]
                    case = {'batch': batch, 'calls': calls, 'config': selected, 'records': records,
                            'median_max_rank_us': st.median(maxima), 'max_rank_us': maxima,
                            'host_start_skew_us': skew, 'all_exact': all(r['exact'] for r in records)}
                    report['cases'].append(case)
                    Path(args.output).write_text(json.dumps(report, indent=2))
                    print(f'CASE {args.backend} B{batch} n{calls} config={selected} us={st.median(maxima):.4f} exact={case["all_exact"]}', flush=True)
                del graph
    if workspace is not None:
        workspace.destroy()
    if rank == 0:
        report['clocks_after'] = clocks()
        Path(args.output).write_text(json.dumps(report, indent=2))
        print('DONE', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
