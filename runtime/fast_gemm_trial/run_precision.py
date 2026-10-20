"""Bounded real-model free-generation and teacher-forced decode validation.

Logit capture adds synchronization and CPU work. These runs are accuracy
experiments; their elapsed times must not be used as performance benchmarks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback
import urllib.request

import parallel_matrix as matrix
from run_pp2_wholebatch import read_waves
from precision_workload import build_workload


def save(path, value):
    matrix.save(Path(path), value)


def request_one(item, output_tokens, barrier, mode, phase, capture, trace=None):
    extra = {}
    if capture:
        extra = dict(precision_key=item['key'],
                     precision_group='repeat' if phase == 'repeat' else 'capture',
                     precision_capture=int(item['capture_full']))
    if trace is not None:
        extra['precision_trace'] = trace
    payload = dict(model='deepseek-v41-flash', prompt=item['prompt_token_ids'],
                   request_id=f"precision-{mode}-{phase}-{item['key']}",
                   temperature=0, seed=0, max_tokens=output_tokens,
                   ignore_eos=True, stream=True, return_token_ids=True,
                   stream_options=dict(include_usage=True), vllm_xargs=extra)
    req = urllib.request.Request(matrix.URL + '/v1/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    barrier.wait()
    result = dict(key=item['key'], output_ids=[], text='', events=[], forced=trace is not None)
    with urllib.request.urlopen(req, timeout=1800) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                break
            chunk = json.loads(raw)
            if chunk.get('error'):
                raise RuntimeError(chunk['error'])
            if chunk.get('usage'):
                result['usage'] = chunk['usage']
            for choice in chunk.get('choices', []):
                ids = choice.get('token_ids') or []
                result['output_ids'].extend(ids)
                result['text'] += choice.get('text') or ''
                if ids:
                    result['events'].append(dict(t=time.monotonic(), token_ids=ids))
                if choice.get('finish_reason'):
                    result['finish_reason'] = choice['finish_reason']
    if len(result['output_ids']) != output_tokens:
        raise RuntimeError(f"Wrong output length: {item['key']}: {len(result['output_ids'])}")
    if trace is not None and result['output_ids'] != trace:
        raise RuntimeError(f"Trace replay failed: {item['key']}")
    result['prompt_sha256'] = hashlib.sha256(
        json.dumps(item['prompt_token_ids']).encode()).hexdigest()
    return result


def compare_outputs(reference, candidate):
    if set(reference) != set(candidate):
        raise RuntimeError('Request keys differ')
    rows = []
    for key in sorted(reference):
        left, right = reference[key], candidate[key]
        if left['prompt_sha256'] != right['prompt_sha256']:
            raise RuntimeError(f'Prompt differs: {key}')
        a, b = left['output_ids'], right['output_ids']
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        rows.append(dict(key=key, exact=a == b, first_difference=first,
                         common_prefix=len(a) if first is None else first,
                         tokens=len(a), positional_equal=sum(x == y for x, y in zip(a, b))))
    return dict(requests=len(rows), exact_requests=sum(r['exact'] for r in rows),
                tokens=sum(r['tokens'] for r in rows),
                positional_equal=sum(r['positional_equal'] for r in rows),
                common_prefix_tokens=sum(r['common_prefix'] for r in rows), rows=rows)


def verify_ranks(work, case_records):
    by_rank = {}
    for rank in range(8):
        files = list((work / 'rank_timings').glob(f'rank_{rank}_*.jsonl'))
        if len(files) != 1:
            raise RuntimeError(f'Missing/ambiguous rank timing: {rank}')
        by_rank[rank] = [json.loads(line) for line in files[0].read_text().splitlines()]
    checks = []
    for case in case_records:
        waves, ids = read_waves(Path(case['wave_file']), case['batch'], case['output_tokens'])
        for rank, records in by_rank.items():
            rows = [r for r in records if not r['prefill'] and set(r['request_ids']) == ids]
            if len(rows) != len(waves):
                raise RuntimeError(f"Wrong decode step count: {case['label']} rank {rank}")
            if any(r['n'] != case['batch'] or r['padded_tokens'] != case['batch']
                   or r['graph_mode'] != 'FULL' for r in rows):
                raise RuntimeError(f"Wrong graph/batch: {case['label']} rank {rank}")
            if len({r['sequence'] for r in rows}) != len(rows):
                raise RuntimeError('Duplicate rank sequence')
            for row, wave in zip(rows, waves):
                if not wave['wave_start'] <= row['cpu_start'] <= wave['wave_end']:
                    raise RuntimeError('Rank step does not align with scheduler wave')
        checks.append(dict(label=case['label'], batch=case['batch'],
                           decode_steps=len(waves), ranks=8, graph='FULL'))
    return checks


def verify_capture_case(work, group, case, outputs):
    from precision_hooks import _index, SELECTED_ORDINALS
    directory = work / 'captures' / group
    rows = _index(directory, 'all_rows.jsonl')
    full = _index(directory, 'index.jsonl')
    expected_rows = set()
    expected_full = set()
    case_keys = {r['key'] for r in case['requests']}
    for index, item in enumerate(case['requests']):
        ids = outputs[item['key']]['output_ids']
        for ordinal in range(2, case['output_tokens'] + 1):
            key = (item['key'], ordinal)
            expected_rows.add(key)
            if index < 4 and ordinal in SELECTED_ORDINALS:
                expected_full.add(key)
            if key not in rows:
                raise RuntimeError(f'Missing decode capture: {key}')
            meta = rows[key][1]
            if meta['position'] != len(item['prompt_token_ids']) + ordinal - 2:
                raise RuntimeError(f'Wrong decode position: {key}')
            if meta['input_token_id'] != ids[ordinal - 2] or meta['target_token'] != ids[ordinal - 1]:
                raise RuntimeError(f'Wrong input/target token alignment: {key}')
    if {k for k in rows if k[0] in case_keys} != expected_rows:
        raise RuntimeError('Unexpected decode captures')
    if {k for k in full if k[0] in case_keys} != expected_full:
        raise RuntimeError('Full-vocabulary sampling coverage differs from protocol')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('baseline', 'dense', 'all'), required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--free-repeat', action='store_true')
    parser.add_argument('--self-reference', action='store_true',
                        help='Compare the selected backend against its own captured history')
    args = parser.parse_args()
    self_reference = args.mode == 'baseline' or args.self_reference
    run_lock = (matrix.ROOT / 'precision_validation.lock').open('a')
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    work = args.directory
    work.mkdir(parents=True, exist_ok=False)
    save(work / 'controller_pid.json', dict(pid=os.getpid()))
    matrix.URL = 'http://127.0.0.1:8024'
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 8024)) == 0:
            raise RuntimeError('Accuracy port is occupied')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU jobs are already active')
    control = work / 'control.json'
    save(control, dict(batch=1, size=1, log=str(work / 'waves' / 'initial.jsonl')))
    config = dict(mode=args.mode, pp=2, tp=4, ep=4, layers=[20, 20],
                  graph='FULL_DECODE_ONLY', max_model_len=4096,
                  sparse_order=os.environ.get('DSV41_SPARSE_ORDER', 'unmodified'),
                  stable_topk=os.environ.get('DSV41_STABLE_TOPK_DIAGNOSTIC') == '1',
                  prefill_admission_barrier=os.environ.get('DSV41_PRECISION_PREFILL_BARRIER') == '1',
                  marlin_canonical_order=os.environ.get('DSV41_MARLIN_CANONICAL_ORDER') == '1',
                  self_reference=self_reference,
                  free_repeat=args.free_repeat,
                  workload=str(args.workload), reference=str(args.reference),
                  scope='accuracy only; instrumented timings are not performance data',
                  free_outputs='unforced greedy generation',
                  forced_outputs='baseline trace; never used as evidence of generation agreement',
                  logits='full vocabulary before sampling; selected real decode positions')
    config['diagnostic_source_sha256'] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('run_precision.py', 'precision_hooks.py', 'sparse_order_hooks.py',
                     'stable_topk_diagnostic.py', 'precision_prefill_scheduler.py',
                     'marlin_order_diagnostic.py', 'sitecustomize.py', 'serve_trial.py')
        if Path(__file__).with_name(name).exists()}
    save(work / 'configuration.json', config)

    def status(phase, **extra):
        value = dict(phase=phase, mode=args.mode, time=time.time(), **extra)
        save(work / 'status.json', value)
        print(json.dumps(value), flush=True)

    def interrupt(signum, frame):
        raise RuntimeError(f'Interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    signal.alarm(7200)
    process = None
    cases = []
    all_free = {}
    all_forced = {}
    all_free_again = {}
    reference = None
    try:
        env = os.environ.copy()
        env.update(DSV41_PRECISION_HOOK='1', DSV41_PRECISION_DIR=str(work / 'captures'),
                   DSV41_PRECISION_ROLE='baseline' if self_reference else 'candidate',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [str(matrix.ROOT / 'env/bin/python'),
                   str(Path(__file__).with_name('serve_trial.py')), '--mode', args.mode,
                   '--timing-dir', str(work / 'rank_timings'),
                   '--fixed-batch-control', str(control), '--enable-trace-replay']
        with (work / 'server.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(work / 'server_pid.json', dict(pid=process.pid, command=command))
        status('starting', pid=process.pid)
        deadline = time.monotonic() + 1800
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'Server exited with {process.returncode}')
            if matrix.alive():
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Model startup exceeded 1800 seconds')
            time.sleep(3)
        if args.workload.exists():
            workload = json.loads(args.workload.read_text())
        elif args.mode == 'baseline':
            def tokenize(text):
                return matrix.api('/tokenize', dict(model='deepseek-v41-flash', prompt=text,
                                                   add_special_tokens=False))['tokens']
            workload = build_workload(tokenize)
            save(args.workload, workload)
        else:
            raise RuntimeError('Reference workload is missing')
        save(work / 'workload.json', workload)
        if not self_reference:
            if args.reference is None or not (args.reference / 'COMPLETE.json').exists():
                raise RuntimeError('Completed baseline is required')
            reference = json.loads((args.reference / 'free_outputs.json').read_text())

        for case in workload['cases']:
            batch = case['batch']
            requests = case['requests']
            if len(requests) != batch:
                raise RuntimeError('Workload batch mismatch')
            selected = set(range(min(4, batch)))
            for index, item in enumerate(requests):
                item['capture_full'] = index in selected
            phases = ['free', 'repeat' if self_reference else 'forced']
            if args.free_repeat:
                phases.append('free_again')
            for phase in phases:
                label = f"{case['label']}_{phase}"
                wave = work / 'waves' / f'{label}.jsonl'
                save(control, dict(batch=batch, size=batch, log=str(wave)))
                status('testing', label=label, batch=batch)
                barrier = threading.Barrier(batch)
                capture = phase != 'free_again' and (phase != 'free' or self_reference)
                source = all_free if self_reference else reference
                with ThreadPoolExecutor(max_workers=batch) as pool:
                    futures = [pool.submit(request_one, item, case['output_tokens'], barrier,
                                           args.mode, phase, capture,
                                           None if phase in ('free', 'free_again') else source[item['key']]['output_ids'])
                               for item in requests]
                    results = [future.result() for future in futures]
                read_waves(wave, batch, case['output_tokens'])
                target = all_free if phase == 'free' else all_free_again if phase == 'free_again' else all_forced
                target.update({row['key']: row for row in results})
                if capture:
                    group = 'repeat' if phase == 'repeat' else 'capture'
                    verify_capture_case(work, group, case, target)
                save(work / 'requests' / f'{label}.json', results)
                output_file = 'free_outputs.json' if phase == 'free' else 'free_repeat_outputs.json' if phase == 'free_again' else 'forced_outputs.json'
                save(work / output_file, target)
                cases.append(dict(label=label, batch=batch, output_tokens=case['output_tokens'],
                                  wave_file=str(wave), phase=phase))
                save(work / 'cases.json', cases)
                status('case_complete', label=label, batch=batch)
        # A final request flushes the asynchronous per-rank event writers.
        save(control, dict(batch=1, size=1, log=str(work / 'waves' / 'flush.jsonl')))
        request_one(workload['cases'][0]['requests'][0], 4, threading.Barrier(1),
                    args.mode, 'flush', False)
    except BaseException as exc:
        save(work / 'FAILED.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        status('failed', error=repr(exc))
        raise
    finally:
        if process is not None:
            matrix.stop_server(process)
        signal.alarm(0)
        print('PRECISION_SERVER_CLEANUP_COMPLETE', flush=True)

    status('analyzing')
    if args.free_repeat:
        save(work / 'free_repeat_comparison.json', compare_outputs(all_free, all_free_again))
    save(work / 'rank_checks.json', verify_ranks(work, cases))
    from precision_hooks import compare_captures
    if self_reference:
        comparison = compare_captures(work / 'captures/capture', work / 'captures/repeat')
        save(work / 'repeat_logits.json', comparison)
    else:
        save(work / 'free_comparison.json', compare_outputs(reference, all_free))
        comparison = compare_captures(args.reference / 'captures/capture', work / 'captures/capture')
        save(work / 'logits_comparison.json', comparison)
    save(work / 'COMPLETE.json', dict(completed=True, cases=len(cases), requests=len(all_free),
                                     accuracy_pass_not_assumed=True))
    status('complete', requests=len(all_free), cases=len(cases))


if __name__ == '__main__':
    main()
