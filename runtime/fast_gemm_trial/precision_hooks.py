"""Bounded decode checks on complete vocabulary rows at the V2 sampler.

Capture precedes sampling. Teacher forcing uses vLLM's TraceReplayState.
Importing this module from sitecustomize does not initialize torch or CUDA.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys

SELECTED_ORDINALS = set(range(2, 18)) | {32, 64, 96, 128, 160}


class PrecisionCapture:
    def __init__(self):
        import torch.distributed as dist
        if not dist.is_initialized():
            raise RuntimeError('Capture requires initialized distributed ranks')
        self.rank = dist.get_rank()
        self.root = Path(os.environ['DSV41_PRECISION_DIR'])
        self.requests = {}
        self.groups = {}
        self.seen = set()
        self.bytes = 0
        self.max_bytes = int(os.environ.get('DSV41_PRECISION_MAX_BYTES', str(2 * 1024**3)))

    def add(self, req_idx, prompt_len, params):
        self.requests.pop(int(req_idx), None)
        extra = params.extra_args or {}
        key = extra.get('precision_key')
        if key is None:
            return
        group = extra.get('precision_group', 'capture')
        if not isinstance(key, str) or not key or len(key) > 256:
            raise ValueError('Invalid precision request key')
        if not isinstance(group, str) or not group or any(
                c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in group):
            raise ValueError('Invalid precision group')
        self.requests[int(req_idx)] = dict(key=key, group=group, prompt_len=int(prompt_len),
                                          full=bool(extra.get('precision_capture', False)),
                                          trace=extra.get('precision_trace'))

    def group(self, name):
        if name not in self.groups:
            directory = self.root / name / f'rank_{self.rank}'
            directory.mkdir(parents=True, exist_ok=True)
            config = dict(rank=self.rank, source='Sampler.__call__ before processing/trace',
                          full_vocabulary=True, decode_only=True,
                          selected_output_ordinals=sorted(SELECTED_ORDINALS),
                          forced_tokens_are_not_generation_agreement=True)
            (directory / 'configuration.json').write_text(json.dumps(config, indent=2))
            self.groups[name] = (directory, (directory / 'index.jsonl').open('a', buffering=1),
                                (directory / 'all_rows.jsonl').open('a', buffering=1))
        return self.groups[name]

    def record(self, logits, batch, sampler):
        import numpy as np
        import torch
        if batch.has_prefill or batch.num_draft_tokens:
            return
        selected = [(i, self.requests[int(slot)]) for i, slot in enumerate(batch.idx_mapping_np)
                    if int(slot) in self.requests]
        if not selected:
            return
        if os.environ.get('DSV41_SPARSE_AUDIT') == '1':
            from sparse_order_hooks import record
            record(batch, self.requests)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Capture must stay outside the CUDA graph')
        if not np.all(batch.num_scheduled_tokens == 1):
            raise RuntimeError('Capture received non-decode rows')
        if logits.ndim != 2 or logits.shape[0] != len(batch.idx_mapping_np):
            raise RuntimeError('Sampler row mapping mismatch')
        if logits.shape[1] != sampler.sampling_states.vocab_size:
            raise RuntimeError('Logits do not cover the full vocabulary')
        positions = batch.positions[batch.logits_indices].cpu().tolist()
        inputs = batch.input_ids[batch.logits_indices].cpu().tolist()
        values = logits.float()
        finite = torch.isfinite(values).all(dim=1).cpu().tolist()
        top_values = torch.topk(values, 2, dim=1).values.cpu().tolist()
        # Greedy sampling selects the first maximum; topk does not promise tie order.
        argmax_ids = torch.argmax(values, dim=1).cpu().tolist()
        log_z = torch.logsumexp(values, dim=1).cpu().tolist()
        for row, meta in selected:
            ordinal = int(positions[row]) - meta['prompt_len'] + 2
            if ordinal < 2:
                raise RuntimeError('Prefill token misclassified as decode')
            identity = (meta['group'], meta['key'], ordinal)
            if identity in self.seen:
                raise RuntimeError(f'Duplicate capture: {identity}')
            self.seen.add(identity)
            trace = meta['trace']
            target = int(trace[ordinal - 1]) if trace is not None else int(argmax_ids[row])
            record = dict(precision_key=meta['key'], output_ordinal=ordinal,
                          position=int(positions[row]), input_token_id=int(inputs[row]),
                          prompt_len=meta['prompt_len'], argmax=int(argmax_ids[row]),
                          top1_margin=float(top_values[row][0] - top_values[row][1]),
                          finite=bool(finite[row]), target_token=target,
                          target_logprob=float(values[row, target].item() - log_z[row]),
                          forced=trace is not None, rank=self.rank, dtype=str(logits.dtype))
            directory, index, rows = self.group(meta['group'])
            rows.write(json.dumps(record) + '\n')
            if not finite[row]:
                raise RuntimeError(f'Non-finite raw logits at {identity}')
            if meta['full'] and ordinal in SELECTED_ORDINALS:
                array = values[row].cpu().numpy()
                if self.bytes + array.nbytes > self.max_bytes:
                    raise RuntimeError('Capture byte budget exceeded')
                key_hash = hashlib.sha256(meta['key'].encode()).hexdigest()[:24]
                path = directory / key_hash / f'ordinal_{ordinal:06d}.npy'
                path.parent.mkdir(exist_ok=True)
                if path.exists():
                    raise RuntimeError(f'Capture overwrite attempted: {path}')
                np.save(path, array, allow_pickle=False)
                self.bytes += array.nbytes
                record.update(file=str(path.relative_to(directory)), shape=list(array.shape))
                index.write(json.dumps(record) + '\n')


def install_precision_hook(module):
    cls = module.Sampler
    if getattr(cls, '_dsv41_precision', False):
        return
    original_add, original_call = cls.add_request, cls.__call__

    @functools.wraps(original_add)
    def add(self, req_idx, prompt_len, sampling_params):
        extra = sampling_params.extra_args or {}
        trace = extra.get('precision_trace')
        if trace is not None:
            if self.trace_replay_state is None:
                raise RuntimeError('Teacher forcing requires --enable-trace-replay')
            if not isinstance(trace, list) or not trace or len(trace) > 512 or any(
                    type(token) is not int or token < 0 or token >= self.sampling_states.vocab_size
                    for token in trace):
                raise ValueError('Invalid precision trace')
            sampling_params.trace_decode_token_ids = trace
        capture = getattr(self, '_dsv41_precision_capture', None)
        if capture is None and extra.get('precision_key') is not None:
            capture = PrecisionCapture()
            self._dsv41_precision_capture = capture
        if capture is not None:
            capture.add(req_idx, prompt_len, sampling_params)
        return original_add(self, req_idx, prompt_len, sampling_params)

    @functools.wraps(original_call)
    def call(self, logits, input_batch):
        capture = getattr(self, '_dsv41_precision_capture', None)
        if capture is not None:
            if not hasattr(self, '_dsv41_precision_owner'):
                raise RuntimeError('Sampler capture ownership was not initialized')
            if self._dsv41_precision_owner:
                capture.record(logits, input_batch, self)
        return original_call(self, logits, input_batch)

    cls.add_request, cls.__call__ = add, call
    cls._dsv41_precision = True
    print('DSV41_PRECISION_SAMPLER_INSTALLED', flush=True)


def install_model_runner_hook(module):
    cls = module.GPUModelRunner
    if getattr(cls, '_dsv41_precision_owner_hook', False):
        return
    original = cls.load_model

    @functools.wraps(original)
    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self.sampler is not None:
            from vllm.distributed import get_tensor_model_parallel_rank
            # The model may disable configured sharding if it lacks local logits.
            sharded = self.batch_sharder is not None
            self.sampler._dsv41_precision_owner = sharded or get_tensor_model_parallel_rank() == 0
            print(json.dumps(dict(precision_capture_owner=self.sampler._dsv41_precision_owner,
                                  batch_sharded_sampling=sharded,
                                  tp_rank=get_tensor_model_parallel_rank())), flush=True)
        return result

    cls.load_model = load_model
    cls._dsv41_precision_owner_hook = True


def install_import_hook():
    targets = {'vllm.v1.worker.gpu.sample.sampler': install_precision_hook,
               'vllm.v1.worker.gpu.model_runner': install_model_runner_hook}
    if any(getattr(f, '_dsv41_precision', False) for f in sys.meta_path):
        return

    class Finder:
        _dsv41_precision = True

        def find_spec(self, fullname, path=None, target_module=None):
            if fullname not in targets:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader

            class Loader:
                def create_module(self, module_spec):
                    return loader.create_module(module_spec) if hasattr(loader, 'create_module') else None

                def exec_module(self, module):
                    loader.exec_module(module)
                    targets[fullname](module)

            spec.loader = Loader()
            return spec

    sys.meta_path.insert(0, Finder())
    for target, install in targets.items():
        if target in sys.modules:
            install(sys.modules[target])


def _index(root, name):
    records = {}
    for path in sorted(Path(root).glob(f'rank_*/{name}')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row['precision_key'], row['output_ordinal'])
            if key in records:
                raise ValueError(f'Duplicate key {key} in {root}')
            records[key] = (path.parent, row)
    if not records:
        raise ValueError(f'Empty {name} in {root}')
    return records


def _assert_aligned(left, right):
    if set(left) != set(right):
        raise ValueError(f'Capture key sets differ: missing={list(set(left)-set(right))[:5]}, '
                         f'extra={list(set(right)-set(left))[:5]}')
    for key in left:
        a, b = left[key][1], right[key][1]
        for field in ('position', 'input_token_id', 'prompt_len', 'target_token'):
            if a[field] != b[field]:
                raise ValueError(f'Input/target mismatch {key}: {field}')
        if not a['finite'] or not b['finite']:
            raise ValueError(f'Non-finite logits {key}')


def compare_captures(baseline, candidate):
    import numpy as np
    left, right = _index(baseline, 'index.jsonl'), _index(candidate, 'index.jsonl')
    all_left, all_right = _index(baseline, 'all_rows.jsonl'), _index(candidate, 'all_rows.jsonl')
    _assert_aligned(left, right)
    _assert_aligned(all_left, all_right)
    comparisons = []
    for key in sorted(left):
        pa, ma = left[key]
        pb, mb = right[key]
        a = np.load(pa / ma['file'], allow_pickle=False).astype(np.float64)
        b = np.load(pb / mb['file'], allow_pickle=False).astype(np.float64)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'Invalid logits at {key}')
        delta = b - a
        la = a - a.max()
        lb = b - b.max()
        la -= np.log(np.exp(la).sum())
        lb -= np.log(np.exp(lb).sum())
        prob = np.exp(la)
        norm = np.linalg.norm(a)
        denom = norm * np.linalg.norm(b)
        comparisons.append(dict(key=key[0], output_ordinal=key[1],
            max_abs=float(np.max(np.abs(delta))), relative_l2=float(np.linalg.norm(delta) / max(norm, 1e-12)),
            centered_rmse=float(np.sqrt(np.mean((delta-delta.mean())**2))),
            cosine=float(np.dot(a, b) / max(denom, 1e-12)),
            kl=float(np.sum(prob * (la-lb))),
            top10_overlap=len(set(np.argpartition(a, -10)[-10:]) & set(np.argpartition(b, -10)[-10:])) / 10,
            argmax_equal=bool(np.argmax(a) == np.argmax(b)),
            baseline_margin=ma['top1_margin'], candidate_margin=mb['top1_margin']))
    metrics = {}
    for field in ('max_abs', 'relative_l2', 'centered_rmse', 'cosine', 'kl', 'top10_overlap'):
        v = np.asarray([r[field] for r in comparisons])
        metrics[field] = dict(mean=float(v.mean()), p50=float(np.median(v)),
                              p95=float(np.quantile(v, .95)), max=float(v.max()), min=float(v.min()))
    all_diffs = []
    margins = []
    for key in sorted(all_left):
        a, b = all_left[key][1], all_right[key][1]
        all_diffs.append(b['target_logprob'] - a['target_logprob'])
        if a['argmax'] != b['argmax']:
            margins.append(dict(key=key[0], output_ordinal=key[1], baseline_argmax=a['argmax'],
                                candidate_argmax=b['argmax'], baseline_margin=a['top1_margin'],
                                candidate_margin=b['top1_margin']))
    return dict(full_vocab_samples=len(comparisons), all_decode_rows=len(all_left),
                exact_input_alignment=True, all_finite=True,
                raw_argmax_equal=len(all_left)-len(margins), raw_argmax_differences=margins,
                selected_argmax_agreement=float(np.mean([r['argmax_equal'] for r in comparisons])),
                mean_baseline_target_logprob_delta=float(np.mean(all_diffs)),
                max_abs_baseline_target_logprob_delta=float(np.max(np.abs(all_diffs))),
                metrics=metrics, samples=comparisons)
