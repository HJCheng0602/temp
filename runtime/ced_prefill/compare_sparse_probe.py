"""Compare saved sparse attention inputs without loading a model or using CUDA."""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from parallel_matrix import environment
    os.environ.update(environment())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if os.environ.get("CED_COMPARE_BOOTSTRAPPED") != "1":
        os.environ["CED_COMPARE_BOOTSTRAPPED"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)
    import torch
    torch.set_num_threads(2)

    def diff(a, b):
        if a.shape != b.shape:
            return {"shapes": [list(a.shape), list(b.shape)]}
        equal = a.view(torch.uint8) == b.view(torch.uint8)
        count = int((~equal).sum())
        result = {"different_bytes": count, "total_bytes": equal.numel()}
        if count:
            delta = a.float() - b.float()
            result.update(max_abs=float(delta.abs().max()),
                          rmse=float(delta.square().mean().sqrt()))
        return result

    pairs = [
        ("repeat", "decode_rank0/warmup_replay_0", "decode_rank0/warmup_replay_1"),
        ("encoder", "prefill_rank0/warmup_encoder", "decode_rank0/reference_encoder_encoder"),
    ]
    report = {}
    for label, left, right in pairs:
        a = torch.load(args.root / (left + ".pt"), weights_only=True)
        b = torch.load(args.root / (right + ".pt"), weights_only=True)
        row = {}
        for key, first in a["inputs"].items():
            second = b["inputs"].get(key)
            if second is None:
                row[key] = {"missing": True}
            elif hasattr(first, "shape"):
                row[key] = diff(first, second)
                if first.shape == second.shape and "cache" not in key and first.ndim:
                    row[key]["last_row"] = diff(first[-1:], second[-1:])
        for index_key, lens_key, cache_key in [
            ("sparse_indices", "swa_topk_lens", "swa_kv_cache"),
            ("extra_sparse_indices", "extra_sparse_topk_lens", "compressed_kv_cache"),
        ]:
            x, y = a["inputs"][index_key], b["inputs"][index_key]
            lengths_a, lengths_b = a["inputs"][lens_key], b["inputs"][lens_key]
            # Compare the final query's actual candidate set, excluding padding.
            x_last = x.reshape(-1, x.shape[-1])[-1, :int(lengths_a.reshape(-1)[-1])]
            y_last = y.reshape(-1, y.shape[-1])[-1, :int(lengths_b.reshape(-1)[-1])]
            row[index_key]["last_valid"] = {
                "lengths": [x_last.numel(), y_last.numel()],
                "set_equal": torch.equal(x_last.sort().values, y_last.sort().values),
                "head_a": x_last[:16].tolist(), "head_b": y_last[:16].tolist(),
            }
            ca, cb = a["inputs"][cache_key], b["inputs"][cache_key]
            slots = torch.unique(torch.cat([x_last, y_last])).long()
            slots = slots[slots >= 0]
            if slots.numel() and ca.shape[1] == cb.shape[1]:
                width = ca.shape[1]
                row[cache_key]["last_query_used_slots"] = diff(
                    ca[slots // width, slots % width], cb[slots // width, slots % width])
        row["output_last"] = diff(a["outputs_first"]["output"][-1:],
                                  b["outputs_first"]["output"][-1:])
        report[label] = row
        print(label, json.dumps(row, separators=(",", ":")), flush=True)
    (args.root / "sparse_input_comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
