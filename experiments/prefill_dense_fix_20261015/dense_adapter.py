"""32K projection A/B on the existing P4+D4 runner."""
from adapter import ComparisonRunner


class DenseRunner(ComparisonRunner):
    def configure_dense(self, enabled, profile=False, verify=False):
        from prefill_dense import install
        self.projections = install()
        self.projections.configure(enabled, profile, verify)
        return self.projections.snapshot()

    def projection_snapshot(self):
        return self.projections.snapshot()

    def record_dense_accuracy(self, name):
        return self.save_path(name, self.last_replay_sample, continuation_steps=3)

    def compare_dense_accuracy(self, left_name="default", right_name="repaired"):
        import torch
        from accuracy_checks import compare_paths, tensor_metrics, kl_divergence
        left, right = self.accuracy_paths[left_name], self.accuracy_paths[right_name]
        paths = compare_paths({left_name: left, right_name: right})
        rows = []
        same_history = True
        for a, b in zip([left] + left["continuations"],
                        [right] + right["continuations"]):
            same_token = a["sample"]["token_ids"] == b["sample"]["token_ids"]
            finite = bool(torch.isfinite(a["logits"]).all() and torch.isfinite(b["logits"]).all())
            rows.append({"same_history": same_history, "token_equal": same_token,
                         "default_token_ids": a["sample"]["token_ids"],
                         "repaired_token_ids": b["sample"]["token_ids"], "finite": finite,
                         "logits": tensor_metrics(a["logits"], b["logits"]) if same_history else None,
                         "kl": kl_divergence(a["logits"], b["logits"]) if same_history else None})
            same_history = same_history and same_token
        result = {"rank": self.rank, "left": left_name, "right": right_name,
                  "steps": rows, "paths": paths,
                  "passed": all(r["finite"] and r["token_equal"] and r["logits"]
                                and r["logits"]["allclose"] for r in rows)}
        return result


def entry(connection, settings, role, rank, gpu_ids, port):
    import report_worker
    report_worker.ReportRunner = DenseRunner
    report_worker.worker_entry(connection, settings, role, rank, gpu_ids, port)
