"""Compare graph-populated logical sparse indices without loading logits."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def analyze(directory):
    groups = defaultdict(dict)
    path = directory / 'sparse_order_audit.jsonl'
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row['key'], row['ordinal'], row['prefix'])
        if key in groups[row['group']]:
            raise ValueError(f'Duplicate sparse record: {key}')
        groups[row['group']][key] = row
    left, right = groups['capture'], groups['repeat']
    common = left.keys() & right.keys()
    if not common:
        raise ValueError('No paired sparse captures')
    rows = []
    for key in sorted(common):
        a = left[key]
        b = right[key]
        av, bv = a['indices'], b['indices']
        rows.append(dict(key=key[0], ordinal=key[1], position=a['position'],
                         raw_equal=av == bv, set_equal=sorted(av) == sorted(bv),
                         query_equal=a['query_sha256'] == b['query_sha256'],
                         output_equal=a['output_sha256'] == b['output_sha256'],
                         valid_a=sum(x >= 0 for x in av), valid_b=sum(x >= 0 for x in bv),
                         duplicate_valid_a=len([x for x in av if x >= 0]) - len({x for x in av if x >= 0}),
                         duplicate_valid_b=len([x for x in bv if x >= 0]) - len({x for x in bv if x >= 0})))
    report = dict(samples=len(rows), rows=rows, unpaired_capture=len(left.keys() - right.keys()),
                  unpaired_repeat=len(right.keys() - left.keys()),
                  sampling='first four physical batch rows; only matching request/position pairs compared')
    for field in ('raw_equal', 'set_equal', 'query_equal', 'output_equal'):
        report[field] = sum(r[field] for r in rows)
    (directory / 'sparse_order_comparison.json').write_text(json.dumps(report, indent=2))
    return {k: v for k, v in report.items() if k != 'rows'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        print(json.dumps(dict(directory=str(directory), **analyze(directory))))
