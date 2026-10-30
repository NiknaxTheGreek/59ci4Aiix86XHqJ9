"""Compare reconstructed O/S/T/ST split counts to the original submitted dataset manifest.

This is a dataset-provenance contract check, not a model-performance test.
Run from project root after installing requirements; no image pixels or W&B account needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.splits import make_splits

EXPECTED = {
    'O': dict(train=2152, validation=240, test=597),
    'S': dict(train=2147, validation=232, test=610),
    'T': dict(train=2387, validation=301, test=301),
    'ST': dict(train=1904, validation=475, context_unseen_early=489, test=121),
}


def main() -> int:
    dataset = pd.read_csv(ROOT/'manifests'/'datasets'/'DATA-001.csv')
    if len(dataset) != 2989:
        raise AssertionError('Expected original 2,989-frame dataset manifest')

    dataset['source_key'] = dataset.video_id.str.replace('/', '::', regex=False)
    dataset['video_id'] = dataset.video_id.str.split('/').str[-1]
    dataset['relative_path'] = dataset.archive_member.str.removeprefix('images/')
    dataset['path'] = 'dataset-not-required-for-split-contract'
    dataset['target'] = (dataset.label == 'flip').astype(int)

    splits = make_splits(dataset, seed=42)
    for regime, expected in EXPECTED.items():
        actual = splits[regime].role.value_counts().to_dict()
        if actual != expected:
            raise AssertionError(f'{regime}: original count mismatch {actual} != {expected}')
        print(f'{regime}: {actual}')
    print('PASS: Original O/S/T/ST split sizes reproduced from the archived 2,989-frame manifest')
    return 0


if __name__ == '__main__':
    sys.exit(main())
