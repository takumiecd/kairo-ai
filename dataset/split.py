"""Split generated JSONL examples into train/valid/test files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records(
    records: list[dict],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not records:
        return [], [], []
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")
    if min(train_ratio, valid_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_size = int(len(shuffled) * train_ratio)
    valid_size = int(len(shuffled) * valid_ratio)

    # Keep non-zero valid/test splits when possible for small datasets.
    if valid_ratio > 0 and valid_size == 0 and len(shuffled) >= 3:
        valid_size = 1
    test_size = len(shuffled) - train_size - valid_size
    if test_ratio > 0 and test_size == 0 and len(shuffled) >= 3:
        test_size = 1
        train_size = max(1, len(shuffled) - valid_size - test_size)

    train_records = shuffled[:train_size]
    valid_records = shuffled[train_size : train_size + valid_size]
    test_records = shuffled[train_size + valid_size :]
    return train_records, valid_records, test_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    train_records, valid_records, test_records = split_records(
        records,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_jsonl(args.output_dir / "train.jsonl", train_records)
    write_jsonl(args.output_dir / "valid.jsonl", valid_records)
    write_jsonl(args.output_dir / "test.jsonl", test_records)
    print(
        f"train={len(train_records)} "
        f"valid={len(valid_records)} "
        f"test={len(test_records)}"
    )


if __name__ == "__main__":
    main()
