"""Evaluate a trained RNN-T artifact on a JSONL dataset with greedy decode.

Computes corpus-level CER (total edits / total reference chars), sentence
accuracy, and a per-noise-type breakdown when records carry a ``noise`` field.

Example (full test set on GPU):

    python -m eval.run_test \
      --artifact-dir artifacts/rnnt-trf-v1 \
      --data data/combined/all_sources/test.jsonl \
      --device cuda

Use ``--limit`` for a quick sample, ``--predictions-output`` to dump
per-example predictions as JSONL.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from decode.greedy import greedy_decode
from decode.greedy import load_model_from_artifact
from eval.metrics import cer_result


def load_records(path: Path, limit: int | None) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <artifact-dir>/checkpoints/best.pt",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-symbols-per-step", type=int, default=4)
    parser.add_argument("--max-output-length", type=int, default=128)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, input_vocab, output_vocab = load_model_from_artifact(
        args.artifact_dir,
        checkpoint=args.checkpoint,
    )
    model.to(torch.device(args.device))
    model.eval()

    records = load_records(args.data, args.limit)
    print(f"Evaluating {len(records)} examples from {args.data}", flush=True)

    total_edits = 0
    total_reference = 0
    exact_matches = 0
    noise_edits: dict[str, int] = defaultdict(int)
    noise_reference: dict[str, int] = defaultdict(int)
    noise_counts: dict[str, int] = defaultdict(int)

    predictions_file = (
        args.predictions_output.open("w", encoding="utf-8")
        if args.predictions_output
        else None
    )
    started = time.time()
    try:
        for index, record in enumerate(records, start=1):
            input_ids = input_vocab.encode(record["input"])
            prediction = greedy_decode(
                model,
                input_ids,
                output_vocab,
                max_symbols_per_step=args.max_symbols_per_step,
                max_output_length=args.max_output_length,
            )
            result = cer_result(prediction, record["target"])
            total_edits += result.edits
            total_reference += result.reference_length
            exact_matches += prediction == record["target"]

            noise = record.get("noise", "unknown")
            noise_edits[noise] += result.edits
            noise_reference[noise] += result.reference_length
            noise_counts[noise] += 1

            if predictions_file is not None:
                predictions_file.write(
                    json.dumps(
                        {
                            "input": record["input"],
                            "target": record["target"],
                            "prediction": prediction,
                            "cer": result.value,
                            "noise": noise,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if index % args.log_every == 0:
                running_cer = total_edits / max(total_reference, 1)
                elapsed = time.time() - started
                print(
                    f"{index}/{len(records)}  running CER={running_cer:.4f}  "
                    f"({elapsed:.0f}s, {elapsed / index:.2f}s/ex)",
                    flush=True,
                )
    finally:
        if predictions_file is not None:
            predictions_file.close()

    corpus_cer = total_edits / max(total_reference, 1)
    elapsed = time.time() - started
    print()
    print(f"examples:          {len(records)}")
    print(f"corpus CER:        {corpus_cer:.4f} ({corpus_cer * 100:.2f}%)")
    print(f"sentence accuracy: {exact_matches / max(len(records), 1):.4f}")
    print(f"time:              {elapsed:.0f}s")
    print()
    print("CER by noise type:")
    for noise in sorted(noise_counts, key=lambda key: -noise_counts[key]):
        noise_cer = noise_edits[noise] / max(noise_reference[noise], 1)
        print(f"  {noise:<16} n={noise_counts[noise]:<7} CER={noise_cer * 100:.2f}%")


if __name__ == "__main__":
    main()
