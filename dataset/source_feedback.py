"""Turn confirmed IME feedback into RNN-T training records.

Reads one or more ``feedback.jsonl`` files (the canonical event log written by
the Kairo IME -- see the feedback schema in ``docs/FEEDBACK_SCHEMA.md``) and
emits ``{"input": <romaji>, "target": <japanese>}`` records ready for
``train.rnnt`` (and LoRA fine-tuning via ``train.rnnt.lora``).

Aggregation:
- pairs are deduped on ``(input, output)`` and counted;
- pairs seen fewer than ``--min-count`` times are dropped as noise;
- a confirmation with ``candidate_rank > 0`` means the user rejected the model's
  top guess and picked another candidate -- a strong correction signal. Such
  pairs bypass the ``--min-count`` floor and are emitted ``--repeat-corrections``
  times so the fine-tune weights them more heavily.

This is the shared ingest for both local-only fine-tuning and any future server
that aggregates the same schema: the input format is identical.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PairStats:
    count: int = 0
    corrections: int = 0  # confirmations where candidate_rank > 0


def load_feedback_events(paths: list[Path]) -> list[dict]:
    """Read canonical feedback events from JSONL files, skipping bad lines."""
    events: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def aggregate(events: list[dict]) -> dict[tuple[str, str], PairStats]:
    """Aggregate events into per-(input, output) statistics."""
    stats: dict[tuple[str, str], PairStats] = defaultdict(PairStats)
    for event in events:
        if not event.get("accepted", True):
            continue
        input_text = str(event.get("input", "")).strip()
        output_text = str(event.get("output", "")).strip()
        if not input_text or not output_text:
            continue
        entry = stats[(input_text, output_text)]
        entry.count += 1
        if int(event.get("candidate_rank", 0) or 0) > 0:
            entry.corrections += 1
    return stats


def build_records(
    stats: dict[tuple[str, str], PairStats],
    min_count: int,
    repeat_corrections: int,
) -> list[dict[str, str]]:
    """Build training records, dropping rare pairs and upweighting corrections."""
    records: list[dict[str, str]] = []
    for (input_text, output_text), entry in sorted(stats.items()):
        # Corrections are strong signals and bypass the min-count floor.
        if entry.count < min_count and entry.corrections == 0:
            continue
        repeats = 1
        if entry.corrections > 0:
            repeats = max(1, repeat_corrections)
        record = {"input": input_text, "target": output_text}
        records.extend(record.copy() for _ in range(repeats))
    return records


def write_records(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more feedback.jsonl files (e.g. ~/.config/kairo/feedback.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL of {input, target} training records.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Drop (input, target) pairs confirmed fewer than this many times.",
    )
    parser.add_argument(
        "--repeat-corrections",
        type=int,
        default=1,
        help="Emit pairs that include a candidate_rank>0 correction this many times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_feedback_events(list(args.input))
    stats = aggregate(events)
    records = build_records(
        stats,
        min_count=args.min_count,
        repeat_corrections=args.repeat_corrections,
    )
    write_records(args.output, records)
    print(
        f"events={len(events)} unique_pairs={len(stats)} "
        f"records={len(records)} -> {args.output}"
    )


if __name__ == "__main__":
    main()
