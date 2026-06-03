"""Ingest multiple external text sources from a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from dataset.source_text import clean_aozora_text
from dataset.source_text import extract_text_units
from dataset.source_text import generate_from_text_units
from dataset.source_text import normalize_lines
from dataset.source_text import read_source_text
from dataset.source_text import write_manifest
from dataset.source_text import write_jsonl


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    slug = slug.strip("_")
    return slug or "source"


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        payload = payload.get("sources", [])
    if not isinstance(payload, list):
        raise ValueError("manifest must be a list or an object with a 'sources' list")
    return payload


def ingest_manifest(
    manifest_path: Path,
    output_dir: Path,
    combined_output: Path | None,
    default_augmentations: int,
    default_max_units: int,
    seed: int,
) -> list[dict]:
    sources = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_examples = []
    summaries = []
    for index, source in enumerate(sources):
        source_url = source["source"]
        source_name = source.get("source_name", "aozora")
        source_detail = source.get("source_detail", source_url)
        license_name = source.get("license", "external_terms")
        source_format = source.get("format", "aozora")
        augmentations = int(source.get("augmentations", default_augmentations))
        max_units = int(source.get("max_units", default_max_units))
        min_chars = int(source.get("min_chars", 8))
        max_chars = int(source.get("max_chars", 80))
        output_name = source.get("output_name") or slugify(source_detail)

        raw_text = read_source_text(source_url)
        text = (
            clean_aozora_text(raw_text)
            if source_format == "aozora"
            else normalize_lines(raw_text)
        )
        text_units = extract_text_units(
            text,
            max_units=max_units,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        examples = generate_from_text_units(
            text_units,
            augmentations=augmentations,
            source=source_name,
            source_detail=source_detail,
            license_name=license_name,
            seed=seed + index,
        )

        jsonl_path = output_dir / f"{output_name}.jsonl"
        source_manifest_path = output_dir / f"{output_name}.manifest.json"
        write_jsonl(jsonl_path, examples)
        summary = {
            "source": source_url,
            "source_name": source_name,
            "source_detail": source_detail,
            "license": license_name,
            "format": source_format,
            "text_units": len(text_units),
            "examples": len(examples),
            "output": str(jsonl_path),
        }
        write_manifest(source_manifest_path, summary)
        summaries.append(summary)
        all_examples.extend(examples)
        print(
            f"{output_name}: text_units={len(text_units)} "
            f"examples={len(examples)}"
        )

    if combined_output is not None:
        write_jsonl(combined_output, all_examples)
        print(f"combined={len(all_examples)} output={combined_output}")

    write_manifest(
        output_dir / "sources.manifest.json",
        {
            "manifest": str(manifest_path),
            "sources": summaries,
            "combined_output": str(combined_output) if combined_output else None,
            "examples": len(all_examples),
        },
    )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, default=None)
    parser.add_argument("--augmentations", type=int, default=2)
    parser.add_argument("--max-units", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        combined_output=args.combined_output,
        default_augmentations=args.augmentations,
        default_max_units=args.max_units,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
