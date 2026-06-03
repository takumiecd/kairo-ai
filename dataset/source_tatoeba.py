"""Build training JSONL from Tatoeba sentence downloads."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

from dataset.generate import DatasetGenerator
from dataset.generate import TrainingExample
from dataset.generate import write_jsonl
from dataset.source_text import write_manifest


def read_payload(path_or_url: str) -> bytes:
    if path_or_url.startswith(("http://", "https://")):
        with urlopen(path_or_url) as response:
            return response.read()
    return Path(path_or_url).read_bytes()


def decode_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def read_sentences_text(path_or_url: str) -> str:
    payload = read_payload(path_or_url)
    if path_or_url.endswith((".tar.bz2", ".tar.gz", ".tgz", ".tar")):
        import io

        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and "sentences" in Path(member.name).name
            ]
            if not members:
                members = [member for member in archive.getmembers() if member.isfile()]
            if not members:
                raise ValueError(f"archive has no files: {path_or_url}")
            extracted = archive.extractfile(members[0])
            if extracted is None:
                raise ValueError(f"could not extract {members[0].name}")
            payload = extracted.read()
    return decode_text(payload)


def parse_tatoeba_sentences(
    text: str,
    lang: str = "jpn",
    max_units: int | None = None,
    min_chars: int = 4,
    max_chars: int = 120,
) -> list[str]:
    units: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            parts = line.split(",", 2)
        if len(parts) < 3:
            continue
        _sentence_id, sentence_lang, sentence = parts[0], parts[1], parts[2]
        sentence = sentence.strip()
        if sentence_lang != lang:
            continue
        if len(sentence) < min_chars or len(sentence) > max_chars:
            continue
        if sentence in seen:
            continue
        seen.add(sentence)
        units.append(sentence)
        if max_units is not None and len(units) >= max_units:
            break
    return units


def generate_tatoeba_examples(
    sentences: list[str],
    augmentations: int,
    source_detail: str,
    license_name: str,
    seed: int,
) -> list[TrainingExample]:
    generator = DatasetGenerator(seed=seed)
    return generator.generate_examples(
        sentences,
        num_augmentations=augmentations,
        source="tatoeba",
        source_detail=source_detail,
        license_name=license_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", required=True, help="Tatoeba sentences file URL/path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lang", default="jpn")
    parser.add_argument("--source-detail", default="Tatoeba Japanese sentences")
    parser.add_argument("--license", dest="license_name", default="cc_by_2_0_fr")
    parser.add_argument("--max-units", type=int, default=50000)
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument("--augmentations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = read_sentences_text(args.sentences)
    sentences = parse_tatoeba_sentences(
        text,
        lang=args.lang,
        max_units=args.max_units,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    examples = generate_tatoeba_examples(
        sentences,
        augmentations=args.augmentations,
        source_detail=args.source_detail,
        license_name=args.license_name,
        seed=args.seed,
    )
    write_jsonl(args.output, examples)
    if args.manifest:
        write_manifest(
            args.manifest,
            {
                "source": args.sentences,
                "source_name": "tatoeba",
                "source_detail": args.source_detail,
                "license": args.license_name,
                "lang": args.lang,
                "text_units": len(sentences),
                "examples": len(examples),
            },
        )
    print(f"text_units={len(sentences)} examples={len(examples)}")


if __name__ == "__main__":
    main()
