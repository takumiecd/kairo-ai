"""Build training JSONL from external text sources."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
from urllib.request import urlopen
import zipfile

from dataset.generate import DatasetGenerator
from dataset.generate import TrainingExample
from dataset.generate import write_jsonl


AOZORA_RUBY_RE = re.compile(r"《[^》]+》")
AOZORA_ANNOTATION_RE = re.compile(r"［＃[^］]+］")
HEADER_SEPARATOR = "-------------------------------------------------------"
FOOTER_MARKERS = ("底本：", "入力：", "校正：")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？])")


def read_source_text(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        with urlopen(path_or_url) as response:
            payload = response.read()
        return decode_payload(path_or_url, payload)

    path = Path(path_or_url)
    payload = path.read_bytes()
    return decode_payload(str(path), payload)


def decode_payload(name: str, payload: bytes) -> str:
    if name.endswith(".zip"):
        with zipfile.ZipFile(PathBytes(payload)) as archive:
            text_names = [
                filename
                for filename in archive.namelist()
                if filename.lower().endswith(".txt")
            ]
            if not text_names:
                raise ValueError(f"zip source has no .txt file: {name}")
            payload = archive.read(text_names[0])

    for encoding in ("utf-8-sig", "shift_jis", "cp932"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


class PathBytes:
    """Small file-like wrapper for reading zip payloads from memory."""

    def __init__(self, payload: bytes) -> None:
        from io import BytesIO

        self._buffer = BytesIO(payload)

    def seek(self, *args):
        return self._buffer.seek(*args)

    def tell(self):
        return self._buffer.tell()

    def read(self, *args):
        return self._buffer.read(*args)

    def seekable(self):
        return True


def clean_aozora_text(text: str) -> str:
    text = strip_aozora_header(text)
    text = strip_aozora_footer(text)
    text = AOZORA_RUBY_RE.sub("", text)
    text = AOZORA_ANNOTATION_RE.sub("", text)
    text = text.replace("｜", "")
    return normalize_lines(text)


def strip_aozora_header(text: str) -> str:
    parts = text.split(HEADER_SEPARATOR)
    if len(parts) >= 3:
        return HEADER_SEPARATOR.join(parts[2:])
    return text


def strip_aozora_footer(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if any(line.startswith(marker) for marker in FOOTER_MARKERS):
            return "\n".join(lines[:index])
    return text


def normalize_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_text_units(
    text: str,
    max_units: int | None = None,
    min_chars: int = 8,
    max_chars: int = 80,
) -> list[str]:
    units: list[str] = []
    for line in text.splitlines():
        for sentence in SENTENCE_SPLIT_RE.split(line):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) < min_chars:
                continue
            if len(sentence) > max_chars:
                for start in range(0, len(sentence), max_chars):
                    chunk = sentence[start : start + max_chars]
                    if len(chunk) >= min_chars:
                        units.append(chunk)
            else:
                units.append(sentence)
            if max_units is not None and len(units) >= max_units:
                return units
    return units


def generate_from_text_units(
    text_units: list[str],
    augmentations: int,
    source: str,
    source_detail: str,
    license_name: str,
    seed: int,
) -> list[TrainingExample]:
    generator = DatasetGenerator(seed=seed)
    return generator.generate_examples(
        text_units,
        num_augmentations=augmentations,
        source=source,
        source_detail=source_detail,
        license_name=license_name,
    )


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="URL or local text/zip path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-name", default="external")
    parser.add_argument("--source-detail", default=None)
    parser.add_argument("--license", dest="license_name", default="external_terms")
    parser.add_argument("--format", choices=["plain", "aozora"], default="aozora")
    parser.add_argument("--augmentations", type=int, default=2)
    parser.add_argument("--max-units", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_text = read_source_text(args.source)
    text = clean_aozora_text(raw_text) if args.format == "aozora" else normalize_lines(raw_text)
    text_units = extract_text_units(
        text,
        max_units=args.max_units,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    source_detail = args.source_detail or args.source
    examples = generate_from_text_units(
        text_units,
        augmentations=args.augmentations,
        source=args.source_name,
        source_detail=source_detail,
        license_name=args.license_name,
        seed=args.seed,
    )
    write_jsonl(args.output, examples)
    if args.manifest:
        write_manifest(
            args.manifest,
            {
                "source": args.source,
                "source_name": args.source_name,
                "source_detail": source_detail,
                "license": args.license_name,
                "format": args.format,
                "text_units": len(text_units),
                "examples": len(examples),
            },
        )
    print(f"text_units={len(text_units)} examples={len(examples)}")


if __name__ == "__main__":
    main()
