"""Build training JSONL from Japanese Wikipedia XML dumps."""

from __future__ import annotations

import argparse
import bz2
import contextlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

from dataset.generate import DatasetGenerator
from dataset.generate import TrainingExample
from dataset.generate import write_jsonl
from dataset.source_text import extract_text_units
from dataset.source_text import normalize_lines
from dataset.source_text import write_manifest


TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^/]*/>", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
FILE_LINK_RE = re.compile(r"\[\[(?:ファイル|File|画像|Image):[^\]]+\]\]", re.IGNORECASE)
CATEGORY_LINK_RE = re.compile(r"\[\[(?:Category|カテゴリ):[^\]]+\]\]", re.IGNORECASE)
INTERNAL_LINK_RE = re.compile(r"\[\[([^|\]]*\|)?([^\]]+)\]\]")
EXTERNAL_LINK_RE = re.compile(r"\[(?:https?|ftp)://[^\s\]]+(?:\s+([^\]]+))?\]")
HEADING_RE = re.compile(r"^=+.*=+$")
TABLE_LINE_PREFIXES = ("{|", "|}", "|-", "|", "!")
REDIRECT_PREFIXES = ("#REDIRECT", "#転送")


@contextlib.contextmanager
def open_dump(path_or_url: str):
    if path_or_url.startswith(("http://", "https://")):
        response = urlopen(path_or_url)
        try:
            if path_or_url.endswith(".bz2"):
                with bz2.BZ2File(response) as file:
                    yield file
            else:
                yield response
        finally:
            response.close()
        return

    path = Path(path_or_url)
    if path.suffix == ".bz2":
        with bz2.open(path, "rb") as file:
            yield file
    else:
        with path.open("rb") as file:
            yield file


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def iter_wikipedia_pages(path_or_url: str):
    with open_dump(path_or_url) as file:
        context = ET.iterparse(file, events=("end",))
        for _event, elem in context:
            if strip_namespace(elem.tag) != "page":
                continue

            title = ""
            namespace = "0"
            text = ""
            redirect = False
            for child in elem:
                child_tag = strip_namespace(child.tag)
                if child_tag == "title":
                    title = child.text or ""
                elif child_tag == "ns":
                    namespace = child.text or "0"
                elif child_tag == "redirect":
                    redirect = True
                elif child_tag == "revision":
                    for revision_child in child:
                        if strip_namespace(revision_child.tag) == "text":
                            text = revision_child.text or ""
                            break

            elem.clear()
            if redirect or namespace != "0" or not text:
                continue
            yield title, text


def clean_wikipedia_text(text: str) -> str:
    if text.lstrip().startswith(REDIRECT_PREFIXES):
        return ""

    previous = None
    while previous != text:
        previous = text
        text = TEMPLATE_RE.sub("", text)

    text = REF_RE.sub("", text)
    text = FILE_LINK_RE.sub("", text)
    text = CATEGORY_LINK_RE.sub("", text)
    text = EXTERNAL_LINK_RE.sub(lambda match: match.group(1) or "", text)
    text = INTERNAL_LINK_RE.sub(lambda match: match.group(2), text)
    text = TAG_RE.sub("", text)
    text = text.replace("'''", "").replace("''", "")

    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if HEADING_RE.match(line):
            continue
        if line.startswith(TABLE_LINE_PREFIXES):
            continue
        lines.append(line)

    return normalize_lines("\n".join(lines))


def extract_wikipedia_units(
    dump: str,
    max_units: int,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    units: list[str] = []
    for _title, raw_text in iter_wikipedia_pages(dump):
        cleaned = clean_wikipedia_text(raw_text)
        if not cleaned:
            continue
        remaining = max_units - len(units)
        units.extend(
            extract_text_units(
                cleaned,
                max_units=remaining,
                min_chars=min_chars,
                max_chars=max_chars,
            )
        )
        if len(units) >= max_units:
            break
    return units


def generate_wikipedia_examples(
    text_units: list[str],
    augmentations: int,
    source_detail: str,
    license_name: str,
    seed: int,
) -> list[TrainingExample]:
    generator = DatasetGenerator(seed=seed)
    return generator.generate_examples(
        text_units,
        num_augmentations=augmentations,
        source="wikipedia",
        source_detail=source_detail,
        license_name=license_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="Wikipedia XML dump URL/path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-detail", default="Japanese Wikipedia dump")
    parser.add_argument("--license", dest="license_name", default="cc_by_sa_gfdl")
    parser.add_argument("--max-units", type=int, default=100000)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument("--augmentations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text_units = extract_wikipedia_units(
        args.dump,
        max_units=args.max_units,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    examples = generate_wikipedia_examples(
        text_units,
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
                "source": args.dump,
                "source_name": "wikipedia",
                "source_detail": args.source_detail,
                "license": args.license_name,
                "text_units": len(text_units),
                "examples": len(examples),
            },
        )
    print(f"text_units={len(text_units)} examples={len(examples)}")


if __name__ == "__main__":
    main()
