"""ペルソナ別のコーパス文書ストリームを、実行時と同一のプロファイルビルダー
へ時系列に流し、段階B学習用のレコードを生成する CLI (docs/PROFILE.md §5)。

各ペルソナ(エンジニア型 / 小説執筆型 / Wikipedia 型...)ごとに独立した仮想
ユーザー(``ProfileBuilder`` を u_0 から開始)を1本走らせ、
``user_profile.from_corpus.stream_snapshots`` で得られる
``(snapshot(u_{m-1}), s_m)`` のペアを、既存のローマ字合成器
(:class:`dataset.generate.DatasetGenerator`)+ typo ノイズで打鍵列 x に変換
する。複数ペルソナの出力は1つの jsonl に混ぜて書き出せる(PROFILE.md §5 の
E_π = ペルソナ分布上の期待値、に対応)。

出力レコード(1行1件)::

    {"persona": ..., "input": x, "context": c, "profile": u_{m-1}, "target": s_m}

使い方::

    python -m dataset.profile_stream \\
        --persona engineer=docs/engineer_corpus.txt:code \\
        --persona novelist=aozora/sample.zip:prose \\
        --output data/profile_stream.jsonl --snapshot-every 1
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import random

from dataset.generate import DatasetGenerator
from dataset.source_text import clean_aozora_text
from dataset.source_text import extract_text_units
from dataset.source_text import normalize_lines
from dataset.source_text import read_source_text
from user_profile.builder import ProfileBuilder
from user_profile.from_corpus import stream_snapshots


@dataclass(frozen=True)
class PersonaSource:
    """1ペルソナ分のコーパス指定。``--persona`` 1個に対応する。"""

    name: str
    path: str
    domain: str | None = None
    format: str = "aozora"  # "aozora" | "plain"


@dataclass(frozen=True)
class ProfileStreamExample:
    persona: str
    input: str
    context: str
    profile: dict
    target: str


def parse_persona_arg(raw: str) -> PersonaSource:
    """``name=path[:domain[:format]]`` を解析する。

    ``domain`` は ``code``/``prose``/``chat`` のいずれか(省略時は
    ``ProfileBuilder`` の自動推定に任せる)。``format`` は ``aozora``(既定)
    または ``plain``。
    """
    if "=" not in raw:
        raise ValueError(
            f"invalid --persona value (expected name=path[:domain[:format]]): {raw!r}"
        )
    name, rest = raw.split("=", 1)
    parts = rest.split(":")
    path = parts[0]
    domain = parts[1] if len(parts) > 1 and parts[1] else None
    fmt = parts[2] if len(parts) > 2 and parts[2] else "aozora"
    return PersonaSource(name=name, path=path, domain=domain, format=fmt)


def load_persona_sentences(
    source: PersonaSource,
    max_units: int | None,
    min_chars: int,
    max_chars: int,
) -> list[dict]:
    """1ペルソナ分のコーパスから ``stream_snapshots`` に渡す文書辞書列を作る。"""
    raw_text = read_source_text(source.path)
    text = (
        clean_aozora_text(raw_text)
        if source.format == "aozora"
        else normalize_lines(raw_text)
    )
    units = extract_text_units(
        text, max_units=max_units, min_chars=min_chars, max_chars=max_chars
    )
    return [{"text": unit, "domain": source.domain} for unit in units]


def generate_persona_examples(
    source: PersonaSource,
    sentences: list[dict],
    generator: DatasetGenerator,
    snapshot_every: int,
    synthetic_explicit_rate: float,
    lookahead: int,
    seed: int,
    context_window: int,
) -> list[ProfileStreamExample]:
    """1ペルソナ(=1つの仮想ユーザーストリーム)分のレコードを生成する。"""
    examples: list[ProfileStreamExample] = []
    builder = ProfileBuilder()
    history: list[str] = []
    for snapshot, target in stream_snapshots(
        sentences,
        builder=builder,
        snapshot_every=snapshot_every,
        synthetic_explicit_rate=synthetic_explicit_rate,
        lookahead=lookahead,
        seed=seed,
    ):
        noisy_input, _clean_target = generator.generate_pair(target)
        context = "".join(history[-context_window:]) if context_window > 0 else ""
        examples.append(
            ProfileStreamExample(
                persona=source.name,
                input=noisy_input,
                context=context,
                profile=snapshot,
                target=target,
            )
        )
        history.append(target)
    return examples


def write_jsonl(path: Path, examples: list[ProfileStreamExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persona",
        action="append",
        required=True,
        dest="personas",
        help=(
            "name=path[:domain[:format]] (repeatable; format は aozora|plain, "
            "既定 aozora)。複数指定すると仮想ユーザーストリームを混ぜて出力する。"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument("--synthetic-explicit-rate", type=float, default=0.0)
    parser.add_argument("--lookahead", type=int, default=20)
    parser.add_argument(
        "--context-window",
        type=int,
        default=1,
        help="直前何文を context として残すか(0で無効)。",
    )
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="複数ペルソナのレコードを書き出し前にシャッフルする。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [parse_persona_arg(raw) for raw in args.personas]
    generator = DatasetGenerator(seed=args.seed)

    all_examples: list[ProfileStreamExample] = []
    for index, source in enumerate(sources):
        sentences = load_persona_sentences(
            source, args.max_units, args.min_chars, args.max_chars
        )
        examples = generate_persona_examples(
            source,
            sentences,
            generator,
            snapshot_every=args.snapshot_every,
            synthetic_explicit_rate=args.synthetic_explicit_rate,
            lookahead=args.lookahead,
            seed=args.seed + index,
            context_window=args.context_window,
        )
        all_examples.extend(examples)
        print(f"persona={source.name} sentences={len(sentences)} examples={len(examples)}")

    if args.shuffle:
        random.Random(args.seed).shuffle(all_examples)

    write_jsonl(args.output, all_examples)
    print(f"personas={len(sources)} total_examples={len(all_examples)} -> {args.output}")


if __name__ == "__main__":
    main()
