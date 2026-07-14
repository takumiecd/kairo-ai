"""コーパス文書ストリームを学習時(仮想ユーザー)プロファイルへ流す。

``docs/PROFILE.md`` §5 の仮想ユーザーストリーム学習の実装:
ペルソナから生成した文書ストリーム ``s_1, ..., s_M`` を、実行時と同一の
:class:`~user_profile.builder.ProfileBuilder` へ時系列に流し、各時点で
``u_{m-1}`` のスナップショットと次の文 ``s_m`` のペアを取り出す。
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import random

from user_profile.builder import ProfileBuilder
from user_profile.builder import split_words


CorpusItem = str | dict


def _extract_item(item: CorpusItem) -> tuple[str, str | None, str | None]:
    """ストリーム要素1つから (text, domain, reading) を取り出す。"""
    if isinstance(item, dict):
        text = item.get("text") or item.get("target") or item.get("surface") or ""
        domain = item.get("domain")
        reading = item.get("reading") or item.get("input")
        return str(text), domain, reading
    return str(item), None, None


def iter_corpus_lines(path: Path) -> Iterator[CorpusItem]:
    """jsonl (1行1レコード)またはプレーンテキスト(1行1文)を読む。"""
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            stripped = line.lstrip()
            if stripped.startswith("{"):
                try:
                    yield json.loads(line)
                    continue
                except json.JSONDecodeError:
                    pass
            yield line


def stream_snapshots(
    sentences: Iterable[CorpusItem],
    builder: ProfileBuilder | None = None,
    snapshot_every: int = 1,
    synthetic_explicit_rate: float = 0.0,
    lookahead: int = 20,
    seed: int = 0,
) -> Iterator[tuple[dict, str]]:
    """仮想ユーザーストリームを流し、各時点の (snapshot(u_{m-1}), s_m) を yield する。

    ``sentences`` は文字列、または
    ``{"text"/"target"/"surface", "domain", "reading"/"input"}`` を持つ辞書の
    イテラブル。未来の要素を先読みする必要がある(synthetic explicit)ため、
    内部で一度 ``list`` に確定させる。

    ``synthetic_explicit_rate`` > 0 のとき、直近 ``lookahead`` 文先に出現する
    語を確率的に抽出し、「過去にユーザーが修正・登録した」ことにして
    explicit へ合成注入する(PROFILE.md §5 末尾)。romaji 入力が不明なため
    ``input`` は語そのもの(またはレコードの ``reading``/``input``)を使う。
    """
    sentence_list = list(sentences)
    builder = builder if builder is not None else ProfileBuilder()
    rng = random.Random(seed)
    registered: set[str] = set()

    for index, item in enumerate(sentence_list):
        text, domain, reading = _extract_item(item)
        if not text:
            continue

        if synthetic_explicit_rate > 0:
            _inject_synthetic_explicit(
                builder,
                sentence_list,
                index,
                lookahead,
                synthetic_explicit_rate,
                rng,
                registered,
            )

        if index % snapshot_every == 0:
            yield builder.snapshot(), text

        builder.apply_commit(text, reading=reading, domain=domain)


def _inject_synthetic_explicit(
    builder: ProfileBuilder,
    sentence_list: list[CorpusItem],
    index: int,
    lookahead: int,
    rate: float,
    rng: random.Random,
    registered: set[str],
) -> None:
    future_slice = sentence_list[index + 1 : index + 1 + lookahead]
    for future_item in future_slice:
        future_text, _, future_reading = _extract_item(future_item)
        for word in split_words(future_text):
            if word in registered:
                continue
            if rng.random() < rate:
                registered.add(word)
                builder.apply_correction(future_reading or word, word)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="jsonl または プレーンテキストのコーパス")
    parser.add_argument("--output", type=Path, required=True, help="Destination profile JSON.")
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument("--synthetic-explicit-rate", type=float, default=0.0)
    parser.add_argument("--lookahead", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = list(iter_corpus_lines(args.input))
    builder = ProfileBuilder()
    snapshot_count = 0
    for _ in stream_snapshots(
        lines,
        builder=builder,
        snapshot_every=args.snapshot_every,
        synthetic_explicit_rate=args.synthetic_explicit_rate,
        lookahead=args.lookahead,
        seed=args.seed,
    ):
        snapshot_count += 1
    builder.profile.save_json(args.output)
    print(f"sentences={len(lines)} snapshots={snapshot_count} -> {args.output}")


if __name__ == "__main__":
    main()
