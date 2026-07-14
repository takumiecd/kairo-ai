"""feedback.jsonl から実行時プロファイルを構築する CLI。

``docs/FEEDBACK_SCHEMA.md`` の確定イベントスキーマに従い、
``docs/PROFILE.md`` §5 の「実行時」経路として :class:`ProfileBuilder` へ
イベントを流す。

使い方::

    python -m user_profile.from_feedback \\
        --input ~/.config/kairo/feedback.jsonl \\
        --output profile.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset.source_feedback import load_feedback_events
from user_profile.builder import DEFAULT_HALF_LIFE
from user_profile.builder import DEFAULT_RECENCY_SIZE
from user_profile.builder import DEFAULT_UNIGRAM_CAP
from user_profile.builder import ProfileBuilder
from user_profile.schema import Profile


def apply_events(builder: ProfileBuilder, events: list[dict]) -> None:
    """FEEDBACK_SCHEMA.md のイベント列を順にビルダーへ適用する。

    - ``accepted`` が偽のイベントは却下イベントとして扱う。
    - accepted な確定は、実際にユーザーが確定したテキストなので確定
      イベントとして implicit (unigram/recency/domain/lang/N) を更新する。
    - ``candidate_rank > 0`` の確定は、それに加えて明示的な修正シグナル
      として explicit にも計上する(タスク仕様の指示どおり)。
    """
    for event in events:
        input_text = str(event.get("input", "")).strip()
        output_text = str(event.get("output", "")).strip()
        if not output_text:
            continue

        if not event.get("accepted", True):
            builder.apply_rejection(input_text, output_text)
            continue

        builder.apply_commit(output_text, reading=input_text or None)
        if int(event.get("candidate_rank", 0) or 0) > 0:
            builder.apply_correction(input_text, output_text)


def build_profile_from_feedback(
    paths: list[Path],
    unigram_cap: int = DEFAULT_UNIGRAM_CAP,
    recency_size: int = DEFAULT_RECENCY_SIZE,
    half_life: int = DEFAULT_HALF_LIFE,
) -> Profile:
    events = load_feedback_events(paths)
    builder = ProfileBuilder(
        unigram_cap=unigram_cap,
        recency_size=recency_size,
        half_life=half_life,
    )
    apply_events(builder, events)
    return builder.profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more feedback.jsonl files (e.g. ~/.config/kairo/feedback.jsonl).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination profile JSON.")
    parser.add_argument("--unigram-cap", type=int, default=DEFAULT_UNIGRAM_CAP)
    parser.add_argument("--recency-size", type=int, default=DEFAULT_RECENCY_SIZE)
    parser.add_argument("--half-life", type=int, default=DEFAULT_HALF_LIFE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = build_profile_from_feedback(
        list(args.input),
        unigram_cap=args.unigram_cap,
        recency_size=args.recency_size,
        half_life=args.half_life,
    )
    profile.save_json(args.output)
    print(
        f"total_units={profile.meta.total_units} "
        f"unigrams={len(profile.unigram)} explicit={len(profile.explicit)} -> {args.output}"
    )


if __name__ == "__main__":
    main()
