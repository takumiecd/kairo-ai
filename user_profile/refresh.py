"""feedback.jsonl -> profile.json -> profile_bias.tsv を一発で通すワンショット更新 CLI。

launchd 等の定期実行(15分間隔を想定)から呼ばれる前提の配線。
``python -m user_profile.from_feedback`` + ``python -m user_profile.export_bias``
の2段を毎回フル実行する代わりに、このモジュールは:

- ``profile.json`` が既にあれば、``feedback.jsonl`` の「前回処理済みバイト
  オフセット」以降だけを増分適用する(``feedback.jsonl`` は append-only な
  ので、既に処理した先頭部分を読み直す必要がない)。オフセットは
  ``profile.json`` の ``meta.source_offsets`` に記録する
  (:mod:`user_profile.schema` 参照)。
- ``profile.json`` が存在しない、または壊れている(パース不能)場合は
  フルリビルド(オフセット0からの適用)にフォールバックする。
- 出力(``profile.json`` と ``profile_bias.tsv`` の両方)はテンポラリ
  ファイル -> ``os.replace()`` によるアトミック置換で書く。IME 側
  (``kairo-core`` の ``ProfileBias::load_standard``)が読んでいる最中に
  壊れかけの内容を見せないため。

使い方::

    python -m user_profile.refresh
    python -m user_profile.refresh --config-dir /tmp/kairo-config-test
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from user_profile.builder import DEFAULT_HALF_LIFE
from user_profile.builder import DEFAULT_RECENCY_SIZE
from user_profile.builder import DEFAULT_UNIGRAM_CAP
from user_profile.builder import ProfileBuilder
from user_profile.export_bias import DEFAULT_GAMMA
from user_profile.export_bias import DEFAULT_KAPPA
from user_profile.export_bias import DEFAULT_RHO
from user_profile.export_bias import export_bias
from user_profile.export_bias import format_tsv
from user_profile.from_feedback import apply_events
from user_profile.schema import Profile


# profile.json のパース/構造不正として扱い、フルリビルドへフォールバックする例外群。
_CORRUPT_PROFILE_ERRORS = (json.JSONDecodeError, KeyError, OSError, AttributeError, TypeError, ValueError)


def default_config_dir() -> Path:
    """``$XDG_CONFIG_HOME/kairo`` (設定されていれば)、なければ ``~/.config/kairo``。

    kairo-core 側 (``crates/kairo-core/src/dictionary.rs`` の
    ``config_file_path``)と同じ規則に揃えてある。
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "kairo"
    return Path.home() / ".config" / "kairo"


def _read_new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """``path`` の ``offset`` バイト以降にある、完全な行だけを返す。

    書き込み途中(末尾に改行がない)行は消費せず次回に持ち越す。戻り値は
    ``(新規行のリスト, 今回消費し終えたオフセット)``。
    """
    if not path.exists():
        return [], offset
    size = path.stat().st_size
    if size <= offset:
        # ファイルが縮んでいる(オフセットより小さい)= 別ファイルに入れ替わった
        # 可能性がある。安全側に倒して最初から読み直す。
        if size < offset:
            offset = 0
        else:
            return [], offset

    with path.open("rb") as file:
        file.seek(offset)
        chunk = file.read()
    if not chunk:
        return [], offset

    if chunk.endswith(b"\n"):
        consumed = offset + len(chunk)
        text = chunk.decode("utf-8", errors="replace")
        lines = text.splitlines()
    else:
        last_newline = chunk.rfind(b"\n")
        if last_newline == -1:
            return [], offset  # 完全な行がまだ1つもない
        consumed = offset + last_newline + 1
        text = chunk[:last_newline].decode("utf-8", errors="replace")
        lines = text.splitlines()
    return lines, consumed


def _parse_events(lines: list[str]) -> list[dict]:
    events: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """一時ファイルへ書いてから ``os.replace`` で置き換える(同一ディレクトリ内、
    同一ファイルシステム上なので ``os.replace`` はアトミック)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _load_existing_profile(path: Path) -> Profile | None:
    if not path.exists():
        return None
    try:
        return Profile.load_json(path)
    except _CORRUPT_PROFILE_ERRORS:
        return None


def refresh(
    config_dir: Path,
    unigram_cap: int = DEFAULT_UNIGRAM_CAP,
    recency_size: int = DEFAULT_RECENCY_SIZE,
    half_life: int = DEFAULT_HALF_LIFE,
    gamma: float = DEFAULT_GAMMA,
    kappa: float = DEFAULT_KAPPA,
    rho: float = DEFAULT_RHO,
) -> dict:
    """``config_dir`` 配下の feedback.jsonl -> profile.json -> profile_bias.tsv を更新する。

    戻り値は呼び出し元向けの統計 dict(CLI の出力・テストの両方で使う)。
    """
    feedback_path = config_dir / "feedback.jsonl"
    profile_path = config_dir / "profile.json"
    bias_path = config_dir / "profile_bias.tsv"

    existing_profile = _load_existing_profile(profile_path)

    offset = 0
    if existing_profile is not None:
        offset = int(existing_profile.meta.source_offsets.get(str(feedback_path), 0))

    lines, new_offset = _read_new_lines(feedback_path, offset)
    new_events = _parse_events(lines)

    builder = ProfileBuilder(
        profile=existing_profile,
        unigram_cap=unigram_cap,
        recency_size=recency_size,
        half_life=half_life,
    )
    apply_events(builder, new_events)
    builder.profile.meta.source_offsets[str(feedback_path)] = new_offset

    profile_json = json.dumps(builder.profile.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_bytes(profile_path, profile_json.encode("utf-8"))

    rows = export_bias(builder.profile, gamma=gamma, kappa=kappa, rho=rho, half_life=half_life)
    _atomic_write_bytes(bias_path, format_tsv(rows).encode("utf-8"))

    return {
        "config_dir": str(config_dir),
        "new_events": len(new_events),
        "total_units": builder.profile.meta.total_units,
        "unigrams": len(builder.profile.unigram),
        "explicit": len(builder.profile.explicit),
        "bias_rows": len(rows),
        "incremental": existing_profile is not None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="kairo config dir (default: $XDG_CONFIG_HOME/kairo or ~/.config/kairo).",
    )
    parser.add_argument("--unigram-cap", type=int, default=DEFAULT_UNIGRAM_CAP)
    parser.add_argument("--recency-size", type=int, default=DEFAULT_RECENCY_SIZE)
    parser.add_argument("--half-life", type=int, default=DEFAULT_HALF_LIFE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = (args.config_dir or default_config_dir()).expanduser()
    stats = refresh(
        config_dir,
        unigram_cap=args.unigram_cap,
        recency_size=args.recency_size,
        half_life=args.half_life,
        gamma=args.gamma,
        kappa=args.kappa,
        rho=args.rho,
    )
    mode = "incremental" if stats["incremental"] else "full-rebuild"
    print(
        f"[{mode}] new_events={stats['new_events']} "
        f"total_units={stats['total_units']} unigrams={stats['unigrams']} "
        f"explicit={stats['explicit']} bias_rows={stats['bias_rows']} "
        f"-> {config_dir}"
    )


if __name__ == "__main__":
    main()
