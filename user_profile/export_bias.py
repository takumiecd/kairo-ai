"""profile.json から ``profile_bias.tsv`` を書き出す CLI。

docs/PROFILE.md §3 (段階A: デコーダ側スコア融合) の実装。kairo-core は
依存ゼロで JSON を直接パースできないため、decay を焼き込んだボーナス値
(B_exp, B_imp)をここで計算し、Rust 側はできあがった TSV を読むだけにする。
数式の実装箇所はこの1ファイルに集約する(kairo/docs/12-パーソナライズ §3
の方針どおり)。

    B_exp(w) = log(1 + acc(w)) - gamma * log(1 + rej(w))
    B_imp(w) = min(log(1 + decayed_count(w)), kappa) + rho * 1[w in recency]

出力フォーマット(TAB区切り、ヘッダ行は ``#`` コメント)::

    # surface\treading\tb_exp\tb_imp
    surface1\treading1\t0.693147\t0.000000
    ...

- ``reading`` が無ければ空文字列
- ``b_exp`` と ``b_imp`` が両方 0 の行は出力しない
- 行は surface でソートし、出力を決定的にする

使い方::

    python -m user_profile.export_bias \\
        --profile profile.json \\
        --output ~/.config/kairo/profile_bias.tsv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from user_profile.builder import DEFAULT_HALF_LIFE
from user_profile.builder import ProfileBuilder
from user_profile.schema import Profile


DEFAULT_GAMMA = 1.0  # 却下ペナルティの重み (B_exp)
DEFAULT_KAPPA = 3.0  # 暗黙的シグナルのキャップ (B_imp)
DEFAULT_RHO = 0.5  # recency ボーナス (B_imp)

HEADER = "# surface\treading\tb_exp\tb_imp"

BiasRow = tuple[str, str, float, float]


def compute_explicit_bonus(profile: Profile, gamma: float) -> dict[str, tuple[float, str]]:
    """surface -> (B_exp, reading)。

    explicit エントリは (input, surface) ペアで持たれているが、B_exp(w) は
    surface だけで識別される単語 w のボーナスなので、同じ surface を持つ
    複数エントリの accept/reject は単純合算する。reading 列は最初に出現した
    ものを代表として使う(どの入力から来たかは表示上の参考情報でしかない)。
    """
    acc: dict[str, int] = {}
    rej: dict[str, int] = {}
    reading_of: dict[str, str] = {}
    for entry in profile.explicit:
        acc[entry.surface] = acc.get(entry.surface, 0) + entry.accept_count
        rej[entry.surface] = rej.get(entry.surface, 0) + entry.reject_count
        reading_of.setdefault(entry.surface, entry.input)

    bonus: dict[str, tuple[float, str]] = {}
    for surface in acc.keys() | rej.keys():
        b_exp = math.log1p(acc.get(surface, 0)) - gamma * math.log1p(rej.get(surface, 0))
        bonus[surface] = (b_exp, reading_of.get(surface, ""))
    return bonus


def compute_implicit_bonus(
    profile: Profile,
    builder: ProfileBuilder,
    kappa: float,
    rho: float,
) -> dict[str, tuple[float, str]]:
    """surface -> (B_imp, reading)。decay は ``builder.decayed_count`` を使う。"""
    recency_set = set(profile.recency)
    bonus: dict[str, tuple[float, str]] = {}
    for surface, entry in profile.unigram.items():
        decayed = builder.decayed_count(surface)
        b_imp = min(math.log1p(decayed), kappa)
        if surface in recency_set:
            b_imp += rho
        bonus[surface] = (b_imp, entry.reading or "")
    return bonus


def export_bias(
    profile: Profile,
    gamma: float = DEFAULT_GAMMA,
    kappa: float = DEFAULT_KAPPA,
    rho: float = DEFAULT_RHO,
    half_life: int = DEFAULT_HALF_LIFE,
) -> list[BiasRow]:
    """PROFILE.md §3 の式で (surface, reading, b_exp, b_imp) の行を作る。

    ``half_life`` は export 時点の ``profile.meta.total_units`` を N として
    lazy decay を焼き込むために ``ProfileBuilder.decayed_count`` へ渡す
    (builder と export で decay 式がずれないようにするため、実装は
    :mod:`user_profile.builder` の一箇所に留める)。
    """
    builder = ProfileBuilder(profile=profile, half_life=half_life)
    explicit_bonus = compute_explicit_bonus(profile, gamma)
    implicit_bonus = compute_implicit_bonus(profile, builder, kappa, rho)

    rows: list[BiasRow] = []
    for surface in explicit_bonus.keys() | implicit_bonus.keys():
        b_exp, exp_reading = explicit_bonus.get(surface, (0.0, ""))
        b_imp, imp_reading = implicit_bonus.get(surface, (0.0, ""))
        if b_exp == 0.0 and b_imp == 0.0:
            continue
        reading = imp_reading or exp_reading
        rows.append((surface, reading, b_exp, b_imp))
    rows.sort(key=lambda row: row[0])
    return rows


def format_tsv(rows: list[BiasRow]) -> str:
    lines = [HEADER]
    for surface, reading, b_exp, b_imp in rows:
        lines.append(f"{surface}\t{reading}\t{b_exp:.6f}\t{b_imp:.6f}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="Input profile JSON.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination profile_bias.tsv."
    )
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--half-life", type=int, default=DEFAULT_HALF_LIFE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = Profile.load_json(args.profile.expanduser())
    rows = export_bias(
        profile,
        gamma=args.gamma,
        kappa=args.kappa,
        rho=args.rho,
        half_life=args.half_life,
    )
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_tsv(rows), encoding="utf-8")
    print(f"rows={len(rows)} -> {output}")


if __name__ == "__main__":
    main()
