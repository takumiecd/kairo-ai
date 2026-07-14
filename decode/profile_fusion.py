"""段階A: デコーダ側スコア融合 — トライ融合 (docs/PROFILE.md §3)。

$$\\Phi(y_{1:u}) = \\sum_{w \\in \\mathrm{Words}(y_{1:u})} B(w) + \\beta \\cdot \\mathrm{TrieDepth}(\\mathrm{suffix}(y_{1:u}))$$

語ごとのボーナス B(w) = lambda_exp * B_exp(w) + lambda_imp * B_imp(w) は
``user_profile.export_bias`` の ``compute_explicit_bonus`` /
``compute_implicit_bonus`` をそのまま再利用する(数式の実装箇所を1つに
留めるという export_bias のドキュメント方針を踏襲。ここでは既存関数の
シグネチャは一切変更せず、呼び出すだけ)。

ビームサーチ側は ``ProfileFusion.initial_state()`` で状態を作り、1文字
進むごとに ``ProfileFusion.step(state, char)`` を呼んで
``(new_state, delta)`` を得る。``delta`` がそのまま
``Phi(y ++ char) - Phi(y)`` であり、これを log prob に加算すればよい
(beam.py 側の実装)。

## β (トライ内部ノードの部分ボーナス) の設計

PROFILE.md は「現パス上で到達可能な最大 B(w) を深さで按分」という按分方式と
「シンプルに β·depth 固定」という単純方式の両方を許容している。ここでは
**固定 β·depth 方式**を採用した(理由は末尾コメント/報告を参照): 按分方式は
トライの各ノードに「部分木内で最大の B(w) を持つ完成語とその語長」を
事前計算しておく必要があり、実装・検証コストの割に効果が不確かなため、
まずシンプルな方式で段階Aの効果を測る。将来按分方式に切り替える場合は
``ProfileFusion._partial_bonus`` だけを差し替えればよい設計にしてある。
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from user_profile.builder import DEFAULT_HALF_LIFE
from user_profile.builder import ProfileBuilder
from user_profile.export_bias import DEFAULT_GAMMA
from user_profile.export_bias import DEFAULT_KAPPA
from user_profile.export_bias import DEFAULT_RHO
from user_profile.export_bias import compute_explicit_bonus
from user_profile.export_bias import compute_implicit_bonus
from user_profile.schema import Profile


DEFAULT_LAMBDA_EXP = 2.0
DEFAULT_LAMBDA_IMP = 1.0
DEFAULT_BETA = 0.15  # トライ内部ノードの部分ボーナス係数 (固定 depth 方式)
DEFAULT_IMPLICIT_TOP_K = 2000  # 暗黙的シグナルの語彙上限 (プロファイル語彙 = explicit + implicit top-K)


class TrieNode:
    """プロファイル語彙のトライの1ノード。"""

    __slots__ = ("children", "bonus", "depth")

    def __init__(self, depth: int) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.bonus: float | None = None  # このノードで語が完成する場合の B(w)
        self.depth = depth


def build_trie(word_bonus: dict[str, float]) -> TrieNode:
    """surface -> B(w) の辞書からトライを構築する。"""
    root = TrieNode(depth=0)
    for word, bonus in word_bonus.items():
        if not word:
            continue
        node = root
        for char in word:
            child = node.children.get(char)
            if child is None:
                child = TrieNode(depth=node.depth + 1)
                node.children[char] = child
            node = child
        node.bonus = bonus
    return root


def build_word_bonus(
    profile: Profile,
    lambda_exp: float = DEFAULT_LAMBDA_EXP,
    lambda_imp: float = DEFAULT_LAMBDA_IMP,
    gamma: float = DEFAULT_GAMMA,
    kappa: float = DEFAULT_KAPPA,
    rho: float = DEFAULT_RHO,
    half_life: int = DEFAULT_HALF_LIFE,
    implicit_top_k: int = DEFAULT_IMPLICIT_TOP_K,
) -> dict[str, float]:
    """プロファイルから surface -> B(w) の辞書を作る。

    語彙は explicit 全件 + implicit の上位 K 件(decayed count 順、
    ``export_bias.compute_implicit_bonus`` が返す b_imp で近似)の合併集合。
    B_exp/B_imp の計算そのものは ``user_profile.export_bias`` の既存関数を
    呼ぶだけで、ロジックは複製しない。
    """
    builder = ProfileBuilder(profile=profile, half_life=half_life)
    explicit_bonus = compute_explicit_bonus(profile, gamma)
    implicit_bonus_all = compute_implicit_bonus(profile, builder, kappa, rho)

    top_implicit_surfaces = {
        surface
        for surface, _ in sorted(
            implicit_bonus_all.items(),
            key=lambda item: item[1][0],
            reverse=True,
        )[:implicit_top_k]
    }

    word_bonus: dict[str, float] = {}
    surfaces = set(explicit_bonus) | top_implicit_surfaces
    for surface in surfaces:
        b_exp, _ = explicit_bonus.get(surface, (0.0, ""))
        b_imp = 0.0
        if surface in top_implicit_surfaces:
            b_imp, _ = implicit_bonus_all.get(surface, (0.0, ""))
        combined = lambda_exp * b_exp + lambda_imp * b_imp
        if combined != 0.0:
            word_bonus[surface] = combined
    return word_bonus


@dataclass(frozen=True)
class ProfileFusionState:
    """ビーム仮説1つ分のトライ融合状態。

    ``active`` はその時点で「まだ外れていない」トライ上の到達ノード集合
    (東京/東京都のような並行パスを、ノード単位で複数保持する)。
    ``total_word_bonus`` はこれまでに完成した語のボーナスの累積和。
    ``potential`` はその2つから決まる Phi(y_{1:u}) の値そのもの(キャッシュ)。
    """

    active: frozenset[TrieNode] = field(default_factory=frozenset)
    total_word_bonus: float = 0.0
    potential: float = 0.0


_EMPTY_STATE = ProfileFusionState()


class ProfileFusion:
    """トライ融合のポテンシャル関数 Phi とその差分を計算するクラス。"""

    def __init__(self, root: TrieNode, beta: float = DEFAULT_BETA) -> None:
        self.root = root
        self.beta = beta

    @staticmethod
    def from_word_bonus(word_bonus: dict[str, float], beta: float = DEFAULT_BETA) -> "ProfileFusion":
        return ProfileFusion(build_trie(word_bonus), beta=beta)

    @staticmethod
    def from_profile(
        profile: Profile,
        lambda_exp: float = DEFAULT_LAMBDA_EXP,
        lambda_imp: float = DEFAULT_LAMBDA_IMP,
        gamma: float = DEFAULT_GAMMA,
        kappa: float = DEFAULT_KAPPA,
        rho: float = DEFAULT_RHO,
        half_life: int = DEFAULT_HALF_LIFE,
        implicit_top_k: int = DEFAULT_IMPLICIT_TOP_K,
        beta: float = DEFAULT_BETA,
    ) -> "ProfileFusion":
        word_bonus = build_word_bonus(
            profile,
            lambda_exp=lambda_exp,
            lambda_imp=lambda_imp,
            gamma=gamma,
            kappa=kappa,
            rho=rho,
            half_life=half_life,
            implicit_top_k=implicit_top_k,
        )
        return ProfileFusion.from_word_bonus(word_bonus, beta=beta)

    def initial_state(self) -> ProfileFusionState:
        return _EMPTY_STATE

    def potential(self, state: ProfileFusionState) -> float:
        return state.potential

    def step(self, state: ProfileFusionState, char: str) -> tuple[ProfileFusionState, float]:
        """1文字 ``char`` を追加したときの (新しい状態, Phi の差分) を返す。

        候補となる開始点は、現在アクティブな各ノードに加えて常に root
        (=この文字位置から新しく語を始める場合)。子ノードへ進めなかった
        アクティブパスはここで暗黙的に脱落する(部分ボーナスの自動回収)。
        """
        candidates = state.active | {self.root}
        new_active: set[TrieNode] = set()
        gained_bonus = 0.0
        for node in candidates:
            child = node.children.get(char)
            if child is None:
                continue
            new_active.add(child)
            if child.bonus is not None:
                gained_bonus += child.bonus

        new_total_bonus = state.total_word_bonus + gained_bonus
        depth = max((node.depth for node in new_active), default=0)
        new_potential = new_total_bonus + self.beta * depth

        new_state = ProfileFusionState(
            active=frozenset(new_active),
            total_word_bonus=new_total_bonus,
            potential=new_potential,
        )
        delta = new_potential - state.potential
        return new_state, delta
