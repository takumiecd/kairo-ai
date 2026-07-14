"""プロファイル u を条件ベクトル e(u) へ落とす段階B条件付けエンコーダ
(docs/PROFILE.md §4)。

    e(u) = MLP([ d ; ℓ ; pool{emb(w) : w ∈ top-K(F)} ])

top-K 語の埋め込みは Prediction Network の出力語彙埋め込み(文字埋め込み)を
**再利用**する(専用の埋め込みテーブルは持たない)。そのため
:class:`ProfileEncoder` は ``embedding`` を forward の引数として都度受け取る
(``model.transducer.KairoTransducer`` が ``pred_emb``/``pred_transformer.emb``
を渡す)。

このモジュールは torch 依存の ``ProfileEncoder`` 本体に加えて、
:mod:`user_profile` のスナップショット(dict)から学習/推論バッチのテンソルを
組み立てる純粋関数群も持つ。学習ループ(``train/rnnt/profile_data.py``)と
将来の実行時経路の両方から共有する想定。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from user_profile.builder import DEFAULT_HALF_LIFE
from user_profile.schema import DOMAIN_LABELS
from user_profile.schema import Profile


DEFAULT_TOP_K = 64
LANG_KEYS: tuple[str, str] = ("ja_ratio", "en_token_rate")


def _decayed_count(count: float, last_used: int, now: int, half_life: int) -> float:
    """builder._decay と同一の式(lazy decay, PROFILE.md §2)。

    ``ProfileBuilder`` はミュータブルな状態を持つため、シリアライズ済みの
    スナップショット(dict)だけから読み出したいここでは同じ式を複製する。
    """
    elapsed = max(0, now - last_used)
    return count * (2.0 ** (-elapsed / half_life))


def select_top_k_words(
    profile: dict,
    top_k: int = DEFAULT_TOP_K,
    half_life: int = DEFAULT_HALF_LIFE,
) -> list[str]:
    """profile(スナップショット dict)から decayed count 降順で top-K の語を返す。"""
    now = profile.get("meta", {}).get("total_units", 0)
    unigram = profile.get("implicit", {}).get("unigram", {})
    scored = [
        (
            surface,
            _decayed_count(
                entry.get("count", 0.0), entry.get("last_used", 0), now, half_life
            ),
        )
        for surface, entry in unigram.items()
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [surface for surface, _score in scored[:top_k]]


def extract_domain_vector(profile: dict) -> list[float]:
    domain = profile.get("implicit", {}).get("domain", {})
    return [float(domain.get(label, 0.0)) for label in DOMAIN_LABELS]


def extract_lang_vector(profile: dict) -> list[float]:
    lang = profile.get("implicit", {}).get("lang", {})
    return [float(lang.get(key, 0.0)) for key in LANG_KEYS]


def empty_profile_snapshot() -> dict:
    """u_0(空プロファイル)のスナップショット。プロファイルドロップ時に使う
    (PROFILE.md §5 の頑健化: 確率 p_drop で e(u_{m-1}) を e(u_0) に置換)。
    """
    return Profile().to_dict()


def encode_profile_batch(
    profiles: list[dict],
    vocab,
    top_k: int = DEFAULT_TOP_K,
    half_life: int = DEFAULT_HALF_LIFE,
    max_word_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """プロファイルのリストを :class:`ProfileEncoder` の入力テンソル群に変換する。

    ``vocab`` は Prediction Network と共有する出力文字 vocab
    (:class:`dataset.vocab.CharVocab`)。top-K 語の文字 ID 化にはこの vocab の
    ``encode``/``<pad>`` を使う(埋め込みテーブルを共有するため、ID 空間も
    共有する必要がある)。
    """
    per_profile_words = [select_top_k_words(profile, top_k, half_life) for profile in profiles]
    per_profile_char_ids = [
        [vocab.encode(word) for word in words] for words in per_profile_words
    ]
    max_len = max(
        (len(ids) for word_ids in per_profile_char_ids for ids in word_ids),
        default=1,
    )
    if max_word_len is not None and max_len > max_word_len:
        max_len = max_word_len
    max_len = max(max_len, 1)
    pad_id = vocab.token_to_id["<pad>"]

    batch = len(profiles)
    word_char_ids = torch.full((batch, top_k, max_len), pad_id, dtype=torch.long)
    word_char_mask = torch.zeros((batch, top_k, max_len), dtype=torch.bool)
    word_mask = torch.zeros((batch, top_k), dtype=torch.bool)

    for row, word_ids_list in enumerate(per_profile_char_ids):
        for col, ids in enumerate(word_ids_list[:top_k]):
            trimmed = ids[:max_len]
            length = len(trimmed)
            if length == 0:
                continue
            word_char_ids[row, col, :length] = torch.tensor(trimmed, dtype=torch.long)
            word_char_mask[row, col, :length] = True
            word_mask[row, col] = True

    domain = torch.tensor(
        [extract_domain_vector(profile) for profile in profiles], dtype=torch.float32
    )
    lang = torch.tensor(
        [extract_lang_vector(profile) for profile in profiles], dtype=torch.float32
    )

    return {
        "domain": domain,
        "lang": lang,
        "word_char_ids": word_char_ids,
        "word_char_mask": word_char_mask,
        "word_mask": word_mask,
    }


class ProfileEncoder(nn.Module):
    """e(u) = MLP([ d ; ℓ ; pool{emb(w) : w ∈ top-K(F)} ]) (PROFILE.md §4)。

    top-K 語の文字埋め込みは呼び出し側(:class:`model.transducer.KairoTransducer`)
    が Prediction Network の出力語彙埋め込みを ``embedding`` として forward の
    都度渡す(専用の埋め込みテーブルは持たない = 重み共有)。
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        domain_dim: int = len(DOMAIN_LABELS),
        lang_dim: int = len(LANG_KEYS),
        hidden_dim: int | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.output_dim = output_dim
        hidden_dim = hidden_dim or output_dim
        input_dim = domain_dim + lang_dim + embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        domain: torch.Tensor,
        lang: torch.Tensor,
        word_char_ids: torch.Tensor,
        word_char_mask: torch.Tensor,
        word_mask: torch.Tensor,
        embedding: nn.Embedding,
    ) -> torch.Tensor:
        """
        domain: (B, domain_dim)
        lang: (B, lang_dim)
        word_char_ids: (B, K, L) long
        word_char_mask: (B, K, L) bool -- True は有効な文字
        word_mask: (B, K) bool -- True は有効な語スロット(top-K が埋まっている)
        embedding: Prediction Network の出力語彙埋め込み(重み共有・再利用)

        戻り値: (B, output_dim)
        """
        char_embeds = embedding(word_char_ids)  # (B, K, L, E)
        char_weight = word_char_mask.unsqueeze(-1).to(char_embeds.dtype)
        char_sum = (char_embeds * char_weight).sum(dim=2)
        char_count = char_weight.sum(dim=2).clamp(min=1.0)
        word_vectors = char_sum / char_count  # (B, K, E) 語内プーリング(平均)

        word_weight = word_mask.unsqueeze(-1).to(word_vectors.dtype)
        word_sum = (word_vectors * word_weight).sum(dim=1)
        word_count = word_weight.sum(dim=1).clamp(min=1.0)
        pooled = word_sum / word_count  # (B, E) 語間プーリング(平均)

        features = torch.cat([domain, lang, pooled], dim=-1)
        return self.mlp(features)
