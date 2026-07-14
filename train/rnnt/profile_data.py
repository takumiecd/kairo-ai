"""段階B(プロファイル条件付け)学習用の dataset/collate ユーティリティ。

``dataset/profile_stream.py`` が書き出す jsonl
(``{"input", "context", "profile", "target", ...}``)を読み、
``train.rnnt.data`` の通常の RNN-T バッチに加えて、プロファイル特徴一式を
フラットなキー(``profile_domain`` 等)としてバッチへ足す。

``profile_*`` キーをフラットに持つのは、既存の
``train.common.batch.move_batch_to_device``(``{key: value.to(device)}``)が
そのまま使えるようにするため -- ネストした dict を値に持つとそこだけ特別
扱いが必要になり、共通エンジン(``train/common/engine.py``)に手を入れる
ことになってしまう。

プロファイルドロップ(PROFILE.md §5 の頑健化): 確率 ``p_drop`` で
そのバッチ内の1件のプロファイルを u_0(空プロファイル)に置き換えてから
エンコードする。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset

from model.profile_encoder import DEFAULT_TOP_K
from model.profile_encoder import empty_profile_snapshot
from model.profile_encoder import encode_profile_batch
from train.common.checkpoint import has_saved_vocab
from train.common.checkpoint import load_vocabs
from train.common.data import TrainingVocabs
from train.common.data import build_vocabs_from_records
from train.common.data import load_jsonl_examples
from train.rnnt.data import collate_transducer_batch


DEFAULT_PROFILE_DROP_RATE = 0.3


@dataclass(frozen=True)
class EncodedProfileExample:
    input_ids: list[int]
    target_ids: list[int]
    input_text: str
    target_text: str
    profile: dict


class ProfileTransducerDataset(Dataset):
    def __init__(self, examples: list[EncodedProfileExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedProfileExample:
        return self.examples[index]


def encode_profile_records(
    records: list[dict],
    vocabs: TrainingVocabs,
) -> ProfileTransducerDataset:
    return ProfileTransducerDataset(
        [
            EncodedProfileExample(
                input_ids=vocabs.input_vocab.encode(record["input"]),
                target_ids=vocabs.output_vocab.encode(record["target"]),
                input_text=record["input"],
                target_text=record["target"],
                profile=record.get("profile") or {},
            )
            for record in records
        ]
    )


def load_profile_dataset_and_vocabs(
    path: Path,
    output_tokenizer: str = "char",
    output_vocab_size: int = 4000,
    output_min_token_frequency: int = 2,
    vocab_dir=None,
) -> tuple[ProfileTransducerDataset, TrainingVocabs]:
    """プロファイル付き jsonl (dataset/profile_stream.py の出力)を読む。

    プロファイル埋め込みは Prediction Network の出力語彙埋め込みを再利用する
    ため、vocab は通常の RNN-T 学習と同じ手順(``build_vocabs_from_records``)
    で構築する -- 専用のプロファイル vocab は持たない。
    """
    records = load_jsonl_examples(path)
    if vocab_dir is not None and has_saved_vocab(vocab_dir):
        vocabs = load_vocabs(vocab_dir)
    else:
        vocabs = build_vocabs_from_records(
            records,
            output_tokenizer=output_tokenizer,
            output_vocab_size=output_vocab_size,
            output_min_token_frequency=output_min_token_frequency,
        )
    return encode_profile_records(records, vocabs), vocabs


def collate_profile_transducer_batch(
    examples: list[EncodedProfileExample],
    vocabs: TrainingVocabs,
    top_k: int = DEFAULT_TOP_K,
    half_life: int = 100_000,
    profile_drop_rate: float = 0.0,
    rng: random.Random | None = None,
) -> dict[str, torch.Tensor]:
    """通常の RNN-T バッチにフラットな ``profile_*`` テンソルを足して返す。"""
    base_examples = [
        # collate_transducer_batch はダック型なので EncodedExample と
        # 同じ属性(input_ids/target_ids)を持つ EncodedProfileExample を
        # そのまま渡せる。
        example
        for example in examples
    ]
    batch = collate_transducer_batch(base_examples, vocabs)

    rng = rng if rng is not None else random.Random()
    profiles = []
    for example in examples:
        if profile_drop_rate > 0.0 and rng.random() < profile_drop_rate:
            profiles.append(empty_profile_snapshot())
        else:
            profiles.append(example.profile or empty_profile_snapshot())

    profile_tensors = encode_profile_batch(
        profiles,
        vocab=vocabs.output_vocab,
        top_k=top_k,
        half_life=half_life,
    )
    batch["profile_domain"] = profile_tensors["domain"]
    batch["profile_lang"] = profile_tensors["lang"]
    batch["profile_word_char_ids"] = profile_tensors["word_char_ids"]
    batch["profile_word_char_mask"] = profile_tensors["word_char_mask"]
    batch["profile_word_mask"] = profile_tensors["word_mask"]
    return batch


def extract_profile_features(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """フラット化された ``profile_*`` キーを ``KairoTransducer.forward`` が
    期待する ``profile_features`` dict へ戻す。"""
    return {
        "domain": batch["profile_domain"],
        "lang": batch["profile_lang"],
        "word_char_ids": batch["profile_word_char_ids"],
        "word_char_mask": batch["profile_word_char_mask"],
        "word_mask": batch["profile_word_mask"],
    }
