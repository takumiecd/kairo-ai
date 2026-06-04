"""Dataset utilities for the iterative edit refiner.

最小編集オラクル (build_min_edit_script) を refiner の 3 ヘッドのラベルへ展開する:
  - delete_target  : BOS/previous/EOS を並べた仮説の各トークンの KEEP/DELETE
  - insert_target  : 仮説の各ギャップ (T-1 個) に挿入するトークン数 0..K
  - placeholder 列 + fill_target : 削除とギャップ挿入を適用し、挿入位置を <plh> に
                     した系列と、その <plh> 位置に入る目標トークン

仮説は [BOS] + previous + [EOS] で挟む（両端は編集対象外）。出力語彙は
edit transducer 用に作ったものへ <eos> / <plh> を末尾追加して拡張する
（既存 id を保持）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from dataset.vocab import CharVocab
from train.common.data import TrainingVocabs
from train.common.data import load_jsonl_examples
from train.edit.data import DELETE
from train.edit.data import INSERT
from train.edit.data import KEEP
from train.edit.data import STOP
from train.edit.data import build_edit_vocabs_from_records
from train.edit.data import build_min_edit_script


EOS_TOKEN = "<eos>"
PLH_TOKEN = "<plh>"

DELETE_PAD = -100   # delete CE の ignore_index
INSERT_PAD = -100   # insert-count CE の ignore_index
FILL_PAD = -100     # fill CE の ignore_index


@dataclass(frozen=True)
class EncodedRefineExample:
    input_ids: list[int]
    hypothesis_ids: list[int]       # [BOS] + previous + [EOS]
    delete_target: list[int]        # len == len(hypothesis_ids)
    insert_target: list[int]        # len == len(hypothesis_ids) - 1
    placeholder_ids: list[int]      # 削除/挿入適用後、挿入位置は <plh>
    fill_target: list[int]          # len == len(placeholder_ids), <plh> 以外は FILL_PAD
    input_text: str
    previous_text: str
    target_text: str


class JsonlRefineDataset(Dataset):
    def __init__(self, examples: list[EncodedRefineExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedRefineExample:
        return self.examples[index]


def extend_output_vocab(vocab: CharVocab) -> CharVocab:
    """<eos> / <plh> を末尾に追加した出力語彙を返す（既存 id は不変）。"""
    id_to_token = list(vocab.id_to_token)
    for token in (EOS_TOKEN, PLH_TOKEN):
        if token not in id_to_token:
            id_to_token.append(token)
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return CharVocab(token_to_id=token_to_id, id_to_token=id_to_token)


def build_refine_vocabs_from_records(
    records: list[dict[str, str]],
    output_tokenizer: str = "char",
    output_vocab_size: int = 4000,
    output_min_token_frequency: int = 2,
) -> TrainingVocabs:
    base = build_edit_vocabs_from_records(
        records,
        output_tokenizer=output_tokenizer,
        output_vocab_size=output_vocab_size,
        output_min_token_frequency=output_min_token_frequency,
    )
    return TrainingVocabs(
        input_vocab=base.input_vocab,
        output_vocab=extend_output_vocab(base.output_vocab),
    )


def bos_id(vocabs: TrainingVocabs) -> int:
    return vocabs.output_vocab.token_to_id["<bos>"]


def eos_id(vocabs: TrainingVocabs) -> int:
    return vocabs.output_vocab.token_to_id[EOS_TOKEN]


def placeholder_id(vocabs: TrainingVocabs) -> int:
    return vocabs.output_vocab.token_to_id[PLH_TOKEN]


def build_refine_labels(
    previous_ids: list[int],
    target_ids: list[int],
    bos: int,
    eos: int,
    plh: int,
    max_insertions_per_gap: int,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """最小編集スクリプトから refiner の 3 ヘッドのラベルを作る。"""
    actions = build_min_edit_script(previous_ids, target_ids)
    hypothesis = [bos] + previous_ids + [eos]
    num_tokens = len(hypothesis)

    delete_target = [KEEP] * num_tokens          # BOS/EOS は KEEP のまま
    gap_inserts: list[list[int]] = [[] for _ in range(num_tokens - 1)]

    consumed = 0  # これまでに消費した previous トークン数
    for action in actions:
        if action.op_id == KEEP:
            delete_target[consumed + 1] = KEEP
            consumed += 1
        elif action.op_id == DELETE:
            delete_target[consumed + 1] = DELETE
            consumed += 1
        elif action.op_id == INSERT:
            # 現在のカーソル位置に対応するギャップ（consumed 番目）へ。
            gap_inserts[consumed].append(action.token_id)
        elif action.op_id == STOP:
            break

    # 各ギャップの挿入数は K で頭打ち（モデルの insert ヘッドは 0..K）。
    insert_target = [min(len(tokens), max_insertions_per_gap) for tokens in gap_inserts]

    placeholder_ids = [hypothesis[0]]  # BOS
    fill_target = [FILL_PAD]
    for j in range(1, num_tokens):
        for token in gap_inserts[j - 1][:max_insertions_per_gap]:
            placeholder_ids.append(plh)
            fill_target.append(token)
        if delete_target[j] == KEEP:
            placeholder_ids.append(hypothesis[j])
            fill_target.append(FILL_PAD)

    return hypothesis, delete_target, insert_target, placeholder_ids, fill_target


def encode_refine_example(
    input_text: str,
    previous_text: str,
    target_text: str,
    vocabs: TrainingVocabs,
    max_insertions_per_gap: int,
) -> EncodedRefineExample:
    previous_ids = vocabs.output_vocab.encode(previous_text)
    target_ids = vocabs.output_vocab.encode(target_text)
    hypothesis, delete_target, insert_target, placeholder_ids, fill_target = build_refine_labels(
        previous_ids,
        target_ids,
        bos=bos_id(vocabs),
        eos=eos_id(vocabs),
        plh=placeholder_id(vocabs),
        max_insertions_per_gap=max_insertions_per_gap,
    )
    return EncodedRefineExample(
        input_ids=vocabs.input_vocab.encode(input_text),
        hypothesis_ids=hypothesis,
        delete_target=delete_target,
        insert_target=insert_target,
        placeholder_ids=placeholder_ids,
        fill_target=fill_target,
        input_text=input_text,
        previous_text=previous_text,
        target_text=target_text,
    )


def encode_refine_records(
    records: list[dict[str, str]],
    vocabs: TrainingVocabs,
    max_insertions_per_gap: int,
) -> JsonlRefineDataset:
    return JsonlRefineDataset(
        [
            encode_refine_example(
                input_text=record["input"],
                previous_text=record.get("previous", ""),
                target_text=record["target"],
                vocabs=vocabs,
                max_insertions_per_gap=max_insertions_per_gap,
            )
            for record in records
        ]
    )


def load_train_valid_refine_datasets_and_vocabs(
    train_path,
    valid_path=None,
    output_tokenizer: str = "char",
    output_vocab_size: int = 4000,
    output_min_token_frequency: int = 2,
    max_insertions_per_gap: int = 8,
) -> tuple[JsonlRefineDataset, JsonlRefineDataset | None, TrainingVocabs]:
    train_records = load_jsonl_examples(train_path)
    valid_records = load_jsonl_examples(valid_path) if valid_path is not None else []
    vocabs = build_refine_vocabs_from_records(
        train_records + valid_records,
        output_tokenizer=output_tokenizer,
        output_vocab_size=output_vocab_size,
        output_min_token_frequency=output_min_token_frequency,
    )
    train_dataset = encode_refine_records(train_records, vocabs, max_insertions_per_gap)
    valid_dataset = (
        encode_refine_records(valid_records, vocabs, max_insertions_per_gap)
        if valid_path is not None
        else None
    )
    return train_dataset, valid_dataset, vocabs


def collate_refine_batch(
    examples: list[EncodedRefineExample],
    vocabs: TrainingVocabs,
) -> dict[str, torch.Tensor]:
    batch_size = len(examples)
    input_pad = vocabs.input_pad_id
    output_pad = vocabs.output_pad_id

    max_input = max(len(e.input_ids) for e in examples)
    max_hyp = max(len(e.hypothesis_ids) for e in examples)
    max_plh = max(len(e.placeholder_ids) for e in examples)

    inputs = torch.full((batch_size, max_input), input_pad, dtype=torch.long)
    hypothesis = torch.full((batch_size, max_hyp), output_pad, dtype=torch.long)
    placeholders = torch.full((batch_size, max_plh), output_pad, dtype=torch.long)
    delete_target = torch.full((batch_size, max_hyp), DELETE_PAD, dtype=torch.long)
    insert_target = torch.full((batch_size, max_hyp - 1), INSERT_PAD, dtype=torch.long)
    fill_target = torch.full((batch_size, max_plh), FILL_PAD, dtype=torch.long)
    input_pad_mask = torch.ones((batch_size, max_input), dtype=torch.bool)
    hypothesis_pad_mask = torch.ones((batch_size, max_hyp), dtype=torch.bool)
    placeholder_pad_mask = torch.ones((batch_size, max_plh), dtype=torch.bool)

    for row, e in enumerate(examples):
        n_in, n_hyp, n_plh = len(e.input_ids), len(e.hypothesis_ids), len(e.placeholder_ids)
        inputs[row, :n_in] = torch.tensor(e.input_ids, dtype=torch.long)
        hypothesis[row, :n_hyp] = torch.tensor(e.hypothesis_ids, dtype=torch.long)
        placeholders[row, :n_plh] = torch.tensor(e.placeholder_ids, dtype=torch.long)
        delete_target[row, :n_hyp] = torch.tensor(e.delete_target, dtype=torch.long)
        insert_target[row, : len(e.insert_target)] = torch.tensor(e.insert_target, dtype=torch.long)
        fill_target[row, :n_plh] = torch.tensor(e.fill_target, dtype=torch.long)
        input_pad_mask[row, :n_in] = False
        hypothesis_pad_mask[row, :n_hyp] = False
        placeholder_pad_mask[row, :n_plh] = False

    return {
        "inputs": inputs,
        "hypothesis": hypothesis,
        "placeholders": placeholders,
        "delete_target": delete_target,
        "insert_target": insert_target,
        "fill_target": fill_target,
        "input_pad_mask": input_pad_mask,
        "hypothesis_pad_mask": hypothesis_pad_mask,
        "placeholder_pad_mask": placeholder_pad_mask,
    }
