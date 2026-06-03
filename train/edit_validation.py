"""Validation decoding for neural edit transducer training."""

from __future__ import annotations

import torch

from decode.edit_beam import edit_beam_search_decode
from eval.metrics import mean_cer
from train.validation import unwrap_subset


@torch.no_grad()
def evaluate_edit_decode_cer(
    model,
    dataset,
    output_vocab,
    decoder: str,
    max_samples: int,
    beam_width: int = 5,
    expansion_width: int = 5,
    max_actions: int = 128,
    edit_penalty: float = 0.0,
    insert_penalty: float = 0.0,
) -> float | None:
    if decoder == "none":
        return None
    if decoder not in {"greedy", "beam"}:
        raise ValueError("decoder must be one of: none, greedy, beam")

    base_dataset, indexes = unwrap_subset(dataset)
    selected_indexes = indexes[:max_samples]
    predictions: list[str] = []
    references: list[str] = []
    decode_beam_width = 1 if decoder == "greedy" else beam_width
    decode_expansion_width = 1 if decoder == "greedy" else expansion_width

    model.eval()
    try:
        from tqdm import tqdm
        iterable = tqdm(selected_indexes, desc=f"Validating edit ({decoder})", leave=False)
    except ImportError:
        iterable = selected_indexes

    for index in iterable:
        example = base_dataset[index]
        candidates = edit_beam_search_decode(
            model,
            input_ids=example.input_ids,
            previous_ids=example.previous_ids,
            output_vocab=output_vocab,
            beam_width=decode_beam_width,
            expansion_width=decode_expansion_width,
            max_actions=max_actions,
            edit_penalty=edit_penalty,
            insert_penalty=insert_penalty,
        )
        predictions.append(candidates[0].text if candidates else "")
        references.append(example.target_text)
    model.train()

    return mean_cer(predictions, references)
