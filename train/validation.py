"""Validation decoding metrics for training."""

from __future__ import annotations

import torch

from decode.beam import beam_search_decode
from decode.greedy import greedy_decode
from eval.metrics import mean_cer


def unwrap_subset(dataset):
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return dataset.dataset, list(dataset.indices)
    return dataset, list(range(len(dataset)))


@torch.no_grad()
def evaluate_decode_cer(
    model,
    dataset,
    output_vocab,
    decoder: str,
    max_samples: int,
    beam_width: int = 5,
    expansion_width: int = 5,
    max_symbols_per_step: int = 4,
    max_output_length: int = 128,
) -> float | None:
    if decoder == "none":
        return None
    if decoder not in {"greedy", "beam"}:
        raise ValueError("decoder must be one of: none, greedy, beam")

    base_dataset, indexes = unwrap_subset(dataset)
    selected_indexes = indexes[:max_samples]
    predictions: list[str] = []
    references: list[str] = []

    model.eval()
    for index in selected_indexes:
        example = base_dataset[index]
        if decoder == "greedy":
            prediction = greedy_decode(
                model,
                example.input_ids,
                output_vocab,
                max_symbols_per_step=max_symbols_per_step,
                max_output_length=max_output_length,
            )
        else:
            candidates = beam_search_decode(
                model,
                example.input_ids,
                output_vocab,
                beam_width=beam_width,
                expansion_width=expansion_width,
                max_symbols_per_step=max_symbols_per_step,
                max_output_length=max_output_length,
            )
            prediction = candidates[0].text if candidates else ""
        predictions.append(prediction)
        references.append(example.target_text)
    model.train()

    return mean_cer(predictions, references)
