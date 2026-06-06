"""Encoded-dataset cache shared by every model family.

Vocab reuse lives in ``train.common.checkpoint`` (``has_saved_vocab`` /
``load_vocabs``); this module caches the expensive *encoding* step keyed by a
fingerprint of the source file and the vocab, so resuming a run skips both the
BPE merge rebuild and re-encoding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import torch

from train.common.data import TrainingVocabs


def data_fingerprint(path) -> str:
    stat = Path(path).stat()
    return f"{Path(path).name}-{stat.st_size}-{int(stat.st_mtime)}"


def vocab_fingerprint(vocabs: TrainingVocabs) -> str:
    payload = json.dumps(
        {
            "input": vocabs.input_vocab.id_to_token,
            "output": vocabs.output_vocab.id_to_token,
        },
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_or_encode(
    path,
    vocabs: TrainingVocabs,
    cache_dir,
    split: str,
    encode_fn: Callable[[], object],
    rebuild_fn: Callable[[list], object],
    extra_key: str = "",
):
    """Cache an encoded dataset keyed by (data fingerprint, vocab fingerprint, extra_key).

    ``encode_fn()`` must return a Dataset exposing an ``.examples`` list;
    ``rebuild_fn(examples)`` reconstructs that Dataset from a cached list
    (usually the Dataset class itself). ``extra_key`` distinguishes encodings
    that depend on family-specific parameters (e.g. the refiner's K).
    """
    if cache_dir is None:
        return encode_fn()

    cache_dir = Path(cache_dir)
    suffix = f"-{extra_key}" if extra_key else ""
    key = f"{split}-{data_fingerprint(path)}-{vocab_fingerprint(vocabs)}{suffix}"
    cache_file = cache_dir / f"{key}.pt"
    if cache_file.exists():
        print(f"Loading cached encoding: {cache_file}", flush=True)
        return rebuild_fn(torch.load(cache_file, weights_only=False))

    dataset = encode_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dataset.examples, cache_file)
    print(f"Saved encoding cache: {cache_file}", flush=True)
    return dataset
