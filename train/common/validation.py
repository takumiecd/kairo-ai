"""Validation helpers shared across model families."""

from __future__ import annotations


def unwrap_subset(dataset):
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return dataset.dataset, list(dataset.indices)
    return dataset, list(range(len(dataset)))
