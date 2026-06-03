"""Text evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CerResult:
    edits: int
    reference_length: int

    @property
    def value(self) -> float:
        if self.reference_length == 0:
            return 0.0 if self.edits == 0 else 1.0
        return self.edits / self.reference_length


def edit_distance(prediction: str, reference: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if prediction == reference:
        return 0
    if not prediction:
        return len(reference)
    if not reference:
        return len(prediction)

    previous = list(range(len(reference) + 1))
    for pred_index, pred_char in enumerate(prediction, start=1):
        current = [pred_index]
        for ref_index, ref_char in enumerate(reference, start=1):
            substitution_cost = 0 if pred_char == ref_char else 1
            current.append(
                min(
                    previous[ref_index] + 1,
                    current[ref_index - 1] + 1,
                    previous[ref_index - 1] + substitution_cost,
                )
            )
        previous = current

    return previous[-1]


def cer(prediction: str, reference: str) -> float:
    """Compute character error rate."""
    return cer_result(prediction, reference).value


def cer_result(prediction: str, reference: str) -> CerResult:
    return CerResult(
        edits=edit_distance(prediction, reference),
        reference_length=len(reference),
    )


def mean_cer(predictions: list[str], references: list[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return 0.0

    total_edits = 0
    total_reference_length = 0
    for prediction, reference in zip(predictions, references, strict=True):
        result = cer_result(prediction, reference)
        total_edits += result.edits
        total_reference_length += result.reference_length

    if total_reference_length == 0:
        return 0.0 if total_edits == 0 else 1.0
    return total_edits / total_reference_length
