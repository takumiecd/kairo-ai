"""Score and confidence helpers for decoder candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Candidate:
    text: str
    score: float
    confidence: float = 0.0


def normalize_candidate_confidence(candidates: list[Candidate]) -> list[Candidate]:
    """Convert candidate log scores into beam-local confidence values."""
    if not candidates:
        return []

    max_score = max(candidate.score for candidate in candidates)
    exp_scores = [math.exp(candidate.score - max_score) for candidate in candidates]
    total = sum(exp_scores)
    if total == 0.0:
        uniform = 1.0 / len(candidates)
        return [
            Candidate(text=candidate.text, score=candidate.score, confidence=uniform)
            for candidate in candidates
        ]

    return [
        Candidate(
            text=candidate.text,
            score=candidate.score,
            confidence=exp_score / total,
        )
        for candidate, exp_score in zip(candidates, exp_scores, strict=True)
    ]


def top_k_token_probs(logits, id_to_token: list[str], k: int = 5):
    """Return top-k token probabilities from one logits vector."""
    import torch

    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(k, probs.numel()))
    return [
        (id_to_token[token_id.item()], prob.item())
        for prob, token_id in zip(top_probs, top_ids, strict=True)
    ]
