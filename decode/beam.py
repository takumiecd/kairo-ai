"""Beam search RNN-T decoder."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from dataset.vocab import CharVocab
from decode.greedy import infer_model_device
from decode.greedy import load_model_from_artifact
from decode.scores import Candidate
from decode.scores import normalize_candidate_confidence


@dataclass(frozen=True)
class BeamState:
    token_ids: tuple[int, ...]
    prediction_ids: tuple[int, ...]
    score: float


def _top_nonblank_ids(log_probs, blank_id: int, blocked_ids: set[int], k: int):
    values, indexes = torch.topk(log_probs, k=min(log_probs.numel(), k + len(blocked_ids) + 1))
    output: list[tuple[int, float]] = []
    for value, index in zip(values, indexes, strict=True):
        token_id = int(index.item())
        if token_id == blank_id or token_id in blocked_ids:
            continue
        output.append((token_id, float(value.item())))
        if len(output) >= k:
            break
    return output


def _prune_beams(beams: list[BeamState], beam_width: int) -> list[BeamState]:
    best_by_tokens: dict[tuple[int, ...], BeamState] = {}
    for beam in beams:
        existing = best_by_tokens.get(beam.token_ids)
        if existing is None or beam.score > existing.score:
            best_by_tokens[beam.token_ids] = beam
    return sorted(best_by_tokens.values(), key=lambda beam: beam.score, reverse=True)[:beam_width]


@torch.no_grad()
def beam_search_decode(
    model,
    input_ids: list[int],
    output_vocab: CharVocab,
    beam_width: int = 5,
    expansion_width: int = 5,
    max_symbols_per_step: int = 4,
    max_output_length: int = 128,
) -> list[Candidate]:
    blank_id = output_vocab.token_to_id["<blank>"]
    bos_id = output_vocab.token_to_id["<bos>"]
    blocked_ids = {
        output_vocab.token_to_id[token]
        for token in ("<pad>", "<bos>", "<unk>")
        if token in output_vocab.token_to_id
    }
    device = infer_model_device(model)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    beams = [BeamState(token_ids=(), prediction_ids=(bos_id,), score=0.0)]

    for input_step in range(len(input_ids)):
        active = beams
        advanced: list[BeamState] = []

        for _symbol_step in range(max_symbols_per_step):
            emitted: list[BeamState] = []
            for beam in active:
                y = torch.tensor(
                    [list(beam.prediction_ids)],
                    dtype=torch.long,
                    device=device,
                )
                logits = model(x, y)
                log_probs = torch.log_softmax(
                    logits[0, input_step, len(beam.prediction_ids) - 1],
                    dim=-1,
                )
                advanced.append(
                    BeamState(
                        token_ids=beam.token_ids,
                        prediction_ids=beam.prediction_ids,
                        score=beam.score + float(log_probs[blank_id].item()),
                    )
                )

                if len(beam.token_ids) >= max_output_length:
                    continue
                for token_id, token_score in _top_nonblank_ids(
                    log_probs,
                    blank_id=blank_id,
                    blocked_ids=blocked_ids,
                    k=expansion_width,
                ):
                    emitted.append(
                        BeamState(
                            token_ids=beam.token_ids + (token_id,),
                            prediction_ids=beam.prediction_ids + (token_id,),
                            score=beam.score + token_score,
                        )
                    )

            if not emitted:
                break
            active = _prune_beams(emitted, beam_width)

        if not advanced:
            advanced = active
        beams = _prune_beams(advanced, beam_width)

    candidates = [
        Candidate(
            text="".join(output_vocab.id_to_token[token_id] for token_id in beam.token_ids),
            score=beam.score,
        )
        for beam in beams
    ]
    return normalize_candidate_confidence(candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--expansion-width", type=int, default=5)
    parser.add_argument("--max-symbols-per-step", type=int, default=4)
    parser.add_argument("--max-output-length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, input_vocab, output_vocab = load_model_from_artifact(
        args.artifact_dir,
        checkpoint=args.checkpoint,
    )
    candidates = beam_search_decode(
        model,
        input_vocab.encode(args.input),
        output_vocab,
        beam_width=args.beam_width,
        expansion_width=args.expansion_width,
        max_symbols_per_step=args.max_symbols_per_step,
        max_output_length=args.max_output_length,
    )
    for candidate in candidates:
        print(f"{candidate.confidence:.4f}\t{candidate.score:.4f}\t{candidate.text}")


if __name__ == "__main__":
    main()
