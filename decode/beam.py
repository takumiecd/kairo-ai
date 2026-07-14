"""Beam search RNN-T decoder."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from dataset.vocab import CharVocab
from decode.greedy import infer_model_device
from decode.greedy import load_model_from_artifact
from decode.profile_fusion import ProfileFusion
from decode.profile_fusion import ProfileFusionState
from decode.scores import Candidate
from decode.scores import normalize_candidate_confidence
from user_profile.schema import Profile


DEFAULT_PROFILE_FUSION_WEIGHT = 1.0  # docs/PROFILE.md §3 の s(k|t,u) 式の lambda


@dataclass(frozen=True)
class BeamState:
    token_ids: tuple[int, ...]
    prediction_ids: tuple[int, ...]
    score: float
    # profile=None のとき常に None のまま。従来挙動に一切影響しない。
    fusion_state: ProfileFusionState | None = None


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
    profile: Profile | None = None,
    profile_fusion_weight: float = DEFAULT_PROFILE_FUSION_WEIGHT,
) -> list[Candidate]:
    """RNN-T ビームサーチ。

    ``profile`` が与えられた場合のみ、docs/PROFILE.md §3 のトライ融合
    (段階A)を各展開ステップの log prob に加算する。``profile=None``
    (既定)では従来のプロファイル無しの挙動と完全に一致する。
    """
    blank_id = output_vocab.token_to_id["<blank>"]
    bos_id = output_vocab.token_to_id["<bos>"]
    blocked_ids = {
        output_vocab.token_to_id[token]
        for token in ("<pad>", "<bos>", "<unk>")
        if token in output_vocab.token_to_id
    }
    device = infer_model_device(model)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    fusion = ProfileFusion.from_profile(profile) if profile is not None else None
    initial_fusion_state = fusion.initial_state() if fusion is not None else None
    beams = [
        BeamState(
            token_ids=(),
            prediction_ids=(bos_id,),
            score=0.0,
            fusion_state=initial_fusion_state,
        )
    ]

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
                # blank は出力を進めないので、トライ状態も delta も変化なし。
                advanced.append(
                    BeamState(
                        token_ids=beam.token_ids,
                        prediction_ids=beam.prediction_ids,
                        score=beam.score + float(log_probs[blank_id].item()),
                        fusion_state=beam.fusion_state,
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
                    new_fusion_state = beam.fusion_state
                    fused_score = beam.score + token_score
                    if fusion is not None:
                        char = output_vocab.id_to_token[token_id]
                        new_fusion_state, delta = fusion.step(beam.fusion_state, char)
                        fused_score += profile_fusion_weight * delta
                    emitted.append(
                        BeamState(
                            token_ids=beam.token_ids + (token_id,),
                            prediction_ids=beam.prediction_ids + (token_id,),
                            score=fused_score,
                            fusion_state=new_fusion_state,
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
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Profile JSON for decoder-side trie fusion (docs/PROFILE.md §3). "
        "Omit for the profile-free (default) behavior.",
    )
    parser.add_argument(
        "--profile-fusion-weight",
        type=float,
        default=DEFAULT_PROFILE_FUSION_WEIGHT,
        help="Overall lambda weight applied to the trie-fusion potential delta.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, input_vocab, output_vocab = load_model_from_artifact(
        args.artifact_dir,
        checkpoint=args.checkpoint,
    )
    profile = Profile.load_json(args.profile) if args.profile is not None else None
    candidates = beam_search_decode(
        model,
        input_vocab.encode(args.input),
        output_vocab,
        beam_width=args.beam_width,
        expansion_width=args.expansion_width,
        max_symbols_per_step=args.max_symbols_per_step,
        max_output_length=args.max_output_length,
        profile=profile,
        profile_fusion_weight=args.profile_fusion_weight,
    )
    for candidate in candidates:
        print(f"{candidate.confidence:.4f}\t{candidate.score:.4f}\t{candidate.text}")


if __name__ == "__main__":
    main()
