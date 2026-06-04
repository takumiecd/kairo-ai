"""Beam search decoder for neural edit transducer candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dataset.vocab import CharVocab
from decode.greedy import infer_model_device
from decode.scores import Candidate
from decode.scores import normalize_candidate_confidence
from train.edit.data import ACTION_BOS
from train.edit.data import DELETE
from train.edit.data import INSERT
from train.edit.data import KEEP
from train.edit.data import STOP


@dataclass(frozen=True)
class EditBeamState:
    cursor_index: int
    output_ids: tuple[int, ...]
    op_ids: tuple[int, ...]
    insert_token_ids: tuple[int, ...]
    score: float
    insertions_at_cursor: int = 0


def _append_action(
    state: EditBeamState,
    op_id: int,
    insert_token_id: int,
    score_delta: float,
    output_ids: tuple[int, ...],
    cursor_index: int,
    insertions_at_cursor: int,
) -> EditBeamState:
    return EditBeamState(
        cursor_index=cursor_index,
        output_ids=output_ids,
        op_ids=state.op_ids + (op_id,),
        insert_token_ids=state.insert_token_ids + (insert_token_id,),
        score=state.score + score_delta,
        insertions_at_cursor=insertions_at_cursor,
    )


def _prune_states(states: list[EditBeamState], beam_width: int) -> list[EditBeamState]:
    best_by_signature: dict[tuple[int, tuple[int, ...], tuple[int, ...]], EditBeamState] = {}
    for state in states:
        signature = (state.cursor_index, state.output_ids, state.op_ids)
        existing = best_by_signature.get(signature)
        if existing is None or state.score > existing.score:
            best_by_signature[signature] = state
    return sorted(best_by_signature.values(), key=lambda state: state.score, reverse=True)[:beam_width]


@torch.no_grad()
def edit_beam_search_decode(
    model,
    input_ids: list[int],
    previous_ids: list[int],
    output_vocab: CharVocab,
    beam_width: int = 5,
    expansion_width: int = 5,
    max_actions: int = 128,
    max_insertions_per_position: int = 8,
    edit_penalty: float = 0.0,
    insert_penalty: float = 0.0,
) -> list[Candidate]:
    device = infer_model_device(model)
    output_pad_id = output_vocab.token_to_id["<pad>"]
    blocked_insert_ids = {
        output_vocab.token_to_id[token]
        for token in ("<pad>", "<blank>", "<bos>", "<unk>")
        if token in output_vocab.token_to_id
    }
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    previous = torch.tensor([previous_ids], dtype=torch.long, device=device)
    states = [
        EditBeamState(
            cursor_index=0,
            output_ids=(),
            op_ids=(),
            insert_token_ids=(),
            score=0.0,
        )
    ]
    completed: list[EditBeamState] = []

    for _step in range(max_actions):
        next_states: list[EditBeamState] = []
        for state in states:
            action_ops = torch.tensor(
                [[ACTION_BOS] + list(state.op_ids)],
                dtype=torch.long,
                device=device,
            )
            action_insert_tokens = torch.tensor(
                [[output_pad_id] + list(state.insert_token_ids)],
                dtype=torch.long,
                device=device,
            )
            op_logits, insert_logits = model.predict_next(
                x,
                previous,
                action_ops,
                action_insert_tokens,
            )
            op_log_probs = torch.log_softmax(op_logits[0], dim=-1)

            if state.cursor_index >= len(previous_ids):
                completed.append(
                    _append_action(
                        state,
                        STOP,
                        output_pad_id,
                        float(op_log_probs[STOP].item()),
                        state.output_ids,
                        state.cursor_index,
                        state.insertions_at_cursor,
                    )
                )
            else:
                keep_score = float(op_log_probs[KEEP].item())
                next_states.append(
                    _append_action(
                        state,
                        KEEP,
                        output_pad_id,
                        keep_score,
                        state.output_ids + (previous_ids[state.cursor_index],),
                        state.cursor_index + 1,
                        0,
                    )
                )
                delete_score = float(op_log_probs[DELETE].item()) - edit_penalty
                next_states.append(
                    _append_action(
                        state,
                        DELETE,
                        output_pad_id,
                        delete_score,
                        state.output_ids,
                        state.cursor_index + 1,
                        0,
                    )
                )

            if state.insertions_at_cursor < max_insertions_per_position:
                insert_log_probs = torch.log_softmax(insert_logits[0], dim=-1)
                values, indexes = torch.topk(
                    insert_log_probs,
                    k=min(insert_log_probs.numel(), expansion_width + len(blocked_insert_ids)),
                )
                emitted = 0
                for value, index in zip(values, indexes, strict=True):
                    token_id = int(index.item())
                    if token_id in blocked_insert_ids:
                        continue
                    score_delta = (
                        float(op_log_probs[INSERT].item())
                        + float(value.item())
                        - edit_penalty
                        - insert_penalty
                    )
                    next_states.append(
                        _append_action(
                            state,
                            INSERT,
                            token_id,
                            score_delta,
                            state.output_ids + (token_id,),
                            state.cursor_index,
                            state.insertions_at_cursor + 1,
                        )
                    )
                    emitted += 1
                    if emitted >= expansion_width:
                        break

        if not next_states:
            break
        states = _prune_states(next_states, beam_width)

    completed.extend(state for state in states if state.cursor_index >= len(previous_ids))
    candidates = [
        Candidate(
            text=output_vocab.decode(list(state.output_ids)),
            score=state.score,
        )
        for state in _prune_states(completed, beam_width)
    ]
    return normalize_candidate_confidence(candidates)
