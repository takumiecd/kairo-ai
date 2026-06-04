"""Iterative decoding for the context-aware discrete diffusion model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from decode.greedy import infer_model_device
from decode.greedy import load_vocab
from model.diffusion import KairoDiffusionModel


def load_diffusion_model_from_artifact(
    artifact_dir: Path, checkpoint: Path | None = None
):
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    input_vocab = load_vocab(artifact_dir / "input_vocab.json")
    output_vocab = load_vocab(artifact_dir / "output_vocab.json")
    model = KairoDiffusionModel(
        input_vocab_size=len(input_vocab.id_to_token),
        output_vocab_size=len(output_vocab.id_to_token),
        model_dim=config["model_dim"],
        input_embed_dim=config["input_embed_dim"],
        output_embed_dim=config["output_embed_dim"],
        num_heads=config["num_heads"],
        num_input_layers=config["num_input_layers"],
        num_context_layers=config["num_context_layers"],
        num_canvas_layers=config["num_canvas_layers"],
        feedforward_dim=config["feedforward_dim"],
        dropout=config["dropout"],
        max_positions=config["max_positions"],
        diffusion_steps=config["diffusion_steps"],
    )
    state = torch.load(
        checkpoint or artifact_dir / "checkpoints" / "best.pt", map_location="cpu"
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, input_vocab, output_vocab


@torch.no_grad()
def diffusion_decode_ids(
    model,
    input_ids: list[int],
    context_ids: list[int],
    output_vocab,
    output_length: int | None = None,
) -> list[int]:
    device = infer_model_device(model)
    inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
    contexts = torch.tensor([context_ids], dtype=torch.long, device=device)
    input_pad_mask = torch.zeros(inputs.shape, dtype=torch.bool, device=device)
    context_pad_mask = torch.zeros(contexts.shape, dtype=torch.bool, device=device)
    memory, memory_pad_mask, pooled = model.encode_condition(
        inputs, contexts, input_pad_mask, context_pad_mask
    )

    if output_length is None:
        output_length = int(torch.argmax(model.length_head(pooled)[0]).item())
    output_length = max(1, min(output_length, model.max_positions))

    mask_token_id = output_vocab.token_to_id["<mask>"]
    blocked_ids = [
        token_id
        for token, token_id in output_vocab.token_to_id.items()
        if token.startswith("<")
    ]
    canvas = torch.full(
        (1, output_length), mask_token_id, dtype=torch.long, device=device
    )

    for timestep in range(model.diffusion_steps, 0, -1):
        positions = model._positions(output_length, device)
        hidden = (
            model.output_proj(model.output_emb(canvas))
            + model.output_pos(positions)[None]
            + model.time_emb(torch.tensor([timestep], device=device))[:, None]
        )
        decoded = model.canvas_decoder(hidden, memory, memory_key_padding_mask=memory_pad_mask)
        logits = model.token_head(decoded)
        logits[:, :, blocked_ids] = float("-inf")
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        canvas = predicted

        remask_count = round(output_length * (timestep - 1) / model.diffusion_steps)
        if remask_count > 0:
            low_confidence = torch.topk(
                confidence[0], k=remask_count, largest=False
            ).indices
            canvas[0, low_confidence] = mask_token_id

    return canvas[0].tolist()


@torch.no_grad()
def diffusion_decode(
    model,
    input_ids: list[int],
    context_ids: list[int],
    output_vocab,
    output_length: int | None = None,
) -> str:
    return output_vocab.decode(
        diffusion_decode_ids(model, input_ids, context_ids, output_vocab, output_length)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--output-length", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, input_vocab, output_vocab = load_diffusion_model_from_artifact(
        args.artifact_dir, args.checkpoint
    )
    print(
        diffusion_decode(
            model,
            input_vocab.encode(args.input),
            output_vocab.encode(args.context),
            output_vocab,
            args.output_length,
        )
    )


if __name__ == "__main__":
    main()
