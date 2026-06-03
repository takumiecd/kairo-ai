"""Greedy RNN-T decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset.vocab import CharVocab
from dataset.vocab import vocab_from_token_to_id
from decode.scores import top_k_token_probs
from model.transducer import KairoTransducer


def load_vocab(path: Path) -> CharVocab:
    with path.open("r", encoding="utf-8") as file:
        token_to_id = json.load(file)
    return vocab_from_token_to_id(
        {token: int(token_id) for token, token_id in token_to_id.items()}
    )


def load_model_from_artifact(artifact_dir: Path, checkpoint: Path | None = None):
    config_path = artifact_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    input_vocab = load_vocab(artifact_dir / "input_vocab.json")
    output_vocab = load_vocab(artifact_dir / "output_vocab.json")
    checkpoint_path = checkpoint or artifact_dir / "checkpoints" / "best.pt"
    state = torch.load(checkpoint_path, map_location="cpu")

    model = KairoTransducer(
        input_vocab_size=len(input_vocab.id_to_token),
        output_vocab_size=len(output_vocab.id_to_token),
        embed_dim=int(config["embed_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, input_vocab, output_vocab


@torch.no_grad()
def greedy_decode(
    model,
    input_ids: list[int],
    output_vocab: CharVocab,
    max_symbols_per_step: int = 4,
    max_output_length: int = 128,
) -> str:
    blank_id = output_vocab.token_to_id["<blank>"]
    bos_id = output_vocab.token_to_id["<bos>"]
    special_ids = {
        output_vocab.token_to_id[token]
        for token in ("<pad>", "<blank>", "<bos>", "<unk>")
        if token in output_vocab.token_to_id
    }

    x = torch.tensor([input_ids], dtype=torch.long)
    emitted_ids: list[int] = []
    prediction_ids = [bos_id]

    input_step = 0
    while input_step < len(input_ids) and len(emitted_ids) < max_output_length:
        symbols_emitted = 0
        while symbols_emitted < max_symbols_per_step and len(emitted_ids) < max_output_length:
            y = torch.tensor([prediction_ids], dtype=torch.long)
            logits = model(x, y)
            step_logits = logits[0, input_step, len(prediction_ids) - 1]
            token_id = int(torch.argmax(step_logits).item())

            if token_id == blank_id:
                break
            if token_id not in special_ids:
                emitted_ids.append(token_id)
            prediction_ids.append(token_id)
            symbols_emitted += 1

        input_step += 1

    return "".join(output_vocab.id_to_token[token_id] for token_id in emitted_ids)


@torch.no_grad()
def inspect_next_token_probs(
    model,
    input_ids: list[int],
    output_vocab: CharVocab,
    output_prefix: str = "",
    k: int = 5,
):
    bos_id = output_vocab.token_to_id["<bos>"]
    prefix_ids = output_vocab.encode(output_prefix)
    x = torch.tensor([input_ids], dtype=torch.long)
    y = torch.tensor([[bos_id] + prefix_ids], dtype=torch.long)
    logits = model(x, y)
    return top_k_token_probs(logits[0, 0, len(prefix_ids)], output_vocab.id_to_token, k=k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-symbols-per-step", type=int, default=4)
    parser.add_argument("--max-output-length", type=int, default=128)
    parser.add_argument(
        "--show-next-token-probs",
        action="store_true",
        help="Print top token probabilities at the first input step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, input_vocab, output_vocab = load_model_from_artifact(
        args.artifact_dir,
        checkpoint=args.checkpoint,
    )
    input_ids = input_vocab.encode(args.input)
    decoded = greedy_decode(
        model,
        input_ids,
        output_vocab,
        max_symbols_per_step=args.max_symbols_per_step,
        max_output_length=args.max_output_length,
    )
    print(decoded)

    if args.show_next_token_probs:
        for token, probability in inspect_next_token_probs(model, input_ids, output_vocab):
            print(f"{token}\t{probability:.4f}")


if __name__ == "__main__":
    main()
