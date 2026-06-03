"""Generate Kairo AI training examples."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path

from dataset.examples import SYNTHETIC_EXAMPLES
from dataset.noise import InputNoiser
from dataset.reading import JapaneseRomajiConverter
from dataset.vocab import build_input_vocab
from dataset.vocab import build_output_vocab


@dataclass(frozen=True)
class TrainingExample:
    source: str
    source_detail: str
    license: str
    input: str
    target: str
    clean_input: str
    noise: str


class DatasetGenerator:
    def __init__(self, seed: int = 0) -> None:
        self.converter = JapaneseRomajiConverter()
        self.noiser = InputNoiser(seed=seed)

    def generate_pair(self, text: str) -> tuple[str, str]:
        return self.converter.convert_text(text), text

    def generate_input_segments(self, text: str) -> list[tuple[str, bool]]:
        return [
            (span.input, span.kind == "japanese")
            for span in self.converter.convert_spans(text)
        ]

    def generate_examples(
        self,
        texts: list[str],
        num_augmentations: int = 2,
        source: str = "synthetic",
        source_detail: str = "engineer_templates",
        license_name: str = "project_owned",
    ) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for text in texts:
            target = text
            input_segments = self.generate_input_segments(text)
            clean_input = "".join(segment for segment, _mutable in input_segments)
            for noisy_input in self.noiser.augment_segments(
                input_segments,
                num_augmentations,
            ):
                examples.append(
                    TrainingExample(
                        source=source,
                        source_detail=source_detail,
                        license=license_name,
                        input=noisy_input.text,
                        target=target,
                        clean_input=clean_input,
                        noise=noisy_input.noise,
                    )
                )
        return examples


def write_jsonl(path: Path, examples: list[TrainingExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL output path. Prints examples when omitted.",
    )
    parser.add_argument(
        "--augmentations",
        type=int,
        default=2,
        help="Number of noisy variants to add per clean example.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--show-vocab",
        action="store_true",
        help="Print generated input/output char vocab sizes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = DatasetGenerator(seed=args.seed)
    examples = generator.generate_examples(
        SYNTHETIC_EXAMPLES,
        num_augmentations=args.augmentations,
    )

    if args.output:
        write_jsonl(args.output, examples)
    else:
        for example in examples[:10]:
            print(json.dumps(asdict(example), ensure_ascii=False))

    if args.show_vocab:
        input_vocab = build_input_vocab([example.input for example in examples])
        output_vocab = build_output_vocab([example.target for example in examples])
        print(f"input_vocab_size={len(input_vocab.id_to_token)}")
        print(f"output_vocab_size={len(output_vocab.id_to_token)}")


if __name__ == "__main__":
    main()
