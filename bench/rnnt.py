"""Benchmark RNN-T training and decode latency.

This measures the current implementation, not an idealized decoder. Use it to
compare artifacts, batching strategies, tokenizers, and GPU choices with the
same command line.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Callable

import torch

from decode.beam import beam_search_decode
from decode.greedy import greedy_decode
from decode.greedy import load_model_from_artifact
from train.common.batch import move_batch_to_device
from train.common.engine import build_loader
from train.common.engine import select_device
from train.rnnt.data import collate_transducer_batch
from train.rnnt.data import encode_records
from train.rnnt.data import JsonlTransducerDataset
from train.common.data import TrainingVocabs
from train.common.data import load_jsonl_examples
from train.rnnt.loss import compute_rnnt_loss


@dataclass(frozen=True)
class TimingSummary:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class TrainBenchmark:
    batches: int
    examples: int
    lattice_cells: int
    target_tokens: int
    input_tokens: int
    step: TimingSummary
    examples_per_sec: float
    lattice_cells_per_sec: float
    target_tokens_per_sec: float
    peak_cuda_memory_mb: float | None


@dataclass(frozen=True)
class DecodeBenchmark:
    decoder: str
    samples: int
    input_chars: int
    output_chars: int
    latency: TimingSummary
    samples_per_sec: float
    input_chars_per_sec: float


@dataclass(frozen=True)
class ValueSummary:
    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    min: int
    max: int


@dataclass(frozen=True)
class DatasetSummary:
    examples: int
    input_len: ValueSummary
    target_len: ValueSummary
    lattice_cells: ValueSummary


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_values(values: list[int]) -> ValueSummary:
    if not values:
        raise ValueError("values must not be empty")
    float_values = [float(value) for value in values]
    return ValueSummary(
        count=len(values),
        mean=sum(values) / len(values),
        p50=percentile(float_values, 50),
        p95=percentile(float_values, 95),
        p99=percentile(float_values, 99),
        min=min(values),
        max=max(values),
    )


def summarize_seconds(times: list[float]) -> TimingSummary:
    if not times:
        raise ValueError("times must not be empty")
    times_ms = [value * 1000.0 for value in times]
    return TimingSummary(
        count=len(times_ms),
        mean_ms=sum(times_ms) / len(times_ms),
        p50_ms=percentile(times_ms, 50),
        p95_ms=percentile(times_ms, 95),
        p99_ms=percentile(times_ms, 99),
        min_ms=min(times_ms),
        max_ms=max(times_ms),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def example_lattice_cells(example) -> int:
    return len(example.input_ids) * (len(example.target_ids) + 1)


def summarize_dataset(dataset) -> DatasetSummary:
    return DatasetSummary(
        examples=len(dataset),
        input_len=summarize_values([len(example.input_ids) for example in dataset.examples]),
        target_len=summarize_values([len(example.target_ids) for example in dataset.examples]),
        lattice_cells=summarize_values(
            [example_lattice_cells(example) for example in dataset.examples]
        ),
    )


def filter_dataset(
    dataset,
    *,
    max_input_len: int | None,
    max_target_len: int | None,
    max_lattice_cells: int | None,
) -> JsonlTransducerDataset:
    examples = []
    for example in dataset.examples:
        if max_input_len is not None and len(example.input_ids) > max_input_len:
            continue
        if max_target_len is not None and len(example.target_ids) > max_target_len:
            continue
        if max_lattice_cells is not None and example_lattice_cells(example) > max_lattice_cells:
            continue
        examples.append(example)
    return JsonlTransducerDataset(examples)


def load_dataset(
    path: Path,
    vocabs: TrainingVocabs,
    limit: int | None,
    *,
    max_input_len: int | None,
    max_target_len: int | None,
    max_lattice_cells: int | None,
    sort_by_lattice: bool,
):
    records = load_jsonl_examples(path)
    dataset = encode_records(records, vocabs)
    dataset = filter_dataset(
        dataset,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
        max_lattice_cells=max_lattice_cells,
    )
    if sort_by_lattice:
        dataset = JsonlTransducerDataset(
            sorted(dataset.examples, key=example_lattice_cells)
        )
    if limit is not None:
        dataset = JsonlTransducerDataset(dataset.examples[:limit])
    if len(dataset) == 0:
        raise ValueError("dataset is empty after applying benchmark filters")
    return dataset


def batch_stats(batch: dict[str, torch.Tensor]) -> dict[str, int]:
    input_lengths = batch["input_lengths"].to("cpu", non_blocking=False)
    target_lengths = batch["target_lengths"].to("cpu", non_blocking=False)
    batch_size = int(input_lengths.numel())
    max_input_len = int(input_lengths.max().item())
    max_target_len = int(target_lengths.max().item())
    return {
        "batch_size": batch_size,
        "max_input_len": max_input_len,
        "max_target_len": max_target_len,
        "padded_lattice_cells": batch_size * max_input_len * (max_target_len + 1),
        "actual_lattice_cells": int((input_lengths * (target_lengths + 1)).sum().item()),
    }


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "cuda out of memory" in message or (
        "out of memory" in message and "cuda" in message
    )


def benchmark_train_steps(
    *,
    model,
    vocabs: TrainingVocabs,
    dataset,
    device: torch.device,
    batch_size: int,
    warmup_batches: int,
    max_batches: int,
    amp: bool,
) -> TrainBenchmark:
    loader = build_loader(
        dataset,
        lambda examples: collate_transducer_batch(examples, vocabs),
        batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    model.train()
    times: list[float] = []
    examples = 0
    lattice_cells = 0
    target_tokens = 0
    input_tokens = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for index, batch in enumerate(loader):
        if index >= warmup_batches + max_batches:
            break
        stats = batch_stats(batch)
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        started = time.perf_counter()
        try:
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                loss = compute_rnnt_loss(model, batch, vocabs.blank_id)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            synchronize(device)
        except RuntimeError as error:
            if device.type == "cuda" and is_cuda_oom(error):
                torch.cuda.empty_cache()
                detail = " ".join(f"{key}={value}" for key, value in stats.items())
                raise RuntimeError(
                    f"CUDA OOM in train batch {index}: {detail}. "
                    "Try --batch-size 4, --amp, --sort-by-lattice, "
                    "--max-lattice-cells, or --skip-train for decode-only latency."
                ) from error
            raise
        elapsed = time.perf_counter() - started

        if index < warmup_batches:
            continue

        times.append(elapsed)
        batch_size_actual = int(batch["inputs"].shape[0])
        examples += batch_size_actual
        input_lengths = batch["input_lengths"].to("cpu", non_blocking=False)
        target_lengths = batch["target_lengths"].to("cpu", non_blocking=False)
        input_tokens += int(input_lengths.sum().item())
        target_tokens += int(target_lengths.sum().item())
        lattice_cells += int((input_lengths * (target_lengths + 1)).sum().item())

    if not times:
        raise ValueError("no batches were benchmarked; increase --limit-examples or lower warmup")

    total_time = sum(times)
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else None
    )
    return TrainBenchmark(
        batches=len(times),
        examples=examples,
        lattice_cells=lattice_cells,
        target_tokens=target_tokens,
        input_tokens=input_tokens,
        step=summarize_seconds(times),
        examples_per_sec=examples / total_time,
        lattice_cells_per_sec=lattice_cells / total_time,
        target_tokens_per_sec=target_tokens / total_time,
        peak_cuda_memory_mb=peak_memory,
    )


def benchmark_decode(
    *,
    decoder: str,
    decode_fn: Callable[[list[int]], str],
    examples,
    warmup_samples: int,
    max_samples: int,
    device: torch.device,
) -> DecodeBenchmark:
    selected = examples[: warmup_samples + max_samples]
    if len(selected) <= warmup_samples:
        raise ValueError("no samples were benchmarked; increase --limit-examples")

    times: list[float] = []
    input_chars = 0
    output_chars = 0
    for index, example in enumerate(selected):
        synchronize(device)
        started = time.perf_counter()
        decoded = decode_fn(example.input_ids)
        synchronize(device)
        elapsed = time.perf_counter() - started

        if index < warmup_samples:
            continue
        times.append(elapsed)
        input_chars += len(example.input_text)
        output_chars += len(decoded)

    total_time = sum(times)
    return DecodeBenchmark(
        decoder=decoder,
        samples=len(times),
        input_chars=input_chars,
        output_chars=output_chars,
        latency=summarize_seconds(times),
        samples_per_sec=len(times) / total_time,
        input_chars_per_sec=input_chars / total_time,
    )


def benchmark_prefix_decode(
    *,
    decoder: str,
    decode_fn: Callable[[list[int]], str],
    encode_input: Callable[[str], list[int]],
    examples,
    warmup_examples: int,
    max_examples: int,
    prefix_stride: int,
    min_prefix_chars: int,
    device: torch.device,
) -> DecodeBenchmark:
    selected = examples[: warmup_examples + max_examples]
    if len(selected) <= warmup_examples:
        raise ValueError("no prefix samples were benchmarked; increase --limit-examples")

    times: list[float] = []
    input_chars = 0
    output_chars = 0
    for example_index, example in enumerate(selected):
        text = example.input_text
        start = min(len(text), max(1, min_prefix_chars))
        prefix_ends = list(range(start, len(text) + 1, max(1, prefix_stride)))
        if prefix_ends and prefix_ends[-1] != len(text):
            prefix_ends.append(len(text))
        for end in prefix_ends:
            input_ids = encode_input(text[:end])
            synchronize(device)
            started = time.perf_counter()
            decoded = decode_fn(input_ids)
            synchronize(device)
            elapsed = time.perf_counter() - started

            if example_index < warmup_examples:
                continue
            times.append(elapsed)
            input_chars += end
            output_chars += len(decoded)

    if not times:
        raise ValueError("no prefixes were benchmarked; check --prefix-min-chars")

    total_time = sum(times)
    return DecodeBenchmark(
        decoder=f"{decoder}_prefix",
        samples=len(times),
        input_chars=input_chars,
        output_chars=output_chars,
        latency=summarize_seconds(times),
        samples_per_sec=len(times) / total_time,
        input_chars_per_sec=input_chars / total_time,
    )


def print_train_result(result: TrainBenchmark) -> None:
    print("train_step")
    print(
        f"  batches={result.batches} examples={result.examples} "
        f"examples/s={result.examples_per_sec:.2f}"
    )
    print(
        f"  step_ms mean={result.step.mean_ms:.2f} p50={result.step.p50_ms:.2f} "
        f"p95={result.step.p95_ms:.2f} p99={result.step.p99_ms:.2f}"
    )
    print(
        f"  lattice_cells/s={result.lattice_cells_per_sec:.0f} "
        f"target_tokens/s={result.target_tokens_per_sec:.0f}"
    )
    if result.peak_cuda_memory_mb is not None:
        print(f"  peak_cuda_memory_mb={result.peak_cuda_memory_mb:.1f}")


def print_decode_result(result: DecodeBenchmark) -> None:
    print(f"decode_{result.decoder}")
    print(
        f"  samples={result.samples} samples/s={result.samples_per_sec:.2f} "
        f"input_chars/s={result.input_chars_per_sec:.0f}"
    )
    print(
        f"  latency_ms mean={result.latency.mean_ms:.2f} p50={result.latency.p50_ms:.2f} "
        f"p95={result.latency.p95_ms:.2f} p99={result.latency.p99_ms:.2f}"
    )


def print_dataset_summary(summary: DatasetSummary) -> None:
    print("dataset")
    print(f"  examples={summary.examples}")
    for name in ("input_len", "target_len", "lattice_cells"):
        value = getattr(summary, name)
        print(
            f"  {name} mean={value.mean:.1f} p50={value.p50:.0f} "
            f"p95={value.p95:.0f} p99={value.p99:.0f} max={value.max}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data", type=Path, required=True, help="JSONL examples to benchmark.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-examples", type=int, default=512)
    parser.add_argument("--max-input-len", type=int, default=None)
    parser.add_argument("--max-target-len", type=int, default=None)
    parser.add_argument(
        "--max-lattice-cells",
        type=int,
        default=None,
        help="Drop examples where input_len * (target_len + 1) exceeds this value.",
    )
    parser.add_argument(
        "--sort-by-lattice",
        action="store_true",
        help="Sort examples by RNN-T lattice size before batching to reduce padding spikes.",
    )
    parser.add_argument("--train-batches", type=int, default=50)
    parser.add_argument("--train-warmup-batches", type=int, default=5)
    parser.add_argument("--decode-samples", type=int, default=100)
    parser.add_argument("--decode-warmup-samples", type=int, default=5)
    parser.add_argument(
        "--decode",
        choices=["none", "greedy", "beam", "both"],
        default="greedy",
        help="Decode latency benchmark to run after train-step benchmarking.",
    )
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--expansion-width", type=int, default=5)
    parser.add_argument("--max-symbols-per-step", type=int, default=4)
    parser.add_argument("--max-output-length", type=int, default=128)
    parser.add_argument(
        "--prefix-decode",
        action="store_true",
        help="Also decode every input prefix to approximate live IME recomputation latency.",
    )
    parser.add_argument("--prefix-samples", type=int, default=20)
    parser.add_argument("--prefix-warmup-samples", type=int, default=2)
    parser.add_argument("--prefix-stride", type=int, default=1)
    parser.add_argument("--prefix-min-chars", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Use CUDA AMP for train steps.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--continue-on-oom",
        action="store_true",
        help="If train-step benchmark hits CUDA OOM, record it and continue to decode benchmarks.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, input_vocab, output_vocab = load_model_from_artifact(
        args.artifact_dir, checkpoint=args.checkpoint
    )
    model.to(device)
    vocabs = TrainingVocabs(input_vocab=input_vocab, output_vocab=output_vocab)
    dataset = load_dataset(
        args.data,
        vocabs,
        args.limit_examples,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
        max_lattice_cells=args.max_lattice_cells,
        sort_by_lattice=args.sort_by_lattice,
    )
    dataset_summary = summarize_dataset(dataset)
    results: dict[str, object] = {
        "artifact_dir": str(args.artifact_dir),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "data": str(args.data),
        "device": str(device),
        "batch_size": args.batch_size,
        "limit_examples": args.limit_examples,
        "input_vocab_size": len(input_vocab.id_to_token),
        "output_vocab_size": len(output_vocab.id_to_token),
        "dataset": asdict(dataset_summary),
    }

    print(
        f"benchmark artifact={args.artifact_dir} data={args.data} "
        f"device={device} examples={len(dataset)}",
        flush=True,
    )
    print_dataset_summary(dataset_summary)

    if device.type == "mps" and not args.skip_train:
        raise ValueError("RNN-T loss supports CPU/CUDA here; use --device cpu or --skip-train")

    if not args.skip_train:
        try:
            train_result = benchmark_train_steps(
                model=model,
                vocabs=vocabs,
                dataset=dataset,
                device=device,
                batch_size=args.batch_size,
                warmup_batches=args.train_warmup_batches,
                max_batches=args.train_batches,
                amp=args.amp,
            )
            print_train_result(train_result)
            results["train_step"] = asdict(train_result)
        except RuntimeError as error:
            if not args.continue_on_oom or not is_cuda_oom(error):
                raise
            print(f"train_step_error={error}", flush=True)
            results["train_step_error"] = str(error)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if args.decode != "none":
        decode_modes = ["greedy", "beam"] if args.decode == "both" else [args.decode]
        model.eval()
        for decode_mode in decode_modes:
            if decode_mode == "greedy":
                decode_fn = lambda input_ids: greedy_decode(
                    model,
                    input_ids,
                    output_vocab,
                    max_symbols_per_step=args.max_symbols_per_step,
                    max_output_length=args.max_output_length,
                )
            else:
                decode_fn = lambda input_ids: beam_search_decode(
                    model,
                    input_ids,
                    output_vocab,
                    beam_width=args.beam_width,
                    expansion_width=args.expansion_width,
                    max_symbols_per_step=args.max_symbols_per_step,
                    max_output_length=args.max_output_length,
                )[0].text
            decode_result = benchmark_decode(
                decoder=decode_mode,
                decode_fn=decode_fn,
                examples=dataset.examples,
                warmup_samples=args.decode_warmup_samples,
                max_samples=args.decode_samples,
                device=device,
            )
            print_decode_result(decode_result)
            results[f"decode_{decode_mode}"] = asdict(decode_result)

            if args.prefix_decode:
                prefix_result = benchmark_prefix_decode(
                    decoder=decode_mode,
                    decode_fn=decode_fn,
                    encode_input=input_vocab.encode,
                    examples=dataset.examples,
                    warmup_examples=args.prefix_warmup_samples,
                    max_examples=args.prefix_samples,
                    prefix_stride=args.prefix_stride,
                    min_prefix_chars=args.prefix_min_chars,
                    device=device,
                )
                print_decode_result(prefix_result)
                results[f"decode_{decode_mode}_prefix"] = asdict(prefix_result)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote_json={args.json_output}")


if __name__ == "__main__":
    main()
