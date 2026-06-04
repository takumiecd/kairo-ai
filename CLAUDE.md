# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Kairo AI is the ML backend for "Kairo", a local AI IME for engineers. The pipeline goes: generate `romaji-input → kanji/kana-output` datasets, train a Transformer-Transducer (RNN-T) for live conversion, then export light models (ONNX/Safetensors) into the Rust backend (`kairo` repo, separate). README and `docs/` are written in Japanese; match that language when editing user-facing docs.

Target UX (drives design decisions): space key is literal whitespace (not convert), English/Japanese mixing is auto-detected from context, arrow keys pick alternate beam candidates, and confirmed corrections feed online fine-tuning (LoRA).

## Environment & commands

Python 3.11 (conda env `kairo-ai`). Modules are run with `python -m <package>.<module>` — everything is a runnable module, there is no single CLI entrypoint.

```bash
python -m pip install .                      # install (editable-ish via hatchling)
python -m unittest discover -s tests         # run all tests
python -m unittest tests.test_model_transducer            # one test module
python -m unittest tests.test_model_transducer.ClassName.test_method   # one test
```

Tests use `unittest` (not pytest, despite the `.pytest_cache` dir). There is no configured linter/formatter.

Common workflows are documented with full flag examples in `README.md`: `dataset.generate` / `dataset.split` (synthetic data), `dataset.source_*` (Aozora, Tatoeba, Wikipedia ingest), `train.overfit` (smoke test), `train.train` (real training → `artifacts/<run>/`), `decode.greedy` / `decode.beam`.

## Architecture

Two parallel model families share the dataset/vocab tooling but are otherwise independent. Files are prefixed `edit_*` for the second one.

**1. RNN-T conversion model** (the primary, README-documented path):
- `model/transducer.py` — `KairoTransducer`: Encoder (LSTM over romaji input), Prediction Network (LSTM over emitted Japanese), Joint Network combining them via broadcast → per-step output distribution. LSTMs are placeholders; comments mark where Transformer variants go.
- `train/loss.py` — `compute_rnnt_loss` wraps `torchaudio.functional.rnnt_loss`.
- `train/{data,train,validation,overfit}.py`, `train/checkpoint.py` — training loop, batching, checkpointing (saves model + config + vocabs into the artifact dir).
- `decode/{greedy,beam}.py` + `decode/scores.py` — decoders restore model+vocab from an artifact dir; `scores.py` defines `Candidate` and beam-local confidence normalization.

**2. Neural edit transducer** (newer prototype, recent commits):
- `model/edit_transducer.py` — `KairoEditTransducer`: predicts cursor-based **edit actions** (op head + insert-token head) to revise an existing IME hypothesis, rather than emitting output left-to-right.
- `train/edit_{data,loss,train,validation}.py`, `decode/edit_beam.py` — the parallel data/loss/train/decode stack. Action ops use sentinels from `train/edit_data.py` (`ACTION_PAD`, `ACTION_BOS`, `INSERT`, `INSERT_PAD`); `edit_loss` ignores `ACTION_PAD` in cross-entropy.

**Shared dataset layer** (`dataset/`):
- `vocab.py` — `CharVocab` with special tokens: input `<pad> <unk>`; output `<pad> <blank> <bos> <unk>`. `<blank>` is the RNN-T blank emission. `encode` supports multi-char tokens (longest-match) when the vocab contains any.
- `generate.py` / `examples.py` / `noise.py` / `reading.py` — synthesize romaji↔kanji pairs: kanji→reading via SudachiPy, kana→romaji via pykakasi, then add typo/noise to Japanese-derived romaji while preserving commands/paths/identifiers.
- `source_*.py` + `split.py` — external corpus ingest and train/valid splitting.

`eval/metrics.py` — CER (Character Error Rate) for comparing decoded strings to targets, used in validation.

## Data policy

Source code is MIT. Datasets, trained models, external-corpus artifacts, and user learning data are kept separate from source and must not be committed — see `docs/DATA_POLICY.md`. Generated data lands in `data/` and `artifacts/` (gitignored).
