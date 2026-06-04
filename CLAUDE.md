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

Common workflows are documented with full flag examples in `README.md`: `dataset.generate` / `dataset.split` (synthetic data), `dataset.source_*` (Aozora, Tatoeba, Wikipedia ingest), `train.rnnt.overfit` (smoke test), `train.rnnt.train` (real training → `artifacts/<run>/`), `decode.greedy` / `decode.beam`.

The `train/` package is split by model family: `train/common/` holds model-agnostic infra (`engine.py` — `Trainer` + the shared epoch/checkpoint/metrics/resume loop and common argparse; `checkpoint.py`; `data.py` — `TrainingVocabs`, `build_vocabs_from_records`, `load_jsonl_examples`; `batch.py`; `validation.py` — `unwrap_subset`). The per-family stacks live in `train/rnnt/`, `train/edit/`, `train/refiner/`, each with `data.py` / `loss.py` / `train.py` (run as `python -m train.<family>.train`). A family's `train.py` is a thin entrypoint: it supplies family-specific args, dataset/collate, model construction, and `loss_fn` / `cer_fn` closures, then calls `Trainer.fit(...)`.

## Architecture

Three model families share the dataset/vocab tooling but are otherwise independent.

**1. RNN-T conversion model** (the primary, README-documented path):
- `model/transducer.py` — `KairoTransducer`: Encoder (LSTM over romaji input), Prediction Network (LSTM over emitted Japanese), Joint Network combining them via broadcast → per-step output distribution. LSTMs are placeholders; comments mark where Transformer variants go.
- `train/rnnt/loss.py` — `compute_rnnt_loss` wraps `torchaudio.functional.rnnt_loss`.
- `train/rnnt/{data,train,validation,overfit}.py` — training loop, batching, decode-CER validation (on top of `train/common/`).
- `decode/{greedy,beam}.py` + `decode/scores.py` — decoders restore model+vocab from an artifact dir; `scores.py` defines `Candidate` and beam-local confidence normalization.

**2. Neural edit transducer** (prototype):
- `model/edit_transducer.py` — `KairoEditTransducer`: predicts cursor-based **edit actions** (op head + insert-token head) to revise an existing IME hypothesis, rather than emitting output left-to-right.
- `train/edit/{data,loss,train,validation}.py`, `decode/edit_beam.py` — the parallel data/loss/train/decode stack. Action ops use sentinels from `train/edit/data.py` (`ACTION_PAD`, `ACTION_BOS`, `INSERT`, `INSERT_PAD`); `edit/loss.py` ignores `ACTION_PAD` in cross-entropy.

**3. Iterative edit refiner** (newest prototype):
- `model/edit_refiner.py` — `KairoEditRefiner`: Levenshtein-Transformer-style **non-autoregressive** editor. Parallel delete / insert-count / fill heads over per-position representations, with cross-attention from the hypothesis encoder into the romaji input; separate input/output embedding dims projected to a shared `model_dim`. `refine()` warm-starts from the previous hypothesis and runs a bounded number of rounds (latency ∝ rounds, not length).
- `train/refiner/{data,loss,train}.py` — labels (delete / per-gap insert-count / `<plh>` fill) are derived from the min-edit oracle (`build_min_edit_script`), wrapping the hypothesis in `<bos>`/`<eos>` and extending the output vocab with `<eos>`/`<plh>`.

**Shared dataset layer** (`dataset/`):
- `vocab.py` — `CharVocab` with special tokens: input `<pad> <unk>`; output `<pad> <blank> <bos> <unk>`. `<blank>` is the RNN-T blank emission. `encode` supports multi-char tokens (longest-match) when the vocab contains any.
- `generate.py` / `examples.py` / `noise.py` / `reading.py` — synthesize romaji↔kanji pairs: kanji→reading via SudachiPy, kana→romaji via pykakasi, then add typo/noise to Japanese-derived romaji while preserving commands/paths/identifiers.
- `source_*.py` + `split.py` — external corpus ingest and train/valid splitting.

`eval/metrics.py` — CER (Character Error Rate) for comparing decoded strings to targets, used in validation.

## Data policy

Source code is MIT. Datasets, trained models, external-corpus artifacts, and user learning data are kept separate from source and must not be committed — see `docs/DATA_POLICY.md`. Generated data lands in `data/` and `artifacts/` (gitignored).
