# Kairo AI Roadmap

Kairo AI is the machine learning backend for a local, engineer-focused IME.
The first target is a small, debuggable RNN-T system that can learn live
romanized input to mixed Japanese/ASCII output conversion.

## Core Direction

- Start with an LSTM-based RNN-T.
- Use character-level input and output vocabularies first.
- Keep ASCII/code fragments as output tokens instead of forcing everything into Japanese.
- Add typing noise and romanization variants in the dataset generator.
- Delay SentencePiece/BPE or other learned tokenizers until the RNN-T pipeline works.

## Important Model Boundary

The model should learn probabilities for the next output token, including when to emit
`<blank>`. Beam search should not be treated as learned model parameters.

Instead:

- The model owns token probabilities.
- The decoder owns search, ranking, and candidate assembly.
- The exported runtime can package model + decoder together as one IME inference unit.

This means Kairo can still expose a single "model-like" API to the Rust backend:

```text
input prefix + committed output context
  -> top-k conversion candidates
  -> scores/confidence
  -> underlined live candidate
```

The beam search behavior can be configured and shipped with the model artifact, but
the probability of a word or phrase should come from the RNN-T logits and accumulated
sequence score.

## Phase 0: Dataset MVP

Goal: turn raw text into clean/noisy training pairs.

- Convert only Japanese spans to romaji.
- Preserve ASCII/code spans such as:
  - `git`
  - `commit`
  - `-m`
  - `src/main.rs`
  - `user_id`
  - `README.md`
- Generate clean input/output pairs.
- Generate noisy variants from each clean input.
- Build character-level vocabularies from generated examples.
- Add fixed tests for Japanese, shell commands, paths, identifiers, and mixed text.

Example:

```text
target:
git commit -m "バグを修正した"

clean input:
git commit -m "baguwoshuuseishita"

noisy inputs:
git commit -m "baguwosyuuseisita"
git commit -m "baguwoshuuseishta"
git commit -m "baguwoshuuseishitaa"
```

## Phase 1: RNN-T Minimal Training

Goal: prove the model and loss are wired correctly.

- Add small synthetic training data.
- Implement padding and tensor collation.
- Use `torchaudio.functional.rnnt_loss`.
- Define special tokens:
  - input: `<pad>`, `<unk>`
  - output: `<pad>`, `<blank>`, `<bos>`, `<unk>`
- Overfit a tiny dataset first.
- Implement greedy decoding.
- Verify loss decreases and simple examples decode correctly.

## Phase 2: Decoder and Beam Search

Goal: produce multiple IME candidates from the model.

- Implement RNN-T beam search.
- Return top-k candidates with accumulated log scores.
- Keep decoder state cacheable for live typing.
- Support committed prefix + live uncommitted candidate.
- Define the inference API expected by the external `kairo` Rust repository.

The decoder should be part of the inference runtime, even if it is not learned
inside the neural network weights.

## Phase 3: Engineer Context Dataset

Goal: make the data match the product.

- Add command-line examples.
- Add commit messages.
- Add Markdown snippets.
- Add source code comments.
- Add file paths and identifiers.
- Add mixed English/Japanese examples.
- Add typo/no-typo evaluation splits.

Noise types:

- QWERTY-adjacent typo
- Deletion
- Duplication
- Adjacent character swap
- Romanization variants:
  - `shi` / `si`
  - `chi` / `ti`
  - `tsu` / `tu`
  - `fu` / `hu`
  - `ja` / `jya`
  - `shu` / `syu`
  - `cho` / `tyo`
  - `wo` / `o`
  - `nn` / `n`

## Phase 4: Export and Rust Integration

Goal: make the model usable by Kairo.

- Export model weights to a portable format.
- Decide whether decoder runs in Python, ONNX-compatible code, or Rust.
- Measure latency with realistic live typing.
- Tune beam width, hidden size, and vocab size.
- Keep the API stable enough for the Rust backend.

## Phase 5: Personalization

Goal: adapt to the user's history via an external profile. Model parameters
stay fixed per user — no local fine-tuning (LoRA-style updates are
abandoned; see [PROFILE.md](PROFILE.md)).

- Build the profile builder (feedback.jsonl → profile at runtime; corpus
  stream → virtual profile at training time; same code for both).
- Stage A: trie-based score fusion in `decode/beam.py` (no retraining).
- Stage B (after measuring Stage A): profile-conditioned pretraining with
  virtual user streams; inject profile embedding into the prediction network.
- Keep personalization private and local-first.

## Immediate Implementation Plan

Create the dataset layer first:

```text
dataset/
  reading.py
  noise.py
  generate.py
  vocab.py

tests/
  test_dataset_generation.py
  test_vocab.py
```

First success condition:

```text
raw text
  -> clean input/output pair
  -> noisy input variants
  -> char vocab
  -> encoded tensors
```

Only after this is stable should the training loop and RNN-T loss become the main focus.
