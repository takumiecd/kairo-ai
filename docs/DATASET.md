# Dataset Generation

Kairo AI starts with synthetic engineer-focused examples before adding external
corpora. The goal is to make the training data match the IME product: mixed
Japanese, English, commands, paths, identifiers, and typing mistakes.

## Quick Start

Generate examples from the built-in synthetic templates:

```bash
python -m dataset.generate --output data/synthetic.jsonl --augmentations 4 --show-vocab
```

Split generated examples into train/valid/test files:

```bash
python -m dataset.split \
  --input data/synthetic.jsonl \
  --output-dir data/synthetic \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1
```

Build examples from an external Aozora-style text or zip source:

```bash
python -m dataset.source_text \
  --source https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip \
  --output data/external/aozora_wagahai.jsonl \
  --source-name aozora \
  --source-detail "Natsume Soseki: Wagahai wa Neko de Aru" \
  --license aozora_public_domain_checked \
  --format aozora \
  --max-units 200 \
  --augmentations 2 \
  --manifest data/external/aozora_wagahai.manifest.json
```

For many Aozora sources, create a manifest and ingest all sources in one command:

```bash
cp docs/AOZORA_SOURCES.example.json data/aozora_sources.json
```

Edit `data/aozora_sources.json`, then run:

```bash
python -m dataset.source_manifest \
  --manifest data/aozora_sources.json \
  --output-dir data/external/aozora \
  --combined-output data/combined/aozora_all.jsonl \
  --max-units 1000 \
  --augmentations 2 \
  --seed 0
```

Then split the combined output:

```bash
python -m dataset.split \
  --input data/combined/aozora_all.jsonl \
  --output-dir data/combined/aozora \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 0
```

For practical training, target at least tens of thousands of generated examples
first, then scale toward hundreds of thousands. Increase the number of Aozora
sources and prefer `augmentations=1` or `2` until validation CER is being tracked.

## Tatoeba

Tatoeba sentence downloads use tab-separated rows:

```text
Sentence id [tab] Lang [tab] Text
```

Download `sentences.csv` or `sentences.tar.bz2` from Tatoeba, then extract Japanese
sentences:

```bash
python -m dataset.source_tatoeba \
  --sentences data/raw/tatoeba/sentences.tar.bz2 \
  --output data/external/tatoeba_ja.jsonl \
  --lang jpn \
  --source-detail "Tatoeba Japanese sentences" \
  --license cc_by_2_0_fr \
  --max-units 50000 \
  --augmentations 1 \
  --manifest data/external/tatoeba_ja.manifest.json
```

Tatoeba text is generally under CC BY 2.0 FR, so keep attribution metadata in
the generated manifest.

## Wikipedia

Download a Japanese Wikipedia pages/articles dump, then extract article text:

```bash
python -m dataset.source_wikipedia \
  --dump data/raw/wiki/jawiki-latest-pages-articles.xml.bz2 \
  --output data/external/wiki_ja.jsonl \
  --source-detail "Japanese Wikipedia pages/articles dump" \
  --license cc_by_sa_gfdl \
  --max-units 100000 \
  --augmentations 1 \
  --manifest data/external/wiki_ja.manifest.json
```

The extractor streams the XML dump, skips redirects and non-article namespaces,
removes common wiki markup, and keeps attribution/license metadata in the manifest.

Preview examples without writing a file:

```bash
python -m dataset.generate --augmentations 2 --show-vocab
```

Disable typo noise for mutable English spans:

```bash
python -m dataset.generate --no-noise-literals
```

## Output Format

The generator writes JSONL records:

```json
{
  "source": "synthetic",
  "source_detail": "engineer_templates",
  "license": "project_owned",
  "input": "git commit -m \"baguwoshuuseishita\"",
  "target": "git commit -m \"バグを修正した\"",
  "clean_input": "git commit -m \"baguwoshuuseishita\"",
  "noise": "none"
}
```

Fields:

- `source`: broad data source category.
- `source_detail`: specific source or generator name.
- `license`: source/license label for auditability.
- `input`: romanized/noisy IME input string.
- `target`: expected mixed Japanese/ASCII output string.
- `clean_input`: non-noisy input generated from the target.
- `noise`: applied noise type, or `none`.

## Current Pipeline

1. Start from synthetic target text.
2. Split mixed text into Japanese and literal spans.
3. Convert only Japanese spans to romaji.
4. Preserve command/path/identifier-like literal spans.
5. Apply noise to Japanese-derived romaji spans and mutable English spans.
6. Emit clean and noisy training examples.
7. Build character-level input/output vocabularies from generated examples.
8. Split generated JSONL into train/valid/test files before training.

Example:

```text
target:
git commit -m "バグを修正した"

clean input:
git commit -m "baguwoshuuseishita"

noisy inputs:
git commit -m "baguuwoshuuseishita"
git commit -m "baguwosyuuseishita"
git commit -m "baguwohsuuseishita"
```

## Literal Noise

Kairo should tolerate typos in both Japanese romaji and English. The generator
therefore applies typing noise to mutable English spans by default.

Protected literal spans include:

- common command words such as `git`, `commit`, `cargo`, `docker`, and `pytest`
- flags such as `-m`
- paths such as `src/main.rs`
- identifiers such as `user_id`

Mutable English spans can still receive noise. For example, `tokenizer` or
`README` may become typo variants in noisy examples.

## Noise Types

The current generator supports:

- QWERTY-adjacent typo
- deletion
- duplication
- adjacent character swap
- romaji variants such as `shi`/`si`, `shu`/`syu`, `wo`/`o`

## Vocab

The first training phase uses character-level vocabularies:

- input vocab: `<pad>`, `<unk>`, ASCII input characters
- output vocab: `<pad>`, `<blank>`, `<bos>`, `<unk>`, Japanese characters, ASCII characters

Learned tokenizers such as SentencePiece/BPE are intentionally delayed until
the RNN-T training and decoding pipeline is working.

## External Corpora

External corpora should be added after the synthetic pipeline is stable.

Initial preferred sources:

- project-owned synthetic examples
- public-domain or clearly reusable Japanese texts
- Aozora Bunko texts after checking per-text copyright and usage notes

Wikipedia dumps may be evaluated later, but attribution and license handling
must be tracked before publishing derived datasets or model artifacts.

Avoid arbitrary web scraping in the initial project.

See `docs/DATA_POLICY.md` for the broader data and license policy.
