# Data Policy

Kairo AI is intended to be an open source, local-first IME machine learning
project. The source code is licensed under the MIT License, but datasets,
trained weights, and user personalization data may have different handling
requirements.

This document is a project policy, not legal advice.

## License Separation

- Source code: MIT License.
- Synthetic datasets created by this project: MIT or CC0, to be decided per release.
- External corpus-derived datasets: governed by the original corpus terms.
- Trained model weights: distributed under terms declared with each model release.
- User personalization data: local user data, not part of the public repository.

## Initial Data Sources

The initial dataset pipeline should prefer sources with clear usage conditions:

- Synthetic engineer-focused examples created by the project.
- Public-domain or clearly reusable Japanese text corpora.
- Aozora Bunko texts, after checking each text's copyright and usage notes.

Wikipedia dumps may be evaluated later, but their attribution and license
requirements must be tracked before derived datasets or model artifacts are
published.

## Sources to Avoid Initially

The project should not start with arbitrary web scraping. In particular, avoid:

- Blog posts and news articles without explicit reusable terms.
- Forum posts, social posts, and private logs.
- Common Crawl-derived data as a default source before license and quality
  handling are designed.

## GitHub-Derived Sources

`dataset/source_github.py` builds an engineer-domain source corpus from GitHub
repositories. The corpus may contribute RawExamples and causal profile
transitions for training, evaluation, and inference-time profile construction.
It must not be assigned one fixed "engineer profile" wholesale: profile state
for a training row must be built only from preceding or disjoint records and
must not contain that row's target.

```text
GitHub README/commits -> source_github -> RawExamples
                                      -> causal ProfileTransitions
                                      -> inference profiles
```

The RawExample, profile linkage/transition, and train/validation/test split are
separate artifacts. Group by repository or source document before splitting;
then construct profile transitions inside each split to prevent target leakage.

- **Repository license (README and commit messages only):** only repositories
  under a permissive allow-list are used —
  MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, CC0-1.0, Unlicense.
  Repositories with no LICENSE file (`NOASSERTION`), copyleft licenses
  (GPL/LGPL/AGPL), or NC/ND-style terms are rejected outright, since the
  repository license is the legal basis for reuse. GitHub's detected SPDX id
  is only an initial gate, not proof of compliance: each ingest run must keep
  the fetched license text and follow any attribution or notice requirements.
- **Excluded sources:** source code, code comments, issues, pull requests, and
  comments are not collected. In particular, issue ingestion is intentionally
  unsupported rather than justified through Copyright Act Article 30-4.

Commit and README text is scrubbed on a best-effort basis for email addresses,
URLs, GitHub `@mentions`, common identity-bearing commit trailers, and long
token-like strings. This is not a guarantee of anonymization; names, addresses,
phone numbers, or sensitive prose can remain and require review before use.
Each ingest run writes a mandatory manifest containing the repository URL,
SPDX license id, full detected license text, commit SHA, fetch timestamp, legal
basis, and usage scope. The corpus and manifest must remain together, and the
original license and notice requirements apply to any redistribution.

## Dataset Metadata

Generated examples should carry source metadata so datasets can be audited later.

Recommended JSONL shape:

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

For external corpus-derived examples, include enough detail to identify the
source corpus, license, and original text record when practical.

## Personalization Data

Kairo's online learning loop should be local-first:

- Do not upload user input or correction history by default.
- Store personalization examples locally.
- Provide a way to delete local personalization data.
- If telemetry or shared training data is ever added, it must be explicit opt-in.

## Model Releases

Each model release should include a model card or release note describing:

- Training data sources.
- Dataset licenses or terms.
- Whether synthetic, external, or user-contributed data was used.
- Evaluation sets.
- Known limitations and privacy assumptions.
