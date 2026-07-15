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

`dataset/source_github.py` builds persona/domain corpora (e.g. the "engineer"
profile-stream persona) from GitHub repositories. Two separate safeguards
apply, for two separate risks:

- **Repository license (README, commit messages, source code if ever added):**
  only repositories under a permissive allow-list are used —
  MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, CC0-1.0, Unlicense.
  Repositories with no LICENSE file (`NOASSERTION`), copyleft licenses
  (GPL/LGPL/AGPL), or NC/ND-style terms are rejected outright, since the
  repository owner is presumed to be the actual copyright holder of these
  artifacts and the license they chose governs reuse.
- **Issues (`--include-issues`, opt-in):** issue titles/bodies may be written
  by any GitHub user, not just the repository owner, so the repository's
  LICENSE cannot be assumed to cover them, and GitHub's Terms of Service only
  grant a narrow in-service viewing license — not a general reuse license.
  Ingesting them is therefore justified on a different basis: Japan's
  Copyright Act Article 30-4 (information analysis exception), which permits
  use of copyrighted works for machine learning / information analysis
  without the rightsholder's permission, independent of the work's declared
  license. This is a narrower, purpose-specific justification than the
  license allow-list above, and it does not extend to redistributing the
  raw issue text itself — only to training on it.

In both cases, author identity (name/email/GitHub login) is never collected,
and issue/commit/README text is scrubbed of email addresses, URLs, GitHub
`@mentions`, and long token-like strings before use. Each ingest run writes a
manifest (repo, SPDX license id, commit SHA, fetch timestamp) so sourcing
decisions stay auditable, mirroring the Aozora Bunko manifest convention.

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
