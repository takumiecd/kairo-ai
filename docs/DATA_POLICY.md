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
