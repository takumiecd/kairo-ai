# Kairo AI (Neural IME Backend)

次世代のエンジニア向けローカルAI IME「Kairo」の機械学習リポジトリです。
このリポジトリでは、データセットの自動生成から変換モデルの学習までを行います。変換モデルは、タイピング中に逐次変換する「リアルタイムモデル」と、文を丸ごと一括変換する「一発出力モデル」の2系統を用意します（詳細は後述）。学習済みモデルは軽量化（ONNX/Safetensors）され、Rustバックエンド（`kairo` リポジトリ）に組み込まれます。

## 🎯 目指すUX（ユーザー体験）
1. **スペースキーは変換に使わない**：スペースキーはただの空白として入力されます（コーディングやコマンド入力に最適化）。
2. **英語・日本語の混在を文脈で自動判定**：`git commit -m "shuseishita"` は `git commit -m "修正した"` に自動で変換（アンダーライン表示）。
3. **矢印キーによる候補選択**：AIの推論（トップ候補）が意図と異なる場合のみ、上下矢印キーで裏で計算されている別のビームサーチ候補を選択します。
4. **学習のループ（Online Learning）**：ユーザーが矢印キーで候補を修正して確定した履歴は、AIにとって最高の正解データになります。定期的にローカルで微調整（LoRA等）を行い、ユーザー専用のIMEに進化します。

## 🧠 モデルの2系統
変換の出し方が異なる2系統のモデルを用意します。役割で分けるだけで、それぞれの内部アーキテクチャは固定せず、実験しながら選定します。設計方針の詳細は [`docs/MODEL_DESIGN.md`](docs/MODEL_DESIGN.md) を参照してください。

### 1. リアルタイムモデル（逐次出力）
タイピング中に、打った分から逐次的に変換結果を出し続けるモデルです。ライブ変換・矢印キーでのbeam候補選択など、入力途中のUXを担います。

- **Encoder**: タイピングされたローマ字（アルファベット）を受け取り、特徴量に変換します。
- **Prediction Network**: これまでに確定・出力された日本語（過去の文脈）を保持し、次に来る文字を予測します。
- **Joint Network**: Encoderの出力とPrediction Networkの出力を結合し、「次の出力文字」または「空白（これ以上出力しない＝次のタイピングを待つ）」を予測します。

現状はRNN-T（Transformer-Transducer）系で実装していますが、Encoder/Prediction の具体構造（LSTM/Transformer等）は固定せず差し替え可能にします。

### 2. 一発出力モデル（一括変換）
入力が一区切りついた時点で、文を丸ごと一括で変換・推敲するモデルです。逐次モデルの結果を磨いて精度を上げる後段としても使えます。

- 非自己回帰（並列に全位置を出力）や反復編集など、丸ごと出すことに適した構造を想定します。
- このリポジトリには `model/edit_refiner.py`（反復編集）や discrete diffusion 実験など、この系統の試作が含まれます。

将来的には「リアルタイムモデルで即時変換 → 手が止まったら一発モデルで推敲」というパイプライン構成も視野に入れます。

## 📂 ディレクトリ構成
- `dataset/`: 日本語コーパス（青空文庫やWikipedia）から、「タイピング（ローマ字）→ 正解（漢字かな混じり）」のペアを自動生成するスクリプト群。
- `model/`: PyTorchによるTransformer-Transducerのモデル定義。
- `train/`: 学習ループ、損失計算（RNN-T Loss）、評価用スクリプト。
- `tests/`: ユニットテスト。

## 🚀 環境構築
[uv](https://docs.astral.sh/uv/) で管理します。`uv.lock` をコミットしているため、開発環境（macOS）と学習サーバー（Linux/CUDA）で同一の依存バージョンが再現されます。

```bash
uv sync
uv run python -m unittest discover -s tests
```

以降のコマンド例の `python -m ...` は `uv run python -m ...` と読み替えてください。

<details>
<summary>conda を使う場合（代替手段）</summary>

```bash
conda create -y -n kairo-ai python=3.11 pip
conda activate kairo-ai
python -m pip install .
python -m unittest discover -s tests
```

※ lockファイルによるバージョン固定は効かないため、再現性が必要な場面では uv を推奨します。

</details>

## 🤗 学習済みモデルを試す（Quickstart）

学習済みモデル **rnnt-trf-v1** を [Hugging Face](https://huggingface.co/takumiecd/kairo-rnnt-trf-v1) で公開しています。
Wikipedia・Tatoeba・青空文庫から構築した約30万ペアで学習したモデルで、checkpoint・config・vocab が揃っているため、このリポジトリのコードだけですぐ推論できます（CPUで動作可）。

```bash
# モデル一式（checkpoint + config + vocab）をダウンロード
# （hf CLI は uv sync で入る huggingface_hub に同梱）
uv run hf download takumiecd/kairo-rnnt-trf-v1 --local-dir artifacts/rnnt-trf-v1

# greedy decode で推論
python -m decode.greedy \
  --artifact-dir artifacts/rnnt-trf-v1 \
  --checkpoint artifacts/rnnt-trf-v1/epoch_020.pt \
  --input "wagahaihanekodearu."
# => 吾輩は猫である。
```

タイプミスを含む入力も復元できます。

```bash
python -m decode.greedy \
  --artifact-dir artifacts/rnnt-trf-v1 \
  --checkpoint artifacts/rnnt-trf-v1/epoch_020.pt \
  --input "wagaahaihanekodearu."   # 'a' が重複したtypo入り
# => 吾輩は猫である。
```

モデルの詳細（データセット構成・学習設定・評価結果）は [モデルカード](https://huggingface.co/takumiecd/kairo-rnnt-trf-v1) を参照してください。

## 🧪 データセット生成
まずはエンジニア向けの synthetic examples から、ローマ字入力と漢字かな混じり出力の JSONL を生成します。

```bash
python -m dataset.generate --output data/synthetic.jsonl --augmentations 4 --show-vocab
python -m dataset.split --input data/synthetic.jsonl --output-dir data/synthetic
```

生成器は日本語spanをローマ字化し、コマンド・パス・識別子を保持しつつ、日本語由来のromajiと一部の英語spanにtypo/noiseを追加します。詳細は [`docs/DATASET.md`](docs/DATASET.md) を参照してください。

外部テキストは、明示したURLまたはローカルファイルから取り込みます。まずは青空文庫形式のテキスト/zipを対象にします。

```bash
python -m dataset.source_text \
  --source https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip \
  --output data/external/aozora_wagahai.jsonl \
  --source-name aozora \
  --license aozora_public_domain_checked \
  --format aozora
```

複数作品をまとめて取り込む場合は、`docs/AOZORA_SOURCES.example.json` を元にURL一覧を作り、manifest ingestを使います。

```bash
python -m dataset.source_manifest \
  --manifest data/aozora_sources.json \
  --output-dir data/external/aozora \
  --combined-output data/combined/aozora_all.jsonl
```

Tatoebaの短文データも取り込めます。

```bash
python -m dataset.source_tatoeba \
  --sentences data/raw/tatoeba/sentences.tar.bz2 \
  --output data/external/tatoeba_ja.jsonl \
  --lang jpn \
  --max-units 50000 \
  --augmentations 1
```

Wikipedia dumpから現代語・固有名詞・技術語を増やせます。

```bash
python -m dataset.source_wikipedia \
  --dump data/raw/wiki/jawiki-latest-pages-articles.xml.bz2 \
  --output data/external/wiki_ja.jsonl \
  --license cc_by_sa_gfdl \
  --max-units 100000 \
  --augmentations 1
```

## 🏋️ Train Smoke Test
生成した JSONL で、RNN-T loss と backward が通ることを確認します。小さい subset に絞ると、overfit できるかも確認できます。

```bash
python -m dataset.generate --output data/synthetic.jsonl --augmentations 1
python -m train.rnnt.overfit --data data/synthetic.jsonl --steps 30 --batch-size 2 --embed-dim 16 --hidden-dim 32 --learning-rate 0.01 --max-examples 4 --eval-every 10
```

評価指標として、文字列同士の CER(Character Error Rate) を `eval.metrics` に用意しています。decode実装後は予測文字列とtargetの比較に使います。

## 🚂 Training
checkpoint、config、vocab を保存する通常の学習 entrypoint です。

```bash
python -m train.rnnt.train \
  --data data/synthetic/train.jsonl \
  --valid-data data/synthetic/valid.jsonl \
  --output-dir artifacts/run1 \
  --epochs 5 \
  --batch-size 4 \
  --embed-dim 64 \
  --hidden-dim 128 \
  --learning-rate 0.001
```

validation中にdecodeしてCERも確認できます。`greedy` は軽く、`beam` は重いのでサンプル数と頻度を制限します。

```bash
python -m train.rnnt.train \
  --data data/combined/aozora/train.jsonl \
  --valid-data data/combined/aozora/valid.jsonl \
  --output-dir artifacts/aozora-rnnt-v1 \
  --epochs 100 \
  --device cuda \
  --valid-decode greedy \
  --valid-cer-samples 100 \
  --valid-cer-every 5
```

学習成果物は `artifacts/run1/` 以下に保存されます。

途中から再開する場合:

```bash
python -m train.rnnt.train \
  --data data/synthetic/train.jsonl \
  --valid-data data/synthetic/valid.jsonl \
  --output-dir artifacts/run1 \
  --epochs 20 \
  --resume artifacts/run1/checkpoints/epoch_010.pt
```

## 🔎 Greedy Decode
学習済みartifactからモデルとvocabを復元し、greedy decodeで予測文字列を確認します。

```bash
python -m decode.greedy \
  --artifact-dir artifacts/run1 \
  --input 'git commit -m "baguwoshuuseishita"' \
  --show-next-token-probs
```

初期段階の短い学習では出力品質より、checkpoint復元と推論経路が通ることを確認します。

Beam searchで候補とbeam内confidenceを見る場合:

```bash
python -m decode.beam \
  --artifact-dir artifacts/run1 \
  --input 'git commit -m "baguwoshuuseishita"' \
  --beam-width 5
```

## 🫧 Discrete Diffusion Experiment

`train.diffusion.train` は、ローマ字入力・任意の前文 context・ノイズ化した
出力 canvas を単一モデルで参照し、正解文を反復復元します。初期 canvas は
予測長ぶんの `<mask>` なので、空の状態から非自己回帰で生成できます。

小規模な学習確認:

```bash
python -m train.diffusion.train \
  --data data/synthetic/train.jsonl \
  --valid-data data/synthetic/valid.jsonl \
  --output-dir artifacts/kairo-diffusion-v1 \
  --epochs 20 \
  --batch-size 16 \
  --device cuda \
  --diffusion-steps 8 \
  --valid-decode greedy \
  --valid-cer-samples 100
```

全 `<mask>` から反復 decode:

```bash
python -m decode.diffusion \
  --artifact-dir artifacts/kairo-diffusion-v1 \
  --input 'git commit -m "baguwoshuuseishita"' \
  --context '直前の確定済み文章'
```

JSONL に任意の `context` フィールドがあれば文脈encoderへ渡します。現段階の
実験モデルは length head が最初に系列長を決め、denoising中は長さを固定します。
可変長の insert/delete diffusion は、固定長denoisingの安定性を確認した後に追加します。

## 📜 ライセンスとデータ方針
このリポジトリのソースコードは MIT License で公開します。

データセット、学習済みモデル、外部コーパス由来の成果物、ユーザーの個人学習データは、ソースコードとは分けて扱います。詳細は [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md) を参照してください。
