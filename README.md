# Kairo AI (Neural IME Backend)

次世代のエンジニア向けローカルAI IME「Kairo」の機械学習リポジトリです。
このリポジトリでは、データセットの自動生成から、ライブ変換を可能にするRNN-T（Transformer-Transducer）モデルの学習までを行います。学習済みモデルは軽量化（ONNX/Safetensors）され、Rustバックエンド（`kairo` リポジトリ）に組み込まれます。

## 🎯 目指すUX（ユーザー体験）
1. **スペースキーは変換に使わない**：スペースキーはただの空白として入力されます（コーディングやコマンド入力に最適化）。
2. **英語・日本語の混在を文脈で自動判定**：`git commit -m "shuseishita"` は `git commit -m "修正した"` に自動で変換（アンダーライン表示）。
3. **矢印キーによる候補選択**：AIの推論（トップ候補）が意図と異なる場合のみ、上下矢印キーで裏で計算されている別のビームサーチ候補を選択します。
4. **学習のループ（Online Learning）**：ユーザーが矢印キーで候補を修正して確定した履歴は、AIにとって最高の正解データになります。定期的にローカルで微調整（LoRA等）を行い、ユーザー専用のIMEに進化します。

## 🧠 アーキテクチャ選定：Transformer-Transducer
タイピング中にリアルタイム（数ミリ秒）で文脈を予測し続けるため、RNN-T（現代版：Transformer-Transducer）を採用します。
- **Encoder (Acoustic Model)**: タイピングされたローマ字（アルファベット）を受け取り、特徴量に変換します。
- **Prediction Network (Language Model)**: これまでに確定・出力された日本語（過去の文脈）を保持し、次に来る文字を予測します。
- **Joint Network**: Encoderの出力とPrediction Networkの出力を足し合わせ、Softmaxで「次の出力文字」または「空白（これ以上出力しない＝次のタイピングを待つ）」を予測します。

## 📂 ディレクトリ構成
- `dataset/`: 日本語コーパス（青空文庫やWikipedia）から、「タイピング（ローマ字）→ 正解（漢字かな混じり）」のペアを自動生成するスクリプト群。
- `model/`: PyTorchによるTransformer-Transducerのモデル定義。
- `train/`: 学習ループ、損失計算（RNN-T Loss）、評価用スクリプト。
- `tests/`: ユニットテスト。

## 🚀 環境構築
Anaconda/Miniconda で Python 3.11 環境を作成します。

```bash
conda create -y -n kairo-ai python=3.11 pip
conda activate kairo-ai
python -m pip install .
python -m unittest discover -s tests
```

## 🧪 データセット生成
まずはエンジニア向けの synthetic examples から、ローマ字入力と漢字かな混じり出力の JSONL を生成します。

```bash
python -m dataset.generate --output data/synthetic.jsonl --augmentations 4 --show-vocab
python -m dataset.split --input data/synthetic.jsonl --output-dir data/synthetic
```

生成器は日本語spanをローマ字化し、コマンド・パス・識別子を保持しつつ、日本語由来のromajiと一部の英語spanにtypo/noiseを追加します。詳細は [`docs/DATASET.md`](docs/DATASET.md) を参照してください。

## 🏋️ Train Smoke Test
生成した JSONL で、RNN-T loss と backward が通ることを確認します。小さい subset に絞ると、overfit できるかも確認できます。

```bash
python -m dataset.generate --output data/synthetic.jsonl --augmentations 1
python -m train.overfit --data data/synthetic.jsonl --steps 30 --batch-size 2 --embed-dim 16 --hidden-dim 32 --learning-rate 0.01 --max-examples 4 --eval-every 10
```

評価指標として、文字列同士の CER(Character Error Rate) を `eval.metrics` に用意しています。decode実装後は予測文字列とtargetの比較に使います。

## 🚂 Training
checkpoint、config、vocab を保存する通常の学習 entrypoint です。

```bash
python -m train.train \
  --data data/synthetic.jsonl \
  --output-dir artifacts/run1 \
  --epochs 5 \
  --batch-size 4 \
  --embed-dim 64 \
  --hidden-dim 128 \
  --learning-rate 0.001
```

学習成果物は `artifacts/run1/` 以下に保存されます。

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

## 📜 ライセンスとデータ方針
このリポジトリのソースコードは MIT License で公開します。

データセット、学習済みモデル、外部コーパス由来の成果物、ユーザーの個人学習データは、ソースコードとは分けて扱います。詳細は [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md) を参照してください。
