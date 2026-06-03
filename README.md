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
[uv](https://github.com/astral-sh/uv) などのモダンなパッケージマネージャを利用して構築します。
```bash
uv pip install -r pyproject.toml
```

## 📜 ライセンスとデータ方針
このリポジトリのソースコードは MIT License で公開します。

データセット、学習済みモデル、外部コーパス由来の成果物、ユーザーの個人学習データは、ソースコードとは分けて扱います。詳細は [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md) を参照してください。
