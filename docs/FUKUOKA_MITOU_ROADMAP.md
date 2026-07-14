# 福岡未踏 × IPA未踏 ロードマップ

> 2026年6月16日時点の計画。対話ベースで策定。

## 全体戦略

2段構えで挑む：

1. **福岡未踏 Grow 2026** → Kairo（AI IME）のネイティブ推論化
2. **IPA未踏 2027** → SCD（Sparse Candidate Discovery）で独立テーマとして応募

福岡未踏の経験（PM伴走・発表・修了生コミュニティ）を礎に、IPA未踏へステップアップする。Grow修了後はJumpコース（継続メンタリング）も活用可能。

---

## 福岡未踏 Grow 提案テーマ

### Kairo: エンジニア向けAI日本語入力エンジンのネイティブ推論化

#### 現状

- **kairo-ai** (Python): RNN-T (Transformer-Transducer) 学習済み。rnnt-trf-v1 (CER 5.87%) を HuggingFace に公開。ビームサーチ、LoRA個人化、Edit Refiner / Discrete Diffusion のプロトタイプも存在。
- **kairo** (Rust): 変換パイプライン完成 (tokenizer → protector → typo → backend → scorer → join)。辞書バックエンドが標準、RemoteBackend で Python AI サーバーとHTTP接続。FFIクレートあり。

#### 課題

推論が Python サーバー経由。エンジニアが日常使いするIMEとして、Pythonプロセスを常駐させるのは非実用的。

#### Grow期間でやること

RNN-T推論をONNX/Rustベースのネイティブバイナリに移行し、Pythonサーバー不要でリアルタイム変換（<50ms/キーストローク）を実現する。

---

## 提案書の骨格（福岡未踏の審査基準に対応）

### 何を作るか

エンジニア向けAI日本語入力エンジンKairoの推論をネイティブ化する。現在はPyTorchモデル＋PythonサーバーのRNN-T変換を、ONNX/Rustベースの単一バイナリに変え、リアルタイムで動作する推論ランタイムを完成させる。

### 何が斬新か
d
「ただONNXを使う」のではない。以下の技術的チャレンジがある：

1. **RNN-Tは動的制御フロー** — emit（出力する）か wait（次の入力を待つ）をステップごとに判断する。静的グラフへのエクスポートが自明でない。encoder / prediction network / joint network を個別にONNXエクスポートし、制御ループをRustで書く。
2. **ストリーミング推論** — IMEでは1キーストロークごとに再推論が必要。Encoder KVキャッシュのPrefix再利用なしには実用にならない。
3. **ビームサーチのRust実装** — モデルはforward passだけ提供し、探索ロジックはRust側。PyTorchのビームサーチをそのまま移植できない。
4. **量子化と日本語精度のトレードオフ** — INT8量子化でCER（文字誤り率）が劣化しないかの検証。
5. **全パイプラインのネイティブ化** — protector → typo → neural → scorer が Python 不要で動く。

### 何が変わるか

- エンジニアが `git commit -m "baguwoshuuseishita"` と打つと `git commit -m "バグを修正した"` に即時変換。コマンドやパスは壊れない。
- Pythonもクラウドも不要、ネイティブバイナリだけで動く。
- 既存IME（Google日本語入力、macOS標準）はエンジニアのコンテキスト（コード混在）を理解しない。Kairoはそこに特化。

### 5W1H

| 項目 | 内容 |
|------|------|
| **Who** | ソフトウェアエンジニア（日本語でコミット、チャット、ドキュメントを書く人） |
| **When** | コーディング中、常時。1日数百回のタイピングごとに推論が走る |
| **Where** | ローカルマシン上（macOS → 将来的にクロスプラットフォーム） |
| **What** | 日本語入力時にコード・英語が壊れる問題を解決 |
| **Why** | 既存IMEはエンジニアのコンテキストを理解しない。スペース変換がコーディングと干渉する |
| **How** | RNN-Tモデルのネイティブ推論。「世の中に足りない」のはエンジニア向けに特化したローカルAI変換エンジン |

### 要素技術の分解と独自性

```
提案の構成要素:
├── ローマ字→日本語変換モデル (RNN-T)        ← 学習済み、HF公開済み
├── tokenizer + protector (コード保護)        ← Rustで実装済み
├── ONNX エクスポート                         ← ★ Phase 0 で PoC
├── Rust 推論ランタイム (ort or candle)       ← ★ Grow の核心
├── ストリーミング推論 + KV キャッシュ         ← ★ Grow の核心
├── Rust ビームサーチ                          ← ★ Grow の核心
├── 量子化 (INT8)                             ← ★ Grow で検証
└── OS IME 統合                               ← Grow 後（将来）

太刀打ちできない系（やらない）: 汎用IME、LLMベースの変換
選択肢あり系（差別化困難）: GUI、認証、CRUD
────────────────────────────────────
残った領域 = Kairo の勝ち筋:
  「RNN-Tストリーミング推論のRustネイティブ実装 + エンジニア文脈のコード保護」
```

---

## Phase 0: 応募準備（6/16 → 8/8 エントリー → 8/23 面接）

### 目標

応募時点で「ONNXエクスポート + Rust greedy decode が動く PoC」を持っている状態にする。

### 週次計画

#### W1 (6/16-22): ONNX エクスポート

- `torch.onnx.export` で encoder / prediction network / joint network を個別エクスポート
- dynamic axes 設定（batch, seq_len）
- Transformer の attention mask / positional encoding の ONNX 互換性確認
- PyTorch 出力と ONNX Runtime 出力の数値一致を検証
- **成果物**: 3つの `.onnx` ファイルが ONNX Runtime で推論可能

#### W2 (6/23-29): Rust ランタイム比較

- `ort` crate で ONNX モデルの forward pass を実行
- `candle` で同等のモデル実装を試行（可能な範囲で）
- 比較: レイテンシ、バイナリサイズ、API の使い勝手
- **成果物**: ランタイム選定の判断材料（比較表）

#### W3 (6/30-7/6): Rust Greedy Decode PoC

- 選定したランタイムで RNN-T の greedy decode を Rust 実装
- emit / wait ループを Rust で制御、モデルは forward pass のみ
- E2E: romaji 文字列 → Rust → ONNX forward → greedy decode → 日本語文字列
- PyTorch の greedy decoder と出力一致を検証
- **成果物**: `cargo run -- "wagahaihanekodearu."` → `吾輩は猫である。`

#### W4 (7/7-13): PoC 整理 + ベンチマーク

- Python 推論 vs Rust/ONNX 推論のレイテンシ比較表
- モデルサイズ比較（PyTorch checkpoint vs ONNX）
- 提案書の構成検討開始（5W1H + 要素技術分解）
- **イベント**: 7/12 説明会（参加推奨）

#### W5 (7/14-20): 提案書ドラフト v1

- 技術PoC: レイテンシ改善余地の調査（KVキャッシュ、量子化の初期実験）
- 提案書: 指定フォーマットで v1 ドラフト作成
- **成果物**: 提案書の骨格完成

#### W6 (7/21-27): 提案書 v2 推敲

- PoC の数値で技術的主張を裏付け（「現状 ○○ms → 目標 ○○ms」の実測）
- 提案書にベンチマークデータを埋め込み

#### W7 (7/28-8/3): エントリー提出

- エントリーフォーム提出（事前エントリー締切 8/3 頃）
- 提案書最終版の推敲・校正

#### W8 (8/4-8): 提案書提出 + バッファ

- 提案書提出（締切 8/7-8 頃）

#### 面接準備 (8/9-23)

- プレゼン資料作成
- デモ準備: ターミナルで romaji 入力 → 日本語出力を見せる
- Q&A 練習: 「なぜ ONNX？」「candle ではなく？」「5ヶ月で本当にできる？」
- **イベント**: 8/23 二次審査（対面プレゼンテーション）

---

## Phase 1: Grow 開発期間（2026年9月 → 2027年1月）

### M1 (9月): ONNX 完全エクスポート

- Phase 0 の PoC を本格化
- エクスポートの自動化（学習 → ONNX → 検証のパイプライン）
- edge case（長い入力、空入力、ASCII only）の検証

### M2 (10月): Rust 推論ランタイム

- ort / candle 統合を kairo crate に組み込み
- `OnnxBackend` を `KanaKanjiBackend` trait として実装
- greedy decode の本格実装 + テスト
- **イベント**: 9/19-20 中間発表合宿

### M3 (11月): ビームサーチ + ストリーミング

- Rust でビームサーチ実装（PyTorch 版と出力一致検証）
- Encoder KV キャッシュによる Prefix 再利用（1キー入力ごとの増分推論）
- 1キーストロークあたりのレイテンシ計測
- **イベント**: 11/21 中間発表会（公開）

### M4 (12月): 量子化 + 最適化

- INT8 動的量子化
- 量子化前後の CER 比較
- レイテンシ目標: <50ms/キーストローク on Apple Silicon
- **イベント**: 12/27-29 成果報告合宿

### M5 (1月): 統合 + デモ + 成果報告

- `OnnxBackend` を kairo のデフォルトバックエンドとして統合
- Python 完全不要の E2E デモ
- ベンチマーク公開（レイテンシ、モデルサイズ、CER）
- **イベント**: 1/30 成果報告会（公開・最終ゴール）

### 成果

**Pythonサーバー不要で、ローマ字→日本語のリアルタイム変換が動く Rust バイナリ。**

---

## Phase 2: IPA未踏 2027（SCD）

- 福岡未踏修了後、Jump コース（継続メンタリング）に応募可能
- SCD（Sparse Candidate Discovery）を独立テーマとして IPA 未踏に提案
- Kairo とは完全に別の研究テーマ（ニューラルネットのスパース学習）
- 目標: IPA 未踏クリエータ認定

---

## 注意事項・TODO

- [ ] 2026年度1期の正確な日程を公式サイトで確認（スライドの日程は予定）
- [ ] 提案書フォーマット（.docx テンプレート）をダウンロード
- [ ] 7/12 の説明会参加を検討
- [ ] チーム構成の検討（1人 or 最大3人）
- [ ] エントリーフォーム URL の確認（2026年度版が公開され次第）

---

## 参考リンク

- [福岡未踏 公式サイト](https://mitou-fukuoka.org/)
- [公募要領](https://mitou-fukuoka.org/koubo/)
- [2026 学生向け説明資料](https://www.docswell.com/s/yara/K278J1-fukuoka-mitou-202605-for-applicants)
- [Kairo RNN-T v1 モデル (HuggingFace)](https://huggingface.co/takumiecd/kairo-rnnt-trf-v1)
