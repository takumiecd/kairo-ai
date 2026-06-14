# FEEDBACK_SCHEMA — 確定フィードバックの正準スキーマ

Kairo の IME（`kairo` リポジトリ、Swift）でユーザが変換を**確定**したときに記録する
イベントの、唯一の正準スキーマです。`kairo`（記録側）と `kairo-ai`（学習側）が
この1本の契約を共有します。サーバ収集・データセット化・LoRA 微調整・登録辞書の
すべてが、このイベント列からの**派生**として組み立てられます。

> このファイルは**スキーマ仕様**であり、ユーザ学習データそのものではありません。
> 実データ（`feedback.jsonl` 等）は `docs/DATA_POLICY.md` に従い別管理・非コミットです。

## 1. ストレージ

確定イベントは **append-only JSONL**（1行1イベント）として書き出します。

```text
~/.config/kairo/feedback.jsonl        # 正準: append-only イベント列（このスキーマ）
~/.config/kairo/feedback_counts.json  # 派生: 集計（left_context|input -> {output: count}）
~/.config/kairo/accepted.tsv          # 派生: 高頻度エントリの登録辞書（input<TAB>output...）
```

`feedback.jsonl` が真実の源で、`feedback_counts.json` と `accepted.tsv` はそこから
再生成可能な派生物です。

## 2. イベントスキーマ（v1）

```json
{
  "v": 1,
  "ts": "2026-06-14T12:34:56Z",
  "client_id": "anon-1f2e3d4c-....",
  "input": "wagahaihanekodearu.",
  "output": "吾輩は猫である。",
  "left_context": "",
  "right_context": "",
  "source": "neural",
  "candidate_rank": 0,
  "accepted": true
}
```

| フィールド | 型 | 必須 | 意味 |
|------------|----|:----:|------|
| `v` | int | ✓ | スキーマバージョン。互換のための移行に使う。現行 `1`。 |
| `ts` | string | ✓ | 確定時刻。ISO 8601 / UTC（例 `2026-06-14T12:34:56Z`）。 |
| `client_id` | string | ✓ | インストール単位の**匿名**安定 ID（`anon-<uuid>`）。PII を含めない。 |
| `input` | string | ✓ | 変換対象として backend に渡したローマ字列（reading）。 |
| `output` | string | ✓ | 確定された日本語テキスト。 |
| `left_context` | string | ✓ | 直前の確定済みテキスト（文脈依存学習用）。無ければ空文字。 |
| `right_context` | string | ✓ | 直後のテキスト。現状はほぼ空文字。 |
| `source` | string | ✓ | 候補の出所: `neural` / `dict` / `kana` / `raw`。 |
| `candidate_rank` | int | ✓ | 採用した beam 候補の順位（0 = top-1）。`>0` は「ユーザが別候補を選んだ」= 強い補正シグナル。 |
| `accepted` | bool | ✓ | そのまま確定なら `true`。将来、編集を伴う確定は `corrected_from` 等の拡張で表す。 |

### 設計上の性質

- **append-only**: 既存行は書き換えない。並行書き込み・端末間マージに強い。
- **counts は加算的**: `left_context|input -> {output: count}` は加算なので、複数端末・複数ファイルを
  衝突なくマージできる（CRDT 的）。これが将来のサーバ集約／多端末同期の土台。
- **補正シグナルの識別**: `candidate_rank > 0` や（将来の）編集確定は、top-1 をそのまま受けた
  ケースより学習価値が高い。データセット化・LoRA 時の重み付けに使う。
- **privacy**: `client_id` は匿名 UUID のみ。`feedback.jsonl` のローカル記録は常時行うが、
  外部（将来のサーバ）への送信は**オプトイン**。`docs/DATA_POLICY.md` 準拠で非コミット・別管理。

## 3. 派生物の作り方

### feedback_counts.json（集計）
キー `"{left_context}|{input}"` ごとに `{output: count}` を加算。既存の IME 実装が
書いている形式に合わせる。

### accepted.tsv（登録辞書）
`(input, output)` の出現回数が閾値以上のものを TSV エントリ
（`input<TAB>output<TAB>type<TAB>priority`）として書き出す。`kairo` 側はこれを user 辞書と
同様に読み込み、登録単語の**確定変換**（hard-override）に使う。手動登録（`user.tsv`）と
自動登録（`accepted.tsv`）の両輪。

### 学習レコード（kairo-ai）
`dataset/source_feedback.py`（予定）で `feedback.jsonl` を
`{"input": <romaji>, "target": <japanese>}` の学習レコードへ変換し、count / confidence で
選別・dedup の上、既存の `dataset.split` パイプラインに合流させて LoRA 微調整に使う。

## 4. バージョニング

スキーマ変更時は `v` をインクリメントし、本ドキュメントに差分を追記する。読み込み側は
未知フィールドを無視し、欠落フィールドは既定値で補う（前方・後方互換を優先）。
