# ユーザープロファイル設計 (Profile Design)

パーソナライズの正式方針とその定式化。設計判断の背景は kairo リポジトリの
[docs/12-パーソナライズ.md](../../kairo/docs/12-パーソナライズ.md) を参照。

**核となる決定:**

1. **モデルはユーザーごとに学習し直さない。** パラメータ $\theta$ は全ユーザー
   共通・固定。ユーザー適応は外部プロファイル $u$ だけに宿る。
2. **変換は常に一本道** $f_\theta(x, c, u)$。強制辞書などパーソナライズ専用の
   別経路は持たない。
3. **実行時プロファイルと学習時の仮想プロファイルを、同一のビルダーで作る。**
   これが崩れると「推論時と同じ状況を学習時に再現する」が成立しない。

> **旧方針からの変更:** ユーザーの feedback で LoRA をローカル微調整する経路
> (`train.rnnt.lora`)は**廃止**する。θ を動かすため、破滅的忘却・モデル汚染・
> 評価不能性の問題を持ち込むからである。`feedback.jsonl` の収集経路とスキーマ
> ([FEEDBACK_SCHEMA.md](FEEDBACK_SCHEMA.md))は**存続** — プロファイル
> ビルダーの入力として新方針の中核データになる。

## 0. 全体像

$$\hat{y} \;=\; \arg\max_{y}\;\; \underbrace{\log p_\theta(y \mid x,\, c,\, e(u))}_{\text{固定モデル（段階B条件付け）}} \;+\; \underbrace{\lambda_{\mathrm{exp}}\, \Phi_{\mathrm{exp}}(y; u) \;+\; \lambda_{\mathrm{imp}}\, \Phi_{\mathrm{imp}}(y; u)}_{\text{デコーダ側スコア融合（段階A）}}$$

- $x = x_{1:T}$: 打鍵列、$c$: 周辺文脈、$y = y_{1:U}$: 出力文字列
- $\theta$: 全ユーザー共通・固定。$u$: ユーザープロファイル(モデル外の状態)
- **段階A**(スコア融合)は学習不要でモデル無変更。$e(u)$ を落とし
  $\log p_\theta(y \mid x, c)$ を素のモデルとして先行実装する
- **段階B**(条件付け)は段階Aで効果を測ってから投資する

## 1. 固定モデル本体（RNN-T）

ブランク $\varnothing$ を含むアラインメント $a$ の周辺化:

$$p_\theta(y \mid x) \;=\; \sum_{a \,\in\, \mathcal{B}^{-1}(y)} \; \prod_{i} P\big(a_i \,\big|\, t(i),\, u(i)\big)$$

各格子点 $(t, u)$ のトークン分布は三つの部品で決まる
([model/transducer.py](../model/transducer.py)):

$$h_t = \mathrm{Enc}_\theta(x_{1:T})_t \qquad \text{（双方向・入力は確定済み）}$$

$$g_u = \mathrm{Pred}_\theta(y_{<u}) \qquad \text{（causal・既出力のみ）}$$

$$z_{t,u} = W_2 \tanh\!\big(W_1 [\,h_t \,;\, g_u\,]\big), \qquad P(k \mid t, u) = \mathrm{softmax}(z_{t,u})_k, \quad k \in \mathcal{V} \cup \{\varnothing\}$$

出力語彙は当面**文字レベル**を維持する。トライ融合(§3)との相性が良く、
RNN-T 語彙を小さく保て、専門用語の OOV 問題はプロファイル側で吸収する、
という役割分担が方針と噛み合うため。

## 2. プロファイルの構造と更新則

$$u \;=\; \big(\underbrace{E}_{\text{明示}},\; \underbrace{F}_{\text{頻度}},\; \underbrace{r}_{\text{recency}},\; \underbrace{d}_{\text{domain}},\; \underbrace{\ell}_{\text{lang}},\; \underbrace{N}_{\text{総確定量}}\big)$$

```
profile
├─ meta:      version, base_profile_id, total_units N（総確定文字数）
├─ explicit:  E（明示的シグナル・減衰なし）
│   entries: [ { input, surface, accept数, reject数, source: 修正|登録 } ]
├─ implicit:  （暗黙的シグナル・減衰あり）
│   ├─ unigram F:  surface → { count: float, reading, last_used: N時点 }
│   │              ※上位K件でキャップ（例 10k）
│   ├─ recency r:  直近 R 語のリングバッファ（例 500）
│   ├─ domain d:   { code, prose, chat } 正規化ベクトル
│   └─ lang ℓ:     { ja比率, 英トークン率 }
```

更新はイベント(確定・修正・却下)ごとの作用素で書ける:

$$u_{N} \;=\; \mathcal{U}(u_{N-1},\, \mathrm{event}_N)$$

暗黙的カウントは **lazy decay** 付きで読む(半減期 $H$ は総確定文字数単位。
書き込み O(1)、全エントリの定期スキャン不要):

$$\tilde{c}(w; N) \;=\; c(w) \cdot 2^{-\big(N - N_{\mathrm{last}}(w)\big)/H}$$

明示的エントリ $E$ は減衰させない。

**シグナルの非対称性:** 惰性の採用(暗黙)より、修正・却下(明示的な意思表示)を
強く反映する。却下は $E$ 内の負のシグナルとして持つ(§3 の $\gamma$ 項)。

## 3. 段階A — デコーダ側スコア融合（先行実装）

単語 $w$ ごとのボーナス:

$$B_{\mathrm{exp}}(w) = \log\!\big(1 + \mathrm{acc}(w)\big) \;-\; \gamma \log\!\big(1 + \mathrm{rej}(w)\big)$$

$$B_{\mathrm{imp}}(w) = \min\!\Big(\log\!\big(1 + \tilde{c}(w)\big),\; \kappa\Big) \;+\; \rho \cdot \mathbb{1}[\,w \in r\,]$$

- $\kappa$: 自己増幅ループ抑制のキャップ
- $\gamma$: 却下ペナルティの重み
- $\rho$: recency ボーナス

文字レベルのビームに載せるため、プロファイル語彙をトライ(接頭辞木)に持ち、
**ポテンシャル関数**として定義する:

$$\Phi(y_{1:u}) \;=\; \sum_{w \,\in\, \mathrm{Words}(y_{1:u})} B(w) \;+\; \beta \cdot \mathrm{TrieDepth}\big(\mathrm{suffix}(y_{1:u})\big)$$

ビーム展開([decode/beam.py](../decode/beam.py))の1ステップに加算するのは
その**差分**:

$$s(k \mid t, u) \;=\; \log P(k \mid t, u) \;+\; \lambda \Big[\Phi\big(y_{1:u} \oplus k\big) - \Phi\big(y_{1:u}\big)\Big]$$

ポテンシャル差分にしておくと、トライを途中まで辿って外れた仮説は
$\mathrm{TrieDepth}$ 項が消えて**部分ボーナスが自動的に回収される**
(rollback の明示的な実装が不要)。

旧ハードオーバーライド(`accepted.tsv`)はこの式の極限として回収される:

$$\lambda_{\mathrm{exp}} \to \infty \;\implies\; \text{登録変換が必ず勝つ（旧挙動）}$$

つまり「必ずこの変換」は独立した仕組みではなく、**保証の強さがパラメータに
なった**だけである。既定では「文脈が圧倒的に反対しない限り勝つ」程度の
有限の強い重みとする。

## 4. 段階B — プロファイル条件付け（効果検証後）

プロファイルを固定次元の条件ベクトルに落とす:

$$e(u) \;=\; \mathrm{MLP}\Big(\big[\, d \,;\, \ell \,;\, \mathrm{pool}\{\mathrm{emb}(w) : w \in \mathrm{top}\text{-}K(F)\}\,\big]\Big)$$

注入点は **Prediction Network**(投資先は LM 側、kairo docs/11-定式化 §5 の
帰結。Encoder は小さいまま):

$$g_u = \mathrm{Pred}_\theta\big(y_{<u} \,;\, e(u)\big) \qquad \text{（プレフィックストークンとして先頭に注入）}$$

CLAS 拡張(さらに効果があれば)はプロファイルエントリ集合へのアテンション:

$$g_u^{\mathrm{ctx}} = g_u + \mathrm{Attn}\big(g_u,\; \{\mathrm{emb}(w_j)\}_{w_j \in E \cup \mathrm{top}\text{-}K(F)}\big)$$

進め方: 段階A(トライ融合) → B-1(粗い埋め込み $e(u)$) → B-2(CLAS)。
各段で前段をベースラインに効果を測ってから次へ投資する。

## 5. 学習 — 仮想ユーザーストリーム

ペルソナ $\pi$(エンジニア型 / 小説執筆型 / Wikipedia 型 / 日常会話型 /
日英混在型...)からコーパス文書ストリーム $s_1, \dots, s_M$ をサンプルし、
**実行時と同一のビルダー $\mathcal{U}$** を時系列に流す:

$$u_m \;=\; \mathcal{U}(u_{m-1},\, s_m), \qquad u_0 = \text{空 or 初期プロファイル}$$

学習サンプルは「直前までのスナップショット → 次の文」:

$$\mathcal{L}(\theta) \;=\; \mathbb{E}_{\pi}\; \mathbb{E}_{m}\; \Big[\, \mathcal{L}_{\mathrm{RNNT}}\big(\underbrace{E(s_m) + p_{\mathrm{typo}}}_{\text{合成打鍵}},\;\; c_m,\;\; u_{m-1}\,;\; s_m \big) \Big]$$

この方式の利点:

- **スパースプロファイル問題が自動的に解ける。** $m$ が小さいスナップショットは
  自然に「まばらで成長途中」なので、コールドスタート分布は時系列を守るだけで
  学習データに入る。実ユーザーがゼロから育てる過程と同じ分布になる。
- **ペルソナ生成が「コーパスの選択」に還元される。** 青空文庫→小説家型、
  コード混じりドキュメント→エンジニア型。既存の `dataset/` 合成器に
  ビルダーを一段挟むだけでよい。

頑健化(プロファイル過依存の防止): 確率 $p_{\mathrm{drop}}$ で
$e(u_{m-1}) \to e(u_0)$ に置換する。

明示的シグナル $E$ は実行時にしか発生しないので合成する: ストリームの未来
$s_{>m}$ に出現する語をサンプルし「過去にユーザーが修正・登録した」ことに
して $E$ に注入する。これで「登録した語がちゃんと出る」能力を学習時から
仕込める。

## 6. 一行での要約

**$\theta$ に入るのは期待値(ペルソナ分布上の平均的な適応能力)、$u$ に入るのは
個人の実現値。** 学習と推論でビルダー $\mathcal{U}$ を共有することが、この
分離を成立させる唯一の接着剤である。

## 7. 実装順序

1. **プロファイルビルダー $\mathcal{U}$**: feedback.jsonl → profile(実行時
   モード)と、コーパスストリーム → profile(学習時モード)の両対応
2. **段階A トライ融合**: `decode/beam.py` にポテンシャル差分のスコア加算
3. **評価**: ペルソナ別評価セット、ユーザー修正率、明示的登録の遵守率、
   コールドスタート回帰
4. **段階B**: B-1(粗い埋め込み) → 効果があれば B-2(CLAS)。仮想ユーザー
   ストリーム学習はここで導入
