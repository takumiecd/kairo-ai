# ユーザープロファイル設計 (Profile Design)

パーソナライズの正式方針と定式化。設計判断の背景は kairo リポジトリの
`docs/12-パーソナライズ.md` を参照。

## 0. 核となる決定

1. **デフォルトは一様な定数ではない。** 想定ユーザー母集団における、入力ごとの
   条件付き分布 $\mu(y\mid x)$ をデフォルトとする。
2. **個別プロファイルはデフォルトとの差分で表す。** ユーザー $u$ の時点 $t$ の
   状態を $\delta_{u,t}$ とし、履歴が無いときは $\delta_{u,0}=0$ とする。
3. **ユーザーごとのモデルやLoRAを作らない。** オフライン学習では、全profileで
   共有する一つのモデルがデフォルト経路とprofile差分を同時に学ぶ。実行時には
   共有モデルを固定し、外部プロファイル状態だけを更新する。
4. **学習対象は完成プロファイルではなく更新則とスコア差分。** 更新前後の
   transitionから、選択候補だけを局所的に動かす方法を学ぶ。
5. **入力 $x$ は構造化入力全体を指す。** 現在のローマ字列だけでなく、利用可能な
   確定済み文脈も $x$ に含む。モデル内部では別経路で符号化してよい。

旧LoRA方針は廃止する。既存のprofile-conditioned RNN-T (`train.rnnt.profile`)
は旧段階Bの実験実装として残るが、本章の主経路ではない。

## 1. デフォルト分布

デフォルトは、どの入力にも同じ値を返すprofileではない。想定母集団における
条件付き分布である。

$$
\mu(y\mid x)=P_{\mathrm{population}}(Y=y\mid X=x)
$$

有限データからは次の経験分布を推定する。

$$
\hat\mu_N(y\mid x)
=
\frac{\sum_{i=1}^{N}\mathbf{1}[X_i=x,\,Y_i=y]}
     {\sum_{i=1}^{N}\mathbf{1}[X_i=x]}
$$

実際には同一の $x$ が疎なので、RNN-T等の共有パラメータモデルで近い入力間を
一般化する。データ量を無限に増やしても一様分布にはならない。収集元の混合比へ
収束するため、母集団は明示的な重みで定義する。

$$
P_{\mathrm{population}}
=\sum_s w_s P_s,
\qquad \sum_s w_s=1
$$

$s$ は synthetic / Wikipedia / Aozora / GitHub / conversation 等のsource群。
単純なレコード数ではなく、Kairoが想定する利用分布に基づいて $w_s$ を決める。

## 2. 個別プロファイルは分布差分

ユーザー $u$ の分布を、デフォルト分布の指数傾斜として表す。

$$
p_u(y\mid x)
=
\frac{\mu(y\mid x)\exp\delta_u(x,y)}
     {\sum_{y'}\mu(y'\mid x)\exp\delta_u(x,y')}
$$

したがって差分は対数確率比に対応する。

$$
\delta_u(x,y)=\log p_u(y\mid x)-\log\mu(y\mid x)+C(x,u)
$$

$\delta_u=0$ なら必ず $p_u=\mu$ となる。複数階層を持つ場合は加法分解する。

$$
\delta_{u,t}
=\delta_{\mathrm{persona}(u)}
 +\delta_{\mathrm{user}(u)}
 +\delta_{\mathrm{recent}(u,t)}
$$

たとえば全体平均との差、engineer平均との差、個人の長期差、直近セッション差を
独立に保持できる。識別可能性のため、母集団上の差分平均をゼロに制約する。

$$
\mathbb{E}_{u}[\delta_u]=0
$$

## 3. 未確定プロファイル

履歴が少ない段階で $\delta_u$ を一点推定しない。平均ゼロの事前分布から始め、
ユーザー履歴 $D_u$ によって事後分布を更新する。

$$
\delta_u\sim\mathcal{N}(0,\Sigma),
\qquad
p(\delta_u\mid D_u)\propto p(D_u\mid\delta_u)p(\delta_u)
$$

予測は事後分布で周辺化する。

$$
p(y\mid x,D_u)
=\int p(y\mid x,\delta_u)p(\delta_u\mid D_u)d\delta_u
$$

候補集合が有限なら、デフォルト分布を擬似カウント事前分布として使える。

$$
p_u(y\mid x,D_u)
=
\frac{\alpha\mu(y\mid x)+n_u(x,y)}
     {\alpha+\sum_{y'}n_u(x,y')}
$$

$\alpha$ はデフォルトを何観測分信頼するかを表す。履歴ゼロではデフォルトと一致し、
履歴が増えるほど個人分布へ連続的に移る。

## 4. 実行時スコア

共有モデルのデフォルトスコアを $b_\theta(y\mid x)$、プロファイル差分を
$\Delta_\theta(y,x,\delta_{u,t})$ とする。両者は別ユーザー用の重みではなく、
同じ共有パラメータ $\theta$ の二つの経路として共同学習する。

$$
s(y\mid x,u,t)
=b_\theta(y\mid x)
 +g_\theta(x)\,\Delta_\theta(y,x,\delta_{u,t})
$$

$g(x)$ は基盤候補のエントロピーや1位・2位のmarginから決める曖昧性ゲート。
入力だけで候補が明確なら補正を弱め、候補が衝突するときだけ強める。

空プロファイルの補正は構造上ゼロにする。

$$
\Delta_\theta(y,x,0)=0
$$

非線形モデルでは、次の差として定義すればこの条件を厳密に満たせる。

$$
\Delta_\theta(y,x,\delta)
=F_\theta(y,x,\mu+\delta)-F_\theta(y,x,\mu)
$$

既存の `decode.profile_fusion` は、頻度・recency・明示採用/却下から
$\Delta$ を規則計算する先行実装と位置付ける。学習版では、すべてのprofileを
混ぜた共通損失で $b_\theta$ と $\Delta_\theta$ を共同最適化する。profileごとの
LoRA、adapter、モデル複製は作らない。

## 5. プロファイル更新

プロファイルはイベント列を前向きに再生して作る。現在状態から過去状態を逆算
しない。

$$
\delta_{u,t}
=\rho_t\delta_{u,t-1}+K_\psi(e_t)
$$

- $e_t$: 確定・採用・修正・却下・明示登録イベント
- $K_\psi$: イベントを疎なprofile差分へ変換する更新則
- $\rho_t$: 長期減衰とrecencyを表す係数

学習・評価サンプルは必ず更新前状態を使う。

```text
profile_before = snapshot(u)
predict current target with profile_before
apply current event to u
profile_after = snapshot(u)
```

現在targetをprofileへ追加してから同じtargetを予測してはならない。

## 6. 学習データ

Raw DatasetとProfile Transitionを分離する。

```text
RawExample
  example_id, source_id, input, target, license/provenance

ProfileTransition
  stream_id, sequence_index, example_id,
  profile_before, event, profile_after,
  candidates, selected, rejected
```

Raw Datasetはsynthetic / Wikipedia / Aozora / GitHub等からprofile非依存に作る。
その後、独立したストリーム生成工程でprofileとの因果的な対応を付ける。コーパス
全体に固定の `engineer` ラベルを貼るだけでは、profileではなくsource分類を
学習してしまうため禁止する。

同じ入力に複数のtargetがあるcollision groupを重要サンプルとする。

```text
koushou -> {交渉, 工廠, 高尚, 校章}
```

同じ入力・ほぼ同じ文脈に対し、`profile_before`だけが異なる最小対立ペアを作る。
一方、曖昧でない一般変換ではprofileを入れ替えても出力が変わらない不変例を作る。

## 7. 差分学習

デフォルト経路とprofile経路は、最初から同じ共有モデルとして共同学習する。

$$
\mathcal{L}
=\mathcal{L}_{\mathrm{base}}
 +\lambda_p\mathcal{L}_{\mathrm{profile}}
 +\lambda_d\mathcal{L}_{\mathrm{delta}}
 +\lambda_l\mathcal{L}_{\mathrm{local}}
$$

- $\mathcal{L}_{\mathrm{base}}$: 空profileでの通常変換損失
- $\mathcal{L}_{\mathrm{profile}}$: 現在profileでの候補順位/変換損失
- $\mathcal{L}_{\mathrm{delta}}$: 更新後にselectedがrejectedより上がるmargin損失
- $\mathcal{L}_{\mathrm{local}}$: 更新と無関係な候補の差分を変えない制約

各optimizer stepへ、空profile、成長途中のprofile、異なるstream/profileを
混ぜて入れる。これによりデフォルト経路だけ、または特定profileだけが直近batchへ
過適合することを防ぐ。必要なら短いwarm-upは使えるが、以後デフォルト経路を
freezeして別モデルを学ぶ段階には分けない。更新される $\theta$ は常に全profile
共通であり、profile固有の学習済みパラメータは持たない。ユーザー実行時には
勾配更新せず、$\delta_{u,t}$だけを更新する。

## 8. Splitと評価

train / validation / testは文単位ではなく、原典文書・リポジトリ・stream単位で
先に分割する。その後、split内だけでtransitionを生成する。同じtargetを含む
文書からprofileを作って同じtargetを予測する漏洩を防ぐ。

評価指標:

- 全体CERとprofile無し回帰
- collision groupのTop-1精度 / MRR
- Beam Recall@K
- profile swapで適切に順位が変わる割合
- profile蓄積量に対する改善曲線
- $H(Y\mid X)-H(Y\mid X,U)=I(Y;U\mid X)$ の推定

Profileの価値は、入力だけでは残る不確実性をどれだけ減らせるかで測る。

## 9. 実装移行順序

1. 既存Stage Aをデフォルトスコア＋規則ベース差分として評価
2. RawExample / ProfileTransitionスキーマとgroup splitを実装
3. collision groupと最小対立ペアを生成
4. 疎な $\delta$ と未確定度を扱うprofile補正器を実装
5. 空profileと成長中profileを混ぜた共有モデルの共同学習を実装
6. 旧 `train.rnnt.profile` と新補正器を同一評価セットで比較し、旧段階Bの存廃を決定

## 10. 一行での要約

**デフォルトは母集団の条件付き分布 $\mu(y\mid x)$、個別profileはそこからの
未確定な差分 $\delta_{u,t}$、学習対象は完成profileではなく更新前後の局所的な
候補スコア差分である。**
