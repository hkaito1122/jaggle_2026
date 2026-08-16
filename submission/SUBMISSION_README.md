# 提出プログラムと出力結果の対応

最終提出ファイルと、それを生成するプログラムの対応を記載する。

- 実行環境: Google Colab（`50_`〜`62_`, `64_`, `66_` は CPU / `63_`, `65_`, `68_`, `70_` は GPU推奨・CPU可）
- データ配置: `data/input/`（配布ファイルをそのまま）
- 出力先: `data/output/<実行日>/`
- 実験の全経緯: `data/output/submit_result_report.md`

---

## 1. 最終提出ファイル

| 項目 | 内容 |
|---|---|
| ファイル | `data/output/20260816/20260816_pool_top150_hire_fixed_avg.csv` |
| Public スコア | **0.5086987224728512** |
| 生成プログラム | **`src/66_final_blend.ipynb`（第5節）** |

### レシピ

```
pooled = 0.5 × (プールC × TabPFN(top150)) + 0.5 × (プールC × TabPFN+TabICL(hire_fixed、TabDPT除く))
       = プールC × 0.70 + TabPFN(top150) × 0.15 + [TabPFN+TabICL(hire_fixed)平均] × 0.15
```

`68_foundation_models_3way` で、TabPFN・TabICLが月次列を全て除いた`hire_fixed`（78列）で
突出した検証結果を示した（[[monthly-data-information-ceiling]]と整合、TabDPTのみ逆傾向）。
`70_hire_fixed_fm_submission`でTabPFN+TabICLを`hire_fixed`でTest予測し（TabDPTは検証で
Publicが悪化すると確認されたため除外）、AutoGluonプールとブレンド（w=0.70固定）した
`pool_fm2_hire_fixed_w70`（単体でPublic 0.509339）を作成。これと下記1bの旧最良候補
（`pool_tabpfn_blend_w70`、単体でPublic 0.508793）は相関0.9971・MAD 0.01635の独立した
同水準候補であり、Jensenの不等式（loglossは確率について凸）より期待loglossは単純平均以下
になることが理論的に保証される（[[private-lb-variance-strategy]]）。実際にPublicでも
両者の単純平均(0.508699)が良い方の単体(0.508793)をわずかに上回った
（差-0.0000944はノイズ床未満だが、pooling戦略が悪化させないことの初の実証）。

`66_final_blend.ipynb` は、提出済みCSVとの**完全一致を assert で検証する**セルを含む
（第1〜3節: `pool_tabpfn_blend_w70`との最大絶対差1.665e-16／第5節: この`pool_top150_hire_fixed_avg`
との最大絶対差2.2e-16、いずれも浮動小数点誤差のみ）。

---

## 1b. 旧最良（参考、単体でも高スコア）

| 項目 | 内容 |
|---|---|
| ファイル | `data/output/20260816/20260816_pool_tabpfn_blend_w70.csv` |
| Public スコア | 0.5087930948139492 |
| 生成プログラム | `src/66_final_blend.ipynb`（第1〜3節） |

```
最終予測 = 0.70 × AutoGluonプールC + 0.30 × TabPFN(top150)
```

重み 0.70 は `63_` のブレンド重み走査に基づく。曲線が平坦（w=0.5とw=0.7の差は+0.0006のみ）で
あることを確認した上で、ブレンド相手が単層CatBoostより強いAutoGluonプールであることを
考慮して決めた。**Publicスコアを見て走査した値ではない**（Private評価に対する過学習を避けるため）。
現在は`1.`のpooled候補の材料の半分として使われている。

---

## 2. 実行順序と、各プログラムの出力

`66_` は再学習を行わず、保存済みの予測を合成するだけ（数秒）。1〜6・8〜9 は互いに独立で、
順不同に実行してよい。

### 出力A（`1b.`、Public 0.508793）を再現するのに必要なノートブック

| # | プログラム | 主な出力（`data/output/<日付>/`配下） | 役割 |
|---|---|---|---|
| 1 | `src/50_autogluon_memofix.ipynb` | `..._AG50_full441_weighted.csv` | AutoGluon 441列 |
| 2 | `src/51_autogluon_catboost_bias.ipynb` | `..._AG51_full441_weighted.csv` | + NN_TORCH/KNN 除外 |
| 3 | `src/53_autogluon_dystack.ipynb` | `..._AG53_full441_weighted.csv` | + DyStack 有効 |
| 4 | `src/61_autogluon_extended_time.ipynb` | `..._AG61_full441_weighted.csv` | + time_limit 6時間 |
| 5 | `src/62_autogluon_seed_averaging.ipynb` | `..._AG62_bag16_weighted.csv`<br>`..._full441_seed{42,2024,7}_weighted_testpreds.npy` | シード3本 + bag16 |
| 6 | `src/63_tabpfn_ensemble.ipynb` | `..._tabpfn_top150_testpreds.npy` | TabPFN v2（上位150列・3シード）|
| 7 | **`src/66_final_blend.ipynb`（第1〜3節）** | **`..._pool8_tabpfn_w70.csv`**<br>`..._provenance.csv` | **1〜6 を合成** |

### 出力B（`1.`、Public 0.508699・現最良）を再現するのに追加で必要なノートブック

上記1〜6に加えて:

| # | プログラム | 主な出力 | 役割 |
|---|---|---|---|
| 8 | `src/68_foundation_models_3way.ipynb` | 検証専用（Test予測なし）| TabPFN/TabICL/TabDPT×5特徴量セットの比較。`hire_fixed`を特定 |
| 9 | `src/70_hire_fixed_fm_submission.ipynb` | `..._tabpfn_hire_fixed_testpreds.npy`<br>`..._tabicl_hire_fixed_testpreds.npy` | TabPFN・TabICLを`hire_fixed`(78列)でTest予測 |
| 7' | **`src/66_final_blend.ipynb`（第5節）** | **`..._pool_top150_hire_fixed_avg.csv`** | **出力Aのblendと出力9の平均を50:50で合成** |

`66_` は材料の由来一覧（`provenance.csv`）も出力するので、
どの出力がどのノートブック由来かを後から追える。

### AutoGluonプールC の内訳（8本の単純平均）

上表 1〜5 が出力した **441列・WeightedEnsemble** の予測8本。
`56_autogluon_lm_block` は特徴量が444列（LMブロック込み）で異質なため**除外**している。

---

## 3. 特徴量パイプライン

1〜6 のノートブックは**それぞれが同一の特徴量生成コードを自己完結で持つ**（441列、ブロックL2）。
共通ライブラリ化していないのは、Colab上で各ノートブックを単独実行できるようにするため。

系譜:

| 出自 | 内容 |
|---|---|
| `12_`〜`18_` | 月次集約（16指標×14統計）、テキストTF-IDF+SVD、四半期/加速度特徴量、部署Target Encoding |
| `28_` | ブロックL2（入社時メモの「転居許容 × 希望勤務地一致」交互作用）|
| `49_` | メモパーサーのカバレッジ修正（見出しが無い書式Bの5.24%を救済）|

リーク対策として、部署Target Encoding は**学習期間のID集合のみでfit**している
（`prepare_split()` の `fit_ids` 引数）。

---

## 4. 環境・依存

| ライブラリ | バージョン指定 | 備考 |
|---|---|---|
| `autogluon.tabular` | 指定なし（最新）| **`ray` は入れない**（`47_`で pyarrow のバイナリ非互換により全モデルが学習前に落ちた）|
| `tabpfn` | **`==2.2.1` 固定必須** | 最新(8.x)はライセンス同意＋APIトークンを要求し `TabPFNLicenseError` で落ちる。2.x系がTabPFN v2世代でApache 2.0ベース |
| `catboost`, `optuna` | 指定なし | |

`tabpfn` を入れ替えた場合は**ランタイムの再起動が必要**（`sys.modules` のキャッシュのため）。
`63_`/`65_` はバージョン不一致を検知して、原因（pip未適用か再起動漏れか）を切り分けて停止する。

---

## 5. 補助・検証用のプログラム（旧最良`1b.`の再現には`68_`/`70_`は不要。現最良`1.`には必要）

| プログラム | 内容 |
|---|---|
| `src/54_l2_m_interaction.ipynb` | 単層CatBoostの最良構成（Public 0.515030）|
| `src/57_kitchen_sink_all_features.ipynb` | 全特徴量統合と組み合わせ探索（新規特徴量は0件）|
| `src/64_autogluon_more_seeds.ipynb` | AutoGluonプールをシード追加で厚くする |
| `src/65_tabpfn_scaleup.ipynb` | TabPFN側をシード数・特徴量数の複数構成で厚くする |
| `src/67_foundation_model_features.ipynb` | TabPFN/TabICLの特徴量選択方針を比較（`68_`の前身、TabDPT未対応版）|
| `src/68_foundation_models_3way.ipynb` | TabPFN/TabICL/TabDPT×5特徴量セットの比較。**`1.`の生成に必要** |
| `src/69_catboost_native_text.ipynb` | CatBoostネイティブtext_features・在籍月数回帰ブレンド+Platt較正を検証。4構成ともPublicで不採用確定 |
| `src/70_hire_fixed_fm_submission.ipynb` | `hire_fixed`でのTabPFN/TabICL/TabDPT Test予測。**`1.`の生成に必要** |
| その他 `src/*.ipynb` | 各種の検証・棄却された仮説。経緯は `submit_result_report.md` を参照 |

---

## 6. ライセンス適合性（SIGNATE参加規約 第2条8項への対応）

参加規約 第2条8項は「**提出物に商業利用が禁止されているOSSを利用又は組み込んではならない**」と
定めている。本提出物が組み込む機械学習ライブラリのライセンスを、
**インストール済みパッケージのLICENSE全文**で確認した結果は以下のとおり。

| ライブラリ | バージョン | ライセンス（実物で確認） | 商用利用 | 判定 |
|---|---|---|---|---|
| **tabpfn** | **2.2.1** | **Prior Labs License v1.1**（Apache 2.0の派生。第10条で帰属表示を追加）| 可 | ✅ 適合 |
| autogluon.tabular | — | Apache-2.0 | 可 | ✅ 適合 |
| catboost | — | Apache-2.0 | 可 | ✅ 適合 |
| scikit-learn / pandas / numpy | — | BSD-3-Clause | 可 | ✅ 適合 |

### ⚠️ tabpfn のバージョン固定は規約適合上の必須要件

`tabpfn` の **2.5 / 2.6 / 3系（PyPI上の 6.x〜8.x）は非商用ライセンス**であり、
第2条8項に抵触するため**使用できない**。本提出物は `tabpfn==2.2.1`（v2世代、Apache 2.0派生）に
固定しており、`63_` および `66_` のインストールセルでバージョンを明示、
さらに実行時に `tabpfn.__version__.startswith("2.")` を検証して不一致なら停止する。

### Prior Labs License 第10条に基づく帰属表示

> **Built with PriorLabs-TabPFN**

本提出物の予測は、Prior Labs 社の TabPFN（v2、Prior Labs License v1.1）を構成要素として含む。
同ライセンスの写しは、インストールされた `tabpfn` パッケージの `LICENSE` ファイルに含まれる。

### tabicl（現最良`1.`で実際に使用）

| ライブラリ | ライセンス | 商用利用 |
|---|---|---|
| **tabicl** | **BSD 3-Clause（Inria Soda team）** | **可** |

`70_hire_fixed_fm_submission.ipynb` が現最良`1.`（`pool_top150_hire_fixed_avg.csv`）の
材料としてTabICLを実際に使用している。BSD 3-Clauseは商用利用可であり、
第2条8項に抵触しない。

### tabdpt（検証のみ、最終提出物・候補ともに不使用）

| ライブラリ | ライセンス | 商用利用 |
|---|---|---|
| tabdpt | Apache-2.0 | 可 |

`68_foundation_models_3way.ipynb` で評価したが、`hire_fixed`特徴量セットでは他モデルより
明確に劣り（第102節）、`70_`のTest予測でも含めるとPublicが悪化すると確認された（第103-104節）。
そのため`1.`・`1b.`とも不採用。ライセンス自体はApache-2.0で商用利用可。

### 補足

上記はパッケージ同梱のLICENSE全文を確認した事実の整理であり、法的助言ではない。
判断に迷う場合は規約第2条7項に従い、ホストへ確認すること。
