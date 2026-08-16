# Claude 永続メモリ（このプロジェクト分、全件）

本ファイルは `~/.claude/projects/.../memory/` 配下のメモリファイル全件を、一言一句そのまま集約したものです。要約・言い換えは行っていません。
総ファイル数: 35（索引 `MEMORY.md` 含む）
最終同期: 2026-08-16

---

## MEMORY.md（索引）

```markdown
- [Ensemble OOF overfitting](ensemble_oof_overfitting.md) — hand-learned weights on raw models overfit OOF and lose on Public; but AutoGluon's built-in WeightedEnsemble atop bagging+stacking is different — it BEAT best_single on Public in 51_ (0.514263 vs 0.515123), now the rule is scoped to "naive weight learning" only.
- [Best submission status](best_submission_status.md) — 現在の最良: プールC×TabPFNブレンド Public 0.508793（AutoGluon 8実行平均 × TabPFN v2 top150 を w_AG=0.70）。1位との差0.012643。
- [Validation gap finding](validation_gap_finding.md) — 2-split stability tracks Public for interpretable business-logic features (20_) but not for high-dim embeddings (19_); cross-hardware CatBoost noise can exceed GPU-only noise floor (21_).
- [Local test not decisive](local_test_not_decisive_for_ablation.md) — don't call a feature-ablation result "confirmed" from a local-machine run; only Colab + Public settle it.
- [Edit no-op error gotcha](edit_noop_error_gotcha.md) — Edit's "old_string and new_string are the same" error means the call itself was a no-op mistake, not that the file was already correct; always re-grep.
- [Dept target-encoding leak](dept_target_encoding_leak.md) — small-group target encoding must be fit on train-period IDs only, not full train, under chronological holdout CV.
- [Feature importance vs generalization](feature_importance_vs_generalization.md) — high CatBoost importance doesn't mean the feature generalizes; ablate each block solo before trusting it.
- [CatBoost GPU non-determinism](catboost_gpu_nondeterminism.md) — same-config runs across sessions differ by ~0.003-0.006 val logloss; don't trust smaller deltas.
- [Future action: text embeddings](future_action_text_embeddings.md) — try pretrained Japanese sentence embeddings instead of TF-IDF (user-requested backlog item).
- [EDA v3 findings](eda_v3_findings.md) — memo-text parsing & prior-exp/grade consistency promising; new: self-study (自己学習実施月数) column, unused before, promising too.
- [EDA v4 findings](eda_v4_findings.md) — location-match block (J) confirmed on Public; early-raise timing (K) looked strong but failed on Public — see hire-time-fixed vs monthly-behavioral hypothesis.
- [EDA v5 findings](eda_v5_findings.md) — relocation-tolerance × location-match interaction: 38.3pt gap, CONFIRMED on Public as new best (0.529672). 5 other "hire-time-fixed" candidates were null/redundant.
- [Checkpoint Drive-sync gotcha](checkpoint_drive_sync_gotcha.md) — local Mac and Colab share the same Drive-mounted checkpoint path; a local pre-run can make Colab silently skip computation and return stale results.
- [GBDT interaction type matters](gbdt_interaction_type_matters.md) — cross-field equality features (L/J) help GBDT a lot; plain AND-of-two-raw-categories (block M, 30.5pt gap but p=4e-30) made CatBoost worse since trees already learn that natively.
- [Test set is survivor-filtered](test_set_is_survivor_filtered.md) — Test has zero employees who quit within 24 months, Train has 129; drop them from the validation set (gap 0.018→0.007), not from training.
- [Horizon-dependent feature effects](horizon_dependent_feature_effects.md) — hire-time-fixed features strengthen with horizon, behavioral ones weaken (残業 r=-0.50@24m → -0.30@120m); this is why G/H/K/N/O kept failing.
- [Modeling levers beat new features](modeling_levers_beat_new_features.md) — CONFIRMED on Public: full-train refit -0.0065, seed averaging -0.0003 (insurance, not expected gain); seed sd 0.005-0.008 exceeds the effect sizes being judged.
- [EDA v6 findings](eda_v6_findings.md) — feedback text ruled out by exhaustive n-gram mining (0 survive Bonferroni); manager effect, cohort size, self-study portability all confounded proxies.
- [Ablation cannot settle feature blocks](ablation_cannot_settle_feature_blocks.md) — only raw EDA effect sizes (E/J/L, 25-38pt gaps) ever transferred to Public; G passed the strictest validation ever (4/4 bootstrap CIs) and still flipped sign — nested splits + winner's curse.
- [Hire-year extrapolation](hire_year_extrapolation.md) — 入社年 is a feature but Train(2011-2014.3) and Test(2014.4-2017.3) are disjoint; all 2,502 Test rows fall past the training range into a 60-sample leaf.
- [Early-turnover literature review](literature_review_early_turnover.md) — literature/ 配下のレビュー: 産業・職業は効かず規模と賃金カーブが効く(小林2016)。文献は新特徴量の供給源としては枯れており、価値は既存設計の裏づけとデータ側の欠陥発見だった。
- [Hyperparameter retuning exhausted](hyperparameter_retuning_exhausted.md) — retuning found depth 4→5 at 113 cols but T1b's predictions returned to baseline: a different path to the same function, not a different function.
- [CatBoost beats other GBDT](catboost_beats_other_gbdt.md) — CatBoost +0.032 vs LightGBM, +0.026 vs XGBoost on the 113-col set; the top features are categorical. First decisive validation result in the project.
- [Refit chaos noise floor](refit_chaos_noise_floor.md) — any feature-matrix change shifts Test predictions ~0.014 MAD; that equals a seed redraw. Submit only if MAD > 0.02122.
- [Validation asymmetry](validation_asymmetry.md) — 6 val/Public pairs: validation was right 1/5 times it said "better", 1/1 when it said "worse". Use it only as a floor filter.
- [AutoGluon stacking wins](autogluon_stacking_wins.md) — 8-fold bagging + stack level 2 on 441 cols beat the hand-built single-layer CatBoost by 0.004; retracts "AutoGluon doesn't work" and "441→113 reduction helped".
- [Memo parser coverage bug](memo_parser_coverage_bug.md) — 5.24% of memos lacked the header the L2 parser assumed, silently nulling the strongest block; fix CONFIRMED on Public (-0.0022 to -0.0035, single-layer CatBoost); not yet merged into the AutoGluon winner.
- [Job embeddedness/onboarding literature](job_embeddedness_onboarding_literature.md) — 10+ hypotheses from Mitchell/Bauer/Kristof-Brown all rejected; a borderline effect (19.5pt/p=2.5e-18, just under the 20pt/p<1e-20 bar) failed on Public too, reinforcing the threshold.
- [L2×M risk count confirmed](l2_m_risk_count_confirmed.md) — works great in single-layer CatBoost (-0.004 Public), but folding it into AutoGluon (56_) made both weighted and best_single WORSE on Public (2/2) — a feature win doesn't always transfer across architectures.
- [TabPFN ensemble partner](tabpfn_ensemble_partner.md) — TabPFN v2(150列)は単体でCatBoostと互角(+0.0013)・相関0.954で、事前登録の足切りを通った初のアンサンブル相手。検証-0.004、w=0.1〜0.9の全域で改善。pip install は tabpfn==2.2.1 に固定必須。
- [Reference notebook: 1st place base](reference_notebook_1st_place_base.md) — 0816版は現1位のベース。text_features・在籍月数回帰ブレンド+Platt較正とも4構成ともPublicで不採用確定。検証-0.0047の改善はPublicで+0.0018悪化に反転（本番同一構成・探索なしでも検証の改善は信頼できない）。
- [Private LB variance strategy](private_lb_variance_strategy.md) — Private評価あり&最終提出は自分で選べる。Public最良は運の良い引き(Jensenで証明済)なので選ばず、分散最小のプール平均を選ぶ。fold割当の乱数>あらゆる構成変更。
- [Monthly data information ceiling](monthly_data_information_ceiling.md) — 月次データは10年定着をほとんど予測しない（outcomeに3年近い窓を使っても定数予測から0.009しか改善しない）。評価列は持続性ゼロ。生存時間モデリング一族は完全に閉じた。
- [Kitchen sink combination search](kitchen_sink_combination_search.md) — all 546 features combined + 7,673 pairwise combos + a 3-factor candidate tested: zero new independent features survived Public. Everything traced to known confounds (eval-calendar artifacts, job-type missingness, 新卒/中途 composition) or, for the 3-factor score, redundancy with an already-modeled raw feature.
```

---

## ensemble_oof_overfitting.md

```markdown
---
name: ensemble-oof-overfitting
description: OOFに合わせて重みを学習するアンサンブル手法(Stacking/Optimized Weighted)は本プロジェクトのPublicスコアを悪化させる
metadata: 
  node_type: memory
  type: project
  originSessionId: eadf9cc6-c7df-4302-8bc8-d88cd57c0d21
  modified: 2026-08-15T00:46:02.957Z
---

10年定着予測プロジェクトで、CatBoost/LightGBM/XGBoostのアンサンブルにおいて、OOF Log Lossを最小化するように重みを「学習」する手法（Stacking、Optimized Weighted Average）は、OOFスコアは最良になるがPublicスコアでは最悪になるという完全な順位逆転が実測された（`12_ensemble_gbdt_eda_driven_features`の提出結果、2026-08-08）。

- OOFランキング: Optimized Weighted(0.5849) > Stacking(0.5849) > Weighted Avg(0.5877) > Simple Avg(0.5878)
- Publicランキング: Weighted Avg(0.6057) > Simple Avg(0.6057) > Optimized Weighted(0.6058) > Stacking(0.6071)

**Why**: 3モデルの予測は同一特徴量セットで学習しているため強く相関しており、限られたOOFサンプル（TimeSeriesSplitのため実質サンプル数がさらに減る）に対して重みを最適化すると、そのOOF区間固有のノイズに適合してしまう。Train/Testで入社時期が異なる分布シフトがある（[[eda_train_test_shift]]参照）ため、「OOFに最適な重み」は「Test期間に最適な重み」と一致しない。

**How to apply**: このプロジェクトでは今後、Simple Average / Weighted Average（固定ヒューリスティック、データから重みを学習しない）を基本の提出候補とする。Stacking・Optimized Weighted Averageのような学習型アンサンブルは理論上魅力的でも本データでは過学習しやすいため、採用する場合は均等重みへの縮約（正則化）や、OOFサンプル数を増やす工夫（fold数増加、複数時系列ホールドアウトの平均）とセットで検討する。

**追記（2026-08-09、13_での再確認）**: `13_autogluon_prototype`でもAutoGluonのWeighted Ensemble（検証0.567022, Public 0.575345）とCatBoost単体（検証0.567941, Public 0.574165）を比較したところ、検証では僅かにWeighted Ensembleが良かったにもかかわらずPublicでは単体CatBoostが上回るという**同じ方向の逆転**を再確認した（差は約0.001〜0.002と12_より小さい）。効果が小さかったのは、AutoGluonの貪欲法アンサンブルが11種の多様なベースモデルからCatBoostへほぼ重みを集中させており「実質ほぼ単体」に近かったため。2つの独立したケース（12_の3モデル手動アンサンブル、13_のAutoGluon 11モデルアンサンブル）で同方向の現象が確認できたため、この教訓の一般性への確信は強まった。→ 今後、新しいアンサンブル手法を試す際は必ず「単体最良モデル」の提出も並行して残す運用を標準にする（[[best_submission_status]]参照）。

**追記2（2026-08-09、18_：固定の単純平均でも悪化しうる）**: `18_multi_model_expanded_d`で、CatBoost(val 0.5370)・LightGBM(0.5664)・XGBoost(0.5694)の**単純平均（学習された重みではなく固定1/3ずつ）**を計算したところ、val 0.5492となりCatBoost単体より悪化した。これまでの教訓は「学習された重みがOOFに過学習する」という話だったが、今回は**重みを学習していない固定平均でも、モデル間の実力差が大きい場合は劣るモデルを混ぜるだけで悪化する**ことが分かった。3モデルの実力差が大きい状況では、均等平均は安全策にならない。アンサンブルする場合は実力差を反映した重み付け（例: OOFスコアの逆数、ただし過学習リスクとのトレードオフ）か、そもそも単体最良モデルを使うべき。

**追記4（2026-08-11、32_：CatBoost×線形モデルの固定重みブレンドも「弱いモデルを混ぜると悪化」を再確認）**: `32_ensemble_catboost_linear`で、CatBoost(L_v2、val~0.50〜0.53)と却下済みブロック(M・G・I等)を含むL2ロジスティック回帰(val~0.57〜0.59)を、学習された重みではなく固定重み（CatBoost 100/95/90/85/80/70%）でブレンドしたところ、**CatBoost単体(重み1.0)が両splitで最良**で、線形モデルの重みを増やすほど単調に悪化した。追記2（18_の固定単純平均）と同じ「モデル間の実力差が大きい場合、固定重みでも劣るモデルを混ぜると悪化する」パターンの再現。線形モデル側は却下済み特徴量(M等)を単体では捉えられていた（[[gbdt_interaction_type_matters]]参照）が、モデル全体の絶対性能が低すぎてブレンドの恩恵より希釈効果が上回った。**How to apply**: 「弱いモデルが特定の情報を捉えている」ことと「アンサンブルすると改善する」ことは別問題。ブレンド前に弱いモデル単体の絶対性能差を確認し、差が大きい（今回は0.05〜0.09程度）場合は確率空間の単純平均ではなく、ロジット空間でのブレンドや特徴量選択の絞り込みで弱モデルの質を上げる方を先に検討する。

**追記3（2026-08-10、22_：特徴量ブロックの組み合わせでも同じ現象、事前の過学習チェックが機能した）**: `22_ensemble_weight_tuning`で、`20_`の5構成（baseline/E_memo/F_grade_consistency/G_self_study/combo_EFG）の検証予測に対し`scipy.optimize`でOptimized Weighted Averageを計算したところ、80/20・75/25の2 splitで最適化された重みベクトルが大きく異なった（combo_EFGへの配分が0.224→0.723等）。このため過学習と判定し提出を見送った。並行して、**特徴量ブロックの「組み合わせ」自体（combo_EFG = E+F+G）でも同じ現象が起きていたことが判明**: 検証ではcombo_EFGが最良だったが、Public提出の結果**E_memo単体（0.534829）がcombo_EFG（0.537215）を上回った**（[[best_submission_status]]追記10参照）。モデルアンサンブルだけでなく、特徴量ブロックの組み合わせについても「検証で最良の組み合わせ」が必ずしもPublicで最良とは限らない。**How to apply**: 新しい特徴量ブロックを追加する際は、単体の効果だけでなく組み合わせた場合のPublic確認も行い、「足すほど良い」という前提を置かない。アンサンブルの重み最適化は、提出前に必ず2-split（以上）での重みベクトルの一貫性チェックを行う運用は今回機能した（過学習を検出し無駄な提出を避けられた）。

**追記5（2026-08-15、51_：この教訓は「素の重み学習」限定だった。AutoGluon内蔵のWeightedEnsembleは逆にPublicで単体を上回った）**: `13_`以来「単体モデルを本命とする」方針を続けてきたが、`51_autogluon_catboost_bias`で`AG51_full441_weighted`（AutoGluonのWeightedEnsemble_L3）がPublic 0.514263を出し、同構成の単体`AG51_full441_best_single`（0.515123）を**上回った**。検証でも一貫してWeighted > 単体だった（44_・50_・51_のholdout/full全てで）ので、今回は検証の向きがそのままPublicに転移した。

**Why違ったか**: `12_`・`13_`・`32_`はいずれも「生カテゴリ/素の特徴量セットに対する、少数モデル間の重み学習・単純平均」だった。`51_`は441列に整形済みの特徴量 + 8-fold bagging + stack_level2という土台の上で、AutoGluon自身がL2の複数モデルのOOF予測から貪欲法で重みを学習したもの——**入力の多様性・モデル数・スタッキング構造が全く違う**。「重み学習は過学習する」という一般則は、素の少数モデルの手動アンサンブルには今も当てはまるが、**十分に整備されたbagging+stacking基盤の上でのAutoGluon内蔵アンサンブルには当てはまらない**と修正する。

**How to applyの修正**: 「アンサンブルは単体に負ける」を無条件の前提にしない。素の重み学習（手動でOOFに最適化）は引き続き疑うべきだが、bagging+stackingのような多層構造の中でのWeightedEnsembleは、むしろ試す価値がある。新しい基盤ができたら、単体だけでなくWeighted版も両方Publicで確認する運用に変える。

**追記6（2026-08-15、55_：多様性は本物でも実力差が大きすぎるとブレンド不可）**: `55_sequence_model_gru`で、月次24ヶ月の生系列を軽量GRUに直接投入したモデルを作った。CatBoost（holdout 0.512909）との予測相関は0.346・MAD 0.189（ノイズ床0.02122の約9倍）で**多様性は本物**だったが、GRU単体のholdoutが0.649248とCatBoostに0.136も劣っており、固定重み0.05〜0.30のどの重みで混ぜてもCatBoost単体を上回れなかった（w=0.05で早くも+0.0015悪化、以降単調悪化）。追記2（`18_`）と同じ「実力差が大きいと固定重みでも希釈負けする」パターンの再現。**How to apply**: アンサンブル候補を評価する際は「多様性があるか」と「単体性能がどれだけ近いか」の両方を見る。多様性だけでは不十分——目安として、単体性能の差がholdout loglossで0.05を超えるような弱いモデルは、どんな固定重みでも足を引っ張る可能性が高い。
```

---

## best_submission_status.md

```markdown
---
name: best-submission-status
description: Current best submission and how it was obtained
metadata:
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-16T11:16:20.049Z
---

**現在の最良: pooled（top150ブレンド × hire_fixed(fm2)ブレンドの50:50平均）、Public 0.508699**
（2026-08-16 提出）。ファイル: `data/output/20260816/20260816_pool_top150_hire_fixed_avg.csv`
生成: `src/66_final_blend.ipynb`（第5節）。

**Publicリーダーボード順位（2026-08-16 16:35時点、ユーザー確認）: 5位。**

| 順位 | チーム | スコア |
|---|---|---|
| 1 | crmaine（初学者）| 0.49330 |
| 2 | mu（初学者）| 0.49436 |
| 3 | Yashi（経験者）| 0.49467 |
| 4 | 博多ラーメン（初学者）| 0.50825 |
| **5** | **（自分）** | **0.508699** |

1位crmaineは[[reference-notebook-1st-place-base]]のreference/0816ノートブックのベースを
使っていたと見られる参加者。4位との差は0.0004ときわめて僅少（ノイズ床未満）。
**この日の投稿上限に到達済み（2026-08-16時点、ユーザー報告）**——1日の提出回数に上限があるため、
新しい候補はPublicで即座に検証できず、次回投稿枠まで待つ必要がある。

```
pooled = 0.5 × (プールC×TabPFN(top150)) + 0.5 × (プールC×TabPFN+TabICL(hire_fixed、TabDPT除く))
       = プールC × 0.70 + TabPFN(top150) × 0.15 + [TabPFN+TabICL(hire_fixed)平均] × 0.15
```

旧最良（プールC×TabPFN(top150)単体、0.508793）比 **-0.0000944**。この差自体はノイズ床
（weighted MAD 0.02381、[[refit-chaos-noise-floor]]）以下で「改善した」と単独では主張できないが、
[[private-lb-variance-strategy]]のJensen不等式に基づくpooling戦略が「少なくとも悪化させない」
ことをPublicで確認できた初のデータ点。相関0.9971の2つの独立候補（top150ブレンドとhire_fixed
(fm2)ブレンド、詳細は第104-105節）を単純平均したもの。

### 旧最良（プールC × TabPFN ブレンド単体、Public 0.508793、2026-08-16）

= AutoGluon full441 weighted の保存済み8実行の平均（プールC）× TabPFN v2 top150（3シード平均）を
w_AG=0.70 で線形ブレンドしたもの。`50_`(0.513108)比 -0.004315、プールC(0.514050)比 -0.005257。
同一レシピの実行間ばらつき(0.0023)の約2倍で、ノイズでは説明できない本物の改善。
詳細は [[tabpfn-ensemble-partner]] と [[private-lb-variance-strategy]]。

**1位（Yashi 0.49615、2026-08-15 22:02時点）との差: 0.012549（新最良基準）。**

### 旧最良の記録（参考）

`51_`のweighted（0.514263、val 0.485880）より`50_`のweighted（val 0.486051、valはやや劣る）の方が
Publicは良かった。NN_TORCH/KNN除外（51_の変更点）はbest_singleでは改善・weightedでは悪化と
方向が割れており、ノイズ床以下の差は方向すら予測できないことを示す一例。

**単体モデル(`AG51_full441_best_single`, 0.515123)をWeighted Ensembleが上回った。**
`13_`以来「単体を本命とする」方針だったが、今回は逆転（[[ensemble-oof-overfitting]]の
「重み学習は過学習する」という教訓は、441列+bagging+stackingという土台の上でのAutoGluon
自身のWeightedEnsembleには当てはまらなかった。生特徴量に対する素の重み学習と、
整形済み特徴量+stackingの上でのWeightedEnsembleは別物と修正）。

`50_`（0.515949）に`excluded_model_types`へ`NN_TORCH`・`KNN`追加だけの変更。検証はほぼ横ばい
（誤差±0.0002）だったが、Publicで-0.000826改善した（[[validation-asymmetry]]に「検証が
『ほぼ差なし』と言ってもPublicで改善しうる」という新パターン追加）。

**AutoGluon（441列, CatBoost_r137_BAG_L2）に `49_` のメモパーサー修正を統合した構成が土台。**

| | `44_`（AutoGluon初出） | `50_` | `51_`（現最良） |
|---|---|---|---|
| 特徴量パイプライン | 旧パーサー（バグあり） | `49_`のパーサー修正込み | 同左 |
| `excluded_model_types` | `["FASTAI"]` | 同左 | **+ `NN_TORCH`, `KNN`** |
| `time_limit` | 3600秒/fit | 7200秒/fit | 同左 |
| full441_full 完走モデル数 | 41 | 72 | 61 |
| Public | 0.517685 | 0.515949 | **0.515123** |

`28_`(0.529454)からの累計 −0.020661。

**スコア履歴**: 0.529454(`28_`) → 0.522659(`37_`) → 0.521729(`40_`) → 0.517685(`44_`)
→ 0.515949(`50_`) → 0.515123(`51_`) → 0.514263(`51_`w) → 0.513108(`50_`w)
→ 0.514050(プールC 8本平均) → 0.508793（プールC × TabPFN） →
**0.508699（pooled: top150ブレンド × hire_fixed(fm2)ブレンドの50:50平均）**

※ 0.515949〜0.513108 の区間は[[refit-chaos-noise-floor]]の通りノイズと運の混在。
今回の -0.004315 はそれらとは質的に異なる幅。

**なぜ44_→50_で伸びたか**: `44_`は環境不具合（FASTAI全滅・XGBoost一部失敗・DyStackで900秒浪費）
込みで41モデルしか完走せず、それでも勝っていた。`48_`でこれらを修正し`time_limit`も倍増した結果、
`50_`は同じ441列で72モデル完走。さらに`49_`のパーサー修正でL2の情報もフルに使えるようになった。

**なぜ50_→51_で伸びたか（謎）**: `51_`は`50_`よりモデル完走数が少ない（61<72）上、検証では
ほぼ横ばい（誤差±0.0002）だった。それでもPublicは改善した。単一要因で説明できる差ではなく、
モデル選択のわずかな違いがTestへの汎化にたまたま良く効いた可能性が高い
（[[refit-chaos-noise-floor]]の範囲内）。

**単層CatBoostの最良（`54_`、2026-08-15、変わらず）**: L2×Mのリスク要因数（0/1/2、Job Embeddedness
理論の「複数の埋め込み不足の重なり」）を追加し、441列で Public **0.515030**（`49_`の0.519149から
-0.004119改善）。AutoGluon現最良（0.513108）まで0.0019に接近。
**AutoGluonへの統合は`56_`で試したがPublicで悪化（2/2確定、[[l2-m-risk-count-confirmed]]参照）**。

**`59_`/`60_`（reference/ノートブック由来の5候補、2026-08-15）は全滅**: `59_`のSLブロック
（自己学習パース特徴量）は検証で-0.0115という大幅改善だったがPublicは**+0.002273悪化**
（0.517303、[[validation-asymmetry]]の大きな反転例）。KNN局所ターゲットエンコーディング・Cox生存
時間特徴量（`60_`）・Plain/Orderedブレンド+較正（`59_`）は検証時点で悪化方向だったため未提出。
単層CatBoostの最良は引き続き`54_`（0.515030）のまま。

**lean113系は一貫してfull441に劣る**: `51_`のlean113_best_singleはPublic 0.520590で、
`50_`の同構成(0.519236)より悪化。lean113方面の追加提出は優先度低い。

**未提出で残っているもの**: `50_`/`51_`のlean113_weighted（val 0.479324 / 0.481026）。
lean113はbest_singleで一貫してfull441に劣るため優先度は低い。

関連: [[autogluon-stacking-wins]] [[catboost-beats-other-gbdt]] [[validation-asymmetry]]
[[refit-chaos-noise-floor]] [[modeling-levers-beat-new-features]] [[memo-parser-coverage-bug]]
```

---

## validation_gap_finding.md

```markdown
---
name: validation-gap-finding
description: 単一時系列ホールドアウト検証が6-fold TimeSeriesSplit OOFよりPublicスコアとの整合性が高かった観察（未検証の仮説）
metadata: 
  node_type: memory
  type: project
  originSessionId: eadf9cc6-c7df-4302-8bc8-d88cd57c0d21
  modified: 2026-08-11T03:12:06.401Z
---

`13_autogluon_prototype`（単一時系列ホールドアウト、Train直近20%で検証）は検証-Publicギャップ0.0083だったのに対し、`11_`/`12_`（6-fold TimeSeriesSplit OOF平均）はギャップ0.018〜0.022と2倍以上大きかった（[[best_submission_status]]参照）。

**Why（仮説、未検証）**: TimeSeriesSplitは学習データが少ない初期foldも平均に含めるため、十分な学習データがある状態を想定するTestの実態より悲観的なOOFになりやすい。単一の「直近期間ホールドアウト」の方が実際のTrain→Testの関係（過去で学習し未来を予測）に構造が近い。ただし特徴量エンジニアリングやモデル自体も13_と11_/12_で異なるため、検証方法の違いだけが原因と断定はできていない。

**How to apply**: 今後CVスキームを設計する際、6-fold TimeSeriesSplitに固執せず、単一の直近期間ホールドアウトも並行して試し、どちらがPublicとの整合性が高いか継続的に観察する。手動GBDTパイプライン（11_/12_系）に単一時系列ホールドアウト検証を追加して、この仮説をより厳密に検証する余地がある（特徴量・モデルを揃えた上でCV方式だけを変える比較実験が望ましい）。

**追記（2026-08-09）**: `13_`のCatBoost単体提出はギャップ0.0062と、これまでの全提出中で最小（Weighted Ensembleの0.0083よりさらに小さい）。単一時系列ホールドアウト検証は、学習された重みを持たない単体モデルと組み合わせたときに最もPublicとの整合性が高くなる傾向が見えてきている。

**追記2（2026-08-09、14_）**: より深いCatBoost（depth=8, [[best_submission_status]]参照）はギャップが0.0105に拡大した（13_の0.0062より大きい）。ただし絶対Publicスコアは明確に改善（0.574165→0.563076）しているため、「モデルの表現力を上げるとギャップはやや拡大するが、絶対性能は改善する」というトレードオフがある可能性。ギャップの大小だけでなく絶対スコアも合わせて判断すべき。

**追記3（2026-08-09、決定的な検証）**: `common/`ラッパーを使わない自前のCatBoost+Optuna実装（同じ13_特徴量、同じ単一時系列ホールドアウト）が、AutoGluonのCatBoost_4とPublicスコアで0.000129しか差がなかった（0.563205 vs 0.563076）。**単一時系列ホールドアウト検証の有効性はAutoGluon固有ではなく、モデル実装によらず再現することが確定**。12_→13_/14_の約7%改善の主因は「AutoGluonの魔法」ではなく「(a)単一時系列ホールドアウト検証への変更、(b)LabelEncoding/多項式/ビン化/複合交互作用を廃した特徴量の単純化」の2点であると結論づけた。

**追記4（2026-08-09、テキスト系特徴量は例外的にギャップが大きい）**: `16_`のA_only（テキストTF-IDF追加）は検証Log Lossが14_から4.15%改善したが、Publicでの実際の改善は0.62%のみで、検証-Publicギャップ0.0299は全提出中最大となった。単一時系列ホールドアウト検証は概ねPublicとの整合性が高いという傾向（追記2・3）に対し、**高次元・疎なテキスト由来特徴量（TF-IDF+SVD等）は例外的に単一ホールドアウトへ過学習気味に良いスコアが出やすい**。今後テキスト系特徴量を拡張する際は、検証スコアの改善幅を割り引いて評価し、可能なら複数ホールドアウトでの確認を検討する。

**追記5（2026-08-09、17_で機構解明）**: `17_`で80/20と75/25の2つの時系列split双方を比較したところ、**baselineのsplit間std(0.0044)に対しTF-IDF追加後は3設定ともstdが2〜3倍(0.009〜0.012)に拡大**した。さらにTF-IDFのbaseline比改善幅はsplit_75_25で0.49%、split_80_20で2.83%と分割によって大きく異なった。**16_で使った80/20分割は、たまたまTF-IDFの効果が過大に出やすい分割だった**と考えられ、これが追記4のギャップの直接的な機構的説明になる。次元削減（8次元・5次元に縮小）はstdの改善に寄与しなかった（テキスト特徴量を使う限り一定のブレは避けられない）。今後の教訓: **テキスト系特徴量の効果を判断する際は複数splitの平均を使い、単一splitの改善幅は割り引いて見る**。GPU非決定性（[[catboost_gpu_nondeterminism]]）と合わせて、小さな改善幅（0.005未満程度）は信頼しないこと。

**追記6（2026-08-10、2-split検証の限界: 分布シフトは別問題）**: `19_text_embeddings`で、TF-IDFより2-split間のstdが小さい（=Train内のサブサンプリングに安定）文埋め込み特徴量が、実際のPublicではTF-IDFより悪化し（0.564355 vs 0.550352）、ギャップも拡大した（0.0228 vs 0.0133、[[future_action_text_embeddings]]参照）。**2-split（80/20, 75/25）のstdは「Train内でどこを検証に切り出すか」への感度を測るものであり、Train(2011-2014年入社)とTest(2014-2017年入社)という本質的に異なる母集団への汎化（分布シフトへの頑健性）とは別の問題**であることが判明した。「Train内で安定 = Testに強い」とは限らない。今後、Train内の頑健性チェック（2-split等）はスクリーニングとして有用だが、最終判断は実際のPublic提出でしか下せないことを念頭に置く。

**追記7（2026-08-10、ただし特徴量の性質によっては2-split安定性がPublicとも一致する）**: `20_eda_v3_driven_features`（EDA v3由来のメモ構造化・経験等級整合性・自己学習実施月数を組み合わせたcombo_EFG）では、baseline比で2-split間のstdが0.0108→0.0013へ大幅縮小し、**その通りPublicも改善（0.550352→0.537215）、ギャップも縮小（0.0133→0.0112）した**（[[best_submission_status]]追記7参照）。これは追記6（19_の文埋め込み）と正反対の結果である。両者の違いは特徴量の性質にあると考えられる: 19_は384次元の統計的埋め込みをPCA圧縮した**間接的・高次元**な表現だったのに対し、20_はメモの構造化カテゴリ・経験等級残差・自己学習実施月数という**少数の解釈可能な業務ロジック由来**の特徴量だった。**教訓**: 2-split内での安定性向上を見たとき、その特徴量が「データ生成ロジックに近い直接的・解釈可能な特徴量」であればPublicとの整合性も期待できるが、「高次元・統計的な間接表現」である場合は追加の警戒が必要（Publicで確認するまで結論を保留する）。

**追記11（2026-08-11、26_：単体ブロック追加でも「両split一貫改善」は信用できないケースがある、性質による使い分けの仮説）**: 早期昇給タイミング（K）を18_ベースラインに単体追加したところ、combo_EG/combo_EIと同様に80/20・75/25の両splitで一貫して改善して見えたが、Publicでは18_ベースラインより悪化（ギャップ0.0251）した。一方、同じ「単体追加・両split改善」パターンだった希望勤務地マッチ度（J）は健全に汎化した（ギャップ0.0095）。両者の違いを分析すると、**「入社時に固定される客観的な条件」（E・J）は汎化し、「0-23ヶ月の行動・時系列パターン」（G・H・K）は汎化しない**という傾向が見える（[[best_submission_status]]追記14参照）。教訓: 「両split一貫した改善」は多ブロック組み合わせだけでなく単体追加でも過信すべきでない。ただし特徴量の性質（入社時固定 vs 月次行動由来）を見れば、Publicでの汎化しやすさをある程度事前に予測できる可能性がある。

**追記9（2026-08-10、23_：E/F/Gブロックの組み合わせ探索で検証-Publicの逆転が2連続発生、探索打ち切り）**: E単体(Public 0.534829)がcombo_EFG(0.537215)を上回ったことを受け、combo_EF・combo_EGを検証したところ、Colab CPUではcombo_EG(-0.0076)・combo_EFG(-0.0102)がE単体より改善して見えた。しかしcombo_EGをPublic提出すると**0.557731（18_のベースラインより悪化）、検証-Publicギャップ0.0317（全提出中最大）**という結果になった（[[best_submission_status]]追記11参照）。combo_EFG・combo_EGと2回連続で「検証で改善→Publicで悪化」が起きたことから、**このE/F/Gブロック組み合わせの実験系列に関しては、CPU 2-split検証がPublicの予測材料として機能していない**と判断し、これ以上の組み合わせ探索を打ち切った。教訓: ある特徴量ファミリーで検証とPublicの矛盾が2回連続で起きたら、そのファミリー内での追加の組み合わせ探索を検証結果だけで判断するのをやめ、単体で確認済みの構成（この場合E単体）を確定させて別方向に切り替えるべき。

**追記10（2026-08-10、24_：E_memoへの追加実験が3回連続で検証-Public逆転、悪化幅も拡大の一途）**: combo_EFG(-0.0024)→combo_EG(-0.0205)→combo_EI(-0.0406)と、E_memoに新ブロックを追加する実験を3回試したが、検証での見え方（改善・中立・弱い矛盾シグナル）に関わらず**毎回Publicで悪化し、しかも悪化幅が回を追うごとに拡大した**（[[best_submission_status]]追記12参照）。個別ブロックの質の問題ではなく、「E_memoの特徴量セットに何か列を追加すること自体」がPublicでの汎化を悪化させている可能性が高い。**教訓**: ある基盤構成（この場合E_memo）に対して追加実験が2回以上連続でPublicを悪化させたら、3回目以降は個別ブロックを疑うのをやめ、「その基盤構成に追加する」というアプローチ自体を疑うべきだった。今後、新しい特徴量は既存の最良構成に追加する形ではなく、確定済みのベースライン（この場合18_、Eなし）に対して単体で検証し、それ単体が既存最良を上回るかで判断する。

**追記8（2026-08-10、21_：CPU/GPU間・異なるCPUアーキテクチャ間の検証ノイズはこれまでの想定より大きい）**: `21_engagement_deepdive_features`（ブロックH: 情報共有件数・360度評価_信頼度の深掘り）を検証したところ、同一の特徴量セット・探索設定にもかかわらず環境によって検証結果の**方向が正反対**になった: ローカルMac CPU(ARM)ではbaseline比+0.0057（悪化）、Colab CPUハイメモリ(x86)では逆に-0.0058（改善、両split一貫）。さらに同じ`combo_EFG`のmean valが GPU(0.525051)・ローカルMac CPU(0.524276)・Colab CPU(0.532538)の3環境で0.008程度もブレた。これは[[catboost_gpu_nondeterminism]]で確認していた「GPU非決定性ノイズ幅0.003〜0.006」（同一GPU・別セッション間）より大きく、**異なるハードウェア/CPUアーキテクチャ間の実装差は、同一GPU上のセッション間ブレより大きくなりうる**ことが判明した。Publicで確認したところcombo_EFGHは実際には悪化しており（0.540507、[[best_submission_status]]追記8参照）、Colab CPUでの「改善」は見せかけだった。**教訓**: 検証環境（ローカルPC・Colab・GPU/CPU）を切り替えて得られた改善幅は、環境間で結果の方向が一致しない限り信用しない。特に環境をまたいだ「back-to-backではない」比較（例: 過去のGPU実行結果と、別セッションのCPU実行結果を比較する）は、GPU非決定性ノイズ幅を上回るブレを生みうるため、疑わしい場合は必ずPublicで決着をつける。
```

---

## local_test_not_decisive_for_ablation.md

```markdown
---
name: local-test-not-decisive-for-ablation
description: Don't declare a CatBoost feature-ablation result "confirmed" from a local-machine test run; only the user's actual execution environment (Colab) and ultimately Kaggle Public count.
metadata:
  type: feedback
  originSessionId: eadf9cc6-c7df-4302-8bc8-d88cd57c0d21
  modified: 2026-08-11T10:34:32.719Z
---

When validating a new notebook locally (no GPU, different CPU architecture than Colab) before handing it to the user, local val-score results are useful for catching bugs/crashes but must NOT be presented as a decisive verdict on whether a feature block helps or hurts.

**Why**: In the `21_engagement_deepdive_features` episode (2026-08-10), I ran the notebook locally on an Apple Silicon Mac with the real (non-reduced) n_trials=25 config and declared block H "confirmed rejected" (+0.0057 worse) before the user had run anything on Colab. When the user actually ran it on Colab CPU (high-RAM), the result flipped sign (-0.0058, looked like an improvement). The underlying `combo_EFG` baseline mean varied by ~0.008 across GPU / local-Mac-CPU / Colab-CPU — larger than the previously-assumed GPU-only noise floor (0.003–0.006, see [[catboost_gpu_nondeterminism]]). Only the actual Kaggle Public submission (0.540507, worse than the 0.537215 baseline) settled it — both validation-only verdicts were wrong in different directions. See [[validation_gap_finding]] addendum 8 for the full technical writeup.

**How to apply**: Local runs (possibly on different hardware/OS than the user's) are for smoke-testing pipeline correctness (no crashes, sane shapes/values), not for accepting or rejecting a feature block. Never say "ローカルで確認済みなので確定的" or equivalent. Always frame local ablation numbers as provisional, and push borderline/ambiguous cases (effect size near the known noise floor, or where the whole point is a marginal gain) through to an actual Colab run and, if still ambiguous or high-stakes, a real Public submission before recording a block as accepted/rejected in the report or best_submission_status memory.

**Addendum (2026-08-11, block N momentum, `34_momentum_density_features`)**: Local Mac showed block N (short-term momentum ratios) improving consistently on both splits (mean_diff -0.00187). Independent Colab run showed it worse on both splits (mean_diff +0.00107) — a full sign flip, same pattern as block H above. Two-for-two now on local-vs-Colab sign flips for small-effect-size blocks. Also relevant: `33_feature_selection_pipeline`'s Step1 (correlation/VIF filter) showed consistent improvement in BOTH local Mac AND Colab validation (not just one), yet still failed on Public (0.533674 vs baseline 0.529454) — so agreement between local and Colab is necessary but not sufficient either; even a clean two-environment validation match can still not survive Public. The likely mechanism there: choosing which of two near-duplicate columns to keep based on which has higher train-period target-correlation is itself a target-informed selection with sampling noise at n~2200, not a "free" dedup. General lesson: for this dataset's scale, no amount of validation-only agreement (single split, both splits, single environment, or even two environments) reliably predicts Public for small/marginal effect sizes — only an actual Public submission does.
```

---

## edit_noop_error_gotcha.md

```markdown
---
name: edit-noop-error-gotcha
description: An Edit tool "old_string and new_string are exactly the same" error means the edit was rejected as a no-op due to a copy-paste mistake in the call itself — it does NOT mean the target text was already fixed. Always re-grep to confirm.
metadata:
  type: feedback
  originSessionId: eadf9cc6-c7df-4302-8bc8-d88cd57c0d21
  modified: 2026-08-10T15:34:05.593Z
---

While converting notebook `25_location_match_and_raise_timing.ipynb`'s CatBoost calls from `task_type="GPU"` to `"CPU"` (2026-08-11), a `replace_all` Edit only matched the dict-style occurrence (`"task_type": "GPU",`), missing the keyword-arg-style second occurrence (`task_type="GPU"`, no quotes around the key — this exact GPU/CPU dual-occurrence pattern recurs across every notebook in this project's `run_model_config` function). I then wrote a follow-up Edit call to fix the second occurrence, but pasted the already-corrected string into both `old_string` and `new_string` by mistake. The tool returned "No changes to make: old_string and new_string are exactly the same" — I misread this as "good, it's already fixed" and moved on, leaving the real GPU reference in place. It only surfaced later when the local correctness test crashed with `CatBoostError: Environment for task type [GPU] not found`.

**Why**: This error message is purely an input-validation rejection (old_string == new_string as literal strings passed to the tool) — it says nothing about whether the target text in the file already matches the desired new_string. It is easy to conflate "the edit was rejected" with "the file is already correct," especially mid-flow when doing a series of near-identical GPU→CPU swaps across many notebooks.

**How to apply**: Whenever an Edit call errors with this message, do not assume success — immediately `grep` the file for the old (unwanted) string to confirm it's actually gone before moving on. This is especially important for repeated find-replace patterns applied across multiple similar files (e.g., the `task_type` GPU→CPU conversion done identically across notebooks 21-25 in this project) — a mistake in one instance can silently slip through since the pattern "looks like it worked" from the surrounding successful edits. The local end-to-end integration test (already standard practice, see [[local_test_not_decisive_for_ablation]]) is what actually caught this — reinforces that the pre-handoff local test is not just about validating ablation logic but also about catching exactly this class of mechanical error.
```

---

## dept_target_encoding_leak.md

```markdown
---
name: dept-target-encoding-leak
description: 小規模グループのTarget Encodingを時系列split後の検証期間を含む全データでfitするとリークする
metadata: 
  node_type: memory
  type: feedback
  originSessionId: eadf9cc6-c7df-4302-8bc8-d88cd57c0d21
  modified: 2026-08-09T00:49:39.831Z
---

部署IDのような小規模グループ（1グループ平均6名）に対するKFold+スムージング付きTarget Encodingを、時系列split（chronological holdout）の検証期間の社員を含む全Trainデータでfitすると、検証期間の社員同士でラベルが漏れ込むリークが発生する。KFoldの分割が日付と無関係なため、同じ部署の検証期間社員が異なるfoldに散らばり、互いのラベルが smoothed mean に混入する。

`15_enriched_feature_engineering`（2026-08-09）で発見: 部署特徴量を追加した直後、検証Log Lossが0.552597→0.396と不自然に改善したため疑い、検証期間内での特徴量とラベルの相関を測定したところ0.466（学習期間のみでfitし直すと0.10程度）という異常値だった。

**Why**: グループサイズが小さいほど（例: 部署=6名/グループ）、KFoldのランダム分割で同一グループの複数メンバーが検証期間側に偏る確率が上がり、スムージング後の値が実質的にそのグループの検証期間メンバー自身のラベル平均に近くなる。職種・等級・入社区分のような大規模グループ（数百名）ではこの効果は希釈されるため問題になりにくい。

**How to apply**: 時系列ホールドアウト（またはいかなるchronological split）を検証に使う場合、Target Encoding等ラベルを使う集計は**必ず学習期間のID集合のみでfitし**、検証期間・Testには「学習期間で作ったマップ」を適用（transformのみ）する。小規模カテゴリ変数（部署・上司等）ほど要注意。大規模カテゴリ変数（職種・等級等）でも一貫性のため同様に扱うのが安全。この修正パターンは`15_enriched_feature_engineering.ipynb`の`create_department_target_encoding`関数（`fit_ids`パラメータ追加）と、その前段の`TRAIN_PERIOD_IDS`確定セルに実装済みなので、以後の新規ノートブックでも流用できる。[[best_submission_status]]も参照。
```

---

## feature_importance_vs_generalization.md

```markdown
---
name: feature-importance-vs-generalization
description: 特徴量重要度が高いことは、その特徴量が汎化に貢献することを意味しない（アブレーションで検証すべき）
metadata: 
  node_type: memory
  type: feedback
  originSessionId: eadf9cc6-c7df-4302-8bc8-d88cd57c0d21
  modified: 2026-08-09T02:59:27.624Z
---

`16_feature_ablation`（2026-08-09）で、CatBoostの特徴量重要度で最上位だった「部署の時系列変化特徴量」（経験部署_target_enc_avg等、[[dept_target_encoding_leak]]でリーク修正済み）を単体で追加すると、検証Log Lossがbaselineより明確に悪化した（0.547645→0.571729）。逆に重要度がほぼゼロだった「ランク特徴量」はbaselineとほぼ同等（無害）だった。

**Why**: 特徴量重要度（例: CatBoostのPredictionValuesChange）は訓練データに対してどれだけ分岐に使われたか・訓練損失を下げたかを示す指標であり、検証/Testでの汎化性能を保証しない。特に小規模グループ由来の特徴量（部署など）は、訓練データにフィットしやすい（重要度が高く出やすい）が、真に汎化する情報量が少ない場合がある。

**How to apply**: 新しい特徴量ブロックを追加する際は、特徴量重要度の高さだけで「効いている」と判断せず、**必ずそのブロックを単体で追加/除外するアブレーションを行い、検証スコアの変化で採否を決める**。複数ブロックを同時に追加すると、一つの有害なブロックが他の有益なブロックの効果を打ち消してしまい、全体の結果だけでは何が効いているか分からなくなる（実際、`15_`で4ブロック同時追加が悪化したのは主に部署時系列ブロックが原因だったことが`16_`のアブレーションで判明した）。[[best_submission_status]]、[[dept_target_encoding_leak]]も参照。
```

---

## catboost_gpu_nondeterminism.md

```markdown
---
name: catboost-gpu-nondeterminism
description: "CatBoost task_type=\"GPU\"はrandom_seed固定でも別セッション間で完全再現しない（ノイズ幅0.003〜0.006）"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: eadf9cc6-c7df-4302-8bc8-d88cd57c0d21
  modified: 2026-08-09T11:54:22.121Z
---

`17_tfidf_tuning_and_combos`（2026-08-09）で、16_の`A_only`（val Log Loss 0.529646）と17_の`A_v1`（全く同一設定: 特徴量・seed=42・Optuna 25試行）でval Log Lossが0.535388となり、0.0058の差が生じた。baseline同士でも0.547645 vs 0.550999で0.0034の差があった。

**Why**: CatBoostの`task_type="GPU"`は`random_seed`を固定していても、GPU上の並列浮動小数点演算（ヒストグラム集計等）の実行順序が完全には決定的でないため、bit-for-bit再現しない。16_と17_は別のColabセッション（＝別のGPUインスタンス）で実行されたため、このズレが生じた。CPU学習（`task_type="CPU"`、デフォルト）ならこの種のブレは通常起きない。

**How to apply**: 別セッション/別ノートブックで「同一設定のはず」の実験を比較する際は、**0.003〜0.006程度の差はノイズとして許容し、これより小さい改善幅は信頼しない**。新しい特徴量やパラメータの効果を判断する際は、この非決定性ノイズ幅より明確に大きい差（できれば2倍以上）が出ているかを確認する。同一セッション内・同一`study.optimize`呼び出し内の比較（例: [[best_submission_status]]のアブレーション実験）はこの問題の影響を受けにくい。
```

---

## future_action_text_embeddings.md

```markdown
---
name: future-action-text-embeddings
description: 将来のアクション候補として、TF-IDFの代わりに日本語事前学習済み文埋め込みモデルを試す
metadata: 
  node_type: memory
  type: project
  originSessionId: eadf9cc6-c7df-4302-8bc8-d88cd57c0d21
  modified: 2026-08-09T23:42:55.233Z
---

現状（2026-08-09時点）はテキスト特徴量（入社時メモ・上司/同僚フィードバック）に文字n-gram TF-IDF + TruncatedSVDを使っている（[[validation_gap_finding]]参照、分割間でスコアがブレやすい特性あり）。将来のアクション候補として、日本語の事前学習済み文埋め込みモデル（multilingual sentence embeddings等）に置き換えることを検討する。

**Why**: TF-IDFは疎な語彙ベースの表現で、分割間の分散が大きい（baselineの2〜3倍のstd）。文埋め込みモデルなら次元を増やさず（むしろ減らして）意味的な情報を捉えられ、疎行列由来の不安定さを減らせる可能性がある。

**How to apply**: 優先度は中程度（ユーザーの明示的な要望により記録）。着手する際はColab上でのモデルダウンロード等セットアップコストが増える点に留意し、既存のTF-IDFパイプライン（`create_tfidf_svd_features`）と同様に「Trainのみでfit/推論し、Testはtransformのみ」の構造を踏襲する。2-split（80/20, 75/25）での頑健性確認とセットで評価すること。

**追記（2026-08-09、19_で実施済み）**: 実施した。日本語特化のSentence-BERT(`sonoisa/sentence-bert-base-ja-mean-tokens-v2`)はMeCab(`fugashi`)必須でColab環境が不安定になるため、分かち書き不要な`intfloat/multilingual-e5-small`(384次元→PCA15次元)に変更。結果、**std(分割間のブレ)はTF-IDFの0.0108→0.0070に改善**し仮説通り安定性は向上したが、**meanはTF-IDF(0.5444)がまだ embedding(0.5466)よりわずかに良い**。

**追記2（2026-08-10、結論: 不採用）**: `embedding`バリアントをPublic提出した結果、**Public 0.564355（TF-IDFの0.550352より約2.54%悪化）、検証-Publicギャップ0.0228（TF-IDFの0.0133より拡大）**となり、2-split内での安定性の高さはPublicへの汎化の良さを意味しないことが判明した（[[validation_gap_finding]]追記6参照）。**本バックログ項目は完了・不採用で終了**。テキスト特徴量は引き続きTF-IDF+SVD(15次元)を採用する。
```

---

## eda_v3_findings.md

```markdown
---
name: eda-v3-findings
description: EDA v3（昇進降格・入社時メモ構造化・離職伝染・多変量プロファイル・経験等級整合性・専攻職種マッチ・自己学習内容分析）の結果
metadata: 
  node_type: memory
  type: project
  originSessionId: eadf9cc6-c7df-4302-8bc8-d88cd57c0d21
  modified: 2026-08-10T09:02:59.077Z
---

`notebooks/eda/data_exploration_v3.ipynb`（2026-08-10）で7つの新規分析を実施した結果:

1. **昇進・降格は0-23ヶ月データでは特徴量化不可**: Train全2,761名のうち0-23ヶ月以内に等級・役割が変化した社員はゼロ。10年フルデータ（`employee_monthly_train_full.csv`）で確認すると、最初の昇進は必ず24ヶ月目以降にしか発生しない（制度設計上の制約）。Train/Testとも0-23ヶ月しか観測できないため、この特徴量軸は原理的に実装不可。
2. **入社時メモの構造化パースは有望**: 正規表現で「転居を伴う異動を許容するか」「在宅勤務を希望するか」「キャリア志向（安定志向/管理職志向/専門職志向/未定）」を抽出。転居許容で定着率+9.7%pt（61.9% vs 52.2%）、在宅希望で+8.3%pt（60.7% vs 52.3%）という明確な差。
3. **部署内の離職伝染効果はほぼ無風**: 初期部署の同僚（自分除く）休職退職率と自分の定着率の相関は-0.0154。単純な指標では同調効果は確認できず。
4. **後期離職者(60-119ヶ月)は0-23ヶ月データのほぼ全指標で定着者と見分けがつかない**: 360度評価5項目・欠勤・給与等12指標で比較、v2の残業時間の知見（後期離職は0-23mo判別困難）が一般化することを確認。早期離職者は情報共有件数・360度評価_信頼度・顧客満足度で明確に低い。
5. **前職経験と等級の整合性は有望**: 中途入社者(n=736)で、前職経験に対し等級が高いグループの定着率76.0% vs 整合的65.5% vs 低いグループ64.3%（差約11-12%pt）。
6. **専攻×初期職種のマッチ/ミスマッチは効果薄**: カイ二乗検定で非有意(p=0.319)。
7. **`自己学習（詳細）`列（新発見、未使用だった）の内容分析は有望**: `employee_dataset_specification.md`再確認で発見した、月次の自発的学習記録（例:「クラウド基盤入門：2.0時間｜統計学基礎：1.5時間」）。既存の「研修時間」とは無相関(0.004)の独立情報源。**自己学習実施月数**と定着の相関0.060、最下位5分位(0-6ヶ月)の定着率48.9% vs 他分位56-62%と明確な差。合計時間(相関0.050)・ユニークテーマ数(相関-0.030)も弱い相関。一方、学習テーマの内容が初期職種と合っているか（テーマ職種マッチ度、35テーマを6カテゴリにマッピング）はカイ二乗検定で非有意(p=0.671)、6と同様に効果薄。パース時、「｜」区切りの複数テーマを正しく分割しないと2つ目以降のテーマ・時間を取りこぼすバグに注意（修正済み）。

**How to apply**: 2・5・7（自己学習実施月数等）は`20_eda_v3_driven_features`で実装・検証し、組み合わせcombo_EFGがPublic 0.537215で総合最良を更新（[[best_submission_status]]参照）。1・3・6・7のテーマ職種マッチ度は見送り/優先度低。**4（早期/中期/後期離職者の多変量プロファイル比較）の深掘り（情報共有件数のゼロ連続月数・360度評価_信頼度の本人内相対偏差・職種内偏差）は`21_engagement_deepdive_features`で試したが悪化し不採用**（[[best_submission_status]]追記8参照）。詳細は`notebooks/eda/report/md/data_exploration_v3_report.md`参照。
```

---

## eda_v4_findings.md

```markdown
---
name: eda-v4-findings
description: EDA v4（希望勤務地マッチ度・昇給タイミング）の結果、本プロジェクト最大級の効果量を発見
metadata:
  type: project
  originSessionId: eadf9cc6-c7df-4302-8bc8-d88cd57c0d21
  modified: 2026-08-11T03:12:17.732Z
---

`notebooks/eda/data_exploration_v4.ipynb`（2026-08-11）で、E_memoに新ブロックを追加すると3回連続でPublicが悪化した教訓（[[best_submission_status]]追記12参照）を踏まえ、「客観的な事実・条件のマッチング」系の情報源を優先的に探索した結果:

1. **希望勤務地マッチ度 → 本プロジェクト最大級の効果量**: `入社時メモ`の希望勤務地（ブロックEで抽出済み）と既存列「初期勤務地」の一致・不一致を見ると、一致群の定着率62.1% vs 不一致群36.8%で**差25.3%pt**（p=1.4×10⁻²⁶）。希望勤務地・初期勤務地は個別には既にモデルに投入済みだが、両者の一致関係自体は未検証だった。
2. **早期昇給フラグ → 有望**: 入社6ヶ月以内に初回昇給があった社員は定着率68.8%、それ以降(多くは12ヶ月目の定期昇給)は54.6%で**差14.2%pt**（p=6.7×10⁻⁷）。
3. **初任給の勤務地内偏差 → 弱い有望**: 相関0.130、最上位分位で定着率69.9%。
4. **転居許容×実際の転居ミスマッチ → null**: 実際に転居した社員は60名(2.2%)のみ、該当ケースはわずか20名でサンプル数不足。
5. **上司・同僚フィードバックのキーワード探索 → null**: 6候補すべて非有意(p>0.15)。「人物所見」と違い長文・多様な言い回しで単純キーワード一致は効かない。
6. **未使用列点検（採用経路・性別・初期勤務地・初期役割） → 新規性なし**: 統計的に有意な差はあるが、いずれも`drop_cols`に含まれておらず既にCatBoostへ生カテゴリ変数として投入済みだった。

**How to apply**: 分析1・2を新しい特徴量エンジニアリングノートブックで検証する。ただし[[validation_gap_finding]]追記10の教訓（E_memoに追加すると3回連続で検証-Public逆転）を踏まえ、**E_memoに追加する形ではなく、まず18_のベースライン（Eなし）に対して単体で検証する**。特に分析1（希望勤務地マッチ度）は最優先。詳細は`notebooks/eda/report/md/data_exploration_v4_report.md`参照。

**追記（2026-08-11、分析1をPublicで確認、G/H/Iとは違う「本物」）**: `25_location_match_and_raise_timing`で分析1（希望勤務地マッチ度）を18_ベースラインに単体追加しPublic提出した結果、0.540648（18_比約1.76%改善、検証-Publicギャップ0.0095と健全）。E_memo(0.534829)には届かなかったが、combo_EG/combo_EIのような検証-Public逆転（ギャップ0.03〜0.04）は起きず、**単体で本物の汎化効果を持つブロックと確認できた**（[[best_submission_status]]追記13参照）。

**追記2（2026-08-11、分析2（早期昇給タイミング）はPublicで失敗、教訓を格上げ）**: 分析2（早期昇給タイミング、K）を18_ベースラインに単体追加したところ、`26_`の検証では両split一貫して改善して見えたが、Public提出の結果0.569681と18_ベースラインより悪化（ギャップ0.0251）した。分析1(J、成功)と分析2(K、失敗)を比較すると、**「入社時に固定される客観的な条件」（希望勤務地マッチ度）は汎化するが、「0-23ヶ月の行動・時系列パターン」（昇給タイミング）は汎化しない**という仮説が浮上した（[[validation_gap_finding]]追記11、[[best_submission_status]]追記14参照）。今後のEDAでは前者の系統を優先的に探索する。
```

---

## eda_v5_findings.md

```markdown
---
name: eda-v5-findings
description: EDA v5（入社時固定・客観条件の深掘り）の結果、転居許容×勤務地マッチの交互作用が本プロジェクト最大の発見に。Publicで新最良スコアとして確定済み。
metadata: 
  node_type: memory
  type: project
  originSessionId: b15106b6-cdc7-4e1e-9c01-1c419c65f3a7
  modified: 2026-08-11T05:48:21.362Z
---

**追記（2026-08-11、Publicで確定・新たな総合最良）**: 下記の最重要発見（転居許容×勤務地マッチ交互作用）を`27_relocation_location_mismatch_interaction`でブロックLとして実装し18_ベースラインに単体追加、Colab CPUで検証（baseline比-0.0388、Jの改善幅の約2倍）した上でPublic提出した結果、**Public 0.529672となり、それまでの最良だったE_memo単体(0.534829)を更新した**。詳細は[[best_submission_status]]追記15参照。

`notebooks/eda/data_exploration_v5.ipynb`（2026-08-11）で、1位のPublicスコア(0.50081)と現在のベスト
（E_memo単体, 0.534829）の差(約0.034)を埋めるべく、[[validation_gap_finding]]追記11・[[best_submission_status]]
追記14で確立した「入社時に固定される客観的な条件」系（E・Jが健全に汎化、G/H/I/Kは悪化）に絞って
6候補を検証した。

**最重要発見: 転居許容×希望勤務地マッチの交互作用**（本プロジェクト最大の効果量）。ブロックE（転居許容
フラグ）とブロックJ（希望勤務地マッチ度、[[eda_v4_findings]]）はそれぞれ単体で有効と確認済みだが、
両者のAND条件「転居を伴う異動を許容しない、かつ希望と異なる勤務地に配属された」社員（n=342, 12.4%）の
定着率はわずか**23.1%**、それ以外は61.4%で**差38.3%pt**（p=1.4×10⁻³⁹、オッズ比0.189）。これはJ単体
(25.3%pt, p=1.4×10⁻²⁶)を上回る、これまで確認された全単一特徴量中で最大の効果量・有意性。加法的効果
ではなく交互作用（転居許容者は勤務地不一致の影響が-7.2%ptと小さいが、転居非許容者は-38.5%ptと激烈）。
Testデータでも同一ロジックで同程度の割合(10.9%)で再現し、入社時に確定する情報のみなのでリークもない。

**5つの負の結果（「入社時固定」系候補でも効かなかった事例）**:
1. **初期部署IDの職種超過情報 → null**: 全460部署が例外なく単一の初期職種に従属（部署は職種の
   完全な下位区分）。正しく職種平均へ縮約すると部署固有の偏差は目的変数とほぼ無相関(-0.003〜-0.006)。
   `data_exploration_v2_report.md`が「強力な候補」と評価したのは、グローバル平均への縮約による
   見かけの効果だったと判明（v2の記述は訂正が必要）。
2. **学歴×年齢整合性 → null**: 新卒の卒業年齢ギャップ(相関0.004)、中途の未説明ブランク月数(相関-0.052)
   いずれも無風。
3. **初任給の勤務地内偏差 → 初任給そのものと重複**: 相関0.129は初任給単体(0.132)とほぼ同じ。
   勤務地×等級を両方コントロールすると相関はほぼ消失(-0.022)。
4. **前職職種×初期職種マッチ(中途) → null**: 68.6% vs 68.7%、p=1.0。専攻×職種マッチ([[eda_v3_findings]])
   と同じく配属は専門性継続より他要因で決まる。
5. **キャリア志向×初期役割トラック整合性 → 既存特徴量への交絡**: 全体では大差(p=2.4e-8)に見えるが、
   初期役割=メンバーに限定すると有意差消失(p=0.13)。単純集計の大差に騙されず、主要な既存特徴量で
   層別してから採否判断すべきという教訓。

**How to apply**: 転居許容×勤務地マッチ交互作用（ブロックL想定）を最優先で実装し、`27_`想定のノートブック
で`18_`ベースライン（Eなし・Jなし）に対して単体アブレーション→Public確認、というE/J確立時と同じ
プロトコルを踏襲する。複数ブロック合成は`combo_EFG`以降の教訓（検証で改善→Publicで悪化が複数回）を
踏まえ、検証だけで判断せず必ずPublicで確認する。今後の特徴量探索では、単純集計で大きく見える差も
主要な既存特徴量（初期職種・初期役割・初任給等）で層別して残るかを必ず確認する運用を徹底する。
詳細は`notebooks/eda/report/md/data_exploration_v5_report.md`参照。
```

---

## checkpoint_drive_sync_gotcha.md

```markdown
---
name: checkpoint-drive-sync-gotcha
description: "Local-Mac and Colab runs share the same Google Drive-mounted checkpoint path, so a local pre-run can silently make Colab skip real computation"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b15106b6-cdc7-4e1e-9c01-1c419c65f3a7
  modified: 2026-08-11T05:48:33.142Z
---

The project's notebooks (`18_`~`27_`) use a fixed-path checkpoint file (`data/output/_checkpoints/{SCRIPT_NAME}_checkpoint.csv`) so an ablation can resume across sessions (`run_or_resume`). `PROJECT_ROOT` on Colab (`/content/drive/MyDrive/jaggle_2026`) and on the local Mac (`~/Library/CloudStorage/GoogleDrive-.../マイドライブ/jaggle_2026`) point at the *same* underlying Google Drive folder for this user.

**Why**: If a script is run locally first, the checkpoint CSV syncs to Drive. When the user then opens the same notebook in Colab expecting an independent run, `run_or_resume` finds the config labels already present and returns the cached (local Mac) results without training anything — completing almost instantly with numbers identical to the local run down to the last decimal. This happened with `27_relocation_location_mismatch_interaction` (2026-08-11): a local validation run followed by a Colab run produced bit-identical output in seconds, which was the tell.

**How to apply**: Before asking the user to run a notebook on Colab specifically to get an *independent* cross-environment confirmation, check whether that notebook's checkpoint file already exists under `data/output/_checkpoints/`. If it does (from a prior local run), move it aside (don't delete — keep as a reference/backup) before the Colab run. A fresh run should take multiple minutes for a multi-config CatBoost+Optuna ablation, not seconds — abnormally fast completion with exactly matching numbers is the signal something was skipped. Separately: this incident also reconfirmed that Colab CPU + fixed seed is fully deterministic session-to-session (this run's `baseline` and `J_location_match` values matched `25_`'s historical Colab run bit-for-bit), so exact reproduction *within* Colab-CPU-to-Colab-CPU is expected and fine — the risk is specifically local-to-Colab contamination via shared Drive storage.
```

---

## gbdt_interaction_type_matters.md

```markdown
---
name: gbdt-interaction-type-matters
description: Explicit interaction features only help GBDT when they encode a cross-field equality/match; plain AND-of-two-raw-categories is already learnable by trees and adding it can hurt
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b15106b6-cdc7-4e1e-9c01-1c419c65f3a7
  modified: 2026-08-11T09:00:36.261Z
---

Not all statistically significant interaction terms are worth engineering as explicit CatBoost features — it depends on whether trees can already represent the interaction from the raw columns.

**Why**: Two systematic pattern searches surfaced strong two-variable interactions on this HR retention dataset:
- Block L/J (relocation tolerance × desired-location-matches-actual-location): a comparison of whether **two independently-encoded categorical variables hold the same value**. Trees split on one feature's threshold/category at a time and cannot natively test "does variable A's value equal variable B's value" — that requires an explicit engineered flag. Making it explicit was a huge win (L: 38.3pt gap → confirmed as new best Public score, [[eda_v5_findings]]).
- Block M (専攻分野=non-analytical AND 初期職種=analytical): despite an even more significant raw chi-square signal (30.5pt gap, p=4e-30) than J, adding it as an explicit feature to `29_major_job_analytical_mismatch` made the CatBoost validation score **worse** (mean_diff +0.0046, both splits raw or flat-to-worse). Both underlying raw columns (専攻分野, 初期職種) were already present in the baseline feature set. A plain AND of two already-present categorical values is exactly what GBDT tree splits do natively (split on job type, then within that leaf split on major) — the explicit flag was redundant and only added noise to the Optuna search space.

**How to apply**: When a systematic/exhaustive interaction search (chi-square scan across categorical pairs, etc.) surfaces a candidate, classify it before spending a submission on it:
- **Cross-field equality/match type** ("does stated preference X equal realized outcome Y", "does A's value match B's value") → high expected value as an explicit feature; GBDTs can't learn this on their own. Prioritize these.
- **Plain AND-of-two-raw-categorical-values type** (both variables already present as raw model inputs) → low expected value; a tree ensemble likely already captures it implicitly. Validate locally first (cheap) before spending Colab time or a submission — a clean local ablation showing no improvement (or both splits flat/worse) is enough to skip Colab/Public entirely, this doesn't need the "local isn't decisive" caveat when the direction is unambiguously non-positive on both splits.

This also means a raw p-value/effect-size ranking from an exhaustive interaction scan should not be the sole prioritization criterion — filter by interaction *type* first.

**Follow-up (2026-08-11, confirmed via 3 experiment tracks)**: Track 1 (`30_catboost_encoding_variants`) tried 3 ways of feeding block M to CatBoost (add alongside raw cols / force one-hot via `one_hot_max_size` / replace raw cols with only the joint category) — all 3 made things worse than baseline, worse in that exact order. Confirms encoding tricks don't revive an AND-type block for a GBDT. Track 2 (`31_linear_model_rejected_features`) then confirmed the flip side: M was the single strongest individual improvement (-0.0073) in an L2 logistic regression built on the same base features, exactly as this hypothesis predicts (linear models can't learn ANY interaction, AND-type or equality-type, without an explicit feature). Track 3 (`32_ensemble_catboost_linear`) tried blending CatBoost(L_v2)+this linear model via fixed-weight probability averaging — this failed too (monotonically worse as linear weight increased) because the linear model's absolute quality (val~0.57-0.59) is far below CatBoost's (~0.50-0.53), so the "weak learner tax" outweighs the diversity benefit in simple probability-space blending. See [[ensemble_oof_overfitting]] for the general pattern of weak-model blends losing to the strong model alone.
```

---

## test_set_is_survivor_filtered.md

```markdown
---
name: test-set-is-survivor-filtered
description: Test contains only employees who survived 24 months; train does not — base rate and validation level differ
metadata: 
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-12T11:16:12.203Z
---

Test の月次データは 2,502名 × ちょうど24ヶ月で、`月末在籍状態=="退職"` の行が **1件も無い**。
Train には 0-23ヶ月に退職した社員が 129名（4.7%）いて全員ラベル0。つまり Test は
「24ヶ月生存者」に絞られた母集団で、Train は絞られていない（2026-08-11 の EDA v6 で発見、
v1〜v5 では未指摘）。

基準定着率: Train全体 0.5647 → 早期退職者を除くと 0.5923。

**Why:** 検証セットには Test に存在しない「自明に当てられる負例」が 3.3% 混ざるため、
検証スコアが構造的に楽観側へずれる。既知Public 6構成で測ると検証-Publicギャップの平均は
0.0184 → 0.0070 に縮む（ただし std と順位相関は改善しないので、これは水準の補正であって
ブロック採否の判定精度を上げるものではない）。

**How to apply:** 検証セットからは早期退職者129名を除く。**学習からは除かない。**
予測確率の平均は既に 0.598 で生存者基準率 0.592 とほぼ一致しているので、
単純な再キャリブレーションによる伸びしろも無い。

**Publicで確定（2026-08-12、`40_` P1_drop_early）**: 学習からも129名を除くと
**+0.001765 悪化**（D3 0.522659 → P1 0.524424。441列・A_PARAMS・560反復・5シードで
学習行だけが違う）。学習曲線の傾き（1名あたり0.0000195）から予測される
129名喪失の損失 +0.002522 とほぼ一致し、**母集団是正の利得はあっても0.0008程度で
データ喪失に負ける**。129名は Test 集団の予測にも役立つ情報を持っていた。

**副次的な観察**: この件では EDA v6のローカル追試(+0.0044)・`40_`の検証(+0.00668)・
Public(+0.00176) が**3つとも同じ符号**だった。特徴量の変更では検証が繰り返し符号を外すのに
（[[ablation-cannot-settle-feature-blocks]]）、**学習行の変更では当たった**。n=1なので運用は変えない。

関連: [[horizon-dependent-feature-effects]] [[validation_gap_finding]] [[best-submission-status]]
```

---

## horizon_dependent_feature_effects.md

```markdown
---
name: horizon-dependent-feature-effects
description: Hire-time-fixed features strengthen with prediction horizon; behavioral features weaken — explains why G/H/K/N/O kept failing
metadata: 
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-11T14:42:11.840Z
---

`employee_monthly_train_full.csv` から退職月を復元し、「hヶ月時点の在籍」を目的変数にして
効果量の推移を測った（EDA v6、2026-08-11）。

| 特徴量 | h24 | h60 | h120 |
|---|---|---|---|
| L（転居×勤務地ミスマッチ, pt差） | -7.7 | -28.3 | -36.4 |
| 希望勤務地マッチ（pt差） | +6.1 | +24.0 | +25.5 |
| 初期役割 / 初任給（相関） | 0.016 / 0.005 | 0.134 / 0.102 | 0.152 / 0.132 |
| **残業時間_mean（相関）** | **-0.496** | -0.434 | **-0.302** |

入社時固定系は h とともに強くなり h72〜84 で飽和。行動系（残業時間）は逆に h24 が最強で
h120 では6割へ減衰する。

**Why:** 行動系ブロック（G 自己学習・H エンゲージメント・K 昇給タイミング・N モメンタム・
O 活動密度）が「検証で改善→Publicで悪化」を繰り返した機構的説明になる。行動系が本当に
予測しているのは短期離職であり、(1) 10年定着への予測力はもともと減衰しており、
(2) その短期離職者が Test には1人もいない（[[test-set-is-survivor-filtered]]）ため、
検証でだけ手柄を立てられる。`submit_result_report.md`セクション34 の「会社の運用・制度が
年月とともに変化しているため」という当時の仮説は、理由としては誤りだったと考えられる。

**How to apply:** 行動系・モメンタム系の特徴量探索は打ち切ってよい。

**ただし「生存時間を目的変数に組み込む」方向には繋がらなかった（実測で不採用）。**
AUC（キャリブレーション非依存）で4方式を比較した結果、ベースライン y120 binary（AUC 0.7878±0.0049）を
上回るものは無し: multi-horizon平均 0.7907±0.0014（差はシード分散の0.6倍かつ4モデル平均による
分散低減と区別不能）、ソフトラベルCE scale0.25 0.7867 / scale0.40 0.7848、退職の早さによる
サンプル重み 0.7835、h84補助目的変数 0.7796。

**罠**: `CatBoostClassifier` に小数ターゲットを渡すと既定の `Logloss` では各小数値を別クラス扱いし
（`classes_` が `[0.3, 1.0]` 等になる）、`predict_proba(...)[:,1]` が無意味になる（AUC 0.2554 で発覚）。
ソフトラベルには `loss_function="CrossEntropy"` を明示すること。
```

---

## modeling_levers_beat_new_features.md

```markdown
---
name: modeling-levers-beat-new-features
description: Training-set size and seed averaging offer bigger gains than any remaining new feature; seed sd exceeds the effect sizes being judged
metadata: 
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-11T14:55:34.273Z
---

EDA v6（2026-08-11）で実測した、特徴量を1つも足さずに得られる改善余地。

**学習量**: `28_`の `run_model_config` は最終モデルを Train の80%だけで学習し、残り20%（553名）を
`eval_set` にしか使っていない。学習曲線は80%地点でも未飽和（60%:0.54470 → 80%:0.53391、
+552件で0.011改善。ただし70%は0.54649 と非単調）。全件学習で 0.005〜0.01 の期待値。

**シード分散**: 同一設定でシードのみ変えると val logloss の sd は 0.0045〜0.0068。
これは L_v1→L_v2 の Public 差（0.0002）や N・O（±0.001）より大きい。
3〜5シード平均で単一シード平均から 0.006〜0.009 改善（8シードで頭打ち）。

**Why:** 単一実行同士を比べて「両split一貫して改善」と判定してきた運用は、シードの引きを
見ていた可能性がある（[[local_test_not_decisive_for_ablation]]・[[catboost_gpu_nondeterminism]]
として記録してきた現象の正体）。シード平均は重みを学習しないので
[[ensemble_oof_overfitting]] の教訓には抵触しない。

**Publicで確定（2026-08-11、`37_`）**: 全件学習とシード平均だけが違う2構成を提出して分離できた。

- **全件学習: -0.00646（予測通り的中）** → 新最良 Public 0.522659（[[best-submission-status]]）
- **シード平均: -0.00034（ほぼ横ばい、予測は外れ）**

シード平均がPublicで効かなかったのは、比較対象の `28_` が使っていた seed=42 が5シード中の
当たりくじだったため（各シードのval: 0.5143/0.5164/0.5204/0.5257/0.5257）。
**シード平均の価値は期待値の改善ではなく、悪いシードを引いたときの保険（分散低減）**と理解すべき。

**How to apply:** (1) 最終モデルは必ず Train 全件で再学習（イテレーション数は80/20で決めた値を
件数比1.25でスケール。スケール係数自体は未検証の任意の選択）、(2) 提出はシード平均（期待値は
上がらないがシード事故を防ぐ）、(3) 今後のアブレーション判定は必ず複数シード平均で行う
——シードsdは本番441列で 0.0047〜0.0078 あり、判定してきた効果量より大きい。

**注意**: 少数の検証データ（535件）に対するOptuna探索は不安定。`37_`では検証セットを18名(3.3%)
変えただけで探索が別領域に着地した（depth 4→8、L2が1/26、best_iteration 584→165）。
検証セットの構成をいじるなら**ハイパーパラメータは固定**すること。
```

---

## eda_v6_findings.md

```markdown
---
name: eda-v6-findings
description: "EDA v6 closed the text/confounder search space — feedback text, manager effect, cohort size, self-study portability all null"
metadata: 
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-11T14:05:35.233Z
---

EDA v6（`notebooks/eda/data_exploration_v6.ipynb` / `report/md/data_exploration_v6_report.md`、
2026-08-11）の null 結果。いずれも「もう探さなくてよい」と確定させたもの。

- **フィードバック2列**: 12文字n-gramを総当たり（上司62個・同僚42個）して Bonferroni 補正すると
  **有意ゼロ**。v4 の手選びキーワードによる null を、極性ごと捕捉できる方法で追認した。
- **入社時メモ**: 有意なn-gram 65個は全て `初期職種`（IT・エンジニアリング）とブロックEの
  在宅希望フラグの言い換えで、新規性ゼロ。テキスト系の探索は枯渇。
- **上司IDのランダム効果**: 観測分散は二項ノイズの1.12倍のみ、LOO相関0.072。部署(0.109)より弱く、
  その部署は v5 で「職種の再パッケージ」と判明済み。不採用。
- **同期入社人数**: 全体 corr -0.137 だが入社区分内・4月入社内では -0.01 台。新卒4月一括採用の代理変数。
- **自己学習テーマの可搬性比率**（Python/SQL/クラウド等の学習比率）: 全体 corr -0.145、
  五分位 0.648→0.460 と綺麗に単調だが、**職種内プール相関 -0.029 で符号も反転**。`初期職種` の代理変数。
- **「在宅希望なのに在宅日数が少ない」L型フラグ**: p=0.63 かつ符号逆。L の成功は勤務地固有で、
  働き方一般には拡張できない。

**Why:** 3件目が特に紛らわしい（解釈しやすいストーリー＋単調性を持つのに実体は既知情報）。
v5 の「単純集計で大きく見える差は主要な既存特徴量で層別してから採否を決める」を必ず通すこと。

関連: [[horizon-dependent-feature-effects]] [[test-set-is-survivor-filtered]]
[[modeling-levers-beat-new-features]] [[eda_v5_findings]]
```

---

## ablation_cannot_settle_feature_blocks.md

```markdown
---
name: ablation-cannot-settle-feature-blocks
description: Validation-based ablation has never predicted Public for this project; only raw EDA effect sizes have
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-12T11:12:32.213Z
---

**Publicで転移したブロックは E・J・L の3つだけで、いずれもEDAの生の効果量で発見された**
（定着率の差 25.3〜38.3%pt、p < 1e-26）。逆に**アブレーションの検証スコア改善を根拠に採用した
ブロックは一つも転移していない**（G・H・I・K・M・N・O、combo_EFG/EG/EI）。

決定打は `39_`（2026-08-12）。Gブロックは本プロジェクト最厳のプロトコル
（ハイパーパラメータ固定・反復数固定・8シード・生存者のみ検証・4 splitすべてでペアード
ブートストラップ95%CIが完全に0未満・陽性対照も正常）を通過したのに、
検証 -0.0109 → **Public +0.0012 と符号反転**した（Gはこれが2度目の失敗）。

**Why:** 事後に検証設計の欠陥が2つ見つかった。
1. **入れ子split**: 検証セットを入社日順の後ろから切るため、小さいsplitは大きいsplitの部分集合。
   split_85_15(401名)は他3つすべてに含まれる。「4/4 splitで有意」は4回の独立検定ではなく、
   実質1回（n=798）を4サイズで報告していただけ。
2. **勝者の呪い**: 13構成を同じ検証セットで比べて最良を選んだのに多重比較を補正していない。
   CI半幅±0.009〜0.011で13回引けば、真の効果0でも-0.011の当たりは出る。

**How to apply:**
- 新規ブロックの採否は**アブレーションの検証スコアではなくEDAの生の効果量**で判断する
  （目安: 定着率差20%pt以上、p<1e-20。E・J・Lは全てこの水準）。
- splitを増やすなら**入れ子にならない設計**（重複しない時間ブロック）にする。
- 複数構成から勝者を選ぶなら多重比較を補正するか、選択バイアス込みだと明記する。
- **adversarial validationは転移の予測に使えない**: E(0.80)・L(0.74)という最大の期間シフトを
  持つブロックが唯一転移し、G(0.495)はシフトゼロなのに転移しない。P(x)ではなくP(y|x)の問題。

**最大の反証事例（`42_`、2026-08-12）**: `40_` のLOO診断は「`agg` 224列を落とすと検証 **−0.01068**」
（14グループ中で最大の改善）と示したが、実際に落とした `S1_no_agg33`(33列) の Public は
**+0.00977**。**符号反転・ずれ約0.020**で、`39_` のGブロック（0.012）を上回る本プロジェクト最大の乖離。
`S2_agg_mean49`(49列) も検証 −0.00607（全構成中最良）→ Public +0.00420 で反転。

**教訓の拡張**: 「検証スコアで採否を決めるな」は**削る対象の選択にも適用される**。
`42_` では「減らす方向はPublicで確認済み」を根拠にしたが、
**"次に何を落とすか" をLOO診断＝検証スコアから取ったのが誤り**だった。
判断すべきは列数ではなく**その列が一次情報か派生・冗長列か**:
成功した441→113は派生列（TF-IDF SVD・四半期・高度統計・aggの冗長統計144）を落としたが、
失敗した113→33は一次情報（aggの中核統計80）を落としていた。

**補強証拠（`40_`、2026-08-12、グループ単位leave-one-out・8シード）**:
441列のうち検証で明確に必要と言えるのは **`L2`(2列, 落とすと+0.0400)** と
**`persona`(21列, +0.0280)** の2つだけ。残り12グループはすべて分解能±0.011以下で、
`agg`(224列＝全体の半分)に至っては**落とすと0.0107改善**する。
**EDA v5の生の効果量で発見したL_v2 2列が、月次集約224列より測定可能な寄与を持っている。**

関連: [[best-submission-status]] [[horizon-dependent-feature-effects]]
[[modeling-levers-beat-new-features]] [[local_test_not_decisive_for_ablation]]
```

---

## hire_year_extrapolation.md

```markdown
---
name: hire-year-extrapolation
description: "入社年 is 100% extrapolating for Test but CatBoost barely uses it — real design flaw, no measurable harm"
metadata: 
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-12T00:44:42.548Z
---

**Train と Test は入社年で完全に分離している**（Train 2011-04〜2014-03 / Test 2014-04〜2017-03）。
にもかかわらず `28_`〜`40_` は **`入社年` を特徴量として渡している**
（`入社日` は `_feature_cols` で除外されるが、`入社年`・`入社月`・`入社四半期` は persona 列として残る）。
80%学習時にモデルが見るのは {2011,2012,2013} のみで、全件学習でも2014年は**60名だけ**。
Test 2,502名は全員 `入社年` ≥ 2014。

**結論（`41_` 実測、2026-08-12）: 設計上の不備は実在するが、実害は無い。**

| 測定 | 結果 |
|---|---|
| 学習範囲内(2011-2013)の定着率の振れ | 3.2%pt、カイ二乗 **p=0.345**（信号なし） |
| 「≥2014」の学習側の証拠 | **60名のみ**、0.6500、95%CI [0.5245, 0.7614]（Fisher p=0.190 で区別不能） |
| **特徴量重要度（113列）** | **98位/113、全体の0.18%** |
| X0(あり) vs X1(なし) 予測の平均絶対差 | 0.01438 |
| **同一構成でシードを引き直したノイズ床** | **0.01246（95%上限 0.02122）** |
| 予測平均の差 | +0.00057（入社年ごとの系統的ズレも無し） |

**`入社年` を外して予測が動く量は、シードを引き直すのと変わらない。**
提出は見送った（現最良との相関0.9973 ＝ `38_` H1 と同じ状況）。

**外挿監査（441列全数）: 閾値0.5以上は `入社年` だけ。次点は0.060。同種の列は存在しない。**
監査コードは `41_` 第11節にあり再利用できる。

**Why:** 当初「Test予測平均0.5875 vs Train定着率0.5647 の差の一因では」と疑ったが**棄却**。
CatBoost（depth=4・強正則化）が無信号の列を実質的に無視していた。

**How to apply:**
- **`入社年` は今後の基準構成から外す**（信号が無く全Testが外挿域＝残す理由が無い、という事前知識による。
  スコアが理由ではない）。`入社月`・`入社四半期` は循環変数で外挿問題が無いので残す。
- **予測差の大小を論じるときは必ず同一構成のシード再抽選と比較する。**
  `41_` では事前に「平均絶対差0.01以上なら大きい」と書いたが、
  ノイズ床を測らずに決めた閾値で誤っていた。
- 外部の景気指標（有効求人倍率）のマージは不要 —— 入社年と1対1対応で情報が増えず、外挿も解消しない。

発見経路: 文献レビュー（`literature/01_early_turnover_literature_review.md` 第3.1節）。
小林徹(2016)の「学卒時の有効求人倍率が3年内離職率の約3%ptを説明」を受けてコホート変数を点検した。

関連: [[best-submission-status]] [[literature-review-early-turnover]]
[[ablation-cannot-settle-feature-blocks]] [[test-set-is-survivor-filtered]]
```

---

## literature_review_early_turnover.md

```markdown
---
name: literature-review-early-turnover
description: "What the early-turnover literature does and does not support for this competition's feature design"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-11T23:34:56.516Z
---

`literature/01_early_turnover_literature_review.md`（2026-08-12作成）に、
若年者の早期離職・定着に関する文献9件のレビューをまとめた。

**最も内容の濃い文献は 小林徹(2016)『日本労働研究雑誌』668**（全文精読済み、東大社研パネルn=2073）。
- **産業・職業の違いは3年内離職に有意な影響なし。企業規模だけが明確に効く**（大企業ほど低い）
- 卒業時の有効求人倍率が約**3%pt**を説明（単一変数として最大）
- メカニズム: 企業特殊的技能が重要な職場ほど**勤続賃金カーブが立ち**、離職率が低い

**文献が支持した既存設計**: `初任給_等級内偏差` 等（JILPT2016「成果に見合った賃金」）、
ブロックLの交互作用エンコード（Kachi et al.2018 は雇用不安の効果が**子ありでのみ**出現＝同型の部分集団効果）、
産業・職業系を重視していないこと、検証セットの生存者補正（Works2023のブランク分析と同じ選択バイアス構造）。

**文献から新規に実行する価値があるのは1つだけ**: 部署×職種単位の**勤続賃金カーブの傾き**
（個人の昇給ではなく所属集団の構造属性である点が、失敗したブロックK「昇給タイミング」と異なる）。
ただし本データは24ヶ月しかなく、まずEDAで生の効果量を測ること。

**却下した仮説**: 職務満足度（古屋2021で19.0pt差だが該当列が本データに無い）、
入社前研修の非単調性（杉山2019はn=21インタビューでエビデンス弱、かつGBDTは非単調を学習済み）、
部署規模（EDA v6で交絡プロキシと判定済み）、外部景気指標のマージ（[[hire-year-extrapolation]]参照）。

**Why:** この文献群はほぼすべて**3年以内**の離職を扱っており、本コンペの目的変数は**10年定着**。
しかも3年内離職者は Test から構造的に排除されている（[[test-set-is-survivor-filtered]]）。
[[horizon-dependent-feature-effects]] を踏まえると、**文献の知見をそのまま持ち込むことはできない**。

**How to apply:** 文献は「新しい特徴量の供給源」としてはほぼ枯れている。
価値があったのは既存設計の裏づけと、**データ側の欠陥発見**（[[hire-year-extrapolation]]）の方。
文献由来の仮説も採否基準は変わらず、EDAの生の効果量（定着率差20%pt以上）で判断する
（[[ablation-cannot-settle-feature-blocks]]）。

**注意**: 近藤絢子(2008)と前田ほか(2010)は配布元PDFにアクセスできず本文未取得。
レポートでは独自要約を書かず、他文献による引用のみ記載してある。
```

---

## hyperparameter_retuning_exhausted.md

```markdown
---
name: hyperparameter-retuning-exhausted
description: Retuning CatBoost hyperparameters found a different region but the same function — stop investing in it
metadata: 
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-12T01:18:17.108Z
---

**ハイパーパラメータ再探索は441列でも113列でも不発。今後investしない。**

| | `38_`（441列） | `42_`（113列） |
|---|---|---|
| 探索結果 | depth 4 のまま（A_PARAMSとほぼ同じ領域） | **depth 4→5、l2 が 1/3.4**（別領域を発見） |
| マルチシードvalの改善 | −0.002（シードsd 0.0047 以下） | −0.00205（典型シードsd 0.00206 とほぼ同値） |
| Public | 提出見送り | 提出見送り |

**`42_` の決定的な発見: 再探索が見つけたのは別の関数ではなく、同じ関数への別の経路だった。**
再探索パラメータは learning_rate が2.5分の1（0.0352→0.0140）で ES best_iteration が2倍（387→801）。

- 560反復（学習不足）で学習した `T1` は T0 との Test平均絶対差 **0.04442**
- 正しい反復数1001で学習した `T1b` は **0.01878** ＝ **T0に戻ってきた**

`41_` で実測したノイズ床（Test 5シード平均どうしで平均0.01246・95%上限0.02122）の範囲内。
**学習率を下げて反復を増やせば、深さと正則化を変えてもほぼ同じ解に収束する。**

**Why:** 事前登録した提出条件（改善幅 > 典型シードsd）は 0.002053127 vs 0.002057871 で
**4.7e-6 差で不成立**という無意味な僅差だった。しかし結論は基準に依存しない:
(1) `T1` は lr 0.014 に対し560反復で構成として誤り、(2) `T1b` の予測差はノイズ床の範囲内。

**How to apply:**
- **反復数560は113列でも最適だった**（曲線の最小点。250〜1200を測定）。今後も560を使う。
- **113列ではシード分散が441列の半分以下**（val_single_sd 0.0011〜0.0050、中央値0.0021 vs
  441列の0.0038〜0.0057）。特徴量を減らすと実験の分解能そのものが上がる。
- 予測差の判定には必ず [[hire-year-extrapolation]] で測ったノイズ床と比較する。
- 事前登録の閾値が僅差で決まりそうなときは、**基準に依存しない別の根拠**を探すこと。

関連: [[best-submission-status]] [[modeling-levers-beat-new-features]]
[[ablation-cannot-settle-feature-blocks]]
```

---

## catboost_beats_other_gbdt.md

```markdown
---
name: catboost-beats-other-gbdt
description: On this feature set CatBoost beats LightGBM by 0.032 and XGBoost by 0.026 — the top features are categorical
metadata: 
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-12T13:25:20.570Z
---

**113列（`40_` R6_lean）で3ファミリを同条件で比較した結果（`43_`、2026-08-12）:**

| モデル | val（生存者535名・8シード平均） | CatBoost差 |
|---|---|---|
| **CatBoost** | **0.514642** | — |
| LightGBM | 0.546655 | **+0.032012** |
| XGBoost | 0.541109 | **+0.026466** |

**これは本プロジェクトで初めての「検証でdecisiveな差」**。効果量が分解能（CI半幅±0.011）の
2.4〜2.9倍あり、判定不能だらけだった他の比較とは質が違う。
**効果が本当に大きければ検証でも判定できる**ことの確認にもなった。

**多様性はあった**: catboost vs lgbm/xgb の予測相関は 0.924/0.938 で、
CatBoost内のシード間相関 0.986 を明確に下回る。だが lgbm vs xgb は 0.982 で実質同じ。
**分かれ目は「勾配ブースティングの実装」ではなく CatBoost かそれ以外か。**

等重み平均を試算するとすべて悪化（cat+lgbm +0.0119 / cat+xgb +0.0098 / 3モデル +0.0155）。
事前登録したゲート（単体valがCatBoost+0.01以内のみ参加）が正しく機能し、**提出候補ゼロ**。

**Why:** `41_` の特徴量重要度の上位が **`専攻分野`・`初期職種`（カテゴリ）**、
そして **`転居x勤務地_状態_v2`（カテゴリ）** である。
CatBoost の ordered target statistics がこの形の情報に強く、LGBM/XGB は両方ともその利点を持たない。
`12_` の「XGBoostが3モデル中最も弱い」も同じ現象を別の特徴量セットで見ていたことになる。

**続報（`46_`、同日）: この「単体valのみ見るゲート」は不適切だった。**
入力を分割して作った `M1_cat`(11列) / `M2_num`(100列) は単体で +0.027 / +0.020 と
同程度に弱いのに、**M0との相関が 0.79 / 0.86 と低く、等重み平均すると検証が改善した**
（E_M0_M1 −0.0021、E_ALL3 −0.0050。M1とM2の相関は 0.63）。
`43_` は相関0.92-0.94だったから悪化しただけで、**等重み平均の可否は
「メンバーの強さ」だけでなく「メンバー間の相関」で決まる**。
ゲートを `43_` から `46_` へコピーしたとき、適用範囲が変わったことを検討しなかったのが誤り。

**How to apply:**
- **モデルファミリを CatBoost から替えない。** ただし
  **アンサンブルの可否は単体の強さだけで判断しない —— 必ず予測相関も見る。**
- 重みは学習しない（[[ensemble-oof-overfitting]]）。等重みでも弱いモデルは平均を引きずる。
- 副産物: LightGBMのシード分散が異常に小さい（sd 0.000583 vs CatBoost 0.005220）。
  XGBoostの予測はfloat32で返る（npyサイズが半分になるので混乱しないこと）。

関連: [[best-submission-status]] [[ensemble-oof-overfitting]] [[hyperparameter-retuning-exhausted]]
```

---

## refit_chaos_noise_floor.md

```markdown
---
name: refit-chaos-noise-floor
description: "Any change to the feature matrix shifts predictions by ~0.014 MAD — that is refit chaos, not an effect"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-16T01:20:31.956Z
---

**特徴量行列に何を変えても、Test予測の平均絶対差は約0.014に落ち着く。**
これは「変更の効果」ではなく**再学習に伴う軌道の揺らぎ**である。

| 変更内容 | 平均絶対差 |
|---|---|
| `41_` X1: `入社年` 1列を除去 | 0.01438 |
| `41_` X2: 入社時系3列を除去 | 0.01402 |
| `45_` N1: 数値欠損を −999 から NaN に | 0.01405 |
| `45_` N2: 96%欠損の6列を除去 | 0.01400 |
| `45_` N3: 上記2つの併用 | 0.01437 |
| **同一構成でシードを引き直したノイズ床** | **0.01246（95%上限 0.02122）** |

変更の中身がまったく違うのに数値がほぼ同じで、しかもシード引き直しと同水準。

**Why:** `41_` で事前に「平均絶対差0.01以上なら大きい」と閾値を書いたが、
**ノイズ床を測らずに決めた閾値で誤っていた**。5例そろって同じ値に落ち着いたことで確定した。

**How to apply:**
- **予測差が0.014前後で出たら「何も起きていない」と読む。** 提出しても現最良の近傍に戻るだけ。
- 提出ゲートは **平均絶対差 > 0.02122（ノイズ床95%上限）** を使う。
  `38_` H1(0.0146)・`41_` X1(0.0144)・`45_` 全構成(0.0140-0.0144) はすべてこれで見送り、
  実際に情報が得られない構成だった。
- 逆に **0.03を超えたら本物の別モデル**（`42_` S1 0.061 / S2 0.043、`43_` cat+lgbm 0.035 など）。
- **効果の大小を論じるときは必ず同一構成のシード再抽選と比較する。**

## AutoGluonの実行間ノイズを実測（`62_`、2026-08-16）— 上の値より更に大きい

上のノイズ床0.02122は**単層CatBoost**で測った値。`62_`で同一構成のAutoGluonを3シード
独立実行して測ったところ、もっと大きかった:

| 対象 | シード間MAD平均 | ノイズ床(0.02122)比 |
|---|---|---|
| **weighted** | **0.02381** | **1.12倍** |
| **best_single** | **0.03719** | **1.75倍** |

**しかもこれは下限。** AutoGluonの`fit()`は`random_seed`非対応で、モデル個別のシードを
変えるには`hyperparameters`上書きが必要（zeroshotポートフォリオを壊すのでやらない）。
上記は**fold割り当ての違いのみ**から生じた変動。内部valも同一構成で0.4818〜0.4879と
**0.006振れる**。

### これで無効化される過去の「結論」

AutoGluon構成間で観測したPublicの差は軒並みこの変動幅より小さい:
`50_`vs`51_` weighted (0.0012)、`56_`のLM統合 (0.0022)、`51_`のNN_TORCH/KNN除外 (0.0008)
——**すべてノイズと区別できない**。特に[[l2-m-risk-count-confirmed]]で
「2構成とも悪化方向で一致、偶然ではなく確定的」と書いたのは**過信**だった
（コイン2枚が同じ面を出す確率は50%）。

### 副産物: WeightedEnsembleがbest_singleに勝ち続けた理由

best_singleのシード間MAD(0.0372)はweighted(0.0238)の**1.56倍**。
WeightedEnsembleは加重平均なので実行ごとのばらつきが打ち消される。
[[ensemble-oof-overfitting]] addendum 5の「weightedが勝つ」は
「重み学習が賢い」のではなく**単に分散が小さいから**と機構的に説明できる。

### How to apply（更新）

- **AutoGluonの構成比較は単発実行では判定できない。** 比較するなら最低3シードずつ回して
  平均で比べる（コスト3倍）。単発実行の提出は実質くじ引き1回。
- **差が0.002程度の結果に機構的な解釈を書かない。まずノイズを疑う。**
- **提出ゲート「mad_vs_best > ノイズ床」はシード平均には適用しない。**
  あれは「別構成を試す価値があるか」の基準であって、分散削減が目的の平均化では
  「現最良と近い」ことはむしろ正常（seedavg3のmad 0.0202 < シード間MAD 0.0238 は
  中心に寄っている証拠）。

関連: [[hire-year-extrapolation]] [[hyperparameter-retuning-exhausted]] [[best-submission-status]]
[[l2-m-risk-count-confirmed]] [[ensemble-oof-overfitting]] [[validation-asymmetry]]
```

---

## validation_asymmetry.md

```markdown
---
name: validation-asymmetry
description: "Validation is trustworthy when it says \"worse\" and almost never when it says \"better\""
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-16T08:19:36.275Z
---

**検証とPublicが両方揃った全6件の記録（2026-08-12時点）:**

| 構成 | 検証のref差 | Publicのref差 | |
|---|---|---|---|
| `39_` Gブロック | −0.01090 | +0.00120 | ✗ 反転 |
| `40_` R6_lean(113列) | −0.00101 | −0.00093 | 一致 |
| `40_` P1_drop_early | +0.00668 | +0.00176 | 一致 |
| `42_` S2_agg_mean49 | −0.00607 | +0.00420 | ✗ 反転 |
| `42_` S1_no_agg33 | −0.00169 | +0.00977 | ✗ 反転 |
| `46_` E_ALL3（入力分割アンサンブル） | −0.00504 | +0.00151 | ✗ 反転 |

**追加（`44_`、2026-08-13）**: AutoGluon の holdout val は `lean113`(0.501765) が
`full441`(0.506815) より +0.00505 良いと言ったが、**Public は full441 が +0.00484 良かった**（6回目の反転）。
ただし full441 の holdout は時間切れで63モデルしか学習できず（lean113は149）、同じ土俵ではない点は注記する。

**追加（`49_`、2026-08-13、[[memo-parser-coverage-bug]]）**: 単一の事前登録済みバグ修正
（メモパーサーのカバレッジ回復）を441列・113列の2構成でだけ測定。**2/2で検証とPublicの向きが一致**
（441列: 検証-0.00274→Public-0.00351、幅も一致／113列: 検証-0.00825→Public-0.00221、
幅は検証の27%に縮小）。7・8件目のデータ点だが、過去6件の反転例はすべて「複数構成から検証スコアで
勝者を選ぶ」探索だったのに対し、今回は探索なしの単一修正だった点が違う。

**一致 4/8。しかも強い非対称がある:**

| 検証の言い分 | 件数 | Publicでも同じ向き |
|---|---|---|
| **「改善する」（探索あり・勝者選択）** | 5 | **1件のみ** |
| **「改善する」（探索なし・単一の事前登録修正）** | 2 | **2件（的中、ただし幅は過大評価しがち）** |
| **「悪化する」** | 1 | **1件（的中）** |

**How to applyの追記**: 検証の「改善」を信じてよいかは、**その改善幅が何個の構成から選ばれたものか**
で判断が変わる。単一の事前登録済み修正（探索なし）なら検証は方向だけは概ね当たる。
複数構成から検証スコアで選んだ「改善」は引き続き信用しない。

**追加（`52_`、2026-08-14）**: 境界線上の効果量（19.5pt/p=2.5e-18、採用基準未達）を持つ
残業時間dip-recovery特徴量を2構成でテスト。441列は検証「改善」-0.0033→Public「悪化」+0.0019
（反転）、113列は検証「悪化」+0.0023→Public「悪化」+0.0005（一致）。
**「改善」は通算3/7的中、「悪化」は通算3/3的中**——単一の事前登録済み修正であっても
「改善」方向の検証は依然半分以下しか当たらない。非対称性は探索の有無だけでは完全に消えない。

**追加（`50_`、2026-08-14）**: AutoGluon（44_→50_、パーサー修正+インフラ修正の統合）で
2構成とも検証・Publicが一致（full441: holdoutでは分解能以下と判定も内部val/完走モデル数の
増加から改善方向、Public -0.001736で的中／lean113: 検証「改善」、Public -0.003292で的中）。
**通算「改善」5/9的中**。この2件は「新規仮説を検証」ではなく「同じレシピを、環境不具合を
直して完走させただけ」という性質が違うタイプの変更で、的中率が上がる傾向がある
（[[memo-parser-coverage-bug]]の単一修正2/2的中と合わせて、"新規性の低い改善"は
検証が当てになりやすいという仮説を補強）。

**追加（`56_`、2026-08-15）**: LMブロックをAutoGluonに統合。full441_weightedで検証「改善」
-0.000937→**Public「悪化」+0.002152（反転）**。単層CatBoost（54_）では同じブロックが
検証・Publicとも一致して改善していたのに、AutoGluonの複雑な構造の中では反転した——
「アーキテクチャが変わると同じ特徴量でも検証の信頼性が変わりうる」という新しい教訓。

**追加（`54_`、2026-08-15）**: L2×Mのリスク要因数を追加。441列は検証「改善」-0.0074→Public
「改善」-0.0041（一致、幅は縮小）。113列は検証「悪化」+0.0020→**Public「改善」-0.0010**
（反転）。「検証が悪化と言うときは信用してよい」という唯一の頼れる非対称性への**初めての反例**。
ただし+0.0020は分解能ぎりぎりの弱いシグナルだった点は注記する（強い悪化シグナルでの反転例はまだ無い）。

**追加（`51_`、2026-08-15）**: `50_`に`excluded_model_types`を追加しただけの変更で、
検証はほぼ横ばい（WeightedEnsemble -0.00017 / best_single +0.00017、モデル完走数も61<72で減少）
——「効果なし」と読める内容だった。**それでもPublicは-0.000826改善し、新最良を更新した。**
検証が「改善でも悪化でもない」と言ったケースでPublicが動いた初めての例。
**新パターン**: 検証の差がノイズ床（±0.0002程度）以下のときは、検証結果を全く当てにできない
（改善・悪化どちらの方向にも転びうる）。これは「差が無い」ことの検証すら信用できないことを示し、
[[refit-chaos-noise-floor]]の教訓（構成を変えれば必ず何かは変わる）と整合する。

**Why:** 改善方向の探索はすべて検証セット535名を見て行うので、当たりくじだけが残る
（[[ablation-cannot-settle-feature-blocks]] の勝者の呪い）。
悪化方向は探索していないのでバイアスがかからない。

**How to apply:**
- **検証は足切り（悪化の検出）にだけ使う。改善の主張は必ず Public で確かめる。**
  現行の運用（`VAL_REJECT_MARGIN` で足切り、順位付けはしない）はこの非対称と整合している。
- 検証が「改善」と言ったことを提出の**根拠**にしない。根拠に使えるのは
  ラベルを見ない指標だけ —— 予測相関、[[refit-chaos-noise-floor]] を超える予測差。
- 唯一の例外は `40_` R6_lean（検証 −0.00101 → Public −0.00093）だが、
  ペアードブートストラップのCIは0を跨いでいたので**偶然の一致とみるべき**。

**追加（`59_`、2026-08-15）**: reference/ノートブック由来の自己学習パース特徴量（SLブロック）を
`54_`ベースラインに追加。検証は-0.011531という、この一連の検証の中でも最大級の改善幅（単一の
事前登録済み修正、探索なし）。**Publicは+0.002273悪化**（0.515030→0.517303）。**通算「改善」
5/10的中**。振れ幅が大きいほど反転しないという期待は成り立たず、むしろ「検証改善幅の大きさは
Publicでの信頼性を保証しない」ことを示す、これまでで最大級の反転例。同日`60_`のKNN/Cox
（検証は悪化方向、+0.0008〜+0.0011）は未提出のまま据え置いた判断は、この結果からも妥当だったと
言える。

**追加（`69_`、2026-08-16）**: reference/0816版（現1位のベース）由来の在籍月数回帰ブレンド+
最終Platt較正を`54_`と完全に同一の444列パイプラインで検証。検証は-0.004677という大幅改善
（単一の事前登録済み4構成要因計画の一部、探索による選別ではない）。**Publicは+0.001831悪化**
（0.515030→0.516861）。**通算「改善」5/11的中**。

このケースが特に重要なのは、「探索なし」「本番と完全に同一の構成」という、検証の信頼性が
最も高いはずの条件を両方満たしていたにもかかわらず反転した点。**「探索していないから」
「本番同一構成だから」という理由だけでは検証の“改善”を信頼する根拠にならない**ことを示す、
これまでで最も厳格な条件下での反転例。振れ幅0.0065は`59_`のSLブロック(0.0138)に次ぐ規模。
なお同時に検証した別構成（text_features、悪化方向）はPublicでも悪化として的中しており、
「悪化」側の信頼性は今回も揺らいでいない。

**自己訂正の教訓もセットで記録**: この検証結果を見た直後、「過去の否定的な簡易テスト結果が
覆った」と早合点して記録を書き換えてしまった（[[reference-notebook-1st-place-base]]参照）。
**検証の「改善」を見て過去の結論を書き換えるのは、Public確認前には絶対にしない。**

関連: [[ablation-cannot-settle-feature-blocks]] [[refit-chaos-noise-floor]]
[[best-submission-status]] [[catboost-beats-other-gbdt]] [[reference-notebook-1st-place-base]]
```

---

## autogluon_stacking_wins.md

```markdown
---
name: autogluon-stacking-wins
description: "AutoGluon's 441-column bagged+stacked CatBoost beat the hand-built single-layer CatBoost by 0.004 — several earlier conclusions are now wrong"
metadata: 
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-12T22:31:05.960Z
---

**`44_`（2026-08-13）で AutoGluon の `CatBoost_r177_BAG_L2`（441列）が Public 0.517685 を出し、
手作りの単層CatBoost 0.521729 を −0.00404 上回った。** 全件学習(−0.0065)に次ぐ2番目の改善。

**効いた構造**: 8-fold bagging + **stack_level 2**
（L1のCatBoost/LightGBM/XGBoost/RF/ExtraTrees/NNのOOF予測を追加特徴量にしたCatBoost）。
`12_`・`13_` で失敗した「OOFから**重み**を学習する」とは別物で、
学習しているのは重みではなく**OOF予測を入力に足した木モデルそのもの**。

## これで覆った過去の結論

| 過去の結論 | 更新 |
|---|---|
| 「AutoGluon/アンサンブルは効かない」(`13_`) | **❌ 撤回** |
| 「441→113の削減が効いた」(`40_`) | **⚠️ 訂正**。差 −0.00093 はシード分散(0.0022)以下で実は「同等」。**強いモデルなら441列が有利**（AutoGluonでは441が113より +0.00484 良い） |
| 「モデル側の自由度は使い切った」(`42_`・`43_`) | **❌ 撤回**。単層CatBoostの範囲で使い切っていただけ |
| 「重みの学習は過学習する」(`12_`・`13_`) | **維持**（今回は重み学習ではない） |

## 注意点

- **検証はまた符号を外した**（[[validation-asymmetry]] 6回目）。
  holdout val は lean113 が +0.00505 良いと言ったが Public は full441 が +0.00484 良かった。
  ただし full441 の holdout は時間切れで63モデルしか学習できず（lean113は149）、同じ土俵ではない。
- **提出ゲート（予測差 > ノイズ床0.02122）が機能した初の成功例**。
  full441 は R6_lean との相関0.9505・平均絶対差0.0631で最も別のモデルだった（[[refit-chaos-noise-floor]]）。
- `44_` は環境不具合込み（FASTAI全滅・XGBoost一部失敗・ray無しで時間切れ、441列は41モデルのみ完走）
  **でも勝った**。修正版 `47_autogluon_fixed.ipynb` に伸びしろがある。
- **`47_` は `RUN_FULL441 = False` が既定。勝った構成を回すには `True` にすること。**
- `AG_full441_weighted` は best_single と相関0.9987なので提出しても情報が少ない。

関連: [[best-submission-status]] [[catboost-beats-other-gbdt]] [[ensemble-oof-overfitting]]
```

---

## memo_parser_coverage_bug.md

```markdown
---
name: memo-parser-coverage-bug
description: "入社時メモの見出し前提パーサーが5.24%の記録でL2ブロックをunknownにしていた欠陥、49_でPublic確認済みの修正"
metadata:
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-13T14:37:33.117Z
---

`extract_workstyle_section()`（`27_`〜`44_`・`48_`で共通利用）は入社時メモの
「勤務地・働き方：」見出しを前提にしていたが、**メモの5.24%（276件）は見出しなしの別書式**
（例: 「・勤務地は東京を希望。転居を伴う異動は希望せず、」）で、この関数は`None`を返していた。
結果、本プロジェクト最大の効果量を持つ[[eda-v5-findings]]のブロックL2が、この276件では
丸ごと`unknown`になっていた（`LOO_drop_L2`の実測: 落とすとval +0.040という唯一の壊滅的ブロック）。

**修正**: 見出しが無い場合、「勤務地・転居・在宅勤務」を含む行を拾って疑似セクションとして返す
フォールバックを追加。カバー率92.9%→98.1%、効果量は弱まらない（39.4pt→40.4pt、p=1.9e-43→1.6e-48）。

**`49_`でPublic確認済み**（単層CatBoost、`40_`と同一プロトコル）:

| 構成 | 修正前Public | 修正後Public | 差 |
|---|---|---|---|
| 441列 | 0.522659 | 0.519149 | **-0.003510** |
| 113列 | 0.521729 | 0.519521 | **-0.002208** |

両構成とも改善、単一の事前登録済みバグ修正がそのまま転移した。
[[validation-asymmetry]]の6件中1件しか一致しない悪い実績に対し、今回は検証と方向が2/2一致
（113列は幅が検証の27%に縮小、441列は幅もほぼ一致）——探索構造が単一修正×2点だったため
勝者の呪いが働きにくかったと解釈。

**副産物**: 同じ「見出し前提」パターンは`39_`のブロックE（`memo_career_cat`、キャリア志向セクション）
にもあった。ただし別途生の効果量を測定したところ career_cat 自体は定着率差 <5pt, p=0.37〜0.64で
**無信号と確定**（パーサー修正の有無に関わらず）。E内の他3列（転居許容/在宅希望/希望勤務地）は
L2と同じ勤務地セクション由来で冗長と`39_`で既に確認済み（[[ablation-cannot-settle-feature-blocks]]）。

**未解決**: `44_`のAutoGluonスタッキング（Public 0.517685、現最良）はこの修正をまだ含んでいない。
`48_`（rayなし版AutoGluon再実行）は旧パーサーのまま実行中だったため停止し、修正を統合した
`50_autogluon_memofix.ipynb`に差し替えて実行予定。両方の改善が乗るかはPublicで確認するまで未知。

関連: [[eda-v5-findings]] [[validation-asymmetry]] [[best-submission-status]]
[[ablation-cannot-settle-feature-blocks]] [[autogluon-stacking-wins]]
```

---

## job_embeddedness_onboarding_literature.md

```markdown
---
name: job-embeddedness-onboarding-literature
description: "Job Embeddedness/オンボーディング曲線/P-O・P-J fit理論から10件超の仮説を検証、全て不採用 — 20pt/p<1e-20の採用基準を境界線ケースで補強"
metadata:
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-13T16:23:13.636Z
---

ユーザーが提示した学術文献9本（Mitchell et al. 2001 Job Embeddedness、Bauer et al. 2025
オンボーディング曲線メタ分析、Erdogan & Bauer 適応指標の時系列、Kristof-Brown et al. 2023
P-O fitレビュー、教員P-O/P-J fit研究、生存分析実務記事、MLベンチマーク2本）から
特徴量仮説を定式化し、全て既存カラムで直接検証した。

## 検証した仮説と結果

| 理論 | 定式化 | 効果量 | 判定 |
|---|---|---|---|
| Links（上司との関係断絶） | 上司ID変更の有無 | 2.8pt, p=0.18 | ❌ 無信号 |
| Fit（専攻職種適合） | 既存ブロックM | 既に不採用（[[gbdt-interaction-type-matters]]） | ❌ |
| Sacrifice（昇給タイミング） | 既存ブロックK | 既に不採用 | ❌ |
| 適応曲線dip-recovery（社会的受容） | 360度評価_親和度等5指標 | 最大4.5pt, p>0.29 | ❌ 全て無信号 |
| 適応曲線dip-recovery（残業時間） | [[memo-parser-coverage-bug]]と同型 | **19.5pt, p=2.5e-18** | **境界未達→52_でPublic不採用確定** |
| P-O/P-J fitの経験年数差 | ブロックMを新卒/中途で層別 | 新卒38.4pt/中途4.6pt | 理論は的中も**既に却下済み信号の内訳**、新規性なし |

## 最重要の副産物: 20pt/p<1e-20の採用基準が境界ケースで検証された

残業時間dip-recovery（19.5pt/p=2.5e-18、基準にわずかに届かず）を`52_`でPublic提出したところ
**両構成とも悪化**（+0.0019 / +0.0005）。既存の確認済みブロック（E/J/L, 25.3〜38.3pt, p<1e-26は
転移、G/H/K/M/N/Oなど基準未達組は不転移）に「基準ギリギリで届かなかったものも不転移」という
点が加わり、**このプロジェクトの20pt/p<1e-20という閾値の妥当性が補強された**。

## Job Embeddednessの3因子（links/fit/sacrifice）は個別に全部決着済み

Mitchell et al.の枠組みは魅力的だが、このデータでの素直な翻訳（上司変更・専攻職種適合・昇給）は
**3つとも既に無信号または却下済み**。P-O fitの理論的主張（認知能力・性格特性を統制しても効く）は
このデータに適性検査・性格データが無いため検証不可能。

関連: [[literature-review-early-turnover]]（1本目の文献レビュー、こちらも5/5棄却）
[[ablation-cannot-settle-feature-blocks]] [[validation-asymmetry]] [[gbdt-interaction-type-matters]]
[[memo-parser-coverage-bug]]
```

---

## l2_m_risk_count_confirmed.md

```markdown
---
name: l2-m-risk-count-confirmed
description: "却下済みブロックMをL2との複合リスク要因数(0/1/2)として再構成、Job Embeddedness理論の予測通りPublicで転移した初めての成功例"
metadata:
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-16T01:20:44.614Z
---

`29_`で単体却下済みのブロックM（専攻職種の分析的ミスマッチ、非分析系専攻→分析系職種）を、
L2（転居×勤務地ミスマッチ、本プロジェクト最強のブロック）との**リスク要因数（0/1/2）**として
組み合わせたところ、`54_`でPublic改善が確認できた。

## ローカルEDA: きれいな用量反応

| リスク要因数 | n | 定着率 |
|---|---|---|
| 0個 | 1885 | 67.0% |
| 1個（L2のみ or Mのみ） | 653 | 28.8% |
| 2個（両方） | 47 | 2.1% |

Cochran-Armitage傾向検定（順序尺度のdose-responseに適した検定）で **z=-18.48, p≈0**。

## Public結果（単層CatBoost、`49_`比）

| 構成 | Public | 差 |
|---|---|---|
| 441列+LM3列 | 0.515030 | **-0.004119** |
| 113列+LM3列 | 0.518492 | **-0.001029** |

両方改善。単層CatBoostの新最良（AutoGluon現最良0.513108まで0.0019に接近）。

**Why転移したか（仮説）**: L2状態・専攻分野・初期職種はいずれも既に生特徴量としてモデルに
入っており、[[gbdt-interaction-type-matters]]の「GBDTは生カテゴリのANDを自力で学習できる」
という懸念（実際`29_`でMを単体で足したときは悪化した）が当てはまりそうに見えた。しかし
**「該当する要因の数」という順序尺度に変換して渡したことが効いた**可能性が高い。木は
個々の条件のANDを再構築できても、複数の弱い信号を単調な「用量」として扱う表現は
明示的な列の方が学習しやすいと考えられる。

## 教訓: 却下済み特徴量は「捨てる」のではなく「原料」として残す

このプロジェクトで却下されたブロック（G/H/I/K/M/N/O）は軒並み単体では転移しなかったが、
**Mだけは、理論的根拠のある形で最強ブロック（L2）と組み合わせることで初めて転移した**。
「単体で効かない=永久に無価値」ではなく、「単体で効かない=単体では試すべきではない」
という程度に理解を修正する。

**ただし総当たりはしない。** 今回L2×Mを選んだのは両方に個別の強い根拠
（L2=生EDA最大効果量、M=29_での明確な機構的却下理由）があったから。
却下済みブロック同士を機械的に全ペア試すのは`39_`と同じ多重比較の罠になる。

## AutoGluonへの統合は確定的に失敗（2026-08-15、`56_`）

`50_`のAutoGluonパイプラインにLMブロックを統合したところ、weighted・best_single**2構成とも
Publicで悪化した**（weighted: 検証改善-0.000937→Public+0.002152悪化 / best_single:
Public+0.003712悪化）。単層CatBoost（`54_`）では明確に転移した同じ特徴量ブロックが、
AutoGluonの中では同じようには効かなかった。

> **【2026-08-16 自己訂正】** 当初ここに「2構成とも悪化方向で一致しており、偶然ではなく
> 確定的な結論」と書いたが、**これは過信だった**。`62_`でAutoGluonのシード間MADを実測すると
> weighted 0.02381 / best_single 0.03719（[[refit-chaos-noise-floor]]）で、
> 上記の差(+0.0022 / +0.0037)は**実行間ノイズと区別できない**。2構成が同じ向きに振れることは
> コイン2枚が同じ面を出すのと同程度（50%）に起こりうる。
> **「LMブロックはAutoGluonでは効かない」は未決と扱うこと。** 判定するには
> 最低3シードずつ回して平均で比較する必要がある。

現時点の総合最良（AutoGluon）は`50_`/`51_` weighted（0.513108）のまま。単層CatBoostの最良は
`54_` R0_memofix_plus_LM（0.515030）で確定。

関連: [[gbdt-interaction-type-matters]] [[memo-parser-coverage-bug]] [[validation-asymmetry]]
[[best-submission-status]] [[job-embeddedness-onboarding-literature]]
```

---

## kitchen_sink_combination_search.md

```markdown
---
name: kitchen-sink-combination-search
description: "1_~56_の全特徴量(546列)を統合し重要度計測→組み合わせ交差検証(低×低6105ペア・高×低1568ペア)を実施、新規の独立した特徴量は0件。既存設計の妥当性を裏付け"
metadata:
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-15T09:33:18.583Z
---

`57_kitchen_sink_all_features`で、これまで却下・未採用だった全ブロック（E/G/H/I/J/K/N/O/L1/
文埋め込み）を`54_`（L2×M込み）に統合し、546列で重要度を計測。ゼロ近傍111特徴量を対象に
低×低（C(111,2)=6,105ペア）・高×低（重要度上位30×近傍ゼロ111=1,568ペア）の交差検定を実施した。

## 重要度計測の結果

**`L2xM_リスク要因数`が546列中断トツの1位**（6.21、2位の専攻分野2.97の倍以上）。
`L2xM_ダブル不適合`（二値AND版）はゼロ近傍に転落——**順序尺度の方が二値ANDより効率的**という
[[l2-m-risk-count-confirmed]]の設計判断が定量的に裏付けられた。

## 組み合わせ探索: 新規の独立した特徴量は0件

低×低22ペア・高×低134ペアが統計的基準（gap≥20pt, p<1e-20）を満たしたが、**全て精査の結果、
以下2種類の既知の交絡の言い換えだった**:

1. **評価スケジュールの暦アーティファクト**: `初回評価月`は入社月の完全な決定論的関数
   （std=0。評価が固定カレンダーで実施されるための機械的な待ち時間の計算結果で、
   employee本人の情報を一切含まない）。`入社四半期`・`is_早期評価`・`is_遅延評価`も同系統
2. **職種依存の構造的欠損**: `担当プロジェクト数`等の欠損率は`初期職種`の完全な決定関数
   （プロジェクト管理系職種=欠損率0.3%未満／非該当職種=欠損率99.7-100%）
3. **`入社区分`(新卒/中途)の再エンコード**: `is_Q2_新卒`は`入社区分`=新卒と完全一致（ズレ0件）。
   L2×`is_Q2_新卒`(42.7pt)も、既に確認済みのL2×入社区分（新卒サブグループ39.5pt）と同一の発見が、
   「残り」母集団に高定着率の中途が混ざることで誇張されて見えているだけ

いずれも既存の生特徴量（`入社日`・`初期職種`・`入社区分`）で既にモデルに入っている情報であり、
明示的な特徴量として追加する価値は無いと判断した。

## 方法論の教訓（自戒）

高×低探索の初回実行で、`勤務地_unique_count`（97.8%が同一値の準定数列）を「悪い側」に誤って
選んでしまうバグがあった（value=1の定着率56.46% vs value=2の56.67%、実質誤差レベルの差なのに
最小値というだけで選ばれた）。**二値化の「悪い側」を選ぶ際は、該当率が10-90%に収まることを
必須条件にする**（絶対数の下限だけでは、大規模データセットで準定数列を見逃す）。

## 3因子リスクスコア候補も`58_`でPublic不採用確定（2026-08-15）

低×低・高×低の探索とは別に、**重要度上位同士の組み合わせ**からL2×Mの2因子リスク要因数
（0/1/2）に**残業時間（後期12-23月・上位25%）**を第3因子として加える案が出た。既存リスク
要因数との相関はわずか0.058で交絡クラスターとは無関係、傾向検定z=-20.24・p≈0という
強いローカル用量反応だったが、**`58_`のPublic検証で不採用が確定**（val +0.000575→
Public +0.000366、どちらも悪化方向で一致）。事前の懸念通り、残業時間は既に生特徴量として
モデルに入っているため、明示的な特徴量として追加しても価値を生まなかった。

## 総括

探索的な組み合わせ探索（低×低・高×低の合計7,673ペア + 高×高の3因子候補）は、
**新しい独立した特徴量を1つも生まなかった。** これは「探索が足りなかった」のではなく、
**現行の441+αの特徴量設計が主要な情報を既に捉え尽くしていることの裏付け**と解釈するのが
妥当。今後、同種の総当たり探索を繰り返す優先度は低い。単層CatBoostの最良は`54_`
R0_memofix_plus_LM（Public 0.515030）で確定とする。

関連: [[l2-m-risk-count-confirmed]] [[gbdt-interaction-type-matters]]
[[ablation-cannot-settle-feature-blocks]] [[hire-year-extrapolation]] [[best-submission-status]]
```

---

## monthly_data_information_ceiling.md

```markdown
---
name: monthly-data-information-ceiling
description: 月次データは10年定着をほとんど予測できない（窓を3年ずらしても差-0.001）。評価列は持続性ゼロ。特徴量探索が10連続で失敗した機構的な理由
metadata: 
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-15T14:58:51.240Z
---

2026-08-15のEDAで、**なぜ特徴量探索が繰り返し失敗するのか**が機構的に解明できた。
結論は「月次データにそもそも10年定着の情報がほとんど無い」という、データ生成過程側の性質。

## 決定的な証拠: 情報の天井（窓をずらしても改善しない）

月60まで生存した2,072名に限れば、月0-23と月36-59の**両方**が観測できる（打ち切りリークなし）。
アウトカムに3年近い窓を使っても予測力はほぼ同じだった。

| 特徴量の窓 | logloss | AUC |
|---|---|---|
| 定数（基準率）| 0.552373 | — |
| 月0-23（Testで使える窓）| 0.544048 | 0.609 |
| 月36-59（outcomeに3年近い）| 0.545184 | 0.627 |
| 両方 | 0.543074 | 0.629 |

**アウトカム直前の窓を使ってすら定数予測から0.009しか改善しない。** 月次の行動・評価データは
10年定着をほとんど説明していない。確認済みの勝ちブロックが全て入社時メモ・persona由来
（E/J/L2/LM）だったことと完全に整合する。[[horizon-dependent-feature-effects]]の正体。

## 持続性の測定（月0-23平均 → 月96-119平均の相関、120ヶ月生存者）

| 変数 | 持続性 | 月次ハザードへの寄与 |
|---|---|---|
| 在宅勤務日数 | 0.977 | +0.014 |
| 月例給与_円 | 0.958 | -0.019 |
| 残業時間 | 0.799 | **+0.038（最強）** |
| 担当プロジェクト数 | 0.757 | -0.013 |
| 顧客満足度評価 | 0.613 | -0.019 |
| **360度評価 全5次元** | **0.0004〜0.138** | -0.005〜-0.024 |
| 欠勤日数 / 研修時間 | 0.027 / 0.076 | +0.006 / +0.004 |

**360度評価は毎期ほぼ独立に引き直される。** 瞬間ハザードには効く（信頼度は2番目に強い）のに、
月0-23の値は10年後の評価を全く予測しない。持続する変数（残業・給与・在宅）は既に大量に
特徴量化済みで、持続しない変数は原理的に長期予測に使えない——探索の余地が無い理由。

## 発見した生成規則（データはやはり合成）

- **月11未満の離職はゼロ**（最小離職月=11）
- **360度評価は6ヶ月周期**で更新（月5・11・17・23で81%）＋約3.5%/月の予定外更新。月0-1は100%欠損
- **昇給は年1回**（月12・24で77%）＋月6/18に4%、その他2%の予定外改定

## 今回すべて棄却した候補

| 候補 | 結果 |
|---|---|
| 離職伝染（部署・上司の同僚離職、full履歴でカレンダー時間ベース）| r=0.002〜-0.024、**完全null**。0-23窓内の129件でなく1,202件を使っても再現 |
| 離散時間ハザード CatBoost →生存積 | +0.0394 悪化 |
| 条件付き生存積（区間2/3/4/5個）| +0.008〜+0.034、**区間が増えるほど単調に悪化**。ブレンドも≤0.0002 |
| 持続性<0.15 の列を削除 | +0.0066 悪化（評価列は初期ハザード経由で効いておりノイズではない）|
| 在宅勤務日数のカテゴリ化 | -0.0003（ノイズ床以下）|
| 生成規則由来のEVブロック（定期評価4点・昇給イベント）| 単独では-0.0041に見えたが、**D_expanded四半期ブロックを入れると-0.0005に崩壊＝重複**。イベント回数は+0.00004 |
| 較正シフト（Testは早期離職者ゼロで基準率0.5923）| 予測平均は既に0.586-0.590で整合。シフトは悪化 |
| メモ4セクションの棚卸し | 全て既存ブロックに包含。「記録なし」は0.04%、キャリア志向の最大差≤10pt |

## How to apply

**月次データからの新規特徴量探索はもう行わない。** [[kitchen-sink-combination-search]]・
[[ablation-cannot-settle-feature-blocks]]が経験的に示していたことを、今回まったく別方向
（生成規則の逆解析＋情報の減衰測定）から機構的に確認した。残る伸びしろは特徴量ではなく
モデリング・アンサンブル側にあると考えるのが妥当。

**生存時間モデリング一族は完全に閉じた**（Coxは`60_`、CatBoostハザードと条件付き積は今回）。
今後この方向の提案が出たら、この記録を根拠に却下してよい。

関連: [[horizon-dependent-feature-effects]] [[kitchen-sink-combination-search]]
[[ablation-cannot-settle-feature-blocks]] [[test-set-is-survivor-filtered]]
[[best-submission-status]] [[gbdt-interaction-type-matters]]
```

---

## private_lb_variance_strategy.md

```markdown
---
name: private-lb-variance-strategy
description: Private評価があり最終提出は自分で選べる。Public最良は運の良い引きなので選ばず、分散最小のプール平均を選ぶ
metadata: 
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-16T01:35:44.905Z
---

**本コンペにはPrivate評価があり、最終提出は自分でN件選ぶ形式**（2026-08-16にユーザー確認）。
この事実と[[refit-chaos-noise-floor]]のAutoGluonノイズ実測を合わせると、
**Publicスコアを追う戦略は誤り**だと確定した。

## 根拠: Jensenの不等式（推測ではなく数学）

loglossは予測確率について凸。したがって
`logloss(mean(p_i)) ≤ mean(logloss(p_i))`。

`62_`のseedavg3（3実行の平均）がPublic 0.515387だったので、
**個々の実行のPublic平均 ≥ 0.515387** が確定する。一方`51_`（同一構成の単発）は
**0.514263**でこの下限より良い。すなわち

> **`51_`の0.514263は、その構成の期待性能より良い「運の良い引き」だった。**

`50_`の0.513108（Public最良）も同様に期待値より良い引きと見るべき。

## なぜPublic最良を最終提出に選んではいけないか

**PublicとPrivateは独立した別の引き。** `50_`がPublicで幸運だったことは、
Privateでも幸運であることを一切保証しない。`50_`の期待Private = その構成の実力 ≈ 0.515。
一方プール平均の期待Privateは、Jensenにより**どの単発実行よりも良い**ことが保証される。

**Public最良を選ぶ = 「もう一度同じ幸運が起きる」に賭けること。定義上それは期待できない。**

## 具体的な運用

- **最終提出はプール平均を選ぶ。** `data/output/20260816/20260816_pool_poolC_weighted.csv`
  （パーサー修正後のfull441 weighted 8本平均、`56_`のLM込み444列だけ特徴量が異質なので除外）
- プールCの8本は完全独立ではない。`50_`/`51_`/`53_`/`61_`は行シャッフルをしていないため
  **fold割り当てを共有**しており、実質的な独立ドローは約5本。
  独立ドローを増やす（`62_`をシードを変えて追加実行する）のが最も確実な期待値改善。
- **Publicスコアの0.002以下の差で判断しない。** シード間MADは weighted 0.02381 /
  best_single 0.03719（[[refit-chaos-noise-floor]]）。

## Publicでの実証（2026-08-16、第94節）

プールCを提出した結果 **0.514050** で、**`50_`(0.513108)以外のすべての提出に勝った**。
「平均する本数を増やすほど良くなる」がPublicで確認できた:

| 平均本数 | Public |
|---|---|
| 1本（`51_`単発）| 0.514263 |
| 3本（`62_` seedavg3）| 0.515387 |
| **8本（プールC）** | **0.514050** |

3本→8本で **-0.001337 改善**。追加学習コストはゼロ（保存済みの予測を平均しただけ）。
`50_`との差は0.000942で、シード間MAD 0.02381 からすれば誤差。

**独立ドローを増やせば`50_`のPublicを上回る可能性も十分ある**
（プールCの8本のうち`50_`/`51_`/`53_`/`61_`はfold割り当てを共有しており、
実質的な独立ドローは約5本しかないため、まだ伸びしろがある）。

## 副次的な確定事項（`93_`）

fold割り当ての乱数 > あらゆる構成変更:

| 比較 | 予測MAD |
|---|---|
| 構成違い（除外モデル・DyStack・時間予算2h→6h）※fold割当は同一 | 0.0118〜0.0145 |
| **fold割り当てのみ違う** | **0.0238** |
| 特徴量が違う（`56_`のLMブロック）| 0.0337〜0.0380 |

`61_`（時間予算を7200→21600秒に拡大）は`51_`とのMADが0.01277しかなく、
シード間変動より小さい＝**null result**。時間予算のレバーは打ち止め。

関連: [[refit-chaos-noise-floor]] [[best-submission-status]] [[validation-asymmetry]]
[[ensemble-oof-overfitting]] [[monthly-data-information-ceiling]]
```

---

## tabpfn_ensemble_partner.md

```markdown
---
name: tabpfn-ensemble-partner
description: TabPFN v2(top150列)は単体でCatBoostと互角かつ相関0.954。事前登録した足切りを通った初めてのアンサンブル相手で検証-0.004
metadata: 
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-16T01:51:52.238Z
---

HuggingFaceの表形式基盤モデル **TabPFN v2** が、このプロジェクトで初めて
**事前登録した足切り基準を通過したアンサンブル相手**になった（`63_`、2026-08-16）。

## 結果（同一split・同一特徴量のCatBoostと比較）

| 構成 | TabPFN単体val | CatBoost比 | 相関 | 最良ブレンド | 改善 |
|---|---|---|---|---|---|
| full441（441列）| 0.522101 | +0.0158 | 0.944 | 0.506045 | -0.0002 |
| **top150（150列）** | **0.507569** | **+0.0013** | 0.954 | **0.502252**(w=0.5) | **-0.004018** |

CatBoostベースライン val = 0.506270。**TabPFNは特徴量が多いと明確に劣化する**
（441列→150列で単体性能が0.0145改善）。150列前後が適正。

## `55_`のGRUとの違い（なぜ今回は効いたか）

| | `55_` GRU | `63_` TabPFN(top150) |
|---|---|---|
| 相関 | 0.31〜0.35（多様性大）| 0.954 |
| 絶対性能差 | **+0.136（致命的）** | **+0.0013（互角）** |
| ブレンド | どの重みでも改善せず | **w=0.1〜0.9の全域で改善** |

[[ensemble-oof-overfitting]] addendum 6 の教訓「多様性だけでは足りない、絶対性能の差が
大きすぎると希釈する」が正しく、**逆に絶対性能が互角なら相関0.95でも十分効く**と分かった。
改善域が w=0.1〜0.9 と極めて広く平坦で、最適点が w=0.5 という非チューニング値だった点も、
[[validation-asymmetry]]の「探索で見つけた改善は信用できない」が当てはまりにくい根拠になる。

## 実装上の要点

- **バージョン固定必須**: `pip install tabpfn` は最新(8.x)を入れ、Prior Labsの
  ライセンス同意＋APIトークンを要求して`TabPFNLicenseError`で落ちる。
  **`tabpfn==2.2.1`（2.x系=v2世代, Apache 2.0ベース）を指定する。**
- pipで入れ替えても**カーネル再起動しないと切り替わらない**（sys.modulesのキャッシュ）。
  インストールはノートブック冒頭のセルに置き、再起動しても特徴量生成をやり直さずに済むようにする。
- **`fillna(-999)`をしない**。NaNをネイティブに扱えるので-999を渡すと外れ値扱いになる。
- カテゴリは序数コード化して`categorical_features_indices`で位置を渡す（one-hotは公式に非推奨）。
- GPU必須（CPUだとサンプル数制限）。学習は1シード40秒〜2分程度と速い。

## 次の一手: AutoGluonプールとのブレンド

`63_`のブレンドは単層CatBoost（Public約0.519相当）が土台なので、-0.004改善しても約0.515。
**本命は AutoGluonプール（Public 0.514050、[[private-lb-variance-strategy]]）とのブレンド。**

- 相関(TabPFN top150, プールC) = **0.9657**、MAD **0.0575**（ノイズ床の2.7倍）＝本物に別物
- AutoGluonの100+モデルにTransformerのin-context learningは含まれていない
- 推奨重み **w_AG=0.70**: ブレンド曲線が平坦（w=0.5とw=0.7で差+0.0006のみ）な一方、
  [[l2-m-risk-count-confirmed]]の前例からAutoGluonへの転移は保証されないため、リスク調整
- 出力済み: `data/output/20260816/20260816_pool_tabpfn_blend_w70.csv`

関連: [[ensemble-oof-overfitting]] [[private-lb-variance-strategy]] [[validation-asymmetry]]
[[l2-m-risk-count-confirmed]] [[catboost-beats-other-gbdt]] [[monthly-data-information-ceiling]]
```

---

## reference_notebook_1st_place_base.md

```markdown
---
name: reference-notebook-1st-place-base
description: reference/長期定着予測_crmaine_0816.ipynbは現1位が過去に使用していたベース。text_features・在籍月数回帰ブレンド+Platt較正とも4構成ともPublicで不採用確定
metadata: 
  node_type: memory
  type: project
  originSessionId: 82f12fda-ca43-4e97-868a-4f341598fbac
  modified: 2026-08-16T08:19:19.359Z
---

`reference/長期定着予測_crmaine_0816.ipynb` は**現1位が過去に使用していたと見られるノートブック**
（2026-08-16、ユーザー確認）。単なる別参加者の記録ではなく参考価値が高いと再定義されたが、
実際に移植した2技法はいずれもPublicで不採用が確定した。

## 0816版に含まれていた自分たちの未検証だった2つの技法（`69_`で検証）

1. **CatBoostネイティブtext_features**: メモをTF-IDF+SVD圧縮せず、janomeで
   名詞・動詞・形容詞だけ残した生テキストをCatBoostのtext_features機構に直接渡す
2. **在籍月数の回帰ブレンド + 最終Platt較正**: `employee_monthly_train_full.csv`から
   最後に観測された経過月数を回帰の目的変数にし、Plattで確率化して分類器と85:15でブレンド、
   さらに最終的にネストKFoldでPlatt較正する

## 結果（`54_`と完全に同一の444列パイプラインで検証・Public確認済み）

| 構成 | val（対baseline 0.505477）| Public（対`54_`champion 0.515030）| 判定 |
|---|---|---|---|
| +text_features (446列) | +0.005705（悪化）| +0.004615（悪化）| 一致（的中）|
| +在籍月数回帰ブレンド+Platt較正 | **-0.004677（大幅改善に見えた）** | **+0.001831（実際は悪化）** | **反転** |
| 両方同時 | +0.003349（悪化）| +0.004615（TXT_BLENDと同一ファイル比較、悪化）| 一致（的中）|

**4構成ともPublicで不採用が確定。** 単層CatBoostの最良は引き続き`54_`（Public 0.515030）のまま。

## 重要な教訓（自己訂正込み）: 検証の“改善”は、本番同一構成・探索なしでも信頼できない

当初「在籍月数回帰ブレンド+Platt較正は`54_`と完全に同一の444列パイプラインで検証したところ
-0.0047の大幅改善を示した。第90節の簡易ローカル再現による否定的結果（+0.0014悪化見積もり）が
覆った」と記録したが、**これは検証の「改善」を根拠に結論を書き換えてしまった誤りだった**。

Public結果は+0.001831悪化——**第90節の粗い見積もり（+0.0014悪化）とほぼ同じ方向・同じ桁**。
第90節の大まかな方向性はむしろ正しく、「本番パイプラインで検証したら覆った」という判断自体が
[[validation-asymmetry]]の「検証の“改善”は信頼できない」に引っかかった一例だった。

このケースが特に重要なのは、**「単一の事前登録済み4構成の要因計画（探索による選別ではない）」
「`54_`と完全に同一の本番パイプライン」という、これまでで最も検証の信頼性が高いはずの条件**
だったにもかかわらず、-0.0047の改善が+0.0018の悪化に反転した（振れ幅0.0065）ことである。
`59_`のSLブロック反転（振れ幅0.0138）に次ぐ規模。

原因の推定: 回帰→確率のPlatt較正器を検証セット自身でin-sample fit&applyする設計
（ノートブックに事前に注意点として明記していた箇所）が、この楽観バイアスの発生源だった
可能性が高い。**わずか1特徴量・2パラメータの単純な変換でも、検証セット自身でfit&applyする
限り、検証スコアを額面通り信じてはいけない。**

**How to apply**: 検証の「改善」を見た直後に「過去の否定的結果が覆った」と結論を書き換えない。
必ずPublicで確認してから記録を更新する。この教訓は[[validation-asymmetry]]・
[[local-test-not-decisive-for-ablation]]の運用ルールを再確認するものであり、新しい例外を
作るものではない。

関連: [[local-test-not-decisive-for-ablation]] [[validation-asymmetry]]
[[best-submission-status]] [[private-lb-variance-strategy]]
```
