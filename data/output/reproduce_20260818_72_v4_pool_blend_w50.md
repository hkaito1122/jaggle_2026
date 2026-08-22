# 手順書: `20260818_72_v4_pool_blend_w50.csv` の再現

対象ファイル: `data/output/20260818/20260818_72_v4_pool_blend_w50.csv`
Public Score: **0.497608**（2026-08-18 11:46提出、現最良）
生成コード: `src/72_reference_pipeline_standalone.ipynb` 第8節（2026-08-18追記）

## 0. まず確認: 本当に全部再現する必要があるか

このファイルは2つの材料を w=0.5 で単純平均しただけ。

```
最終予測 = 0.5 × xxxx_v4 + 0.5 × 20260816_pool_top150_hire_fixed_avg
```

**両方の材料、および材料の材料（下記フェーズ0の全出力）が現時点で `data/output/` にすでに存在する。**
ゼロから再現する必要が無ければ、**フェーズ2（`72_`）だけ**を実行すれば良い
（フェーズ2はDrive上の既存ファイルを読むだけで、フェーズ0・フェーズ1は再実行しない）。

「本当にゼロから全部作り直したい」場合だけフェーズ0・1も実行する。以下は全体を分解した手順。

---

## 全体の依存関係図

```
フェーズ0: 20260816_pool_top150_hire_fixed_avg.csv（現最良プール、Public 0.508699）
  ├─ Branch A: pool8_tabpfn_w70
  │    ├─ 50_autogluon_memofix
  │    ├─ 51_autogluon_catboost_bias
  │    ├─ 53_autogluon_dystack
  │    ├─ 61_autogluon_extended_time
  │    ├─ 62_autogluon_seed_averaging
  │    ├─ 63_tabpfn_ensemble
  │    └─ 66_final_blend（第1〜2節）
  └─ Branch B: blend_fm2
       ├─ (68_foundation_models_3way …任意、歴史的根拠のみ)
       ├─ 70_hire_fixed_fm_submission
       └─ 66_final_blend（第5節、Branch Aの結果も使う）

フェーズ1: xxxx_v4_submission.csv（Public 0.508039）
  └─ xxxx_v4.ipynb（自己完結、他ノートブック依存なし）

フェーズ2: 20260818_72_v4_pool_blend_w50.csv（本手順書のゴール）
  └─ 72_reference_pipeline_standalone.ipynb 第8節
       ├─ 入力: フェーズ1の出力（xxxx_v4_submission.csv）
       ├─ 入力: フェーズ0の出力（pool_top150_hire_fixed_avg.csv）
       └─ 前提: 同ノートブックの第1〜7節を先に実行しておくこと（後述）
```

---

## フェーズ0（任意・通常は不要）: `20260816_pool_top150_hire_fixed_avg.csv` の再現

**現状: 下記すべての中間ファイルがDriveに現存することを確認済み（2026-08-18時点）。
このフェーズは丸ごとスキップしてよい。**

再現が必要になった場合（ファイル消失時など）の実行順:

| 順序 | ノートブック | 役割 | `66_`が読む出力 |
|---|---|---|---|
| 0-1 | `src/50_autogluon_memofix.ipynb` | AutoGluon 441列（`49_`パーサー修正込み） | `AG50_full441_weighted.csv` |
| 0-2 | `src/51_autogluon_catboost_bias.ipynb` | 同上 + NN_TORCH/KNN除外 | `AG51_full441_weighted.csv` |
| 0-3 | `src/53_autogluon_dystack.ipynb` | 同上 + DyStack有効 | `AG53_full441_weighted.csv` |
| 0-4 | `src/61_autogluon_extended_time.ipynb` | 同上 + time_limit 6時間 | `AG61_full441_weighted.csv` |
| 0-5 | `src/62_autogluon_seed_averaging.ipynb` | 同上 + bag16 + 3シード | `AG62_bag16_weighted.csv` + seed42/2024/7の3npy |
| 0-6 | `src/63_tabpfn_ensemble.ipynb` | TabPFN v2、CatBoost重要度上位150列、3シード | `tabpfn_top150_testpreds.npy` |
| 0-7 | `src/70_hire_fixed_fm_submission.ipynb` | TabPFN・TabICLを`hire_fixed`(78列)でTest予測 | `..._tabpfn_hire_fixed_testpreds.npy`, `..._tabicl_hire_fixed_testpreds.npy` |
| 0-8 | `src/66_final_blend.ipynb`（第1〜2節→第5節） | 上記材料を0.70:0.30、さらに50:50で合成 | → `pool_top150_hire_fixed_avg.csv` を出力 |

- 0-1〜0-7は互いに独立（並列実行可）。0-8は0-1〜0-7すべての完了後に実行。
- `(68_foundation_models_3way)` は「`hire_fixed`という特徴量セットを選んだ理由」の検証ノートブックで、
  `66_`は直接読まないため厳密には不要（歴史的根拠のみ）。
- **重要**: `66_`の第1節は `PINNED` という辞書で入力ファイル名を日付込みで固定している
  （`20260813/...`, `20260814/...` 等）。0-1〜0-7を**再実行**すると出力は今日の日付フォルダに
  新しく作られるため、`66_`の`PINNED`/`PINNED_HF`辞書のパスを新しい日付に書き換えないと
  古いファイルを読みにいって失敗する（またはglobフォールバックで「一致しない可能性がある」警告が出る）。
- AutoGluon/TabPFNは実行のたびに結果が微妙に変わりうる（完全な数値再現は保証されない）。

---

## フェーズ1: `xxxx_v4_submission.csv` の再現（Public 0.508039）

| 順序 | ノートブック | 役割 |
|---|---|---|
| 1-1 | `src/xxxx_v4.ipynb` | reference（1st place著者）パイプラインを7シード×6-foldで実行 |

- Colab上で実行。`google.colab.drive.mount` でDrive接続。
- 入力: `data/input/employee_persona_{train,test}.csv`,
  `employee_monthly_{train,test}.csv`, `employee_monthly_train_full.csv`
  （他ノートブックの出力への依存なし、完全に自己完結）
- 出力: `data/output/<実行日>/<実行日>_xxxx_v4_submission.csv`
- 所要時間の目安: 前回実行で約58分（分類器7シード × 回帰7シード、各6-fold）
- カテゴリ列: `入社区分/最終学歴/専攻分野/前職職種/採用経路/性別/初期職種/初期勤務地/初期等級/初期役割/
  最終月の勤務地/最終月の職種/最終月の部署ID/希望勤務地/キャリア志向`、
  テキスト列: `メモtxt/学習txt`（CatBoostネイティブ`text_features`に渡す）

---

## フェーズ2: `72_reference_pipeline_standalone.ipynb` を実行し、ブレンドを作る

**前提**: フェーズ0・フェーズ1の出力（`pool_top150_hire_fixed_avg.csv` と `xxxx_v4_submission.csv`）が
`data/output/`配下のどこかの日付フォルダに存在すること（すでに存在するので通常は満たされている）。

`72_`は単一のColabセッションでノートブックの**先頭から順に**セルを実行する設計。
第8節（ブレンド出力セル）は、第1〜7節で定義される変数（`PROJECT_ROOT`, `OUTPUT_DIR`, `logger`,
`ID_COL`, `_find`, `_csv`, `te_ids` など）に依存するため、以下のいずれかの方法で実行する。

### 方法A（推奨・確実）: ノートブック全体を先頭から実行

| 順序 | セル/節 | 内容 | 目安時間 |
|---|---|---|---|
| 2-1 | セル1 | `pip install catboost janome` | 数十秒 |
| 2-2 | セル2 | Google Drive マウント | 数秒 |
| 2-3 | セル3 | `PROJECT_ROOT`/`SCRIPT_NAME`/`TODAY`/`logger`/`OUTPUT_DIR`/`ID_COL` 設定、入力CSV読み込み | 数秒 |
| 2-4 | 第1節 | 月次集約（生存指示子3列を含む） | 約1分 |
| 2-5 | 第2節 | 入社時特徴量・janome形態素解析・メモパース | 約1分 |
| 2-6 | 第3節 | 特徴量結合（104列） | 数秒 |
| 2-7 | 第4節 | 生存指示子3列の分布確認 | 数秒 |
| 2-8 | 第5節 | `R_full`・`R_nosurv` の学習（reference式OOF + 我々式val + Test予測、3シード×5fold ×2構成） | **約40〜45分**（R_full約22分、R_nosurv約21分） |
| 2-9 | 第6節 | ブレンド曲線判定（54_との比較、argmin_w） | 数秒 |
| 2-10 | 第7節 | プールC・現最良とのcorr/MAD比較、まとめ保存 | 数秒 |
| 2-11 | **第8節**（本手順書のゴール） | `xxxx_v4` と `pool_top150_hire_fixed_avg` を w=0.5 でブレンドし保存 | 数秒 |

- 第8節のコードは内部で `_find("*/*xxxx_v4_submission.csv")` と
  `_find("*/*_pool_top150_hire_fixed_avg.csv")` を使い、`data/output/`配下を日付降順で検索して
  **最新のファイルを自動選択する**（`66_`のようなPINNED方式ではない）。複数の`xxxx_v4`実行結果や
  複数の`pool_top150_hire_fixed_avg`があると、最新のものが使われる点に注意。
- 出力: `data/output/<実行日>/<実行日>_72_reference_pipeline_standalone_v4_pool_blend_w50.csv`
- 第5節（2-8）はブレンド自体には不要な計算（`R_full`/`R_nosurv`の学習）だが、
  ノートブックの変数定義チェーン上、省略すると第8節がエラーになる。

### 方法B（時間短縮・上級者向け）: 第5節をスキップするショートカット

第8節が実際に必要とするのは `PROJECT_ROOT`/`OUTPUT_DIR`/`logger`/`ID_COL`/`_find`/`_csv`/`te_ids`
の7つだけ。以下を新しいセルで独自に定義すれば、約40分かかる第5節（2-8）の学習を丸ごと省略できる。

```python
import sys, datetime
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT = Path("/content/drive/MyDrive/jaggle_2026")
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger
import numpy as np, pandas as pd

ID_COL = "社員ID"
SCRIPT_NAME = "72_reference_pipeline_standalone"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
logger = get_logger(SCRIPT_NAME, log_dir=str(PROJECT_ROOT/"logs"))
OUTPUT_DIR = PROJECT_ROOT/"data"/"output"/TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

入社時データ_予測 = pd.read_csv(PROJECT_ROOT/"data"/"input"/"employee_persona_test.csv")
te_ids = 入社時データ_予測[ID_COL]

OUT_ROOT = PROJECT_ROOT/"data"/"output"
def _find(pat):
    h = sorted(OUT_ROOT.glob(pat)); return h[-1] if h else None
def _csv(p):
    return pd.read_csv(p, header=None, names=[ID_COL, "p"]).set_index(ID_COL).loc[te_ids, "p"].to_numpy()
```

その後、`72_`第8節のコードセルをそのまま実行すれば同じ出力になる。
（ただし`72_`本来の検証内容——R_full/R_nosurv・ブレンド曲線・corr比較——はスキップされるので、
監査目的で完全な実行履歴を残したい場合は方法Aを推奨）

---

## 検証: 生成物が既存ファイルと一致するか確認する

再実行後、以下で既存の提出済みファイルと数値が一致するか確認できる（Colab/ローカルいずれでも可）。

```python
import numpy as np, pandas as pd

new = pd.read_csv("<新しく生成したファイルへのパス>", header=None, names=["社員ID","p"])
ref = pd.read_csv(
    "data/output/20260818/20260818_72_v4_pool_blend_w50.csv",
    header=None, names=["社員ID","p"])

m = new.merge(ref, on="社員ID", suffixes=("_new", "_ref"))
d = (m["p_new"] - m["p_ref"]).abs()
print("最大絶対差:", d.max(), " 平均絶対差:", d.mean())
```

- フェーズ0・1を**再実行せず**方法A/Bで72_のみ実行した場合: 材料が完全に同じファイルなので
  **理論上ビット単位で一致**するはず（差が出るとしたら`_find`が意図と違うファイルを拾った場合のみ）。
- フェーズ1（`xxxx_v4`）を**再実行**した場合: CatBoostのセッション間非決定性により
  差が生じる（`catboost_gpu_nondeterminism`メモリ参照、val loglossで0.003〜0.006程度の実績）。
  ブレンド後の最大絶対差は数%オーダーになりうる。
- フェーズ0を**再実行**した場合: AutoGluon/TabPFNの非決定性がさらに加わり、差は拡大する。

---

## まとめ: 最短経路

今すぐ再現したいだけなら:

1. `xxxx_v4.ipynb` を実行（約58分）→ `xxxx_v4_submission.csv`
2. `72_reference_pipeline_standalone.ipynb` を**方法B（ショートカット）**で第8節まで実行（数分）
   → `..._v4_pool_blend_w50.csv`

フェーズ0（プール側）は再実行不要（既存ファイルをそのまま使う）。
