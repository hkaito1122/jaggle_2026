"""119_stacking_nonlinear_meta

`115_`の8モデルスタッキングは最終層にL2ロジスティック回帰（線形結合）を使っている。
`113_`でPlatt scaling（ロジスティック回帰による事後較正）が無風だったことから、
「単純な線形結合の後にもう一段線形変換を足す」ことには価値が無いと分かったが、
これは「最終層自体を非線形にする」こととは別の問い。8つのベースモデルのlogit予測の間に
線形結合では拾えない交互作用（例: 特定のモデルの組み合わせでのみ強い/弱い領域）がある
可能性を、ごく浅い（過学習を避けるため）勾配ブースティングのメタモデルで検証する。

設計:
  - 115_と同一の8モデルOOF/Test予測・同一のlogit変換・同一の外側5-fold(seed=12345)を使う
  - 最終層をLogisticRegressionCVからHistGradientBoostingClassifier(浅い木・強い正則化・
    内部early stopping)に置き換える。8特徴量・約2100件(外側各foldの学習側)という
    小規模データに対する過学習を避けるため、max_depth=2・max_iter少なめ・
    validation_fraction=0.15の内部early stoppingを併用する
  - 線形スタッキング(115_、val=0.482271)と同一の入れ子CV評価枠組みで比較する

判定基準: 分解能±0.0043を超えて115_を上回った場合のみ採用する。

出力: 分解能を超えて改善した場合のみ stacker8_nonlinear_full_survivor_submission.csv
"""
import datetime
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger
from common.utils.seed import seed_everything

SEED = 42
seed_everything(seed=SEED)

TARGET_COL = "10年定着ラベル"
ID_COL = "社員ID"

SCRIPT_NAME = "119_stacking_nonlinear_meta"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 実験開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIR = PROJECT_ROOT / "data" / "input"

train_persona = pd.read_csv(INPUT_DIR / "employee_persona_train.csv")
test_persona = pd.read_csv(INPUT_DIR / "employee_persona_test.csv")
train_monthly = pd.read_csv(INPUT_DIR / "employee_monthly_train.csv")

y_train = train_persona[TARGET_COL].values
train_ids = train_persona[ID_COL].values
test_ids = test_persona[ID_COL].values

EARLY_LEAVER_IDS = set(train_monthly.loc[train_monthly["月末在籍状態"] == "退職", ID_COL].unique())
surv_mask = np.array([tid not in EARLY_LEAVER_IDS for tid in train_ids])
logger.info(f"生存者(24か月在籍): {surv_mask.sum()} / {len(surv_mask)}")

EPS = 1e-9


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ============================================================
# 115_と同一の8モデルOOF/Test予測を読み込み
# ============================================================
OOF_DIR = PROJECT_ROOT / "data" / "output"
BASE_MODELS = {
    "cb134": ("20260822/20260822_86_tabpfn_on_84_top30pct_cb134_oofpreds.npy",
              "20260822/20260822_86_tabpfn_on_84_top30pct_cb134_testpreds.npy"),
    "tabpfn134": ("20260822/20260822_86_tabpfn_on_84_top30pct_tabpfn134_oofpreds.npy",
                  "20260822/20260822_86_tabpfn_on_84_top30pct_tabpfn134_testpreds.npy"),
    "cb89": ("20260822/20260822_93_tabpfn_top20pct_on_77_cb89_oofpreds.npy",
             "20260822/20260822_93_tabpfn_top20pct_on_77_cb89_testpreds.npy"),
    "tabpfn89": ("20260822/20260822_93_tabpfn_top20pct_on_77_tabpfn89_oofpreds.npy",
                 "20260822/20260822_93_tabpfn_top20pct_on_77_tabpfn89_testpreds.npy"),
    "cb144": ("20260824/20260824_97_ft_transformer_on_xxxxv4_features_cb_oofpreds.npy",
              "20260824/20260824_97_ft_transformer_on_xxxxv4_features_cb_testpreds.npy"),
    "tabpfn144": ("20260824/20260824_99_tabpfn_on_xxxxv4_features_tabpfn_oofpreds.npy",
                  "20260824/20260824_99_tabpfn_on_xxxxv4_features_tabpfn_testpreds.npy"),
    "r_full": ("20260816/20260816_72_reference_pipeline_standalone_R_full_oofpreds.npy",
               "20260816/20260816_72_reference_pipeline_standalone_R_full_testpreds.npy"),
    "lgbm134": ("20260905/20260905_114_lightgbm_on_84_top30pct_lgbm134_oofpreds.npy",
                "20260905/20260905_114_lightgbm_on_84_top30pct_lgbm134_testpreds.npy"),
}
STACK_MODELS = list(BASE_MODELS.keys())

oof = {}
test = {}
for name, (oof_rel, test_rel) in BASE_MODELS.items():
    oof[name] = np.load(OOF_DIR / oof_rel)
    test[name] = np.load(OOF_DIR / test_rel)

surv_idx = np.where(surv_mask)[0]
y_s = y_train[surv_idx]
X_s = np.column_stack([logit(oof[m][surv_idx]) for m in STACK_MODELS])
logger.info(f"スタッキング用行列: {X_s.shape} (列={STACK_MODELS})")

REF_STACK8_LINEAR_VAL = 0.482271  # 115_の線形スタッキング(Public確認済み0.493891)
logger.info(f"[参考] 115_の線形スタッキング(入れ子CV): val={REF_STACK8_LINEAR_VAL:.6f}")

# ============================================================
# 非線形メタモデル(HistGradientBoostingClassifier、浅い木+強い正則化+内部early stopping)
# を線形スタッキングと同一の外側5-fold(seed=12345)で評価
# ============================================================
logger.info("=" * 60)
logger.info("非線形メタモデル(HistGradientBoostingClassifier)を入れ子CVで評価中...")

HGB_PARAMS = dict(
    max_depth=2, max_iter=200, learning_rate=0.05, l2_regularization=1.0,
    max_leaf_nodes=4, min_samples_leaf=30,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
)

OUTER_SEED = 12345  # 105_/107_/109_/115_と同一
outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
meta_oof_nonlinear = np.zeros(len(y_s))
meta_oof_linear = np.zeros(len(y_s))  # 同一fold切りでの線形版も並行して再計算し、公平に比較する

for fold_i, (tr_idx, va_idx) in enumerate(outer_skf.split(X_s, y_s)):
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED)
    hgb.fit(X_s[tr_idx], y_s[tr_idx])
    meta_oof_nonlinear[va_idx] = hgb.predict_proba(X_s[va_idx])[:, 1]

    linreg = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 15), cv=5, penalty="l2", scoring="neg_log_loss",
        max_iter=5000, random_state=SEED,
    )
    linreg.fit(X_s[tr_idx], y_s[tr_idx])
    meta_oof_linear[va_idx] = linreg.predict_proba(X_s[va_idx])[:, 1]

    logger.info(f"    [外側fold{fold_i}] 完了 (n_iter実績={hgb.n_iter_})")

nonlinear_val = log_loss(y_s, meta_oof_nonlinear)
linear_val_recomputed = log_loss(y_s, meta_oof_linear)
logger.info(f"[非線形メタモデル(入れ子CV)] val={nonlinear_val:.6f}")
logger.info(f"[線形メタモデル(同一fold切り、再計算)] val={linear_val_recomputed:.6f}"
            f"（115_のオリジナル値{REF_STACK8_LINEAR_VAL:.6f}とほぼ一致するはず）")
logger.info(f"[比較] 非線形-線形: {nonlinear_val - linear_val_recomputed:+.6f}（分解能±0.0043）")

# ============================================================
# 判定: 分解能を超えて改善した場合のみ、全データで再学習してTest予測を生成
# ============================================================
RESOLUTION = 0.0043
improvement = linear_val_recomputed - nonlinear_val
logger.info("=" * 60)
if improvement > RESOLUTION:
    logger.info(f"[判定] 分解能を超えて改善({improvement:+.6f})——"
                f"全データで最終非線形メタモデルを学習しTest予測を生成する")
    final_hgb = HistGradientBoostingClassifier(**HGB_PARAMS, random_state=SEED)
    final_hgb.fit(X_s, y_s)

    X_test = np.column_stack([logit(test[m]) for m in STACK_MODELS])
    test_pred = final_hgb.predict_proba(X_test)[:, 1]
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_stacker8_nonlinear_full_survivor_submission.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_pred}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={test_pred.mean():.4f}）")
else:
    logger.info(f"[判定] 分解能未満または悪化({improvement:+.6f})——"
                f"非線形メタモデルは不採用。線形スタッキング(115_)を維持する")

logger.info("=" * 60)
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
