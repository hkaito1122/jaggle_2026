"""109_stacking_with_autogluon_oof

`108_`で`51_autogluon_catboost_bias`(プールC構成メンバーの1つ、441列旧パイプライン)の
predictor.pklから`predict_proba_oof()`によりOOFを再学習なしで抽出した
（単体生存者OOF logloss=0.504162、既存7モデルとの相関0.78〜0.93——cb144/tabpfn144
(xxxx_v4系144列プール)とは0.78〜0.84、r_full(72_reference)とは0.85と、既存の
intra-family相関0.94〜0.99より明確に低く、94_add_w25の未統合成分だったAutoGluon
プールC系統を初めてこのスタッキング枠組みに追加できる）。

`107_`と全く同一の設計（logit変換→L2ロジスティック回帰、外側5-fold(seed=12345)+
内側LogisticRegressionCVの二重入れ子）で、r_poolc51を8つ目の特徴量として追加し再評価する。

事前登録した判定基準は105_/107_と同一: 対等重み平均・対107_の7モデルスタッキング(0.486675)
の両方に対して分解能±0.0043を超えて改善した場合のみTest予測を生成する。

出力: 分解能を超えて改善した場合のみ stacker8_full_survivor_submission.csv
"""
import datetime
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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

SCRIPT_NAME = "109_stacking_with_autogluon_oof"
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
# 保存済みOOF/Test予測の読み込み(107_の7モデル + 108_のr_poolc51)
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
    "r_poolc51": ("20260905/20260905_108_extract_autogluon_oof_r_poolc51_oofpreds.npy",
                  "20260814/20260814_51_autogluon_catboost_bias_AG51_full441_weighted_testpreds.npy"),
}
STACK_MODELS = list(BASE_MODELS.keys())

oof = {}
test = {}
for name, (oof_rel, test_rel) in BASE_MODELS.items():
    oof[name] = np.load(OOF_DIR / oof_rel)
    test[name] = np.load(OOF_DIR / test_rel)
    assert len(oof[name]) == len(train_ids)
    assert len(test[name]) == len(test_ids)
    val = log_loss(y_train[surv_mask], oof[name][surv_mask])
    logger.info(f"[単体val] {name}: {val:.6f}")

# ============================================================
# スタッキング(入れ子CV) — 105_/107_と全く同一設計、8モデルに拡張
# ============================================================
logger.info("=" * 60)
logger.info("logit変換8モデルでのスタッキング(入れ子CV)")

surv_idx = np.where(surv_mask)[0]
y_s = y_train[surv_idx]
X_s = np.column_stack([logit(oof[m][surv_idx]) for m in STACK_MODELS])
logger.info(f"  スタッキング用行列: {X_s.shape} (行=生存者, 列={STACK_MODELS})")

equal_weight_pred = sigmoid(X_s.mean(axis=1))
equal_weight_val = log_loss(y_s, equal_weight_pred)
logger.info(f"  [ベースライン1] logit空間等重み平均(8モデル): val={equal_weight_val:.6f}")

REF_STACK7_VAL = 0.486675  # 107_の7モデル入れ子CV結果(Public 0.494517で現最良を更新済み)
logger.info(f"  [ベースライン2] 107_の7モデルスタッキング(入れ子CV、Public確認済み0.494517): val={REF_STACK7_VAL:.6f}")

OUTER_SEED = 12345  # 105_/107_と同一
outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
meta_oof = np.zeros(len(y_s))
fold_coefs = []

for fold_i, (tr_idx, va_idx) in enumerate(outer_skf.split(X_s, y_s)):
    clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 15), cv=5, penalty="l2", scoring="neg_log_loss",
        max_iter=5000, random_state=SEED,
    )
    clf.fit(X_s[tr_idx], y_s[tr_idx])
    meta_oof[va_idx] = clf.predict_proba(X_s[va_idx])[:, 1]
    fold_coefs.append(clf.coef_[0])
    logger.info(f"    [外側fold{fold_i}] C={clf.C_[0]:.4f}, 係数={dict(zip(STACK_MODELS, clf.coef_[0].round(3)))}")

stack_val = log_loss(y_s, meta_oof)
logger.info(f"  [8モデルスタッキング(入れ子CV)] val={stack_val:.6f}")
logger.info(f"  対等重み平均(8モデル): {stack_val - equal_weight_val:+.6f} / "
            f"対107_の7モデルスタッキング: {stack_val - REF_STACK7_VAL:+.6f}")

mean_coefs = np.mean(fold_coefs, axis=0)
logger.info(f"  fold平均係数: {dict(zip(STACK_MODELS, mean_coefs.round(3)))}")

# ============================================================
# 判定: 分解能を超えて改善した場合のみ、全データで再学習してTest予測を生成
# ============================================================
RESOLUTION = 0.0043
improvement_vs_equal = equal_weight_val - stack_val
improvement_vs_stack7 = REF_STACK7_VAL - stack_val
logger.info("=" * 60)
if improvement_vs_equal > RESOLUTION and improvement_vs_stack7 > RESOLUTION:
    logger.info(f"[判定] 分解能を超えて改善(対等重み{improvement_vs_equal:+.6f} / 対7モデル版{improvement_vs_stack7:+.6f})"
                f"——全データで最終スタッカーを学習しTest予測を生成する")
    final_clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 15), cv=5, penalty="l2", scoring="neg_log_loss",
        max_iter=5000, random_state=SEED,
    )
    final_clf.fit(X_s, y_s)
    logger.info(f"  最終係数: {dict(zip(STACK_MODELS, final_clf.coef_[0].round(3)))} / C={final_clf.C_[0]:.4f}")

    X_test = np.column_stack([logit(test[m]) for m in STACK_MODELS])
    test_pred = final_clf.predict_proba(X_test)[:, 1]
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_stacker8_full_survivor_submission.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_pred}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={test_pred.mean():.4f}）")
else:
    logger.info(f"[判定] 分解能未満(対等重み{improvement_vs_equal:+.6f} / 対7モデル版{improvement_vs_stack7:+.6f})"
                f"——8モデル版は不採用。判断はユーザーに委ねる")

logger.info("=" * 60)
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
