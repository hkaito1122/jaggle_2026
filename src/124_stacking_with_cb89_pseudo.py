"""124_stacking_with_cb89_pseudo

`123_`で疑似ラベリング(115_のTest予測のうち高信頼度行をhard labelとしてfold学習に
追加)を試した結果、単体CB89のリークなしOOF(93_と同一のネストされた5-fold設計)が
0.521296→0.515789(閾値グリッドの中で最良のt=0.15、疑似ラベル946行追加)まで改善し、
分解能±0.0043を明確にクリアした(+0.005507)。93_のcb89が既に115_の8モデルスタックの
1つとして含まれているため、cb89をcb89_pseudoに置き換えて105_/107_/115_と全く同一の
設計(logit変換→L2ロジスティック回帰、外側5-fold(seed=12345)+内側LogisticRegressionCV
の二重入れ子)でスタック全体の入れ子CVを再評価する。

単体モデルの改善がスタック全体の改善に直結するとは限らない(`109_`のr_poolc51は単体では
妥当に見えたが分解能未満でPublic反転、`117_`のxgb134はlgbm134と相関0.98で追加効果ゼロ)
ため、cb89とcb89_pseudoの相関を確認した上でスタック単位の入れ子CVで判定する。

判定基準: 分解能±0.0043を正規にクリアした場合のみ採用する(上書き提出は行わない)。

出力: 分解能を超えて改善した場合のみ stacker8_cb89pseudo_full_survivor_submission.csv
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

SCRIPT_NAME = "124_stacking_with_cb89_pseudo"
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
# 保存済みOOF/Test予測の読み込み(115_の8モデルのうちcb89を123_のcb89_pseudoに置換)
# ============================================================
OOF_DIR = PROJECT_ROOT / "data" / "output"
BASE_MODELS = {
    "cb134": ("20260822/20260822_86_tabpfn_on_84_top30pct_cb134_oofpreds.npy",
              "20260822/20260822_86_tabpfn_on_84_top30pct_cb134_testpreds.npy"),
    "tabpfn134": ("20260822/20260822_86_tabpfn_on_84_top30pct_tabpfn134_oofpreds.npy",
                  "20260822/20260822_86_tabpfn_on_84_top30pct_tabpfn134_testpreds.npy"),
    "cb89_pseudo": ("20260905/20260905_123_pseudolabel_cb89_poc_cb89_pseudo_best_oofpreds.npy",
                    "20260905/20260905_123_pseudolabel_cb89_poc_cb89_pseudo_best_testpreds.npy"),
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
    assert len(oof[name]) == len(train_ids)
    assert len(test[name]) == len(test_ids)
    val = log_loss(y_train[surv_mask], oof[name][surv_mask])
    logger.info(f"[単体val] {name}: {val:.6f}")

cb89_orig_oof = np.load(OOF_DIR / "20260822/20260822_93_tabpfn_top20pct_on_77_cb89_oofpreds.npy")
corr_pseudo_vs_orig = float(np.corrcoef(oof["cb89_pseudo"][surv_mask], cb89_orig_oof[surv_mask])[0, 1])
logger.info(f"[相関] cb89_pseudo vs 元のcb89(115_で使用): {corr_pseudo_vs_orig:.4f}")

# ============================================================
# スタッキング(入れ子CV) — 115_と全く同一設計、cb89をcb89_pseudoに置換して再評価
# ============================================================
logger.info("=" * 60)
logger.info("logit変換8モデルでのスタッキング(入れ子CV、cb89→cb89_pseudo置換版)")

surv_idx = np.where(surv_mask)[0]
y_s = y_train[surv_idx]
X_s = np.column_stack([logit(oof[m][surv_idx]) for m in STACK_MODELS])
logger.info(f"  スタッキング用行列: {X_s.shape} (行=生存者, 列={STACK_MODELS})")

equal_weight_pred = sigmoid(X_s.mean(axis=1))
equal_weight_val = log_loss(y_s, equal_weight_pred)
logger.info(f"  [ベースライン1] logit空間等重み平均(8モデル): val={equal_weight_val:.6f}")

REF_STACK8_ORIG_VAL = 0.482271  # 115_の8モデル(元のcb89)入れ子CV結果(Public 0.493891で現最良)
logger.info(f"  [ベースライン2] 115_の8モデルスタッキング(元cb89、Public確認済み0.493891): val={REF_STACK8_ORIG_VAL:.6f}")

OUTER_SEED = 12345  # 105_/107_/109_/115_と同一
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
logger.info(f"  [8モデルスタッキング(cb89_pseudo置換版、入れ子CV)] val={stack_val:.6f}")
logger.info(f"  対等重み平均(8モデル): {stack_val - equal_weight_val:+.6f} / "
            f"対115_の8モデルスタッキング(元cb89): {stack_val - REF_STACK8_ORIG_VAL:+.6f}")

mean_coefs = np.mean(fold_coefs, axis=0)
logger.info(f"  fold平均係数: {dict(zip(STACK_MODELS, mean_coefs.round(3)))}")

# ============================================================
# 判定: 分解能を超えて改善した場合のみ、全データで再学習してTest予測を生成
# ============================================================
RESOLUTION = 0.0043
improvement_vs_equal = equal_weight_val - stack_val
improvement_vs_stack7 = REF_STACK8_ORIG_VAL - stack_val
logger.info("=" * 60)
if improvement_vs_equal > RESOLUTION and improvement_vs_stack7 > RESOLUTION:
    logger.info(f"[判定] 分解能を超えて改善(対等重み{improvement_vs_equal:+.6f} / 対元cb89版{improvement_vs_stack7:+.6f})"
                f"——全データで最終スタッカーを学習しTest予測を生成する")
    final_clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 15), cv=5, penalty="l2", scoring="neg_log_loss",
        max_iter=5000, random_state=SEED,
    )
    final_clf.fit(X_s, y_s)
    logger.info(f"  最終係数: {dict(zip(STACK_MODELS, final_clf.coef_[0].round(3)))} / C={final_clf.C_[0]:.4f}")

    X_test = np.column_stack([logit(test[m]) for m in STACK_MODELS])
    test_pred = final_clf.predict_proba(X_test)[:, 1]
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_stacker8_cb89pseudo_full_survivor_submission.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_pred}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={test_pred.mean():.4f}）")
else:
    logger.info(f"[判定] 分解能未満(対等重み{improvement_vs_equal:+.6f} / 対元cb89版{improvement_vs_stack7:+.6f})"
                f"——cb89_pseudo置換版は不採用。判断はユーザーに委ねる")

logger.info("=" * 60)
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
