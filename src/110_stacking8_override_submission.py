"""110_stacking8_override_submission

`109_stacking_with_autogluon_oof`の8モデルスタッキング(入れ子CV val=0.484029)は、
107_の7モデル版(0.486675)比-0.002646で、事前登録した判定基準（分解能±0.0043超えの
改善）を満たさなかったため提出ファイルを生成していなかった。

ただし`r_poolc51`（108_で抽出したAutoGluonプールC系統のOOF）の係数は入れ子CVの
全5foldで安定して正(0.34〜0.43、r_fullに次ぐ2番目の寄与)であり、構造的には健全な拡張
だった（[[ensemble-oof-overfitting]]追記9参照）。

ユーザー指示: **現最良`107_`(Public 0.494517)はそのまま変更せず**、109_を別の独立した
提出候補として確認する。105_→106_/107_→109_と同じパターンで、この回だけ事前登録ルールを
上書きしてPublicで確認する。

出力1ファイル: stacker8_override_submission.csv
"""
import datetime
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger
from common.utils.seed import seed_everything

SEED = 42
seed_everything(seed=SEED)

TARGET_COL = "10年定着ラベル"
ID_COL = "社員ID"

SCRIPT_NAME = "110_stacking8_override_submission"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 実験開始 ===")
logger.info("※ 109_の事前登録ルール(分解能超え必須)は満たさなかったが、"
            "ユーザー明示指示によりPublicで確認するための上書き提出(107_は変更しない)")

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

surv_idx = np.where(surv_mask)[0]
y_s = y_train[surv_idx]
X_s = np.column_stack([logit(oof[m][surv_idx]) for m in STACK_MODELS])
logger.info(f"スタッキング用行列: {X_s.shape} (列={STACK_MODELS})")

logger.info("=" * 60)
logger.info("全生存者データで最終スタッカーを学習中(109_と同一設計のLogisticRegressionCV)...")
final_clf = LogisticRegressionCV(
    Cs=np.logspace(-3, 1, 15), cv=5, penalty="l2", scoring="neg_log_loss",
    max_iter=5000, random_state=SEED,
)
final_clf.fit(X_s, y_s)
train_fit_val = log_loss(y_s, final_clf.predict_proba(X_s)[:, 1])
logger.info(f"  最終係数: {dict(zip(STACK_MODELS, final_clf.coef_[0].round(3)))} / C={final_clf.C_[0]:.4f}")
logger.info(f"  (参考、in-sample val={train_fit_val:.6f}——過学習込みの楽観値。"
            f"109_の入れ子CV val=0.484029の方が信頼できる推定)")

X_test = np.column_stack([logit(test[m]) for m in STACK_MODELS])
test_pred = final_clf.predict_proba(X_test)[:, 1]

path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_stacker8_override_submission.csv"
pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_pred}).to_csv(path, index=False, header=False)
logger.info(f"提出ファイル保存: {path.name}（予測平均={test_pred.mean():.4f}, std={test_pred.std():.4f}）")

logger.info("=" * 60)
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
