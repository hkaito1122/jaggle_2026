"""106_stacking_override_submission

`105_logit_blend_and_stacking`のタスクBは、正則化+入れ子CVで評価したスタッキングが
入れ子CV val=0.498094を記録し、対等重み平均で-0.003302・対93_参考ブレンドで-0.004325と、
**事前登録した判定基準（両方に対して分解能±0.0043超えの改善）を厳密には満たさなかった**
ため、Test予測を生成せず不採用と判定していた。

ただし外側5foldすべてで係数の符号・大きさが安定しており（tabpfn134/cb144が常に最大級の
正の重み、[[ensemble-oof-overfitting]]追記7参照）、過去の過学習事例（12_/32_の「split間で
重みベクトルが大きく変わる」パターン）とは異なる、構造的に健全そうな結果だった。

コンペ残り2週間・Public10位（9位差0.00004）という状況を踏まえ、**ユーザーの明示的指示により、
105_の事前登録ルールを上書きしてPublicで確認する**。105_と同じ6モデル・同じ入れ子CV設計を
使い、全生存者データで最終スタッカーを再学習してTest予測を生成する（105_内の判定ロジックで
未生成だった提出ファイルを、この回だけ明示的に作る）。

出力1ファイル: stacker_override_submission.csv
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

SCRIPT_NAME = "106_stacking_override_submission"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 実験開始 ===")
logger.info("※ 105_の事前登録ルール(分解能超え必須)は満たさなかったが、"
            "ユーザー明示指示によりPublicで確認するための上書き提出")

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
logger.info("全生存者データで最終スタッカーを学習中(105_と同一設計のLogisticRegressionCV)...")
final_clf = LogisticRegressionCV(
    Cs=np.logspace(-3, 1, 15), cv=5, penalty="l2", scoring="neg_log_loss",
    max_iter=5000, random_state=SEED,
)
final_clf.fit(X_s, y_s)
train_fit_val = log_loss(y_s, final_clf.predict_proba(X_s)[:, 1])
logger.info(f"  最終係数: {dict(zip(STACK_MODELS, final_clf.coef_[0].round(3)))} / C={final_clf.C_[0]:.4f}")
logger.info(f"  (参考、in-sample val={train_fit_val:.6f}——過学習込みの楽観値。"
            f"105_の入れ子CV val=0.498094の方が信頼できる推定)")

X_test = np.column_stack([logit(test[m]) for m in STACK_MODELS])
test_pred = final_clf.predict_proba(X_test)[:, 1]

path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_stacker_override_submission.csv"
pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_pred}).to_csv(path, index=False, header=False)
logger.info(f"提出ファイル保存: {path.name}（予測平均={test_pred.mean():.4f}, std={test_pred.std():.4f}）")

logger.info("=" * 60)
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
