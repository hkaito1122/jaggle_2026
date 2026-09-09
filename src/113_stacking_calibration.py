"""113_stacking_calibration

`107_`のスタック出力（7モデルをlogit変換→L2ロジスティック回帰）は、そもそもログロスを
直接最小化するロジスティック回帰そのものなので、単純な事後較正（Platt/isotonic）が
さらに改善する余地は理論上小さいと予想される（logit空間ブレンドが確率空間と無風だった
`105_`タスクAと同種の予想）。ただし未検証のため、107_のスタック出力に対して事後較正を
追加した場合に分解能を超えて改善するかを確認する。

設計:
  - 107_と全く同一のfold割当・パラメータで7モデルの入れ子CVスタック出力(meta_oof)を
    再現する（107_はmeta_oofを保存していなかったため再計算、決定的なので同じ値になる）
  - meta_oofに対してさらに較正層(Platt scaling=1変数ロジスティック回帰 / isotonic回帰)を
    加える。較正層自体の評価も入れ子にする必要があるため、meta_oofとは別のseed(99999)で
    5-foldに分け、4foldで較正器をfitして残り1foldに適用する二重入れ子構成にする
  - Platt scalingは低分散（1パラメータ）、isotonicは高柔軟性（ノンパラメトリック）。
    データ数への配慮から両方試し、Plattの方を主候補として扱う

判定基準: 較正後OOF loglossが未較正のmeta_oof(107_の入れ子CV val=0.486675と同一のはず)を
分解能±0.0043超えて改善した場合のみ、Test予測に較正を適用したファイルを生成する。

出力: 分解能を超えて改善した場合のみ calibrated_submission.csv
"""
import datetime
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
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

SCRIPT_NAME = "113_stacking_calibration"
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


# ============================================================
# 107_と同一の7モデルOOF/Test予測を読み込み、107_と同一設計でmeta_oofを再現する
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

logger.info("=" * 60)
logger.info("107_と同一設計で7モデルスタック(meta_oof)を再現中...")
OUTER_SEED = 12345  # 107_と同一
outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
meta_oof = np.zeros(len(y_s))

for fold_i, (tr_idx, va_idx) in enumerate(outer_skf.split(X_s, y_s)):
    clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 15), cv=5, penalty="l2", scoring="neg_log_loss",
        max_iter=5000, random_state=SEED,
    )
    clf.fit(X_s[tr_idx], y_s[tr_idx])
    meta_oof[va_idx] = clf.predict_proba(X_s[va_idx])[:, 1]

meta_val = log_loss(y_s, meta_oof)
logger.info(f"[再現確認] meta_oof val={meta_val:.6f}（107_の入れ子CV val=0.486675と一致するはず）")

# ============================================================
# 較正層を二重入れ子で評価（meta_oofの5-foldとは別のseedで新たに5分割）
# ============================================================
logger.info("=" * 60)
logger.info("較正層(Platt scaling / isotonic回帰)を二重入れ子CVで評価中...")
CALIB_SEED = 99999  # meta_oofの分割(12345)・107_のfold割当とは独立させる
calib_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CALIB_SEED)

platt_oof = np.zeros(len(y_s))
isotonic_oof = np.zeros(len(y_s))
meta_logit = logit(meta_oof)

for fold_i, (tr_idx, va_idx) in enumerate(calib_skf.split(meta_oof, y_s)):
    # Platt scaling: meta_oofのlogitを1変数ロジスティック回帰で再較正
    platt = LogisticRegression(C=1e6, max_iter=5000)  # 正則化ほぼ無し(1変数なので過学習リスクは低い)
    platt.fit(meta_logit[tr_idx].reshape(-1, 1), y_s[tr_idx])
    platt_oof[va_idx] = platt.predict_proba(meta_logit[va_idx].reshape(-1, 1))[:, 1]

    # Isotonic回帰: ノンパラメトリックな単調較正
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(meta_oof[tr_idx], y_s[tr_idx])
    isotonic_oof[va_idx] = iso.predict(meta_oof[va_idx])

platt_val = log_loss(y_s, platt_oof)
isotonic_val = log_loss(y_s, isotonic_oof)
logger.info(f"[較正後(入れ子CV)] Platt scaling val={platt_val:.6f}（対未較正 {platt_val - meta_val:+.6f}）")
logger.info(f"[較正後(入れ子CV)] Isotonic回帰 val={isotonic_val:.6f}（対未較正 {isotonic_val - meta_val:+.6f}）")
logger.info(f"（分解能±0.0043。データ数への配慮からPlatt scalingを主候補として扱う）")

# ============================================================
# 判定: 分解能を超えて改善した場合のみ、Test予測に較正を適用して提出ファイルを生成
# ============================================================
RESOLUTION = 0.0043
best_calib_name, best_calib_val = min(
    [("platt", platt_val), ("isotonic", isotonic_val)], key=lambda x: x[1]
)
improvement = meta_val - best_calib_val
logger.info("=" * 60)
if improvement > RESOLUTION:
    logger.info(f"[判定] {best_calib_name}較正が分解能を超えて改善({improvement:+.6f})"
                f"——全データで最終スタッカー+較正器を学習しTest予測を生成する")

    final_stacker = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 15), cv=5, penalty="l2", scoring="neg_log_loss",
        max_iter=5000, random_state=SEED,
    )
    final_stacker.fit(X_s, y_s)
    X_test = np.column_stack([logit(test[m]) for m in STACK_MODELS])
    test_stacked = final_stacker.predict_proba(X_test)[:, 1]

    if best_calib_name == "platt":
        final_calib = LogisticRegression(C=1e6, max_iter=5000)
        final_calib.fit(meta_logit.reshape(-1, 1), y_s)
        test_pred = final_calib.predict_proba(logit(test_stacked).reshape(-1, 1))[:, 1]
    else:
        final_calib = IsotonicRegression(out_of_bounds="clip")
        final_calib.fit(meta_oof, y_s)
        test_pred = final_calib.predict(test_stacked)

    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_calibrated_submission.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_pred}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={test_pred.mean():.4f}）")
else:
    logger.info(f"[判定] 分解能未満または悪化(best={best_calib_name} {improvement:+.6f})"
                f"——較正は不採用。Test予測は生成しない")

logger.info("=" * 60)
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
