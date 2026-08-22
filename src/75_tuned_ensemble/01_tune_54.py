# 独立プロセスとして実行するスクリプト。00_build_features.py が保存したキャッシュを読み込み、
# このモデル1つ分のOptuna探索・最終学習だけを行って終了する（メモリはプロセス終了時にOSへ返る）。
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "catboost", "optuna"], check=True)

import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb
import optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path("/content/drive/MyDrive/jaggle_2026")
assert PROJECT_ROOT.exists(), (
    f"{PROJECT_ROOT} が見つからない。run_all.ipynb で先に drive.mount() を実行してから"
    " このスクリプトをサブプロセスとして呼び出すこと。"
)
sys.path.append(str(PROJECT_ROOT))

from common.utils.logger import get_logger
from common.utils.seed import seed_everything

SEED = 42
seed_everything(seed=SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SCRIPT_NAME = "75_01_tune_54"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL, TARGET_COL = "社員ID", "10年定着ラベル"

# ============================================================
# 0. 00_build_features.py が保存したキャッシュを読み込む
# ============================================================
CACHE_DIR = PROJECT_ROOT / "data" / "output" / "_checkpoints" / "75_tuned_ensemble_cache"
assert CACHE_DIR.exists(), f"{CACHE_DIR} が無い。先に 00_build_features.py を実行すること"

X_full = pd.read_pickle(CACHE_DIR / "X_full.pkl")
X_test = pd.read_pickle(CACHE_DIR / "X_test.pkl")
y_full_arr = np.load(CACHE_DIR / "y_full.npy")
y_full = pd.Series(y_full_arr, index=X_full.index)
IS_SURV = np.load(CACHE_DIR / "IS_SURV.npy")
test_ids = np.load(CACHE_DIR / "test_ids.npy", allow_pickle=True)
with open(CACHE_DIR / "obj_cols.json", encoding="utf-8") as f:
    obj_cols = json.load(f)
with open(CACHE_DIR / "A_PARAMS_ORIGINAL.json", encoding="utf-8") as f:
    A_PARAMS_ORIGINAL = json.load(f)

logger.info(f"キャッシュ読み込み完了: X_full={X_full.shape} X_test={X_test.shape}")
print(f"キャッシュ読み込み完了: X_full={X_full.shape} X_test={X_test.shape}")

# ============================================================
# 1. 強化Optuna探索（3-fold CV平均・150試行）
# ============================================================
N_TRIALS_TUNE = 150      # 旧(25)の6倍。実行時間を短縮したい場合はここを下げる
K_FOLDS_TUNE = 3         # 3-fold CV平均を目的関数にする（単一535名ホールドアウトより安定）
MAX_ITER_TUNE = 1500

_skf_tune = list(StratifiedKFold(n_splits=K_FOLDS_TUNE, shuffle=True, random_state=SEED)
                  .split(X_full, y_full))


def _objective(trial):
    params = {
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
    }
    scores, iters = [], []
    for tri, vai in _skf_tune:
        m = cb.CatBoostClassifier(
            **params, iterations=MAX_ITER_TUNE, random_seed=SEED, verbose=False,
            cat_features=obj_cols, early_stopping_rounds=50, task_type="CPU",
        )
        m.fit(X_full.iloc[tri], y_full.iloc[tri],
              eval_set=(X_full.iloc[vai], y_full.iloc[vai]), use_best_model=True)
        scores.append(log_loss(y_full.iloc[vai], m.predict_proba(X_full.iloc[vai])[:, 1]))
        iters.append(m.get_best_iteration())
    trial.set_user_attr("best_iteration_mean", float(np.mean(iters)))
    return float(np.mean(scores))


t0 = time.time()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(_objective, n_trials=N_TRIALS_TUNE, show_progress_bar=False)
logger.info(f"Optuna完了 ({time.time()-t0:.0f}秒): best_value={study.best_value:.6f}")
logger.info(f"best_params={study.best_params}")

TUNED_PARAMS = study.best_params
TUNED_ITER = int(round(study.best_trial.user_attrs["best_iteration_mean"] * 1.25))
print(f"best_value(3-fold CV平均logloss) = {study.best_value:.6f}")
print(f"best_params = {TUNED_PARAMS}")
print(f"全件学習用の反復数（3-fold平均best_iteration×1.25） = {TUNED_ITER}")
print(f"\n参考: 54_の元のA_PARAMS_ORIGINAL（25試行・単一ホールドアウト）")
print(A_PARAMS_ORIGINAL)

with open(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_tuned_params.json", "w", encoding="utf-8") as f:
    json.dump({"params": TUNED_PARAMS, "iterations": TUNED_ITER,
                "cv_best_value": study.best_value}, f, ensure_ascii=False, indent=2)

# ============================================================
# 2. チューニング済みハイパーパラメータでの最終学習（Train全件×5シード）
# ============================================================
SEEDS_FINAL = [42, 2024, 7, 1234, 99]   # 54_の提出用シードと同一

logger.info("-" * 60)
logger.info(f"チューニング済みパラメータで全件学習を開始（反復数={TUNED_ITER}）")

test_preds_tuned = []
for seed in SEEDS_FINAL:
    t0 = time.time()
    m = cb.CatBoostClassifier(**TUNED_PARAMS, iterations=TUNED_ITER, random_seed=seed,
                               verbose=False, cat_features=obj_cols, task_type="CPU")
    m.fit(X_full, y_full)
    test_preds_tuned.append(m.predict_proba(X_test)[:, 1])
    logger.info(f"  seed={seed}: 完了 ({time.time()-t0:.0f}秒)")

pred_tuned = np.mean(test_preds_tuned, axis=0)

# サニティチェック: 旧54_提出物との相関・乖離を見る
_ref_paths = sorted((PROJECT_ROOT / "data" / "output").glob(
    "*/*_54_l2_m_interaction_R0_memofix_plus_LM.csv"))
if _ref_paths:
    _ref54 = (pd.read_csv(_ref_paths[-1], header=None, names=[ID_COL, "p"])
              .set_index(ID_COL).loc[test_ids, "p"].to_numpy())
    print(f"チューニング済みモデル vs 旧54_提出物: "
          f"corr={np.corrcoef(pred_tuned, _ref54)[0,1]:.4f} "
          f"MAD={np.abs(pred_tuned - _ref54).mean():.5f}（ノイズ床0.02122）")

np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_54tuned_testpreds.npy", pred_tuned)
sub_tuned = pd.DataFrame({ID_COL: test_ids, "定着確率": pred_tuned})
path_tuned = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_54tuned_submission.csv"
sub_tuned.to_csv(path_tuned, index=False, header=False)
logger.info(f"提出ファイルを保存: {path_tuned}")
print(f"\n保存: {path_tuned}")
print(sub_tuned.head())
print(f"\n✅ [{SCRIPT_NAME}] 完了。このプロセスをここで終了する（メモリを解放する）。")
