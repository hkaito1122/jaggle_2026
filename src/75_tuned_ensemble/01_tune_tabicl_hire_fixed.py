# 独立プロセスとして実行するスクリプト。00_build_features.py が保存した441列キャッシュ
# (data/output/_checkpoints/75_tuned_ensemble_cache_fm/) の hire_fixed(78列)を使い、
# TabICL の探索空間(n_estimators)をOptunaで探索する。
#
# 注意: TabICLは`70_`で「n_estimators=8, batch_size=1」に固定されており、コメントに
# 「既定よりメモリ55%減・時間45%減」とある——つまり既に精度ではなくメモリ制約で
# 下げられた値。ここでのチューニングは精度と資源のトレードオフを測るものであり、
# n_estimatorsの探索上限は既定の枠を大きく超えないよう抑える（メモリ枯渇を避けるため）。
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabicl", "optuna"], check=True)

import datetime
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import torch
from tabicl import TabICLClassifier
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path("/content/drive/MyDrive/jaggle_2026")
assert PROJECT_ROOT.exists(), (
    f"{PROJECT_ROOT} が見つからない。run_all.ipynb で先に drive.mount() を実行してから"
    " このスクリプトをサブプロセスとして呼び出すこと。"
)
sys.path.append(str(PROJECT_ROOT))

from common.utils.logger import get_logger

SCRIPT_NAME = "75_01_tune_tabicl_hire_fixed"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL, TARGET_COL = "社員ID", "10年定着ラベル"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEVICE}")
if DEVICE == "cpu":
    print("⚠️ GPUが無い。CPU向けに探索を軽量化する（試行数・n_estimators上限を下げる）。"
          "TabICLはCPUだとTabPFNよりさらに遅い可能性がある点に注意。")

# CPUでは1trialのコストが線形に効くn_estimatorsの上限と試行数を下げる。
if DEVICE == "cpu":
    N_TRIALS_TUNE = 6
    N_ESTIMATORS_MAX = 10   # 既定8をわずかに超える程度に抑える
else:
    N_TRIALS_TUNE = 12      # 既定8の2倍まで。これ以上はメモリ枯渇のリスクが上がる
    N_ESTIMATORS_MAX = 16

# ============================================================
# 0. 441列キャッシュを読み込み、hire_fixed(78列)を選ぶ
# ============================================================
FM_CACHE_DIR = PROJECT_ROOT / "data" / "output" / "_checkpoints" / "75_tuned_ensemble_cache_fm"
assert FM_CACHE_DIR.exists(), f"{FM_CACHE_DIR} が無い。先に 00_build_features.py を実行すること"

ag_train_80b = pd.read_pickle(FM_CACHE_DIR / "ag_train_80b.pkl")
ag_val_surv = pd.read_pickle(FM_CACHE_DIR / "ag_val_surv.pkl")
ag_full = pd.read_pickle(FM_CACHE_DIR / "ag_full.pkl")
test_features_full = pd.read_pickle(FM_CACHE_DIR / "test_features_full.pkl")
test_ids = np.load(FM_CACHE_DIR / "test_ids.npy", allow_pickle=True)
with open(FM_CACHE_DIR / "feature_sets.json", encoding="utf-8") as f:
    FEATURE_SETS = json.load(f)

FEATS = FEATURE_SETS["hire_fixed"]
print(f"hire_fixed: {len(FEATS)}列 / 学習{len(ag_train_80b)}名 / 検証{len(ag_val_surv)}名 / 全件{len(ag_full)}名")


def to_numeric_matrix(train_df, other_dfs, feats):
    """TabICL用の行列を作る（TabPFNと同じ序数コード化、63_/70_と同一方式）。"""
    obj = [c for c in feats if train_df[c].dtype == "object"]
    frames = [train_df] + list(other_dfs)
    cat_map = {}
    for c in obj:
        vals = pd.concat([f[c].astype(str) for f in frames]).unique()
        cat_map[c] = {v: i for i, v in enumerate(sorted(vals))}
    out = []
    for f in frames:
        M = f[feats].copy()
        for c in obj:
            M[c] = f[c].astype(str).map(cat_map[c]).astype(float)
        out.append(M.astype(np.float32).values)
    return out


# ============================================================
# 1. Optuna探索（ホールドアウト単発。探索空間はn_estimatorsのみに絞る）
# ============================================================
Mtr, Mva = to_numeric_matrix(ag_train_80b, [ag_val_surv], FEATS)
ytr = ag_train_80b[TARGET_COL].values
yva = ag_val_surv[TARGET_COL].values

# --- 実測タイミングプローブ: 最小構成(n_estimators=2)で1回fitして所要時間を測る ---
_t_probe = time.time()
_probe_clf = TabICLClassifier(device=DEVICE, random_state=42, n_estimators=2, batch_size=1)
_probe_clf.fit(Mtr, ytr)
_probe_clf.predict_proba(Mva)
del _probe_clf
gc.collect()
_probe_sec = time.time() - _t_probe
print(f"タイミングプローブ: n_estimators=2で {_probe_sec:.0f}秒"
      f"（目安: 1trial ≈ {_probe_sec * (N_ESTIMATORS_MAX/2):.0f}秒、"
      f"{N_TRIALS_TUNE}試行で合計 ≈ {_probe_sec * (N_ESTIMATORS_MAX/2) * N_TRIALS_TUNE / 60:.0f}分）")


def _objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 2, N_ESTIMATORS_MAX)
    clf = TabICLClassifier(device=DEVICE, random_state=42, n_estimators=n_estimators, batch_size=1)
    clf.fit(Mtr, ytr)
    p = clf.predict_proba(Mva)[:, 1]
    del clf
    gc.collect()
    return log_loss(yva, p)


t0 = time.time()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(_objective, n_trials=N_TRIALS_TUNE, show_progress_bar=False)
logger.info(f"Optuna完了 ({time.time()-t0:.0f}秒): best_value={study.best_value:.6f}")
logger.info(f"best_params={study.best_params}")
print(f"best_value(holdout logloss) = {study.best_value:.6f}")
print(f"best_params = {study.best_params}")
print("参考: 70_の既定 n_estimators=8, batch_size=1（メモリ55%減・時間45%減のため既定から下げた値）")

TUNED_PARAMS = dict(study.best_params)
TUNED_PARAMS["batch_size"] = 1   # メモリ制約のため固定（探索しない）
with open(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_tuned_params.json", "w", encoding="utf-8") as f:
    json.dump({"params": TUNED_PARAMS, "holdout_logloss": study.best_value}, f, ensure_ascii=False, indent=2)

# ============================================================
# 2. 最終学習（Train全件、3シード平均、70_と同一シード）
# ============================================================
SEEDS_TABICL = [42, 2024, 7]

Mfull, Mtest = to_numeric_matrix(ag_full, [test_features_full], FEATS)
yfull = ag_full[TARGET_COL].values

test_preds = []
for seed in SEEDS_TABICL:
    t0 = time.time()
    clf = TabICLClassifier(device=DEVICE, random_state=seed, **TUNED_PARAMS)
    clf.fit(Mfull, yfull)
    test_preds.append(clf.predict_proba(Mtest)[:, 1])
    del clf
    gc.collect()
    logger.info(f"  seed={seed}: 完了 ({time.time()-t0:.0f}秒)")

pred_tuned = np.mean(test_preds, axis=0)

np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_tabicl_hire_fixed_tuned_testpreds.npy", pred_tuned)
sub = pd.DataFrame({ID_COL: test_ids, "定着確率": pred_tuned})
path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_tabicl_hire_fixed_tuned_submission.csv"
sub.to_csv(path, index=False, header=False)
logger.info(f"提出ファイルを保存: {path}")
print(f"\n保存: {path}")
print(sub.head())
print(f"\n✅ [{SCRIPT_NAME}] 完了。このプロセスをここで終了する（メモリを解放する）。")
