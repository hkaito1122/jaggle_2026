# 独立プロセスとして実行するスクリプト。00_build_features.py が保存した441列キャッシュ
# (data/output/_checkpoints/75_tuned_ensemble_cache_fm/) の hire_fixed(78列)を使い、
# TabPFN v2 の少数のハイパーパラメータ(n_estimators等)をOptunaで探索する。
# GPU推奨だがCPUでも動く設計（device=cpuを検出したら探索を自動的に軽量化する。
# 2026-08-20追記: CPU対応）。
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn==2.2.1", "optuna"], check=True)

import datetime
import json
import time
import warnings
from importlib.metadata import version as _pkgver
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import torch
import tabpfn
from tabpfn import TabPFNClassifier
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

# ライセンス確認（Prior Labs License v1.1系=2.x系のみ許可。参加規約 第2条8項）
_disk, _live = _pkgver("tabpfn"), tabpfn.__version__
if not _live.startswith("2."):
    raise RuntimeError(
        f"tabpfn: ディスク {_disk} / 実行中 {_live}。v2系(2.x)が必要"
        "（2.5/3系は非商用ライセンスで使用不可）。ランタイムを再起動してください。")

SCRIPT_NAME = "75_01_tune_tabpfn_hire_fixed"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL, TARGET_COL = "社員ID", "10年定着ラベル"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"tabpfn {_live} / device = {DEVICE}")
if DEVICE == "cpu":
    print("⚠️ GPUが無い。CPU向けに探索を軽量化する（試行数・n_estimators上限を下げる）。")

# CPUでは1trialのコストが線形に効くn_estimatorsの上限と試行数を下げる。
# GPUなら従来通りの強度で探索する。
if DEVICE == "cpu":
    N_TRIALS_TUNE = 8
    N_ESTIMATORS_MAX = 6
else:
    N_TRIALS_TUNE = 25
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


def to_tabpfn_matrix(train_df, other_dfs, feats):
    """TabPFN用の行列を作る。NaNは埋めない。カテゴリは序数コード化して位置を返す（63_/70_と同一）。"""
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
    cat_idx = [feats.index(c) for c in obj]
    return out, cat_idx


def _make_clf(params, seed, cat_idx):
    kw = dict(device=DEVICE, random_state=seed, **params)
    for extra in ({"categorical_features_indices": cat_idx, "ignore_pretraining_limits": True},
                  {"categorical_features_indices": cat_idx}, {}):
        try:
            return TabPFNClassifier(**kw, **extra)
        except TypeError:
            continue
    raise RuntimeError("TabPFNClassifierの初期化に失敗した")


# ============================================================
# 1. Optuna探索（ホールドアウト単発・小さい探索空間）
# ============================================================
(Mtr, Mva), cat_idx = to_tabpfn_matrix(ag_train_80b, [ag_val_surv], FEATS)
ytr = ag_train_80b[TARGET_COL].values
yva = ag_val_surv[TARGET_COL].values

# --- 実測タイミングプローブ: 最小構成(n_estimators=1)で1回fitして所要時間を測る ---
# 探索を始める前に、CPU/GPUどちらでもtrialあたりの目安時間を実測して表示する。
_t_probe = time.time()
_probe_clf = _make_clf({"n_estimators": 1}, seed=42, cat_idx=cat_idx)
_probe_clf.fit(Mtr, ytr)
_probe_clf.predict_proba(Mva)
_probe_sec = time.time() - _t_probe
print(f"タイミングプローブ: n_estimators=1で {_probe_sec:.0f}秒"
      f"（実際のtrialはn_estimatorsに比例して長くなる。目安: 1trial ≈ {_probe_sec * (N_ESTIMATORS_MAX/2):.0f}秒、"
      f"{N_TRIALS_TUNE}試行で合計 ≈ {_probe_sec * (N_ESTIMATORS_MAX/2) * N_TRIALS_TUNE / 60:.0f}分）")


def _objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 2, N_ESTIMATORS_MAX),
        "softmax_temperature": trial.suggest_float("softmax_temperature", 0.5, 1.5),
        "balance_probabilities": trial.suggest_categorical("balance_probabilities", [True, False]),
        "average_before_softmax": trial.suggest_categorical("average_before_softmax", [True, False]),
    }
    clf = _make_clf(params, seed=42, cat_idx=cat_idx)
    clf.fit(Mtr, ytr)
    p = clf.predict_proba(Mva)[:, 1]
    return log_loss(yva, p)


t0 = time.time()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(_objective, n_trials=N_TRIALS_TUNE, show_progress_bar=False)
logger.info(f"Optuna完了 ({time.time()-t0:.0f}秒): best_value={study.best_value:.6f}")
logger.info(f"best_params={study.best_params}")
print(f"best_value(holdout logloss) = {study.best_value:.6f}")
print(f"best_params = {study.best_params}")
print("参考: 63_のデフォルト設定(n_estimators等はライブラリ既定値のまま)でのTabPFN(top150) val ≈ 0.507569"
      "（列数が違うため直接比較にはならない）")

TUNED_PARAMS = study.best_params
with open(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_tuned_params.json", "w", encoding="utf-8") as f:
    json.dump({"params": TUNED_PARAMS, "holdout_logloss": study.best_value}, f, ensure_ascii=False, indent=2)

# ============================================================
# 2. 最終学習（Train全件、3シード平均、63_/70_と同一シード）
# ============================================================
SEEDS_TABPFN = [42, 2024, 7]

(Mfull, Mtest), cat_idx_full = to_tabpfn_matrix(ag_full, [test_features_full], FEATS)
yfull = ag_full[TARGET_COL].values

test_preds = []
for seed in SEEDS_TABPFN:
    t0 = time.time()
    clf = _make_clf(TUNED_PARAMS, seed=seed, cat_idx=cat_idx_full)
    clf.fit(Mfull, yfull)
    test_preds.append(clf.predict_proba(Mtest)[:, 1])
    logger.info(f"  seed={seed}: 完了 ({time.time()-t0:.0f}秒)")

pred_tuned = np.mean(test_preds, axis=0)

np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_tabpfn_hire_fixed_tuned_testpreds.npy", pred_tuned)
sub = pd.DataFrame({ID_COL: test_ids, "定着確率": pred_tuned})
path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_tabpfn_hire_fixed_tuned_submission.csv"
sub.to_csv(path, index=False, header=False)
logger.info(f"提出ファイルを保存: {path}")
print(f"\n保存: {path}")
print(sub.head())
print(f"\n✅ [{SCRIPT_NAME}] 完了。このプロセスをここで終了する（メモリを解放する）。")
