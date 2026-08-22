# 独立プロセスとして実行するスクリプト。学習は一切行わない（軽量・数秒で終わる）。
# 現最良ブレンド（xxxx_v4 × 現最良プール、Public 0.497608）の xxxx_v4 部分を
# xxxx_tuned（01_tune_xxxx.pyの出力、Public 0.506457で旧xxxx_v4の0.508039を更新）に
# 差し替えるだけ。重みは元のレシピと同じ非チューニングの w=0.5。
import datetime
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/content/drive/MyDrive/jaggle_2026")
assert PROJECT_ROOT.exists(), (
    f"{PROJECT_ROOT} が見つからない。run_all.ipynb で先に drive.mount() を実行してから"
    " このスクリプトをサブプロセスとして呼び出すこと。"
)
sys.path.append(str(PROJECT_ROOT))

from common.utils.logger import get_logger

SCRIPT_NAME = "75_03_xxxx_tuned_pool_blend"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "社員ID"
OUT_ROOT = PROJECT_ROOT / "data" / "output"
W_BLEND = 0.5   # 現最良ブレンド(xxxx_v4×プール)と同じ非チューニングの中点

test_ids = pd.read_csv(PROJECT_ROOT / "data" / "input" / "employee_persona_test.csv")[ID_COL].values


def _find(pat):
    h = sorted(OUT_ROOT.glob(pat))
    return h[-1] if h else None


def _load_test(p):
    if p.suffix == ".csv":
        return pd.read_csv(p, header=None, names=[ID_COL, "p"]).set_index(ID_COL).loc[test_ids, "p"].to_numpy()
    v = np.load(p)
    assert len(v) == len(test_ids)
    return v


p_xxxx_tuned = _find("*/*_75_01_tune_xxxx_xxxx_tuned_testpreds.npy")
assert p_xxxx_tuned is not None, "xxxx_tunedが見つからない（先に01_tune_xxxx.pyを実行すること）"
p_pool = _find("*/*_pool_top150_hire_fixed_avg.csv")
assert p_pool is not None, "現最良プールが見つからない"

xxxx_tuned_test = _load_test(p_xxxx_tuned)
pool_test = _load_test(p_pool)

for label, p in [("xxxx_tuned(Public 0.506457)", p_xxxx_tuned), ("現最良プール(Public 0.508699)", p_pool)]:
    dst = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_input_{label.split('(')[0]}{p.suffix}"
    shutil.copy2(p, dst)
    msg = f"【再利用・複製】{label}: {p.name} → {dst.name}"
    print(msg)
    logger.info(msg)

corr = np.corrcoef(xxxx_tuned_test, pool_test)[0, 1]
mad = np.abs(xxxx_tuned_test - pool_test).mean()
print(f"\ncorr(xxxx_tuned, 現最良プール) = {corr:.4f}  MAD = {mad:.5f}（ノイズ床0.02122）")

blend_test = W_BLEND * xxxx_tuned_test + (1 - W_BLEND) * pool_test

sub = pd.DataFrame({ID_COL: test_ids, "定着確率": blend_test})
path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_xxxx_tuned_pool_blend_w{int(W_BLEND*100)}.csv"
sub.to_csv(path, index=False, header=False)
logger.info(f"xxxx_tuned×プールブレンドを保存: {path}")
print(f"\n保存: {path}")
print(sub.head())
print(f"\n✅ [{SCRIPT_NAME}] 完了。")
