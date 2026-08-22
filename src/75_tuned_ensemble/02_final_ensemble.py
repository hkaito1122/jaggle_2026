# 独立プロセスとして実行するスクリプト。学習は一切行わない（軽量・数秒で終わる）。
# 01_tune_*.py 5本すべての出力（54_tuned / xxxx_tuned / TabPFN・TabICL(hire_fixed)_tuned /
# TabPFN(top150)_tuned）を読み込んで単純平均するだけ。
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

SCRIPT_NAME = "75_02_final_ensemble"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "社員ID"
OUT_ROOT = PROJECT_ROOT / "data" / "output"

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


SOURCES = {
    "54_tuned":                 "*/*_75_01_tune_54_54tuned_testpreds.npy",
    "xxxx_tuned":               "*/*_75_01_tune_xxxx_xxxx_tuned_testpreds.npy",
    "TabPFN(hire_fixed)_tuned": "*/*_75_01_tune_tabpfn_hire_fixed_tabpfn_hire_fixed_tuned_testpreds.npy",
    "TabICL(hire_fixed)_tuned": "*/*_75_01_tune_tabicl_hire_fixed_tabicl_hire_fixed_tuned_testpreds.npy",
    "TabPFN(top150)_tuned":     "*/*_75_01_tune_tabpfn_top150_tabpfn_top150_tuned_testpreds.npy",
}

components = {}
for label, pat in SOURCES.items():
    p = _find(pat)
    assert p is not None, f"見つからない: {pat}（先に該当スクリプト/ノートブックを実行すること）"
    components[label] = _load_test(p)
    dst = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_input_{label.replace('(', '_').replace(')', '')}{p.suffix}"
    shutil.copy2(p, dst)
    msg = f"【再利用・複製】{label}: {p.name} → {dst.name}"
    print(msg)
    logger.info(msg)

names = list(components)
print("\n=== 相関行列（参考、ノイズ床0.02122） ===")
print(f'{"":22s}' + "".join(f"{n[:14]:>16s}" for n in names))
for a in names:
    row = f"{a:22s}"
    for b in names:
        row += f"{np.corrcoef(components[a], components[b])[0,1]:16.4f}"
    print(row)

ensemble_mean = np.mean([components[n] for n in names], axis=0)

sub = pd.DataFrame({ID_COL: test_ids, "定着確率": ensemble_mean})
path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_final{len(names)}_ensemble_mean.csv"
sub.to_csv(path, index=False, header=False)
logger.info(f"最終アンサンブル({len(names)}モデル等重み平均)を保存: {path}")
print(f"\n保存: {path}")
print(f"構成({len(names)}モデル等重み平均): {names}")
print(sub.head())
print(f"\n✅ [{SCRIPT_NAME}] 完了。")
