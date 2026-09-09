"""95_blend_autogluon_into_best

現最良の総合アンサンブル（`94_`のadd_w25、Public 0.495570）に、既存のAutoGluon予測を
さらに追加する価値があるかを検証する。2つのソースを並行して試す:

  - poolC: full441列、AutoGluon 8回学習の平均、Public 0.514050
    （[[refit-chaos-noise-floor]]/[[private-lb-variance-strategy]]の方針で、
    単発の最良スコアではなく分散を抑えた代表値として採用してきた既定路線）
  - ag50: `50_autogluon_memofix`の単発実行、Public 0.513108（poolCより実測は良いが、
    `64_autogluon_more_seeds.ipynb`で「運の良い1回の引き」と位置付けられている）

どちらがPrivate LBでの実力に近いかは分からないため、両方とも試して比較する
（ユーザー判断）。現最良の土台(85_)には既にプールCが70%相当で組み込まれているため、
相関はかなり高くなる可能性が高い。重み探索はせず、事前登録した候補だけを試す
（[[ensemble-oof-overfitting]]）。

出力4ファイル: poolc_w15 / poolc_w25 / ag50_w15 / ag50_w25
（現最良ブレンドに各ソースを15%/25%追加、重み探索は行わない）
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger

SCRIPT_NAME = "95_blend_autogluon_into_best"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 実験開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "社員ID"

# ============================================================
# 入力ファイル(すべてPublic確認済み)
# ============================================================
POOLC_PATH = PROJECT_ROOT / "data" / "output" / "20260816" / "20260816_pool_poolC_weighted.csv"
AG50_PATH = PROJECT_ROOT / "data" / "output" / "20260813" / \
    "20260813_50_autogluon_memofix_AG50_full441_weighted.csv"
CURRENT_BEST_PATH = PROJECT_ROOT / "data" / "output" / "20260823" / \
    "20260823_94_blend_tabpfn89_into_best_add_w25_submission.csv"

PUBLIC_SCORES = {
    "poolc(8run_avg)": 0.514050,
    "ag50(single_run)": 0.513108,
    "current_best(94_add_w25)": 0.495570,
}


def _load(path):
    df = pd.read_csv(path, header=None, names=[ID_COL, "p"])
    return df.set_index(ID_COL)["p"]


logger.info("=" * 60)
logger.info("3つのTest予測を読み込み中...")
poolc = _load(POOLC_PATH)
ag50 = _load(AG50_PATH)
current_best = _load(CURRENT_BEST_PATH)

for name, s in [("poolc", poolc), ("ag50", ag50), ("current_best", current_best)]:
    assert len(s) == 2502, f"{name}: 想定外の行数 {len(s)}"
logger.info("3ファイルとも2502行を確認")

test_ids = current_best.index
poolc = poolc.reindex(test_ids)
ag50 = ag50.reindex(test_ids)

# ============================================================
# 相関・分布の確認(多様性を先に見る、[[private-lb-variance-strategy]])
# ============================================================
logger.info("=" * 60)
corr_poolc = float(np.corrcoef(poolc.values, current_best.values)[0, 1])
corr_ag50 = float(np.corrcoef(ag50.values, current_best.values)[0, 1])
corr_poolc_ag50 = float(np.corrcoef(poolc.values, ag50.values)[0, 1])
logger.info(f"corr(poolc, current_best) = {corr_poolc:.4f}")
logger.info(f"corr(ag50, current_best) = {corr_ag50:.4f}")
logger.info(f"corr(poolc, ag50) = {corr_poolc_ag50:.4f}")
logger.info("(参考: 現最良の土台にはプールCが既に70%相当で組み込まれているため、"
            "相関がかなり高く出る可能性が高い。それでも非チューニングで確認する)")

for name, s in [("poolc", poolc), ("ag50", ag50), ("current_best", current_best)]:
    logger.info(f"[分布] {name}: mean={s.mean():.4f}, std={s.std():.4f}, "
                f"Public={PUBLIC_SCORES.get(name, 'N/A')}")


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, "p": preds.values}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}, std={preds.std():.4f}）")
    return str(path)


# ============================================================
# 候補1/2: 現最良ブレンドにプールC(8回平均、分散抑制路線)を15%/25%追加
# ============================================================
logger.info("=" * 60)
logger.info("[候補1] 現最良ブレンドにpoolCを15%追加 (poolc*0.15 + current_best*0.85)...")
path1 = save_submission(0.15 * poolc + 0.85 * current_best, "poolc_w15")

logger.info("[候補2] 現最良ブレンドにpoolCを25%追加 (poolc*0.25 + current_best*0.75)...")
path2 = save_submission(0.25 * poolc + 0.75 * current_best, "poolc_w25")

# ============================================================
# 候補3/4: 現最良ブレンドにAG50(単発実行、実測最良路線)を15%/25%追加
# ============================================================
logger.info("=" * 60)
logger.info("[候補3] 現最良ブレンドにAG50を15%追加 (ag50*0.15 + current_best*0.85)...")
path3 = save_submission(0.15 * ag50 + 0.85 * current_best, "ag50_w15")

logger.info("[候補4] 現最良ブレンドにAG50を25%追加 (ag50*0.25 + current_best*0.75)...")
path4 = save_submission(0.25 * ag50 + 0.75 * current_best, "ag50_w25")

logger.info("=" * 60)
logger.info("=== 全4ファイル出力完了 ===")
logger.info(f"  {path1}")
logger.info(f"  {path2}")
logger.info(f"  {path3}")
logger.info(f"  {path4}")
logger.info("(いずれも非チューニングの事前登録済み候補。重み探索は行っていない。"
            "Publicで確認し、最も良かった1つを採用する——[[ensemble-oof-overfitting]]参照)")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
