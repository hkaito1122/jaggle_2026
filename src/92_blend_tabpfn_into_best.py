"""92_blend_tabpfn_into_best

`86_tabpfn_on_84_top30pct`のCB134×0.20+TabPFN134×0.80ブレンドはPublic 0.504235——
84_のCB134単体(0.509500)比-0.005265、TabPFN134単体(0.507300)比-0.003065のいずれも
同一レシピ再実行ノイズ床(0.0023)を上回る改善で、このプロジェクトで初めてTabPFNが
単体でCatBoostを上回った（[[tabpfn-ensemble-partner]]2026-08-22追記）。

このスクリプトは、この新しいTabPFN×CatBoostブレンドを現最良の総合アンサンブル
（`85_`のadd_w15、xxxx_v4×プール×0.85+84_CB134×0.15、Public 0.496944）に
組み込む価値があるかを検証する。85_と同じ流儀——OOFが無いため重み探索はせず、
事前登録した少数の非チューニング候補だけを試す（[[ensemble-oof-overfitting]]）。

出力2ファイル: add_w15 / add_w25
（現最良ブレンドに新TabPFNブレンドを15%/25%追加、重み探索は行わない）
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger

SCRIPT_NAME = "92_blend_tabpfn_into_best"
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
NEW_BLEND_PATH = PROJECT_ROOT / "data" / "output" / "20260822" / \
    "20260822_86_tabpfn_on_84_top30pct_blend_argmin_w20pct_submission.csv"
CURRENT_BEST_PATH = PROJECT_ROOT / "data" / "output" / "20260822" / \
    "20260822_85_blend_top30pct_with_v4_pool_add_w15_submission.csv"

PUBLIC_SCORES = {
    "new_blend(cb134x0.2+tabpfn134x0.8)": 0.504235,
    "current_best(85_add_w15)": 0.496944,
}


def _load(path):
    df = pd.read_csv(path, header=None, names=[ID_COL, "p"])
    return df.set_index(ID_COL)["p"]


logger.info("=" * 60)
logger.info("2つのTest予測を読み込み中...")
new_blend = _load(NEW_BLEND_PATH)
current_best = _load(CURRENT_BEST_PATH)

for name, s in [("new_blend", new_blend), ("current_best", current_best)]:
    assert len(s) == 2502, f"{name}: 想定外の行数 {len(s)}"
logger.info("2ファイルとも2502行を確認")

test_ids = new_blend.index
current_best = current_best.reindex(test_ids)

# ============================================================
# 相関・分布の確認(多様性を先に見る、[[private-lb-variance-strategy]])
# ============================================================
logger.info("=" * 60)
corr = float(np.corrcoef(new_blend.values, current_best.values)[0, 1])
logger.info(f"corr(new_blend, current_best) = {corr:.4f}")
logger.info("(参考: xxxx_v4-pool相関0.90、84_top30pct-xxxx_v4相関0.8928、"
            "84_top30pct-プール相関0.9691、[[best-submission-status]])")

for name, s in [("new_blend", new_blend), ("current_best", current_best)]:
    logger.info(f"[分布] {name}: mean={s.mean():.4f}, std={s.std():.4f}, "
                f"Public={PUBLIC_SCORES.get(name, 'N/A')}")


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, "p": preds.values}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}, std={preds.std():.4f}）")
    return str(path)


# ============================================================
# 候補1/2: 現最良ブレンド(85_ add_w15)に新TabPFNブレンドを15%/25%追加
# ============================================================
logger.info("=" * 60)
logger.info("[候補1] 現最良ブレンドに新TabPFNブレンドを15%追加 (new_blend*0.15 + current_best*0.85)...")
cand_add15 = 0.15 * new_blend + 0.85 * current_best
path1 = save_submission(cand_add15, "add_w15")

logger.info("[候補2] 現最良ブレンドに新TabPFNブレンドを25%追加 (new_blend*0.25 + current_best*0.75)...")
cand_add25 = 0.25 * new_blend + 0.75 * current_best
path2 = save_submission(cand_add25, "add_w25")

logger.info("=" * 60)
logger.info("=== 全2ファイル出力完了 ===")
logger.info(f"  {path1}")
logger.info(f"  {path2}")
logger.info("(いずれも非チューニングの事前登録済み候補。重み探索は行っていない。"
            "Publicで確認し、最も良かった1つを採用する——[[ensemble-oof-overfitting]]参照)")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
