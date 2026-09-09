"""94_blend_tabpfn89_into_best

`93_tabpfn_top20pct_on_77`のCB89×0.20+TabPFN89×0.80ブレンドはPublic 0.502608——
`86_`の134列版ブレンド(0.504235)を-0.001627update。局所valでは89列の方が134列より
悪化して見えたが（分解能±0.0043未満）、Publicでは逆に改善しており、TabPFN単体でも
89列版(0.504065)が134列版(0.507300)を-0.003235update した([[tabpfn-ensemble-partner]]
2026-08-23追記)。

`92_`では134列版ブレンドを現最良の総合アンサンブル(`85_`のadd_w15、Public 0.496944)に
15%/25%追加し、25%版がPublic 0.496046で新記録になった。本スクリプトはその134列版
ブレンドを、より強いこの89列版ブレンドに**置き換えて**同様に検証する
（134列版と89列版を両方混ぜるのではなく、上位互換として差し替える——相関が高いはずの
2つを両方入れても希釈するだけのため）。

出力2ファイル: add_w15 / add_w25
（85_の最良ブレンド(0.496944)に新89列版ブレンドを15%/25%追加、重み探索は行わない）
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger

SCRIPT_NAME = "94_blend_tabpfn89_into_best"
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
    "20260822_93_tabpfn_top20pct_on_77_blend_argmin_w20pct_submission.csv"
CURRENT_BEST_PATH = PROJECT_ROOT / "data" / "output" / "20260822" / \
    "20260822_85_blend_top30pct_with_v4_pool_add_w15_submission.csv"

PUBLIC_SCORES = {
    "new_blend89(cb89x0.2+tabpfn89x0.8)": 0.502608,
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
logger.info(f"corr(new_blend89, current_best) = {corr:.4f}")
logger.info("(参考: 92_のnew_blend134-current_best相関0.9715)")

for name, s in [("new_blend", new_blend), ("current_best", current_best)]:
    logger.info(f"[分布] {name}: mean={s.mean():.4f}, std={s.std():.4f}, "
                f"Public={PUBLIC_SCORES.get(name, 'N/A')}")


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, "p": preds.values}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}, std={preds.std():.4f}）")
    return str(path)


# ============================================================
# 候補1/2: 85_の最良ブレンドに新89列版ブレンドを15%/25%追加
# ============================================================
logger.info("=" * 60)
logger.info("[候補1] 85_の最良ブレンドに新89列版ブレンドを15%追加 (new_blend*0.15 + current_best*0.85)...")
cand_add15 = 0.15 * new_blend + 0.85 * current_best
path1 = save_submission(cand_add15, "add_w15")

logger.info("[候補2] 85_の最良ブレンドに新89列版ブレンドを25%追加 (new_blend*0.25 + current_best*0.75)...")
cand_add25 = 0.25 * new_blend + 0.75 * current_best
path2 = save_submission(cand_add25, "add_w25")

logger.info("=" * 60)
logger.info("=== 全2ファイル出力完了 ===")
logger.info(f"  {path1}")
logger.info(f"  {path2}")
logger.info("(いずれも非チューニングの事前登録済み候補。重み探索は行っていない。"
            "Publicで確認し、最も良かった1つを採用する——[[ensemble-oof-overfitting]]参照)")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
