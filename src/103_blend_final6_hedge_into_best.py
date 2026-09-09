"""103_blend_final6_hedge_into_best

Private LB評価があり最終提出は自分で選べる（[[private-lb-variance-strategy]]）。ユーザーは
現最良`94_add_w25`(Public 0.495570)に加え、`74_seed_fold_scaling_on_54`の
`final6_ensemble_mean`(Public 0.501006、AutoGluonを一切含まない6モデル等重み平均)を
2つ目の最終提出候補として検討していた。

`final6_ensemble_mean`単体は現最良よりPublicで+0.005436劣り（同一レシピ再実行のノイズ床
0.0023を明確に超える、本物の性能差）、そのまま2枠目に使うと期待値を落とす。一方で
「AutoGluonに一切依存しない候補を1つ持っておく」という分散低減効果自体は
[[private-lb-variance-strategy]]の観点から本物の価値がある——現最良の22.3%を占める
AutoGluonプールCがPrivateで系統的に崩れた場合の保険になる。

そこで`85_`/`94_`と同じ非チューニング方式（重み探索なし）で、現最良にfinal6_ensemble_meanを
少量（15%）だけ追加した折衷ブレンドを作る。Public期待値をほとんど落とさずに、AutoGluon
非依存という構造的分散低減効果を部分的に得ることを狙う。

出力1ファイル: add_w15（重み探索はしていない、事前登録した単一構成）
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger

SCRIPT_NAME = "103_blend_final6_hedge_into_best"
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
NEW_MODEL_PATH = PROJECT_ROOT / "data" / "output" / "20260819" / \
    "20260819_74_seed_fold_scaling_on_54_final6_ensemble_mean.csv"
CURRENT_BEST_PATH = PROJECT_ROOT / "data" / "output" / "20260823" / \
    "20260823_94_blend_tabpfn89_into_best_add_w25_submission.csv"

PUBLIC_SCORES = {
    "new_model(74_final6_ensemble_mean, AutoGluon非依存)": 0.501006,
    "current_best(94_add_w25)": 0.495570,
}


def _load(path):
    df = pd.read_csv(path, header=None, names=[ID_COL, "p"])
    return df.set_index(ID_COL)["p"]


logger.info("=" * 60)
logger.info("2つのTest予測を読み込み中...")
new_model = _load(NEW_MODEL_PATH)
current_best = _load(CURRENT_BEST_PATH)

for name, s in [("new_model", new_model), ("current_best", current_best)]:
    assert len(s) == 2502, f"{name}: 想定外の行数 {len(s)}"
logger.info("2ファイルとも2502行を確認")

test_ids = new_model.index
current_best = current_best.reindex(test_ids)

# ============================================================
# 相関・分布の確認(多様性を先に見る、[[private-lb-variance-strategy]])
# ============================================================
logger.info("=" * 60)
corr = float(np.corrcoef(new_model.values, current_best.values)[0, 1])
logger.info(f"corr(new_model=74_final6, current_best=94_add_w25) = {corr:.4f}")
logger.info("(参考: best_submission_status.mdに既知の値0.9859の記録あり)")

for name, s in [("new_model", new_model), ("current_best", current_best)]:
    label = "new_model(74_final6_ensemble_mean, AutoGluon非依存)" if name == "new_model" else "current_best(94_add_w25)"
    logger.info(f"[分布] {name}: mean={s.mean():.4f}, std={s.std():.4f}, "
                f"Public={PUBLIC_SCORES.get(label, 'N/A')}")


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, "p": preds.values}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}, std={preds.std():.4f}）")
    return str(path)


# ============================================================
# 候補: 現最良に74_final6(AutoGluon非依存)を15%追加、Private LB向けの保険的2枠目
# ============================================================
logger.info("=" * 60)
logger.info("[候補] 現最良に74_final6を15%追加 (new_model*0.15 + current_best*0.85)...")
cand_add15 = 0.15 * new_model + 0.85 * current_best
path1 = save_submission(cand_add15, "add_w15")

logger.info("=" * 60)
logger.info("=== 出力完了 ===")
logger.info(f"  {path1}")
logger.info("(単一の事前登録済み候補。重み探索は行っていない。Private LB向けの最終提出2枠目の"
            "保険的候補として、Publicでの改善/悪化に関わらずそのまま最終候補にする想定"
            "——[[private-lb-variance-strategy]]の分散低減目的のため、Public最良化が目的ではない)")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
