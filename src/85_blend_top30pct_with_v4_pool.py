"""85_blend_top30pct_with_v4_pool

`84_nested_feature_selection_on_77`のTOP30PCT_nested（447列中134列にネスト選択でプルーニング
した単層CatBoost）はPublic 0.509500——単層CatBoost(54_系)の新記録([[best-submission-status]])。
xxxx_v4単体(0.508039)・現最良プール(top150ブレンド×hire_fixed, 0.508699と同水準)にかなり近い
強さで、かつ特徴量パイプライン・列数(134列)ともに両者と大きく異なる構造のため、
アンサンブル相手として試す価値が高いと判断し、3モデルのブレンドを検証する。

このスクリプトは以下を行う:
  1. 3つのTest予測(新CatBoost・xxxx_v4・現最良プール)を読み込み、ペアワイズ相関を計算・記録
     （[[private-lb-variance-strategy]]・[[tabpfn-ensemble-partner]]の「多様性を先に確認する」流儀）
  2. 現最良ブレンド(xxxx_v4×プール, w=0.5, Public 0.497608)が実際に0.5:0.5の単純平均と
     一致することをサニティチェック
  3. 新CatBoostのOOF・xxxx_v4のOOF・プールのOOFはいずれも保存されていないため
     （[[blend-curve-beats-val-margin-gate]]が要求するargmin探索はできない）、`72_`が
     xxxx_v4×プールの最初のブレンドで採用したのと同じ方針——**重み探索はせず、
     事前登録した少数の非チューニング候補だけを試す**（[[ensemble-oof-overfitting]]の
     「重み探索は過学習する」を避ける）
  4. 候補3つを生成して保存: 3-way等重み平均 / 現最良ブレンドに新モデルを15%追加 /
     現最良ブレンドに新モデルを25%追加

出力3ファイル: equal3way / add_w15 / add_w25
（すべて単一の事前登録済み構成、以後の重み探索は行わない）
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger

SCRIPT_NAME = "85_blend_top30pct_with_v4_pool"
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
NEW_MODEL_PATH = PROJECT_ROOT / "data" / "output" / "20260822" / \
    "20260822_84_nested_feature_selection_on_77_top30pct_nested_classifier_submission.csv"
XXXX_V4_PATH = PROJECT_ROOT / "data" / "output" / "20260817" / "20260817_xxxx_v4_submission.csv"
POOL_PATH = PROJECT_ROOT / "data" / "output" / "20260816" / "20260816_pool_top150_hire_fixed_avg.csv"
CURRENT_BEST_PATH = PROJECT_ROOT / "data" / "output" / "20260818" / "20260818_72_v4_pool_blend_w50.csv"

PUBLIC_SCORES = {
    "new_model(84_top30pct_nested)": 0.509500,
    "xxxx_v4": 0.508039,
    "pool(top150_hire_fixed)": 0.508699,
    "current_best(v4xpool_w50)": 0.497608,
}


def _load(path):
    df = pd.read_csv(path, header=None, names=[ID_COL, "p"])
    return df.set_index(ID_COL)["p"]


logger.info("=" * 60)
logger.info("4つのTest予測を読み込み中...")
new_model = _load(NEW_MODEL_PATH)
xxxx_v4 = _load(XXXX_V4_PATH)
pool = _load(POOL_PATH)
current_best = _load(CURRENT_BEST_PATH)

for name, s in [("new_model", new_model), ("xxxx_v4", xxxx_v4), ("pool", pool), ("current_best", current_best)]:
    assert len(s) == 2502, f"{name}: 想定外の行数 {len(s)}"
logger.info("4ファイルとも2502行を確認")

# 順序を揃える(念のため)
test_ids = new_model.index
xxxx_v4 = xxxx_v4.reindex(test_ids)
pool = pool.reindex(test_ids)
current_best = current_best.reindex(test_ids)

# ============================================================
# サニティチェック: current_best が本当に0.5*xxxx_v4+0.5*poolか
# ============================================================
logger.info("=" * 60)
recon = 0.5 * xxxx_v4 + 0.5 * pool
max_diff = (recon - current_best).abs().max()
logger.info(f"[サニティチェック] current_best vs 0.5*xxxx_v4+0.5*pool の最大絶対差: {max_diff:.8f}")
assert max_diff < 1e-6, "current_bestが0.5:0.5の単純平均と一致しない(別バージョンのファイルの可能性)"
logger.info("  → 一致確認OK")

# ============================================================
# ペアワイズ相関(多様性の確認)
# ============================================================
logger.info("=" * 60)
logger.info("[相関] ペアワイズPearson相関:")
pairs = [
    ("new_model", new_model, "xxxx_v4", xxxx_v4),
    ("new_model", new_model, "pool", pool),
    ("new_model", new_model, "current_best", current_best),
    ("xxxx_v4", xxxx_v4, "pool", pool),
]
for name_a, a, name_b, b in pairs:
    corr = np.corrcoef(a.values, b.values)[0, 1]
    logger.info(f"  corr({name_a}, {name_b}) = {corr:.4f}")

logger.info("(参考: xxxx_v4-pool相関は0.90程度と既知[[best-submission-status]]。"
            "new_modelがこれより低い相関ならアンサンブル相手として有望)")

# ============================================================
# 予測分布の確認
# ============================================================
logger.info("=" * 60)
for name, s in [("new_model", new_model), ("xxxx_v4", xxxx_v4), ("pool", pool), ("current_best", current_best)]:
    logger.info(f"[分布] {name}: mean={s.mean():.4f}, std={s.std():.4f}, "
                f"Public={PUBLIC_SCORES.get(name if name!='current_best' else 'current_best(v4xpool_w50)', 'N/A')}")


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, "p": preds.values}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}, std={preds.std():.4f}）")
    return str(path)


# ============================================================
# 候補1: 3-way等重み平均(new_model, xxxx_v4, poolを1/3ずつ)
# ============================================================
logger.info("=" * 60)
logger.info("[候補1] 3-way等重み平均 (new_model:xxxx_v4:pool = 1:1:1)...")
cand_equal3way = (new_model + xxxx_v4 + pool) / 3.0
path1 = save_submission(cand_equal3way, "equal3way")

# ============================================================
# 候補2/3: 現最良ブレンド(xxxx_v4×プール w50)にnew_modelを15%/25%追加
# ============================================================
logger.info("=" * 60)
logger.info("[候補2] 現最良ブレンドにnew_modelを15%追加 (new_model*0.15 + current_best*0.85)...")
cand_add15 = 0.15 * new_model + 0.85 * current_best
path2 = save_submission(cand_add15, "add_w15")

logger.info("[候補3] 現最良ブレンドにnew_modelを25%追加 (new_model*0.25 + current_best*0.75)...")
cand_add25 = 0.25 * new_model + 0.75 * current_best
path3 = save_submission(cand_add25, "add_w25")

logger.info("=" * 60)
logger.info("=== 全3ファイル出力完了 ===")
logger.info(f"  {path1}")
logger.info(f"  {path2}")
logger.info(f"  {path3}")
logger.info("(いずれも非チューニングの事前登録済み候補。重み探索は行っていない。"
            "Publicで確認し、最も良かった1つを採用する。複数試して一番良いものを"
            "さらに探索する運用はしないこと——[[ensemble-oof-overfitting]]参照)")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
