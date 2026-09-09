"""98_blend_xxxxv4pool_cb_into_best

`97_ft_transformer_on_xxxxv4_features`は「FT-Transformer(新機構) × xxxx_v4独自プール
(新特徴量プール)」を検証する目的だったが、その副産物として**xxxx_v4本来の集約・パースのみ
から作った144列プールに対するCatBoost単体モデル**（`cb_classifier`、局所val=0.528230）も
生成済みである。97_の結論はFT-Transformer機構が弱い（CatBoost比+0.0815、argminブレンド
w_cb=1.00）というものだったが、これは「機構」についての結論であり、「xxxx_v4独自プール
自体がCatBoostとのアンサンブル相手として無価値」ということは意味しない
（[[ft-transformer-96-action-plan]]参照）。

フォーク調査で確認済みの通り、xxxx_v4独自プール（素朴な平均/標準偏差/最終月値集約、独自regex
メモパース、L2xM交互作用・クラスタリング・四半期特徴量・skew/kurtosis/iqrなし）は現行の
447列パイプライン（54_l2_m_interaction系）とほぼ完全に特徴量エンジニアリングが独立している。
すでに`82_`（xxxx_v4由来8列を447列に追加、455列版、Public 0.508039）と`85_`（xxxx_v4本来の
CatBoost単体submission、xxxx_v4×プールのブレンドの一部としてPublic 0.497608に貢献）で、
xxxx_v4系統モデルは総合ブレンドに一貫して価値を追加してきた実績がある
([[best-submission-status]])。97_のcb_classifierは、xxxx_v4本来の集約ロジックを**FULL構成
（重要度による選択なし、144列そのまま）**でCatBoostに通した、これまでで最も"素"に近い
xxxx_v4系統モデルであり、まだ現最良ブレンドに混ぜたことがない独立した予測源である。

本スクリプトは`85_`/`94_`と同じ方針——OOFが現最良ブレンド側に保存されていないため重み探索は
せず、事前登録した少数の非チューニング候補だけを試し、Publicで確認する
（[[ensemble-oof-overfitting]]の「重み探索は過学習する」を避ける）。

出力3ファイル: add_w10 / add_w15 / add_w25
（97_のcb_classifier単体val(0.528230)は85_/94_で過去に混ぜた相手(0.509500/0.502608)より
弱いため、通常の15%/25%に加えてより控えめな10%も候補に加える）
"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger

SCRIPT_NAME = "98_blend_xxxxv4pool_cb_into_best"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 実験開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "社員ID"

# ============================================================
# 入力ファイル(すべてPublic確認済み、new_modelのみ局所OOFのみ既知)
# ============================================================
NEW_MODEL_PATH = PROJECT_ROOT / "data" / "output" / "20260824" / \
    "20260824_97_ft_transformer_on_xxxxv4_features_cb_classifier_submission.csv"
CURRENT_BEST_PATH = PROJECT_ROOT / "data" / "output" / "20260823" / \
    "20260823_94_blend_tabpfn89_into_best_add_w25_submission.csv"

PUBLIC_SCORES = {
    "new_model(97_xxxxv4pool_cb, 局所val=0.528230)": None,
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
logger.info(f"corr(new_model=97_xxxxv4pool_cb, current_best=94_add_w25) = {corr:.4f}")
logger.info("(参考: 85_でのnew_model-current_best相関は既知の値と比較すること。"
            "xxxx_v4系統は独立特徴量エンジニアリングのため低相関が期待される)")

for name, s in [("new_model", new_model), ("current_best", current_best)]:
    label = "new_model(97_xxxxv4pool_cb, 局所val=0.528230)" if name == "new_model" else "current_best(94_add_w25)"
    logger.info(f"[分布] {name}: mean={s.mean():.4f}, std={s.std():.4f}, "
                f"Public={PUBLIC_SCORES.get(label, 'N/A')}")


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, "p": preds.values}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}, std={preds.std():.4f}）")
    return str(path)


# ============================================================
# 候補1/2/3: 現最良ブレンドに97_のxxxx_v4プールCatBoostを10%/15%/25%追加
# ============================================================
logger.info("=" * 60)
logger.info("[候補1] 現最良ブレンドにnew_modelを10%追加 (new_model*0.10 + current_best*0.90)...")
cand_add10 = 0.10 * new_model + 0.90 * current_best
path1 = save_submission(cand_add10, "add_w10")

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
