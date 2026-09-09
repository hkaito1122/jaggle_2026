"""108_extract_autogluon_oof

`107_`の7モデルスタッキングは現最良を更新した(Public 0.494517)が、xxxx_v4・AutoGluonプールC
という94_add_w25の主要成分（合計約54%のウェイト）はOOFを保存していないため、まだこの
スタッキング枠組みに統合できていない。

xxxx_v4自体は`72_reference_pipeline_standalone`のR_full（著者reference独立再現）が既に
107_で採用され改善の主因になったため、xxxx_v4系統は間接的にカバーできている。残る大きな
未統合成分はAutoGluonプールC（441列・旧特徴量パイプライン・bagging+stacking、Public 0.514050）
である。

**重要な発見**: AutoGluonのbaggedモデルは`predict_proba_oof()`メソッドで**再学習不要・
追加のColab計算コストほぼゼロ**でOOF予測を取得できる（内部で8-foldバギングした際に
自動的に計算・保存されている）。`saved_models/`配下に`51_autogluon_catboost_bias`の
`full441_full`（プールC構成メンバーの1つ、単体Public 0.514263、441列旧パイプライン）の
predictor.pklがGoogle Driveに残っているため、これを読み込むだけでOOFを取得できる。

**行の並び順に関する重要な注意**: AutoGluonに渡した学習データは`社員ID`列を持たない
（`_frame()`が`reset_index(drop=True)`するため）。したがって`predict_proba_oof()`が返す
配列の行順は、元の`ag_full`（`51_`のprepare_split(1.0)が返す、**入社日でソートされた**
DataFrame）の行順と一致する。これは`train_persona.sort_values("入社日")`の順序と同一
（51_のprepare_split内の変換順序を再現するため、pd.to_datetime変換も同じ順序で行う）。
特徴量そのものを再構築する必要はなく、IDの並び順さえ再現できればよい。

出力: r_poolc51_oofpreds.npy（train_ids順、107_のスタッキングにそのまま追加できる形式）
※ Test予測は`51_`が既に保存済み(`20260814_51_autogluon_catboost_bias_AG51_full441_weighted_testpreds.npy`、
  test_persona順で他のtest予測と揃っているためそのまま流用、本スクリプトでは再計算しない)
"""
import datetime
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger
from common.utils.seed import seed_everything

SEED = 42
seed_everything(seed=SEED)

TARGET_COL = "10年定着ラベル"
ID_COL = "社員ID"

SCRIPT_NAME = "108_extract_autogluon_oof"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 実験開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIR = PROJECT_ROOT / "data" / "input"

# ============================================================
# 51_のprepare_split(1.0)が作るag_fullの行順(=入社日ソート順)だけを再現する
# 特徴量の再構築は不要——predict_proba_oof()はAutoGluon内部で既に計算済みのOOFを返すため
# ============================================================
train_persona = pd.read_csv(INPUT_DIR / "employee_persona_train.csv")
train_persona["入社日"] = pd.to_datetime(train_persona["入社日"])  # 51_のcell17と同じ変換順序
sorted_persona = train_persona.sort_values("入社日")
id_order_hiredate = sorted_persona[ID_COL].values  # ag_full.indexと同一のはず

train_ids = pd.read_csv(INPUT_DIR / "employee_persona_train.csv")[ID_COL].values  # 元CSV順(107_等の他OOFと揃える基準)
logger.info(f"train_ids(CSV順): {len(train_ids)}件 / id_order_hiredate(入社日順): {len(id_order_hiredate)}件")
assert set(train_ids) == set(id_order_hiredate)

# ============================================================
# AutoGluon predictor読み込み(51_の保存済みfull441_full、再学習しない)
# ============================================================
logger.info("=" * 60)
logger.info("AutoGluon predictor (51_/full441_full) を読み込み中...")
from autogluon.tabular import TabularPredictor

PREDICTOR_PATH = PROJECT_ROOT / "saved_models" / "20260814" / "51_autogluon_catboost_bias" / "full441_full"
# ColabのデフォルトPythonが3.12→3.13に上がり、保存時(3.12)とのバージョン不一致ガードに引っかかる。
# predict_proba_oof()の呼び出しのみ(再学習・再シリアライズなし)なので安全にバイパスする。
predictor = TabularPredictor.load(str(PREDICTOR_PATH), require_py_version_match=False)

try:
    lb = predictor.leaderboard(silent=True)
except TypeError:
    lb = predictor.leaderboard()
weighted_names = lb[lb["model"].str.startswith("WeightedEnsemble")]["model"].tolist()
assert weighted_names, "WeightedEnsembleモデルがleaderboardに見つからない"
model_name = weighted_names[0]
logger.info(f"使用モデル: {model_name}（51_がTest予測保存時に使ったのと同じ選択ロジック）")

# ============================================================
# OOF予測を取得(再学習なし、内部で既に計算済み)
# ============================================================
logger.info("predict_proba_oof()でOOF予測を取得中(再学習なし)...")
oof_df = predictor.predict_proba_oof(model=model_name, as_multiclass=True)
pos_class = predictor.positive_class if hasattr(predictor, "positive_class") else 1
if pos_class not in oof_df.columns:
    pos_class = 1 if 1 in oof_df.columns else oof_df.columns[-1]
oof_values = oof_df[pos_class].values
logger.info(f"OOF予測 shape: {oof_values.shape}（{len(id_order_hiredate)}件のはず）")
assert len(oof_values) == len(id_order_hiredate), (
    f"OOF行数({len(oof_values)})がag_fullの行数({len(id_order_hiredate)})と一致しない。"
    "内部でレア クラス除外等が発生した可能性——train_dataを渡すpredict_proba_oof(train_data=...)で"
    "欠損行を埋める必要がある。"
)

# ============================================================
# 入社日順→CSV順(train_ids順)への並び替え、他のOOFと揃える
# ============================================================
oof_series_hiredate_order = pd.Series(oof_values, index=id_order_hiredate)
oof_csv_order = oof_series_hiredate_order.reindex(train_ids).values
assert not np.isnan(oof_csv_order).any(), "並び替え後にNaNが発生している(IDの対応漏れ)"

# ============================================================
# サニティチェック: 生存者OOF loglossが51_の既知の性能水準(Public 0.514263)と整合するか
# ============================================================
train_monthly = pd.read_csv(INPUT_DIR / "employee_monthly_train.csv")
early = set(train_monthly.loc[train_monthly["月末在籍状態"] == "退職", ID_COL].unique())
y = pd.read_csv(INPUT_DIR / "employee_persona_train.csv")[TARGET_COL].values
surv_mask = np.array([tid not in early for tid in train_ids])
oof_val = log_loss(y[surv_mask], oof_csv_order[surv_mask])
logger.info(f"[サニティチェック] 生存者OOF logloss = {oof_val:.6f}"
            f"（51_のPublic weighted=0.514263と近い水準のはず。大きく外れていたら並び替えを疑うこと）")

# ============================================================
# 保存
# ============================================================
out_path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_r_poolc51_oofpreds.npy"
np.save(out_path, oof_csv_order)
logger.info(f"保存: {out_path.name}")
logger.info("(Test予測は51_の保存済み20260814_51_autogluon_catboost_bias_AG51_full441_weighted_testpreds.npy"
            "をそのまま流用すること——test_persona順で他のtest予測と揃っている)")

logger.info("=" * 60)
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
