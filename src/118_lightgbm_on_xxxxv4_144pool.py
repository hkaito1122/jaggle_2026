"""118_lightgbm_on_xxxxv4_144pool

`115_`のスタック(8モデル)は84_プール(134/89列)由来のCatBoost/TabPFN/LightGBMと、
xxxx_v4独自144列プール由来のCatBoost/TabPFN(97_/99_)・reference再現(r_full、72_)で
構成されている。`114_`/`115_`で「既存の特徴量プール(84_系)×新しい機構(LightGBM)」が
8モデル目として現最良を更新したのに対し、「新しい特徴量プール(xxxx_v4系)×新しい機構
(LightGBM)」——両軸で新しい組み合わせ——はまだ試していない。r_full(reference再現、
CatBoostベース)が最大の貢献者だったのは「新しい特徴量プール×既存の機構」の組み合わせ
だったため、残るセルを埋める。

`97_`のxxxx_v4独自144列プール（月次集約・入社時特徴量・独自regexメモパース+TF-IDF/SVD、
CatBoost選択ステップなしのFULL構成）に対し、LightGBM単体をリークなしKFold OOFで評価する。
xxxx_v4プールは部署頻度エンコーディングのみ(target非依存)のため、84_系と異なり
fold間で特徴量セットを再構築する必要が無い(97_と同じ設計)。

`114_`の教訓（n_estimators固定・early stoppingなしは重度の過学習を招く）を最初から反映し、
CatBoost相当の保守的な設定+fold学習データの一部をearly stopping用の内部検証に回す設計を
採用する。カテゴリ列は114_/116_と同じくTrain+Test全体で確定させたcategoryコード表を使う。

判定基準: 現最良115_(8モデル)にlgbm144を9つ目として追加した入れ子CVスタッキングが、
分解能±0.0043を超えて改善するかで採否を決める（別スクリプトで実施）。本スクリプトは
OOF/Test予測の生成のみを担当する。

出力: lgbm144_classifier（+ OOF・Test予測を.npyで保存し、後日スタッキングに使用）
"""
import datetime
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger
from common.utils.seed import seed_everything

SEED = 42
seed_everything(seed=SEED)

TARGET_COL = "10年定着ラベル"
ID_COL = "社員ID"

SCRIPT_NAME = "118_lightgbm_on_xxxxv4_144pool"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 実験開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models" / TODAY / SCRIPT_NAME
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIR = PROJECT_ROOT / "data" / "input"

train_persona = pd.read_csv(INPUT_DIR / "employee_persona_train.csv")
test_persona = pd.read_csv(INPUT_DIR / "employee_persona_test.csv")
train_monthly = pd.read_csv(INPUT_DIR / "employee_monthly_train.csv")
test_monthly = pd.read_csv(INPUT_DIR / "employee_monthly_test.csv")

y_train = train_persona[TARGET_COL]
train_ids = train_persona[ID_COL].values
test_ids = test_persona[ID_COL].values
logger.info(f"Train Persona: {train_persona.shape}, Test Persona: {test_persona.shape}")

EARLY_LEAVER_IDS = set(train_monthly.loc[train_monthly["月末在籍状態"] == "退職", ID_COL].unique())
_test_early = set(test_monthly.loc[test_monthly["月末在籍状態"] == "退職", ID_COL].unique())
assert len(_test_early) == 0

TEXT_COLS = ["入社時メモ", "上司からのフィードバック", "同僚からのフィードバック"]

surv_mask = np.array([tid not in EARLY_LEAVER_IDS for tid in train_ids])
logger.info(f"生存者(24か月在籍): {surv_mask.sum()} / {len(surv_mask)}")


# ============================================================
# xxxx_v4.ipynb からの移植: 月次集約（自己学習ブロックは73_で不採用確定のため除外）
# ============================================================
等級の数値 = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}
役割の数値 = {"メンバー": 1, "シニア": 2, "エキスパート": 3, "リード": 4,
          "シニアエキスパート": 4, "マネージャー": 5}
行動列 = ["残業時間", "有給取得日数", "欠勤日数", "研修時間",
        "上司との面談実施回数", "情報共有件数", "在宅勤務日数"]
評価列 = ["360度評価_親和度", "360度評価_信頼度", "360度評価_主体度",
        "360度評価_学習度", "360度評価_共有貢献度"]
評価系列 = [*評価列, "顧客満足度評価", "担当プロジェクト数", "360度評価者数"]


def 傾き(値の列):
    有効 = ~np.isnan(値の列)
    if 有効.sum() < 2:
        return 0.0
    return np.polyfit(np.arange(len(値の列))[有効], 値の列[有効], 1)[0]


def create_monthly_agg_xxxxv4(月次データ):
    月次データ = 月次データ.sort_values(["社員ID", "経過月数"]).copy()
    月次データ["等級_num"] = 月次データ["等級"].map(等級の数値)
    月次データ["役割_num"] = 月次データ["役割"].map(役割の数値)
    行一覧 = []
    for 社員, g in 月次データ.groupby("社員ID", sort=False):
        g = g.reset_index(drop=True)
        初月, 最終月 = g.iloc[0], g.iloc[-1]
        行 = {
            "社員ID": 社員,
            "観測月数": len(g),
            "最終月に不在籍": int(最終月["月末在籍状態"] != "在籍"),
            "休職あり": int((g["月末在籍状態"] == "休職").any()),
            "退職済み": int((g["月末在籍状態"] == "退職").any()),
            "最終月の勤務地": 最終月["勤務地"],
            "最終月の職種": 最終月["職種"],
            "最終月の部署ID": 最終月["部署ID"],
            "部署異動回数": g["部署ID"].nunique() - 1,
            "上司交代回数": g["上司ID"].nunique() - 1,
            "役割変化回数": g["役割"].nunique() - 1,
            "初月の等級": 初月["等級_num"],
            "最終月の等級": 最終月["等級_num"],
            "昇級あり": int(最終月["等級_num"] > 初月["等級_num"]),
            "初月の役割": 初月["役割_num"],
            "最終月の役割": 最終月["役割_num"],
            "初月給与": 初月["月例給与_円"],
            "最終月給与": 最終月["月例給与_円"],
            "給与の伸び率": (最終月["月例給与_円"] - 初月["月例給与_円"]) / 初月["月例給与_円"],
        }
        for 列 in 行動列:
            行[f"{列}_平均"] = g[列].mean()
            行[f"{列}_標準偏差"] = g[列].std()
            行[f"{列}_最終月"] = g[列].iloc[-1]
        行["残業時間_傾き"] = 傾き(g["残業時間"].to_numpy(dtype=float))
        for 列 in 評価系列:
            行[f"{列}_平均"] = g[列].mean()
            行[f"{列}_直近"] = g[列].dropna().iloc[-1] if g[列].notna().any() else np.nan
            行[f"{列}_欠損率"] = g[列].isna().mean()
        行["360評価_傾き"] = 傾き(g[評価列].mean(axis=1).to_numpy(dtype=float))
        行["360評価_更新回数"] = g["360度評価更新フラグ"].sum()
        行["360評価_更新率"] = g["360度評価更新フラグ"].mean()
        前半 = g[g["経過月数"] < 12]
        後半 = g[g["経過月数"] >= 12]
        for 列 in ["残業時間", "有給取得日数", "欠勤日数", "情報共有件数", "研修時間", "月例給与_円"]:
            行[f"{列}_後半前半差"] = 後半[列].mean() - 前半[列].mean()
        行["360評価_後半前半差"] = (後半[評価列].mean(axis=1).mean() - 前半[評価列].mean(axis=1).mean())
        行一覧.append(行)
    return pd.DataFrame(行一覧)


# ============================================================
# xxxx_v4.ipynb からの移植: 入社時特徴量
# ============================================================
def create_persona_features_xxxxv4(入社時データ):
    d = 入社時データ.copy()
    d["入社年"] = pd.to_datetime(d["入社日"]).dt.year
    d["前職職種"] = d["前職職種"].fillna("前職なし")
    d["初期等級_num"] = d["初期等級"].map(等級の数値)
    d["初期役割_num"] = d["初期役割"].map(役割の数値)
    for 列 in TEXT_COLS:
        d[f"{列}_文字数"] = d[列].str.len()
    return d


# ============================================================
# xxxx_v4.ipynb からの移植: メモのパース（84_/96_とは別実装の独自regex）
# ============================================================
勤務地一覧 = ["東京", "大阪", "愛知", "福岡", "仙台", "北海道", "その他"]
転居の否定 = r"許容せず|許容しておらず|許容していない|許容しない|希望しておらず|希望せず|希望していない"
在宅の否定 = r"必須条件と(?:は)?していない|不要"


def parse_memo_xxxxv4(メモ):
    メモ = メモ.fillna("")
    働き方の節 = メモ.str.extract(r"・勤務地・働き方：(.+?)(?:\n|$)")[0].fillna("")
    志向の節 = メモ.str.extract(r"・キャリア志向：(.+?)(?:\n|$)")[0].fillna("")
    結果 = pd.DataFrame(index=メモ.index)

    def 最初の勤務地(s):
        候補, 位置 = "不明", 10**9
        for 勤務地 in 勤務地一覧:
            i = s.find(勤務地)
            if 0 <= i < 位置:
                候補, 位置 = 勤務地, i
        return 候補
    結果["希望勤務地"] = 働き方の節.map(最初の勤務地)

    転居の句 = 働き方の節.str.extract(r"(転居[^、。]*)")[0].fillna("")
    転居NG = 転居の句.str.contains(転居の否定, regex=True)
    転居OK = 転居の句.str.contains("許容") & ~転居NG
    結果["転居許容"] = np.where(転居NG, 0, np.where(転居OK, 1, -1))

    在宅の句 = 働き方の節.str.extract(r"(在宅[^、。]*)")[0].fillna("")
    在宅NG = 在宅の句.str.contains(在宅の否定, regex=True)
    在宅OK = 在宅の句.str.contains("希望") & ~在宅NG
    結果["在宅希望"] = np.where(在宅OK, 1, np.where(在宅NG, 0, -1))

    def 志向の分類(s):
        if not s.strip():
            return "不明"
        if ("限定していない" in s or "方向を限定" in s or "限定せず" in s
                or "特定していない" in s or "方向は特定" in s
                or "限定した志向は確認されていない" in s):
            return "限定なし"
        if "専門職" in s:
            return "専門職"
        if "管理職" in s:
            return "管理職"
        if "安定" in s:
            return "安定"
        return "不明"
    結果["キャリア志向"] = 志向の節.map(志向の分類)
    return 結果


# ============================================================
# TF-IDF+SVD（84_/96_と同一関数。xxxx_v4本来はjanome+CatBoost text_featuresだが
# FT-Transformerが生テキストを扱えないためCB/FT共通の数値表現として採用）
# ============================================================
def create_tfidf_svd_features(train_persona, test_persona, col, max_features=300, n_components=15, min_df=3, seed=42):
    train_text = train_persona[col].fillna("").astype(str)
    test_text = test_persona[col].fillna("").astype(str)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=max_features, min_df=min_df)
    train_tfidf = vectorizer.fit_transform(train_text)
    test_tfidf = vectorizer.transform(test_text)
    n_comp = min(n_components, train_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed, algorithm="arpack")
    train_svd = svd.fit_transform(train_tfidf)
    test_svd = svd.transform(test_tfidf)
    col_names = [f"{col}_tfidf_svd_{i}" for i in range(n_comp)]
    train_out = pd.DataFrame(train_svd, columns=col_names)
    train_out[ID_COL] = train_persona[ID_COL].values
    test_out = pd.DataFrame(test_svd, columns=col_names)
    test_out[ID_COL] = test_persona[ID_COL].values
    return train_out, test_out


# ============================================================
# split非依存の特徴量生成（target非依存、リーク無し）
# ============================================================
logger.info("=" * 60)
logger.info("xxxx_v4由来の特徴量を生成中(split非依存)...")
train_ma = create_monthly_agg_xxxxv4(train_monthly)
test_ma = create_monthly_agg_xxxxv4(test_monthly)
train_pf = create_persona_features_xxxxv4(train_persona)
test_pf = create_persona_features_xxxxv4(test_persona)

for d in (train_pf, test_pf):
    parsed = parse_memo_xxxxv4(d["入社時メモ"])
    for 列 in ["希望勤務地", "転居許容", "在宅希望", "キャリア志向"]:
        d[列] = parsed[列].to_numpy()

dept_freq = pd.concat([train_pf["初期部署ID"], test_pf["初期部署ID"]]).value_counts()
train_pf["初期部署ID_頻度"] = train_pf["初期部署ID"].map(dept_freq)
test_pf["初期部署ID_頻度"] = test_pf["初期部署ID"].map(dept_freq)
last_dept_freq = pd.concat([train_ma["最終月の部署ID"], test_ma["最終月の部署ID"]]).value_counts()
train_ma["最終月の部署ID_頻度"] = train_ma["最終月の部署ID"].map(last_dept_freq)
test_ma["最終月の部署ID_頻度"] = test_ma["最終月の部署ID"].map(last_dept_freq)

tfidf_train_list, tfidf_test_list = [], []
for col in TEXT_COLS:
    tr, te = create_tfidf_svd_features(train_persona, test_persona, col, max_features=300, n_components=15, min_df=3, seed=SEED)
    tfidf_train_list.append(tr)
    tfidf_test_list.append(te)
logger.info("xxxx_v4由来の特徴量生成完了")


# ============================================================
# build_features: target依存部分がそもそも無い(部署頻度はtarget非依存)ため
# fold間で作り直す必要は無いが、80_/84_/96_と同じインターフェースに揃えておく
# ============================================================
CAT_COLS = ["入社区分", "最終学歴", "専攻分野", "前職職種", "採用経路", "性別",
            "初期職種", "初期勤務地", "初期等級", "初期役割",
            "最終月の勤務地", "最終月の職種", "最終月の部署ID",
            "希望勤務地", "キャリア志向"]


def build_features():
    tf = train_pf.merge(train_ma, on=ID_COL, how="left")
    for trdf in tfidf_train_list:
        tf = tf.merge(trdf, on=ID_COL, how="left")
    ttf = test_pf.merge(test_ma, on=ID_COL, how="left")
    for tedf in tfidf_test_list:
        ttf = ttf.merge(tedf, on=ID_COL, how="left")

    for c in CAT_COLS:
        tf[c] = tf[c].astype(str).fillna("missing")
        ttf[c] = ttf[c].astype(str).fillna("missing")

    raw_drop = ["入社日", "初期部署ID", *TEXT_COLS]
    tf = tf.drop(columns=[c for c in raw_drop if c in tf.columns]).set_index(ID_COL)
    ttf = ttf.drop(columns=[c for c in raw_drop if c in ttf.columns]).set_index(ID_COL)
    tf[TARGET_COL] = train_persona.set_index(ID_COL).loc[tf.index, TARGET_COL].values
    return tf, ttf


def _feature_cols(df):
    return [c for c in df.columns if c not in [TARGET_COL]]


tf_full, ttf_full = build_features()
FULL_FEATURE_COLS = _feature_cols(tf_full)
logger.info(f"tf_full 特徴量数: {len(FULL_FEATURE_COLS)}（xxxx_v4本来の集約・パース由来99列 + TF-IDF/SVD45列のはず）")
assert len(FULL_FEATURE_COLS) == 144, f"想定外の列数: {len(FULL_FEATURE_COLS)}"
OBJ_COLS_FULL = [c for c in FULL_FEATURE_COLS if tf_full[c].dtype == "object"]
logger.info(f"カテゴリ列数: {len(OBJ_COLS_FULL)}")


SEEDS_SUB = [42, 2024, 7, 1234, 99]
OOF_SEED = 42
OOF_N_SPLITS = 5


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}）")
    return str(path)


# ============================================================
# LightGBM統合: カテゴリ列はcategory dtypeに変換してネイティブ対応させる(114_/116_と同一パターン)
# ============================================================
import lightgbm as lgb
logger.info(f"lightgbm version = {lgb.__version__}")

LGBM_PARAMS = dict(
    n_estimators=2000, learning_rate=0.03, num_leaves=15, max_depth=4,
    colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
    reg_lambda=5.0, reg_alpha=1.0, min_child_samples=30, verbosity=-1,
)
LGBM_EARLY_STOPPING_ROUNDS = 50
LGBM_INTERNAL_VAL_FRAC = 0.15

CAT_CATEGORIES = {
    c: sorted(pd.concat([tf_full[c], ttf_full[c]]).astype(str).unique().tolist())
    for c in OBJ_COLS_FULL
}


def to_lgbm_frame(df, feats, obj_cols):
    X = df[feats].copy()
    for c in obj_cols:
        X[c] = pd.Categorical(X[c].astype(str), categories=CAT_CATEGORIES[c])
    return X


def fit_lgbm(X_tr, y_tr, obj_cols, seed):
    rng = np.random.RandomState(seed)
    n = len(y_tr)
    idx = rng.permutation(n)
    n_val = max(1, int(n * LGBM_INTERNAL_VAL_FRAC))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    X_fit, X_val = X_tr.iloc[fit_idx], X_tr.iloc[val_idx]
    y_fit, y_val = y_tr.iloc[fit_idx], y_tr.iloc[val_idx]

    model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed)
    model.fit(
        X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
        categorical_feature=obj_cols if obj_cols else "auto",
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


# ============================================================
# リークなしKFold OOF: xxxx_v4プールはtarget非依存のためfold間で特徴量再構築は不要
# ============================================================
logger.info("=" * 60)
logger.info("[LightGBM144] xxxx_v4独自144列プールでのOOFを構築中...")
skf = StratifiedKFold(n_splits=OOF_N_SPLITS, shuffle=True, random_state=OOF_SEED)
lgbm_oof = np.zeros(len(train_ids))

X_lgbm_full = to_lgbm_frame(tf_full, FULL_FEATURE_COLS, OBJ_COLS_FULL)
y_full = tf_full[TARGET_COL]

for fold_i, (tr_pos, va_pos) in enumerate(skf.split(train_ids, y_train)):
    fold_train_ids = train_ids[tr_pos]
    fold_val_ids = train_ids[va_pos]
    X_tr = X_lgbm_full.loc[fold_train_ids]
    y_tr = y_full.loc[fold_train_ids]
    X_va = X_lgbm_full.loc[fold_val_ids]

    m_lgbm = fit_lgbm(X_tr, y_tr, OBJ_COLS_FULL, OOF_SEED)
    lgbm_oof[va_pos] = m_lgbm.predict_proba(X_va)[:, 1]
    logger.info(f"    [fold{fold_i}] 完了")

lgbm_score = log_loss(y_train.values[surv_mask], lgbm_oof[surv_mask])
logger.info(f"[LightGBM144] val(OOF, リークなし, n={surv_mask.sum()})={lgbm_score:.6f}")
logger.info("(参考: 97_のCB144=0.528230 / 99_のTabPFN144=0.542703 / "
            "114_のLightGBM134=0.549972。lgbm144が既存モデルとどの程度相関するかは"
            "後日スタッキングスクリプトで確認)")

# ============================================================
# OOFの保存
# ============================================================
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_lgbm144_oofpreds.npy", lgbm_oof)
logger.info("  OOF予測(.npy)を保存しました")

# ============================================================
# 提出: Train全件(=fold構築なしで既にtf_full)でLightGBM(5シード平均)によるTest予測
# ============================================================
logger.info("=" * 60)
logger.info("提出用: Train全件でLightGBM(5シード平均)を学習中...")
X_test_lgbm_full = to_lgbm_frame(ttf_full, FULL_FEATURE_COLS, OBJ_COLS_FULL)

test_preds = []
for seed in SEEDS_SUB:
    m = fit_lgbm(X_lgbm_full, y_full, OBJ_COLS_FULL, seed)
    test_preds.append(m.predict_proba(X_test_lgbm_full)[:, 1])
    logger.info(f"    seed={seed}: 完了")
lgbm_test = np.mean(test_preds, axis=0)
path1 = save_submission(lgbm_test, "lgbm144_classifier")
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_lgbm144_testpreds.npy", lgbm_test)

logger.info("=" * 60)
logger.info(f"局所OOF: LightGBM144={lgbm_score:.6f}")
logger.info(f"=== 出力完了: {path1} ===")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
