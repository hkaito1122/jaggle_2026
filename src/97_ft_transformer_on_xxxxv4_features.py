"""97_ft_transformer_on_xxxxv4_features

`96_`ではFT-Transformer(rtdl_revisiting_models)を84_由来の特徴量プール(447/134/89/67列)で
試し、CatBoost比+0.03〜+0.05で明確に劣ることを3イテレーションで確認した
（[[ft-transformer-96-action-plan]]参照）。ただし96_/84_/93_/86_はいずれも同じ特徴量エンジ
ニアリング（54_l2_m_interaction由来のリークなしKFold集約+TF-IDF/SVD+L2xM交互作用等）を
共有しており、多様性は「モデル機構の違い」だけに依存していた。

一方フォーク調査（本セッション）でCB89/TabPFN89とxxxx_v4.ipynb(著者オリジナルパイプライン)
の派生・交互作用特徴量がほぼ完全に独立していることを確認済み——xxxx_v4は単純な平均/標準偏差/
最終月値・後半前半差という素朴な集約と、独自regexによる勤務地・キャリア志向パース
（84_/96_の`create_relocation_mismatch_features`/`extract_desired_location_v2`とは別実装、
`勤務地一覧`のヒット位置ベース）を使っており、L2xM交互作用・クラスタリング・詳細統計量
（skew/kurtosis/iqr）・四半期特徴量は一切持たない。

本スクリプトは「FT-Transformer(新機構) × xxxx_v4の独自特徴量エンジニアリング(新特徴量プール)」
という、機構と特徴量プールの両軸で同時に多様性を稼げる、まだ試していない組み合わせを検証する
（[[ft-transformer-96-action-plan]]に事前登録済みの次の一手）。

xxxx_v4.ipynbからの移植内容:
  - 月次集約: 平均/標準偏差/最終月値・360度評価の直近値/欠損率・後半前半差・残業時間の傾き
    （`学習時間合計`/`学習テーマ数`等の自己学習ブロックは`73_`で不採用確定済みのため除外、
    `82_`と同じ方針で重複検証しない）
  - 入社時特徴量: 入社年・初期等級/役割の数値化・テキスト文字数
  - メモパース: 独自regex(`勤務地一覧`ヒット位置ベース)による希望勤務地・転居許容・在宅希望・
    キャリア志向（84_/96_の抽出ロジックとは別実装）
  - 部署頻度: 初期部署ID・最終月の部署IDの出現頻度(train+test、target非依存なのでリークなし)
xxxx_v4からの変更点（FT-Transformer対応のため）:
  - xxxx_v4はメモ/学習テキストをjanome形態素解析→CatBoost text_featuresへ生テキストのまま
    渡すが、FT-Transformerは生テキストを扱えないため、84_/96_と同じTF-IDF(char_wb 2-4gram)+
    TruncatedSVD(15次元)を3テキスト列に適用し、CB/FTが同一の数値行列を見るようにする
    （xxxx_v4本来のtext_features設計とは異なる簡略化だが、CB/FT比較の一貫性を優先）
  - 部署の目的変数ターゲットエンコーディングはxxxx_v4に無い(頻度のみ)ためそのまま踏襲

列数: 144（xxxx_v4本来の集約・パース由来99列 + TF-IDF/SVD 45列）。84_のような重要度による
ネスト選択は行わない(xxxx_v4本来がFULL構成のみのため、82_と同じ方針)。

FT-Transformer設定は96_イテレーション3で最良だった構成を踏襲: n_blocks=1(容量縮小)、
FT_EPOCHS=100/FT_PATIENCE=10(元設定)、FT_SEEDS=1(GPU前提で高速反復、良ければ増やす)。

出力3ファイル: cb_classifier / ft_classifier / blend_argmin
（+ OOF・Test予測を.npyで保存し、後日プール/xxxx_v4系との再ブレンドに再利用できるようにする）
"""
import datetime
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb
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

SCRIPT_NAME = "97_ft_transformer_on_xxxxv4_features"
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


# ============================================================
# CatBoost（84_/96_と同一の事前チューニング済みパラメータ、選択ステップなし）
# ============================================================
A_PARAMS = {
    "depth": 4,
    "learning_rate": 0.03518359458951149,
    "l2_leaf_reg": 2.217690447016724,
    "border_count": 218,
    "bagging_temperature": 0.6787467566574921,
    "random_strength": 1.438494697238285,
}
ITER = 560
SEEDS_SUB = [42, 2024, 7, 1234, 99]
OOF_SEED = 42
OOF_N_SPLITS = 5


def _fit_one_classifier(X_tr, y_tr, obj_cols, seed):
    model = cb.CatBoostClassifier(**A_PARAMS, iterations=ITER, random_seed=seed,
                                   verbose=False, cat_features=obj_cols, task_type="CPU")
    model.fit(X_tr, y_tr)
    return model


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}）")
    return str(path)


# ============================================================
# FT-Transformer（rtdl_revisiting_models）。96_イテレーション3で最良だった構成を踏襲:
# n_blocks=1(容量縮小)・FT_EPOCHS=100/FT_PATIENCE=10(元設定)。
# ============================================================
logger.info("=" * 60)
logger.info("FT-Transformer (rtdl_revisiting_models) を読み込み中...")
import torch
import torch.nn as nn
from rtdl_revisiting_models import FTTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"device = {DEVICE}")
if DEVICE == "cpu":
    logger.warning("GPUが無い。96_の経験上CPUだと1イテレーション数時間かかる。ColabのランタイムをGPUに変更すること。")

FT_EPOCHS = 100
FT_BATCH_SIZE = 128
FT_LR = 1e-4
FT_WEIGHT_DECAY = 1e-5
FT_PATIENCE = 10
FT_VAL_FRAC = 0.1
FT_N_BLOCKS = 1  # 96_イテレーション3で確認済みの縮小構成(既定3→1、d_block 192→96)
FT_SEEDS = [42]  # 高速反復のため1シード(良ければ増やす)


def to_ft_matrix(train_df, other_dfs, feats):
    obj_cols = [c for c in feats if train_df[c].dtype == "object"]
    num_cols = [c for c in feats if c not in obj_cols]
    frames = [train_df] + list(other_dfs)

    medians = train_df[num_cols].median()
    scaler = StandardScaler()
    num_arrs = []
    tr_num = train_df[num_cols].fillna(medians).fillna(0.0)
    num_arrs.append(np.clip(scaler.fit_transform(tr_num), -10, 10).astype(np.float32))
    for f in frames[1:]:
        num_arrs.append(np.clip(scaler.transform(f[num_cols].fillna(medians).fillna(0.0)), -10, 10).astype(np.float32))

    cat_map = {}
    cardinalities = []
    for c in obj_cols:
        vals = sorted(train_df[c].astype(str).unique())
        cat_map[c] = {v: i for i, v in enumerate(vals)}
        cardinalities.append(len(vals) + 1)
    cat_arrs = []
    for f in frames:
        if obj_cols:
            M = np.zeros((len(f), len(obj_cols)), dtype=np.int64)
            for j, c in enumerate(obj_cols):
                M[:, j] = f[c].astype(str).map(cat_map[c]).fillna(len(cat_map[c])).astype(np.int64).values
            cat_arrs.append(M)
        else:
            cat_arrs.append(np.zeros((len(f), 0), dtype=np.int64))

    return num_arrs, cat_arrs, cardinalities


def _make_ft_model(n_num_features, cardinalities):
    default_kwargs = FTTransformer.get_default_kwargs(n_blocks=FT_N_BLOCKS)
    model = FTTransformer(
        n_cont_features=n_num_features,
        cat_cardinalities=cardinalities if cardinalities else None,
        d_out=1,
        **default_kwargs,
    )
    return model.to(DEVICE)


def _ft_train_one(X_num_tr, X_cat_tr, y_tr, cardinalities, seed):
    torch.manual_seed(seed)
    n = len(y_tr)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * FT_VAL_FRAC))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    model = _make_ft_model(X_num_tr.shape[1], cardinalities)
    optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    Xn_tr = torch.tensor(X_num_tr[tr_idx], dtype=torch.float32, device=DEVICE)
    Xc_tr = torch.tensor(X_cat_tr[tr_idx], dtype=torch.long, device=DEVICE) if X_cat_tr.shape[1] > 0 else None
    y_tr_t = torch.tensor(y_tr[tr_idx], dtype=torch.float32, device=DEVICE)
    Xn_va = torch.tensor(X_num_tr[val_idx], dtype=torch.float32, device=DEVICE)
    Xc_va = torch.tensor(X_cat_tr[val_idx], dtype=torch.long, device=DEVICE) if X_cat_tr.shape[1] > 0 else None
    y_va_t = torch.tensor(y_tr[val_idx], dtype=torch.float32, device=DEVICE)

    n_tr = len(tr_idx)
    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(FT_EPOCHS):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for start in range(0, n_tr, FT_BATCH_SIZE):
            batch_idx = perm[start:start + FT_BATCH_SIZE]
            xb_num = Xn_tr[batch_idx]
            xb_cat = Xc_tr[batch_idx] if Xc_tr is not None else None
            yb = y_tr_t[batch_idx]
            optimizer.zero_grad()
            logits = model(xb_num, xb_cat).squeeze(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xn_va, Xc_va).squeeze(-1)
            val_loss = loss_fn(val_logits, y_va_t).item()
        if val_loss < best_val_loss - 1e-5:
            best_val_loss, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= FT_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _ft_predict(model, X_num, X_cat):
    model.eval()
    Xn = torch.tensor(X_num, dtype=torch.float32, device=DEVICE)
    Xc = torch.tensor(X_cat, dtype=torch.long, device=DEVICE) if X_cat.shape[1] > 0 else None
    with torch.no_grad():
        logits = model(Xn, Xc).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def ft_fit_predict(train_df, other_dfs, feats, seeds):
    num_arrs, cat_arrs, cardinalities = to_ft_matrix(train_df, other_dfs, feats)
    X_num_tr, *X_num_other = num_arrs
    X_cat_tr, *X_cat_other = cat_arrs
    y_tr = train_df[TARGET_COL].values.astype(np.float32)

    acc = [[] for _ in other_dfs]
    for s in seeds:
        model = _ft_train_one(X_num_tr, X_cat_tr, y_tr, cardinalities, s)
        for k in range(len(other_dfs)):
            acc[k].append(_ft_predict(model, X_num_other[k], X_cat_other[k]))
    return [np.mean(a, axis=0) for a in acc]


# ============================================================
# リークなしKFold OOF: CatBoost/FT-Transformerを同一特徴量プールで比較
# ============================================================
logger.info("=" * 60)
logger.info("[CB/FT] xxxx_v4由来144列でのOOFを構築中...")
skf = StratifiedKFold(n_splits=OOF_N_SPLITS, shuffle=True, random_state=OOF_SEED)

cb_oof = np.zeros(len(train_ids))
ft_oof = np.zeros(len(train_ids))

for fold_i, (tr_pos, va_pos) in enumerate(skf.split(train_ids, y_train)):
    fold_train_ids = train_ids[tr_pos].tolist()
    fold_val_ids = train_ids[va_pos].tolist()

    X_tr = tf_full.loc[fold_train_ids, FULL_FEATURE_COLS].fillna(-999)
    y_tr_cls = tf_full.loc[fold_train_ids, TARGET_COL]
    X_va = tf_full.loc[fold_val_ids, FULL_FEATURE_COLS].fillna(-999)

    m_cb = _fit_one_classifier(X_tr, y_tr_cls, OBJ_COLS_FULL, OOF_SEED)
    cb_oof[va_pos] = m_cb.predict_proba(X_va)[:, 1]

    fold_train_df = tf_full.loc[fold_train_ids]
    fold_val_df = tf_full.loc[fold_val_ids]
    (ft_va,) = ft_fit_predict(fold_train_df, [fold_val_df], FULL_FEATURE_COLS, FT_SEEDS)
    ft_oof[va_pos] = ft_va

    logger.info(f"    [fold{fold_i}] 完了")

cb_score = log_loss(y_train.values[surv_mask], cb_oof[surv_mask])
ft_score = log_loss(y_train.values[surv_mask], ft_oof[surv_mask])
corr = float(np.corrcoef(cb_oof, ft_oof)[0, 1])
logger.info(f"[xxxxv4feat] CB val={cb_score:.6f} / FT val={ft_score:.6f} / 相関={corr:.4f}")

# ============================================================
# ブレンド曲線(argmin)
# ============================================================
logger.info("=" * 60)
logger.info("[ブレンド曲線] w=CatBoost側の重みを0〜1で走査...")
y_surv = y_train.values[surv_mask]
cb_s = cb_oof[surv_mask]
ft_s = ft_oof[surv_mask]
BLEND_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
scan_rows = []
for w in BLEND_WEIGHTS:
    blend = np.clip(w * cb_s + (1 - w) * ft_s, 1e-9, 1 - 1e-9)
    ll = log_loss(y_surv, blend)
    scan_rows.append((w, ll))
best_w, best_ll = min(scan_rows, key=lambda r: r[1])
logger.info(f"[argmin] best_w_cb={best_w:.2f}, best_val={best_ll:.6f} "
            f"(CB単体比 {best_ll - cb_score:+.6f} / FT単体比 {best_ll - ft_score:+.6f})")

# ============================================================
# OOFの保存（後日プール/xxxx_v4系との再ブレンドに再利用するため）
# ============================================================
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_cb_oofpreds.npy", cb_oof)
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_ft_oofpreds.npy", ft_oof)
logger.info("  OOF予測(.npy)を保存しました")

# ============================================================
# 提出: Train全件でCatBoost(5シード)・FT-Transformer(FT_SEEDS)・argminブレンド
# ============================================================
logger.info("=" * 60)
logger.info("提出用: Train全件で学習中...")
X_tr_full = tf_full[FULL_FEATURE_COLS].fillna(-999)
y_tr_full = tf_full[TARGET_COL]
X_test_full = ttf_full[FULL_FEATURE_COLS].fillna(-999)

cb_test_preds = []
last_cb_model = None
for seed in SEEDS_SUB:
    last_cb_model = _fit_one_classifier(X_tr_full, y_tr_full, OBJ_COLS_FULL, seed)
    cb_test_preds.append(last_cb_model.predict_proba(X_test_full)[:, 1])
cb_test = np.mean(cb_test_preds, axis=0)
last_cb_model.save_model(str(SAVED_MODELS_DIR / "cb_full_lastseed.cbm"))
p1 = save_submission(cb_test, "cb_classifier")

logger.info("[FT-full] シード平均で学習・予測中...")
(ft_test,) = ft_fit_predict(tf_full, [ttf_full], FULL_FEATURE_COLS, FT_SEEDS)
p2 = save_submission(ft_test, "ft_classifier")

_num_arrs, _cat_arrs, _cardinalities = to_ft_matrix(tf_full, [], FULL_FEATURE_COLS)
last_ft_model = _ft_train_one(_num_arrs[0], _cat_arrs[0], tf_full[TARGET_COL].values.astype(np.float32),
                               _cardinalities, FT_SEEDS[0])
torch.save(last_ft_model.state_dict(), SAVED_MODELS_DIR / "ft_full_seed42.pt")

blend_test = best_w * cb_test + (1 - best_w) * ft_test
p3 = save_submission(blend_test, f"blend_argmin_w{round(best_w * 100):d}pct")

np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_cb_testpreds.npy", cb_test)
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_ft_testpreds.npy", ft_test)

logger.info("=" * 60)
logger.info(f"局所OOF: CB={cb_score:.6f} / FT={ft_score:.6f} / argminブレンド(w_cb={best_w:.2f})={best_ll:.6f} / 相関={corr:.4f}")
logger.info(f"=== 全3ファイル出力完了 ===")
for p in [p1, p2, p3]:
    logger.info(f"  {p}")
logger.info("(参考: 82_のxxxx_v4由来8列追加+447列(455列)CatBoost Public=0.508039。"
            "本スクリプトはxxxx_v4本来の集約・パースのみを使う別プール(144列)で、82_とは特徴量が異なる)")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
