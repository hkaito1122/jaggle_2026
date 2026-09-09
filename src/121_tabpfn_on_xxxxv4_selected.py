"""121_tabpfn_on_xxxxv4_selected

`99_`でxxxx_v4独自144列プールにTabPFNを適用した際はFULL構成（列選択なし）で、
CatBoost比+0.0144悪化という結果だった（[[tabpfn-ensemble-partner]]）。しかし84_プールでは
「TabPFNは重要度で絞った89列(20%)が最適」という強い知見が確立している(93_)。
xxxx_v4プールに対してはこの列選択を一度も試していない。

`97_`のxxxx_v4独自144列プールに対し、93_と同じ設計（CatBoostの重要度上位20%を各fold
の学習データだけで選び直す、選択リークを避ける、[[validation-asymmetry]]のHow to apply
参照）で約29列に絞り、CatBoostとTabPFNの両方をfitしてリークなしKFold OOFで比較する。
xxxx_v4プールは部署頻度エンコーディングのみ(target非依存)のため、84_系と異なり
特徴量自体はfold間で再構築不要——選択ステップだけをfoldごとにやり直す。

期待: 84_プールで見られたTabPFNの逆転(CatBoost比で単体優位)がxxxx_v4プールでも
起きれば、既存の115_スタックに対して「xxxx_v4系プール由来・低相関・かつ高性能」という、
これまでのr_poolc51/lgbm144（低相関だが弱い）とは異なる有望な追加候補になる。

出力: cb_selected_classifier / tabpfn_selected_classifier / blend_argmin
（+ OOF・Test予測を.npyで保存し、後日スタッキングに使用）
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

SCRIPT_NAME = "121_tabpfn_on_xxxxv4_selected"
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
TABPFN_SEEDS = [42, 2024, 7]  # 63_/93_と同一
OOF_SEED = 42
OOF_N_SPLITS = 5
TOP_PCT_20 = 0.2  # 93_と同一の選択率(84_プールでは447列→89列に相当)
BLEND_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _fit_one_classifier(X_tr, y_tr, obj_cols, seed):
    model = cb.CatBoostClassifier(**A_PARAMS, iterations=ITER, random_seed=seed,
                                   verbose=False, cat_features=obj_cols, task_type="CPU")
    model.fit(X_tr, y_tr)
    return model


def select_top_pct(imp_series, pct):
    n_top = max(1, round(len(imp_series) * pct))
    return imp_series.sort_values(ascending=False).head(n_top).index.tolist()


def select_top20pct(imp_series):
    return select_top_pct(imp_series, TOP_PCT_20)


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}）")
    return str(path)


# ============================================================
# TabPFN v2 統合（63_/93_と同一パターン）
# ============================================================
logger.info("=" * 60)
logger.info("TabPFN v2 (tabpfn==2.2.1) を読み込み中...")
import torch
import tabpfn
from tabpfn import TabPFNClassifier

_live = tabpfn.__version__
if not _live.startswith("2."):
    raise RuntimeError(
        f"tabpfn=={_live} が読み込まれています。v2系(2.x, tabpfn==2.2.1)が必要です。\n"
        "Colabの場合: セル冒頭で `!pip install -q \"tabpfn==2.2.1\"` を実行し、"
        "ランタイムを再起動してから最初のセルに戻ること。"
    )
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"tabpfn version = {_live} / device = {DEVICE}")
if DEVICE == "cpu":
    logger.warning("GPUが無い。TabPFNはCPUだとサンプル数制限に引っかかる。"
                    "ColabのランタイムをGPUに変更してから実行し直すこと。")


def to_tabpfn_matrix(train_df, other_dfs, feats):
    obj = [c for c in feats if train_df[c].dtype == "object"]
    frames = [train_df] + list(other_dfs)
    cat_map = {}
    for c in obj:
        vals = pd.concat([f[c].astype(str) for f in frames]).unique()
        cat_map[c] = {v: i for i, v in enumerate(sorted(vals))}
    out = []
    for f in frames:
        M = f[feats].copy()
        for c in obj:
            M[c] = f[c].astype(str).map(cat_map[c]).astype(float)
        out.append(M.astype(np.float32).values)
    cat_idx = [feats.index(c) for c in obj]
    return out, cat_idx


def tabpfn_fit_predict(train_df, other_dfs, feats, seeds, device=DEVICE):
    (Mtr, *Mo), cat_idx = to_tabpfn_matrix(train_df, other_dfs, feats)
    ytr = train_df[TARGET_COL].values
    acc = [[] for _ in Mo]
    for s in seeds:
        kw = dict(device=device, random_state=s)
        clf = None
        for extra in ({"categorical_features_indices": cat_idx, "ignore_pretraining_limits": True},
                      {"categorical_features_indices": cat_idx},
                      {}):
            try:
                clf = TabPFNClassifier(**kw, **extra)
                break
            except TypeError:
                continue
        clf.fit(Mtr, ytr)
        for k, M in enumerate(Mo):
            acc[k].append(clf.predict_proba(M)[:, 1])
    return [np.mean(a, axis=0) for a in acc]


# ============================================================
# リークなしKFold OOF: 各foldでCatBoost選択(93_と同一設計)→CatBoost/TabPFNを両方評価
# xxxx_v4プールはtarget非依存のため特徴量はfold間で共通(tf_full)、選択だけをfoldごとにやり直す
# ============================================================
logger.info("=" * 60)
logger.info("[CB_selected/TabPFN_selected] ネストされた特徴量選択+CatBoost/TabPFN比較のOOFを構築中...")
skf = StratifiedKFold(n_splits=OOF_N_SPLITS, shuffle=True, random_state=OOF_SEED)
cb_oof = np.zeros(len(train_ids))
tabpfn_oof = np.zeros(len(train_ids))
ncols_per_fold = []

for fold_i, (tr_pos, va_pos) in enumerate(skf.split(train_ids, y_train)):
    fold_train_ids = train_ids[tr_pos]
    fold_val_ids = train_ids[va_pos]

    X_tr_full = tf_full.loc[fold_train_ids, FULL_FEATURE_COLS].fillna(-999)
    y_tr_cls = tf_full.loc[fold_train_ids, TARGET_COL]

    m_selector = _fit_one_classifier(X_tr_full, y_tr_cls, OBJ_COLS_FULL, OOF_SEED)
    imp_series = pd.Series(m_selector.get_feature_importance(), index=FULL_FEATURE_COLS)
    top20_cols = select_top20pct(imp_series)
    ncols_per_fold.append(len(top20_cols))
    obj_top20 = [c for c in top20_cols if c in OBJ_COLS_FULL]

    X_tr20 = tf_full.loc[fold_train_ids, top20_cols].fillna(-999)
    X_va20 = tf_full.loc[fold_val_ids, top20_cols].fillna(-999)
    m_cb20 = _fit_one_classifier(X_tr20, y_tr_cls, obj_top20, OOF_SEED)
    cb_oof[va_pos] = m_cb20.predict_proba(X_va20)[:, 1]

    fold_train_df = tf_full.loc[fold_train_ids]
    fold_val_df = tf_full.loc[fold_val_ids]
    (tp_va,) = tabpfn_fit_predict(fold_train_df, [fold_val_df], top20_cols, TABPFN_SEEDS)
    tabpfn_oof[va_pos] = tp_va

    logger.info(f"    [fold{fold_i}] 完了 (列数={len(top20_cols)})")

cb_score = log_loss(y_train.values[surv_mask], cb_oof[surv_mask])
tabpfn_score = log_loss(y_train.values[surv_mask], tabpfn_oof[surv_mask])
corr_local = float(np.corrcoef(cb_oof, tabpfn_oof)[0, 1])
logger.info(f"[CB_selected] val(OOF, リークなし, n={surv_mask.sum()})={cb_score:.6f} (fold毎列数: {ncols_per_fold})")
logger.info(f"[TabPFN_selected] val(OOF, リークなし, n={surv_mask.sum()})={tabpfn_score:.6f}")
logger.info(f"[CB_selectedとTabPFN_selectedの相関(OOF)] = {corr_local:.4f}")
logger.info("(参考: 97_のCB144(FULL)=0.528230 / 99_のTabPFN144(FULL)=0.542703。"
            "列選択でTabPFNが逆転するかを確認する)")

# ============================================================
# ブレンド曲線(argmin)
# ============================================================
logger.info("=" * 60)
logger.info("[ブレンド曲線] w=CatBoost側の重みを0〜1で走査...")
y_surv = y_train.values[surv_mask]
cb_s = cb_oof[surv_mask]
tp_s = tabpfn_oof[surv_mask]
scan_rows = []
for w in BLEND_WEIGHTS:
    blend = np.clip(w * cb_s + (1 - w) * tp_s, 1e-9, 1 - 1e-9)
    ll = log_loss(y_surv, blend)
    scan_rows.append((w, ll))
best_w, best_ll = min(scan_rows, key=lambda r: r[1])
logger.info(f"[argmin] best_w_cb={best_w:.2f}, best_val={best_ll:.6f} "
            f"(CB_selected単体比 {best_ll - cb_score:+.6f} / TabPFN_selected単体比 {best_ll - tabpfn_score:+.6f})")

# ============================================================
# OOFの保存
# ============================================================
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_cb_selected_oofpreds.npy", cb_oof)
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_tabpfn_selected_oofpreds.npy", tabpfn_oof)
logger.info("  OOF予測(.npy)を保存しました")

# ============================================================
# 提出: Train全件で選択→CatBoost(5シード)・TabPFN(3シード)・argminブレンド
# ============================================================
logger.info("=" * 60)
logger.info("提出用: Train全件で選択ステップを実行中(Testはラベルを持たないためリークなし)...")
X_tr_full_all = tf_full[FULL_FEATURE_COLS].fillna(-999)
y_tr_full = tf_full[TARGET_COL]

m_selector_full = _fit_one_classifier(X_tr_full_all, y_tr_full, OBJ_COLS_FULL, OOF_SEED)
imp_series_full = pd.Series(m_selector_full.get_feature_importance(), index=FULL_FEATURE_COLS)
top20_cols_full = select_top20pct(imp_series_full)
logger.info(f"[提出用] TOP20PCT: {len(top20_cols_full)}列")
obj_top20_full = [c for c in top20_cols_full if c in OBJ_COLS_FULL]

X_tr20_full = tf_full[top20_cols_full].fillna(-999)
X_test20_full = ttf_full[top20_cols_full].fillna(-999)

cb_test_preds = []
for seed in SEEDS_SUB:
    m = _fit_one_classifier(X_tr20_full, y_tr_full, obj_top20_full, seed)
    cb_test_preds.append(m.predict_proba(X_test20_full)[:, 1])
    logger.info(f"    [CB_selected-full] seed={seed}: 完了")
cb_test = np.mean(cb_test_preds, axis=0)
path1 = save_submission(cb_test, "cb_selected_classifier")

logger.info("[TabPFN_selected-full] シード平均で学習・予測中...")
(tabpfn_test,) = tabpfn_fit_predict(tf_full, [ttf_full], top20_cols_full, TABPFN_SEEDS)
path2 = save_submission(tabpfn_test, "tabpfn_selected_classifier")

blend_test = best_w * cb_test + (1 - best_w) * tabpfn_test
path3 = save_submission(blend_test, f"blend_argmin_w{round(best_w * 100):d}pct")

np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_cb_selected_testpreds.npy", cb_test)
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_tabpfn_selected_testpreds.npy", tabpfn_test)

logger.info("=" * 60)
logger.info(f"局所OOF: CB_selected={cb_score:.6f} / TabPFN_selected={tabpfn_score:.6f} / "
            f"argminブレンド(w_cb={best_w:.2f})={best_ll:.6f} / 相関={corr_local:.4f}")
logger.info("=== 全3ファイル出力完了 ===")
for p in [path1, path2, path3]:
    logger.info(f"  {p}")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
