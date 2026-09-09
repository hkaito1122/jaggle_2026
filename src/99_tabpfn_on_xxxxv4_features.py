"""99_tabpfn_on_xxxxv4_features

`97_ft_transformer_on_xxxxv4_features`は「FT-Transformer × xxxx_v4独自144列プール」を検証し、
FT-Transformer機構がCatBoost比+0.0815で明確に劣ると確定した([[ft-transformer-96-action-plan]])。
一方`98_`で97_の副産物のCatBoost単体（xxxx_v4独自144列プール）を現最良ブレンドに追加したが、
相関0.9440が高すぎて悪化した([[best-submission-status]]の98_セクション)。

このプロジェクトで「モデル機構 × 特徴量プール」の2軸のうち、GPU機構でCatBoostを唯一上回った
実績があるのはFT-TransformerではなくTabPFN v2（[[tabpfn-ensemble-partner]]、86_/93_で84_プール
に対して確認済み）。TabPFN × xxxx_v4独自プールはまだ未着手のセルであり、本スクリプトで検証する。

97_と全く同一の特徴量エンジニアリング（xxxx_v4本来の月次集約・入社時特徴量・独自regexメモ
パース＋TF-IDF/SVD45列、計144列）をそのまま流用する。CatBoost単体は97_で既にOOF・Test予測を
`.npy`で保存済み（局所val=0.528230）であり、**foldの切り方（StratifiedKFold, n_splits=5,
random_state=42, train_idsの並びも同一）が97_と完全に一致するため、再学習せず97_の保存済み
`cb_oofpreds.npy`/`cb_testpreds.npy`をそのまま読み込んで比較・ブレンドに使う**
（CatBoostを一切呼ばないため、[[local-catboost-torch-segfault]]のCatBoost→PyTorch/TabPFN
交互呼び出しセグフォルトも本スクリプトでは発生しない）。

TabPFN統合は`93_tabpfn_top20pct_on_77`と同一パターン（`tabpfn==2.2.1`固定・GPU必須・
NaN埋めなし・カテゴリは序数コード化してcategorical_features_indicesで渡す・
シード平均[42, 2024, 7]）。

出力2ファイル: tabpfn_classifier / blend_argmin
（cb_classifierは97_の`20260824_97_ft_transformer_on_xxxxv4_features_cb_classifier_submission.csv`
をそのまま使うため、本スクリプトでは再出力しない）
"""
import datetime
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger
from common.utils.seed import seed_everything

SEED = 42
seed_everything(seed=SEED)

TARGET_COL = "10年定着ラベル"
ID_COL = "社員ID"

SCRIPT_NAME = "99_tabpfn_on_xxxxv4_features"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 実験開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIR = PROJECT_ROOT / "data" / "input"

# 97_が保存したCatBoost(xxxx_v4独自144列プール)のOOF・Test予測(再学習せず流用)
CB97_OOF_PATH = PROJECT_ROOT / "data" / "output" / "20260824" / \
    "20260824_97_ft_transformer_on_xxxxv4_features_cb_oofpreds.npy"
CB97_TEST_PATH = PROJECT_ROOT / "data" / "output" / "20260824" / \
    "20260824_97_ft_transformer_on_xxxxv4_features_cb_testpreds.npy"

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
# 97_からの移植(xxxx_v4.ipynb由来の月次集約・入社時特徴量・メモパース、144列)
# 97_と一字一句同じロジック — foldのOOF位置を揃えるため変更禁止
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


def create_persona_features_xxxxv4(入社時データ):
    d = 入社時データ.copy()
    d["入社年"] = pd.to_datetime(d["入社日"]).dt.year
    d["前職職種"] = d["前職職種"].fillna("前職なし")
    d["初期等級_num"] = d["初期等級"].map(等級の数値)
    d["初期役割_num"] = d["初期役割"].map(役割の数値)
    for 列 in TEXT_COLS:
        d[f"{列}_文字数"] = d[列].str.len()
    return d


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


logger.info("=" * 60)
logger.info("xxxx_v4由来の特徴量を生成中(97_と同一、split非依存)...")
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
logger.info(f"tf_full 特徴量数: {len(FULL_FEATURE_COLS)}（97_と同じ144列のはず）")
assert len(FULL_FEATURE_COLS) == 144, f"想定外の列数: {len(FULL_FEATURE_COLS)}"


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}）")
    return str(path)


# ============================================================
# TabPFN v2 統合(93_/63_と同一パターン)
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

TABPFN_SEEDS = [42, 2024, 7]
OOF_SEED = 42
OOF_N_SPLITS = 5


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
# リークなしKFold OOF: 97_と全く同一のfold割当でTabPFNを評価
# ============================================================
logger.info("=" * 60)
logger.info("[TabPFN] xxxx_v4由来144列でのOOFを構築中(97_と同一fold割当)...")
skf = StratifiedKFold(n_splits=OOF_N_SPLITS, shuffle=True, random_state=OOF_SEED)

tabpfn_oof = np.zeros(len(train_ids))
for fold_i, (tr_pos, va_pos) in enumerate(skf.split(train_ids, y_train)):
    fold_train_ids = train_ids[tr_pos].tolist()
    fold_val_ids = train_ids[va_pos].tolist()
    fold_train_df = tf_full.loc[fold_train_ids]
    fold_val_df = tf_full.loc[fold_val_ids]
    (tp_va,) = tabpfn_fit_predict(fold_train_df, [fold_val_df], FULL_FEATURE_COLS, TABPFN_SEEDS)
    tabpfn_oof[va_pos] = tp_va
    logger.info(f"    [fold{fold_i}] 完了")

# ============================================================
# 97_が保存したCatBoost(同一144列プール)のOOF・Testを読み込んで比較・ブレンド
# ============================================================
logger.info("=" * 60)
logger.info(f"97_保存済みのCB OOF/Testを読み込み中: {CB97_OOF_PATH.name}")
cb_oof = np.load(CB97_OOF_PATH)
cb_test = np.load(CB97_TEST_PATH)
assert len(cb_oof) == len(train_ids), f"97_のOOF長さ不一致: {len(cb_oof)} vs {len(train_ids)}"
assert len(cb_test) == len(test_ids), f"97_のTest長さ不一致: {len(cb_test)} vs {len(test_ids)}"

cb_score = log_loss(y_train.values[surv_mask], cb_oof[surv_mask])
tabpfn_score = log_loss(y_train.values[surv_mask], tabpfn_oof[surv_mask])
corr = float(np.corrcoef(cb_oof, tabpfn_oof)[0, 1])
logger.info(f"[xxxxv4feat] CB val(97_保存値、再確認)={cb_score:.6f} / TabPFN val={tabpfn_score:.6f} / 相関={corr:.4f}")
logger.info("(参考: 97_ログ記載のCB val=0.528230と一致するはず)")

# ============================================================
# ブレンド曲線(argmin)
# ============================================================
logger.info("=" * 60)
logger.info("[ブレンド曲線] w=CatBoost側の重みを0〜1で走査...")
y_surv = y_train.values[surv_mask]
cb_s = cb_oof[surv_mask]
tp_s = tabpfn_oof[surv_mask]
BLEND_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
scan_rows = []
for w in BLEND_WEIGHTS:
    blend = np.clip(w * cb_s + (1 - w) * tp_s, 1e-9, 1 - 1e-9)
    ll = log_loss(y_surv, blend)
    scan_rows.append((w, ll))
    logger.info(f"    w_cb={w:.1f}: val={ll:.6f}")
best_w, best_ll = min(scan_rows, key=lambda r: r[1])
logger.info(f"[argmin] best_w_cb={best_w:.2f}, best_val={best_ll:.6f} "
            f"(CB単体比 {best_ll - cb_score:+.6f} / TabPFN単体比 {best_ll - tabpfn_score:+.6f})")

# ============================================================
# OOFの保存
# ============================================================
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_tabpfn_oofpreds.npy", tabpfn_oof)
logger.info("  OOF予測(.npy)を保存しました")

# ============================================================
# 提出: Train全件でTabPFN(シード平均)・argminブレンド(CB側は97_のTest予測をそのまま使用)
# ============================================================
logger.info("=" * 60)
logger.info("提出用: Train全件でTabPFNを学習・予測中...")
(tabpfn_test,) = tabpfn_fit_predict(tf_full, [ttf_full], FULL_FEATURE_COLS, TABPFN_SEEDS)
path1 = save_submission(tabpfn_test, "tabpfn_classifier")

blend_test = best_w * cb_test + (1 - best_w) * tabpfn_test
path2 = save_submission(blend_test, f"blend_argmin_w{round(best_w * 100):d}pct")

np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_tabpfn_testpreds.npy", tabpfn_test)

logger.info("=" * 60)
logger.info(f"局所OOF: CB={cb_score:.6f} / TabPFN={tabpfn_score:.6f} / argminブレンド(w_cb={best_w:.2f})={best_ll:.6f} / 相関={corr:.4f}")
logger.info("=== 全2ファイル出力完了(cb_classifierは97_のものをそのまま使用) ===")
for p in [path1, path2]:
    logger.info(f"  {p}")
logger.info("(参考: 86_のTabPFN134 Public=0.507300、93_のTabPFN89 Public=0.504065——"
            "84_プールでは列数が少ないほどTabPFNが強い傾向。xxxx_v4独自144列プールでの初検証)")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
