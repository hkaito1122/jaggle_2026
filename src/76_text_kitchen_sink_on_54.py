"""76_text_kitchen_sink_on_54

54_l2_m_interaction (R0_memofix_plus_LM, 444列) に、ユーザーが列挙した9つのテキスト数値化技法のうち
未実施だった3つ（BoW+単語n-gram、事前学習済み埋め込み、キーワード辞書ベースの感情スコア）を追加する。
文字数・正規表現抽出・キーワードフラグ(勤務地/転居)・TF-IDF・PCA/SVD次元削減は54_に既に存在するため
そのまま流用する。

出力6ファイル:
  1. full_classifier              : 444列 + 新規99列(BoW/Emb/Sentiment) の分類器
  2. narrowed_classifier          : 444列 + 新規列のうちCatBoost重要度>0のものだけ(544→537列)
  3. full_regression_blend        : 1と同じ特徴量で、xxxx_v4方式の在籍月数回帰+Platt較正を85:15ブレンド
  4. narrowed_regression_blend    : 2と同じ特徴量で同上
  5. top20pct_classifier          : FULL(543列)の重要度上位20%(109列)だけを使う分類器
  6. top20pct_regression_blend    : 5と同じ特徴量で回帰ブレンド

54_のfeature engineeringロジック(モデル関数含む)は極力そのまま移植し、変更点は
「新規テキスト特徴量ブロックの追加」と「実行環境をColab→ローカルに変更」のみ。
"""
import datetime
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from janome.tokenizer import Tokenizer as JanomeTokenizer

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger
from common.utils.seed import seed_everything

SEED = 42
seed_everything(seed=SEED)

TARGET_COL = "10年定着ラベル"
ID_COL = "社員ID"

SCRIPT_NAME = "76_text_kitchen_sink_on_54"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 実験開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DIR = PROJECT_ROOT / "data" / "input"

train_persona = pd.read_csv(INPUT_DIR / "employee_persona_train.csv")
test_persona = pd.read_csv(INPUT_DIR / "employee_persona_test.csv")
train_monthly = pd.read_csv(INPUT_DIR / "employee_monthly_train.csv")
test_monthly = pd.read_csv(INPUT_DIR / "employee_monthly_test.csv")
train_monthly_full = pd.read_csv(INPUT_DIR / "employee_monthly_train_full.csv")

y_train = train_persona[TARGET_COL]
train_ids = train_persona[ID_COL].values
test_ids = test_persona[ID_COL].values
logger.info(f"Train Persona: {train_persona.shape}, Test Persona: {test_persona.shape}")

EARLY_LEAVER_IDS = set(train_monthly.loc[train_monthly["月末在籍状態"] == "退職", ID_COL].unique())
_test_early = set(test_monthly.loc[test_monthly["月末在籍状態"] == "退職", ID_COL].unique())
assert len(_test_early) == 0

TEXT_COLS = ["入社時メモ", "上司からのフィードバック", "同僚からのフィードバック"]
TEXT_TAG = {"入社時メモ": "memo", "上司からのフィードバック": "sup", "同僚からのフィードバック": "peer"}


# ============================================================
# 54_l2_m_interaction.ipynb と同一の特徴量関数（split非依存）
# ============================================================
def create_monthly_aggregation_features(monthly_df, employee_ids):
    numeric_cols = [
        "残業時間", "有給取得日数", "欠勤日数", "研修時間",
        "上司との面談実施回数", "情報共有件数", "在宅勤務日数",
        "360度評価_親和度", "360度評価_信頼度", "360度評価_主体度",
        "360度評価_学習度", "360度評価_共有貢献度", "360度評価者数",
        "顧客満足度評価", "担当プロジェクト数", "月例給与_円"
    ]
    features_list = []
    for employee_id in employee_ids:
        emp_data = monthly_df[monthly_df["社員ID"] == employee_id].copy()
        emp_data = emp_data.sort_values("経過月数").reset_index(drop=True)
        features = {"社員ID": employee_id}
        for col in numeric_cols:
            if col not in emp_data.columns:
                continue
            values = emp_data[col].values
            valid_values = values[~pd.isna(values)]
            features[f"{col}_mean"] = np.mean(valid_values) if len(valid_values) > 0 else np.nan
            features[f"{col}_std"] = np.std(valid_values) if len(valid_values) > 1 else np.nan
            features[f"{col}_min"] = np.min(valid_values) if len(valid_values) > 0 else np.nan
            features[f"{col}_max"] = np.max(valid_values) if len(valid_values) > 0 else np.nan
            features[f"{col}_median"] = np.median(valid_values) if len(valid_values) > 0 else np.nan
            mean_val = features[f"{col}_mean"]
            std_val = features[f"{col}_std"]
            features[f"{col}_cv"] = std_val / mean_val if (mean_val and mean_val != 0) else np.nan
            early = emp_data[emp_data["経過月数"].between(0, 2)][col]
            mid = emp_data[emp_data["経過月数"].between(3, 11)][col]
            late = emp_data[emp_data["経過月数"].between(12, 23)][col]
            features[f"{col}_early_mean"] = early.mean()
            features[f"{col}_mid_mean"] = mid.mean()
            features[f"{col}_late_mean"] = late.mean()
            features[f"{col}_late_minus_early"] = late.mean() - early.mean()
            features[f"{col}_late_early_ratio"] = (
                late.mean() / early.mean() if early.mean() and early.mean() != 0 else np.nan
            )
            if len(valid_values) >= 2:
                valid_indices = np.where(~pd.isna(values))[0]
                if len(valid_indices) >= 2:
                    slope, _, _, _, _ = stats.linregress(valid_indices, valid_values)
                    features[f"{col}_slope"] = slope
                else:
                    features[f"{col}_slope"] = np.nan
                first_val, last_val = valid_values[0], valid_values[-1]
                features[f"{col}_diff"] = last_val - first_val
                features[f"{col}_ratio"] = last_val / first_val if first_val != 0 else np.nan
            else:
                features[f"{col}_slope"] = features[f"{col}_diff"] = features[f"{col}_ratio"] = np.nan
        features_list.append(features)
    return pd.DataFrame(features_list)


def create_monthly_categorical_change_features(monthly_df, employee_ids):
    categorical_cols = ["部署ID", "職種", "役割", "等級", "勤務地", "上司ID"]
    features_list = []
    for employee_id in employee_ids:
        emp_data = monthly_df[monthly_df["社員ID"] == employee_id].sort_values("経過月数").reset_index(drop=True)
        features = {"社員ID": employee_id}
        for col in categorical_cols:
            if col not in emp_data.columns:
                continue
            values = emp_data[col].values
            changes = sum(1 for i in range(1, len(values)) if pd.notna(values[i]) and pd.notna(values[i - 1]) and values[i] != values[i - 1])
            features[f"{col}_changes"] = changes
            features[f"{col}_unique_count"] = len(pd.Series(values).dropna().unique())
        if "月末在籍状態" in emp_data.columns:
            status_values = emp_data["月末在籍状態"].values
            features["leave_of_absence_flag"] = int("休職" in status_values)
            features["leave_of_absence_months"] = np.sum(status_values == "休職")
        features_list.append(features)
    return pd.DataFrame(features_list)


def create_missing_value_features(monthly_df, employee_ids):
    missing_target_cols = ["360度評価_親和度", "360度評価_信頼度", "顧客満足度評価", "担当プロジェクト数"]
    features_list = []
    for employee_id in employee_ids:
        emp_data = monthly_df[monthly_df["社員ID"] == employee_id].sort_values("経過月数").reset_index(drop=True)
        features = {"社員ID": employee_id}
        for col in missing_target_cols:
            if col in emp_data.columns:
                values = emp_data[col].values
                total_months = len(values)
                features[f"{col}_missing_rate"] = pd.isna(values).sum() / total_months if total_months > 0 else np.nan
        features_list.append(features)
    return pd.DataFrame(features_list)


def create_domain_knowledge_features(monthly_df, employee_ids):
    features_list = []
    for employee_id in employee_ids:
        emp_data = monthly_df[monthly_df["社員ID"] == employee_id].sort_values("経過月数").reset_index(drop=True)
        features = {"社員ID": employee_id}
        eval_cols = ["360度評価_親和度", "360度評価_信頼度", "360度評価_主体度", "360度評価_学習度", "360度評価_共有貢献度"]
        eval_mean_list = [emp_data[col].mean() for col in eval_cols if col in emp_data.columns]
        features["engagement_score"] = np.nanmean(eval_mean_list) if len(eval_mean_list) > 0 else np.nan
        if "残業時間" in emp_data.columns:
            features["overtime_stability"] = emp_data["残業時間"].std()
        if "研修時間" in emp_data.columns and "残業時間" in emp_data.columns:
            training_mean = emp_data["研修時間"].mean()
            overtime_mean = emp_data["残業時間"].mean()
            features["training_overtime_ratio"] = training_mean / overtime_mean if overtime_mean > 0 else np.nan
        features_list.append(features)
    return pd.DataFrame(features_list)


def create_advanced_statistical_features(monthly_df, employee_ids):
    numeric_cols = ["残業時間", "有給取得日数", "研修時間", "360度評価_親和度", "360度評価_信頼度"]
    features_list = []
    for employee_id in employee_ids:
        emp_data = monthly_df[monthly_df["社員ID"] == employee_id].sort_values("経過月数").reset_index(drop=True)
        features = {"社員ID": employee_id}
        for col in numeric_cols:
            if col in emp_data.columns:
                values = emp_data[col].dropna().values
                if len(values) >= 3:
                    features[f"{col}_skew"] = stats.skew(values)
                    features[f"{col}_kurtosis"] = stats.kurtosis(values)
                    features[f"{col}_q25"] = np.percentile(values, 25)
                    features[f"{col}_q75"] = np.percentile(values, 75)
                    features[f"{col}_iqr"] = features[f"{col}_q75"] - features[f"{col}_q25"]
                else:
                    features[f"{col}_skew"] = features[f"{col}_kurtosis"] = np.nan
                    features[f"{col}_q25"] = features[f"{col}_q75"] = features[f"{col}_iqr"] = np.nan
        features_list.append(features)
    return pd.DataFrame(features_list)


def create_cluster_features(monthly_df, employee_ids, n_clusters=5, seed=42):
    key_cols = ["残業時間", "有給取得日数", "研修時間", "360度評価_親和度", "360度評価_信頼度"]
    agg_data = []
    for employee_id in employee_ids:
        emp_data = monthly_df[monthly_df["社員ID"] == employee_id]
        row = {"社員ID": employee_id}
        for col in key_cols:
            if col in emp_data.columns:
                row[col] = emp_data[col].mean()
        agg_data.append(row)
    agg_df = pd.DataFrame(agg_data)
    feature_cols = [c for c in key_cols if c in agg_df.columns]
    X = agg_df[feature_cols].fillna(-999)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    agg_df["cluster"] = kmeans.fit_predict(X_scaled)
    return agg_df[["社員ID", "cluster"]]


def create_eda_driven_features(monthly_df, employee_ids):
    eval_cols = ["360度評価_親和度", "360度評価_信頼度", "360度評価_主体度", "360度評価_学習度", "360度評価_共有貢献度"]
    features_list = []
    for employee_id in employee_ids:
        emp_data = monthly_df[monthly_df["社員ID"] == employee_id].sort_values("経過月数").reset_index(drop=True)
        features = {"社員ID": employee_id}
        absence_vals = emp_data["欠勤日数"].values
        nonzero = absence_vals > 0
        features["欠勤発生月数"] = int(nonzero.sum())
        max_run = cur_run = 0
        for v in nonzero:
            cur_run = cur_run + 1 if v else 0
            max_run = max(max_run, cur_run)
        features["欠勤_最長連続月数"] = max_run
        features["欠勤_連続フラグ"] = int(max_run >= 2)
        flagged = emp_data[emp_data["360度評価更新フラグ"] == 1]
        first_month = flagged["経過月数"].min() if len(flagged) > 0 else np.nan
        features["初回評価月"] = first_month
        features["is_早期評価"] = int(first_month <= 4) if pd.notna(first_month) else 0
        features["is_遅延評価"] = int(first_month >= 7) if pd.notna(first_month) else 0
        features["評価遅延度"] = abs(first_month - 5) if pd.notna(first_month) else np.nan
        for col in ["残業時間", "月例給与_円"]:
            vals = emp_data[col].dropna().values
            features[f"{col}_volatility"] = np.mean(np.abs(np.diff(vals))) if len(vals) >= 2 else np.nan
        n_months = len(emp_data)
        features["有給取得率"] = emp_data["有給取得日数"].sum() / n_months if n_months > 0 else np.nan
        features["評価項目間ばらつき"] = emp_data[eval_cols].std(axis=1).mean()
        features_list.append(features)
    return pd.DataFrame(features_list)


def create_manager_team_size_features(monthly_df, employee_ids):
    month0 = monthly_df[monthly_df["経過月数"] == 0].copy()
    month0["初期上司_部下数"] = month0.groupby("上司ID")["社員ID"].transform("count")
    out = month0[["社員ID", "初期上司_部下数"]]
    return out[out["社員ID"].isin(employee_ids)].reset_index(drop=True)


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
    return train_out, test_out, svd.explained_variance_ratio_.sum()


def create_quarterly_features(monthly_df, employee_ids, metrics, suffix=""):
    quarters = {"q1": (0, 5), "q2": (6, 11), "q3": (12, 17), "q4": (18, 23)}
    features_list = []
    for employee_id in employee_ids:
        emp_data = monthly_df[monthly_df["社員ID"] == employee_id].sort_values("経過月数")
        features = {"社員ID": employee_id}
        for metric in metrics:
            q_means = {}
            for qname, (lo, hi) in quarters.items():
                vals = emp_data[emp_data["経過月数"].between(lo, hi)][metric]
                q_means[qname] = vals.mean()
                features[f"{metric}_{qname}_mean{suffix}"] = q_means[qname]
            first_half_delta = q_means["q2"] - q_means["q1"] if pd.notna(q_means["q1"]) and pd.notna(q_means["q2"]) else np.nan
            second_half_delta = q_means["q4"] - q_means["q3"] if pd.notna(q_means["q3"]) and pd.notna(q_means["q4"]) else np.nan
            features[f"{metric}_acceleration{suffix}"] = (
                second_half_delta - first_half_delta if pd.notna(first_half_delta) and pd.notna(second_half_delta) else np.nan
            )
        features_list.append(features)
    return pd.DataFrame(features_list)


def extract_workstyle_section(text):
    if pd.isna(text):
        return None
    m = re.search(r"勤務地・働き方：(.+?)$", text, re.S)
    if m:
        return m.group(1).strip()
    lines = [l for l in text.strip().splitlines() if re.search(r"勤務地|転居|在宅勤務", l)]
    return "".join(lines) if lines else None


NEG_RELOC = re.compile(r"転居を伴う(異動|勤務地変更)[はも]?(許容せず|許容していない|許容しておらず|希望せず|希望しておらず|希望していない)")
POS_RELOC = re.compile(r"転居を伴う(異動|勤務地変更)[はもを]?(許容し?ており|許容)")


def classify_reloc(s):
    if s is None:
        return None
    if NEG_RELOC.search(s):
        return False
    if POS_RELOC.search(s):
        return True
    return None


def extract_desired_location_v2(s):
    if s is None:
        return None
    m = re.search(r"(?:勤務地は|希望勤務地は)(.+?)(?:を希望|。)", s)
    loc = m.group(1) if m else None
    if loc is None:
        m2 = re.search(r"(.+?)を希望勤務地", s)
        loc = m2.group(1) if m2 else None
    if loc is None:
        m3 = re.search(r"^([一-龥ぁ-んァ-ンー]+?)(?:での勤務|勤務)?を希望。", s)
        loc = m3.group(1) if m3 else None
    if loc is not None:
        loc = loc.strip("「」")
    return loc


def create_relocation_mismatch_features(persona_df, extract_fn, state_col, flag_col):
    ws_section = persona_df["入社時メモ"].apply(extract_workstyle_section)
    reloc_ok_raw = ws_section.apply(classify_reloc)
    desired = ws_section.apply(extract_fn)
    actual = persona_df["初期勤務地"]
    match = (desired == actual) & desired.notna()
    reloc_true = reloc_ok_raw == True
    reloc_false = reloc_ok_raw == False
    valid = desired.notna() & reloc_ok_raw.notna()
    state = pd.Series("unknown", index=persona_df.index)
    state[valid & reloc_true & match] = "許容_一致"
    state[valid & reloc_true & ~match] = "許容_不一致"
    state[valid & reloc_false & match] = "非許容_一致"
    state[valid & reloc_false & ~match] = "非許容_不一致"
    double_bad = (valid & reloc_false & ~match).astype(int)
    return pd.DataFrame({
        "社員ID": persona_df["社員ID"].values,
        state_col: state.values,
        flag_col: double_bad.values,
    })


_ANALYTICAL_MAJOR = {"情報", "理工学"}
_ANALYTICAL_JOB = {"IT・エンジニアリング", "データ・商品企画・コンサルティング"}


def create_l2_m_interaction_features(persona_df, reloc_v2_df):
    is_analytical_major = persona_df["専攻分野"].isin(_ANALYTICAL_MAJOR)
    is_analytical_job = persona_df["初期職種"].isin(_ANALYTICAL_JOB)
    m_bad = (~is_analytical_major & is_analytical_job).astype(int)
    state = reloc_v2_df.set_index("社員ID").loc[persona_df["社員ID"], "転居x勤務地_状態_v2"].values
    l2_bad = (state == "非許容_不一致").astype(int)
    both_bad = (l2_bad & m_bad)
    risk_count = l2_bad + m_bad
    return pd.DataFrame({
        "社員ID": persona_df["社員ID"].values,
        "M_不適合": m_bad,
        "L2xM_ダブル不適合": both_bad,
        "L2xM_リスク要因数": risk_count,
    })


def create_department_target_encoding(train_persona, test_persona, y_train, fit_ids, seed=42, n_splits=5, smoothing=10):
    col = "初期部署ID"
    is_fit = train_persona[ID_COL].isin(fit_ids).values
    dept_all = train_persona[col].values
    y_arr = y_train.values
    global_mean = y_arr[is_fit].mean()
    fit_indices = np.where(is_fit)[0]
    dept_fit = dept_all[fit_indices]
    y_fit = y_arr[fit_indices]
    train_te = np.full(len(train_persona), global_mean)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(fit_indices):
        df_tr = pd.DataFrame({col: dept_fit[tr_idx], "y": y_fit[tr_idx]})
        stats_tr = df_tr.groupby(col)["y"].agg(["mean", "count"])
        smoothed = (stats_tr["count"] * stats_tr["mean"] + smoothing * global_mean) / (stats_tr["count"] + smoothing)
        mapping = smoothed.to_dict()
        actual_val_idx = fit_indices[val_idx]
        train_te[actual_val_idx] = pd.Series(dept_fit[val_idx]).map(mapping).fillna(global_mean).values
    df_full = pd.DataFrame({col: dept_fit, "y": y_fit})
    stats_full = df_full.groupby(col)["y"].agg(["mean", "count"])
    smoothed_full = (stats_full["count"] * stats_full["mean"] + smoothing * global_mean) / (stats_full["count"] + smoothing)
    mapping_full = smoothed_full.to_dict()
    dept_size_map = stats_full["count"].to_dict()
    not_fit_indices = np.where(~is_fit)[0]
    train_te[not_fit_indices] = pd.Series(dept_all[not_fit_indices]).map(mapping_full).fillna(global_mean).values
    test_te = test_persona[col].map(mapping_full).fillna(global_mean).values
    train_out = pd.DataFrame({
        ID_COL: train_persona[ID_COL].values,
        "dept_target_enc": train_te,
        "dept_size": pd.Series(dept_all).map(dept_size_map).fillna(0).values,
    })
    test_out = pd.DataFrame({
        ID_COL: test_persona[ID_COL].values,
        "dept_target_enc": test_te,
        "dept_size": test_persona[col].map(dept_size_map).fillna(0).values,
    })
    return train_out, test_out


# ============================================================
# NEW: BoW + 単語n-gram(1-3) + SVD（54_に無かった技法）
# ============================================================
logger.info("=" * 60)
logger.info("[NEW-1] janome分かち書き → BoW(word, n-gram 1-3) + SVD を生成中...")
_jt = JanomeTokenizer()


def _janome_wakachi(text):
    if pd.isna(text) or text == "":
        return ""
    return " ".join(tok.surface for tok in _jt.tokenize(str(text)))


def create_bow_ngram_features(train_persona, test_persona, col, max_features=300, n_components=15, min_df=3, seed=42):
    train_text = train_persona[col].fillna("").astype(str).apply(_janome_wakachi)
    test_text = test_persona[col].fillna("").astype(str).apply(_janome_wakachi)
    vec = CountVectorizer(analyzer="word", ngram_range=(1, 3), max_features=max_features, min_df=min_df)
    train_bow = vec.fit_transform(train_text)
    test_bow = vec.transform(test_text)
    n_comp = min(n_components, train_bow.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed, algorithm="randomized")
    train_svd = svd.fit_transform(train_bow)
    test_svd = svd.transform(test_bow)
    col_names = [f"{col}_bow_ngram_svd_{i}" for i in range(n_comp)]
    tr = pd.DataFrame(train_svd, columns=col_names)
    tr[ID_COL] = train_persona[ID_COL].values
    te = pd.DataFrame(test_svd, columns=col_names)
    te[ID_COL] = test_persona[ID_COL].values
    return tr, te, svd.explained_variance_ratio_.sum()


bow_train_list, bow_test_list = [], []
for col in TEXT_COLS:
    tr, te, ev = create_bow_ngram_features(train_persona, test_persona, col)
    logger.info(f"  {col}: BoW+n-gram SVD累積寄与率={ev:.3f}")
    bow_train_list.append(tr)
    bow_test_list.append(te)

# ============================================================
# NEW: 事前学習済み文埋め込み（multilingual-e5-small）+ PCA（19_で既に不採用確定した技法だが再検証）
# ============================================================
logger.info("=" * 60)
logger.info("[NEW-2] 事前学習済み文埋め込み(multilingual-e5-small) + PCA を生成中...")
from sentence_transformers import SentenceTransformer

_emb_model = SentenceTransformer("intfloat/multilingual-e5-small")


def create_embedding_features(train_persona, test_persona, col, n_components=15, seed=42):
    train_text = ("passage: " + train_persona[col].fillna("").astype(str)).tolist()
    test_text = ("passage: " + test_persona[col].fillna("").astype(str)).tolist()
    train_vec = _emb_model.encode(train_text, batch_size=64, show_progress_bar=False)
    test_vec = _emb_model.encode(test_text, batch_size=64, show_progress_bar=False)
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="full")
    train_pca = pca.fit_transform(train_vec)
    test_pca = pca.transform(test_vec)
    col_names = [f"{col}_emb_pca_{i}" for i in range(n_components)]
    tr = pd.DataFrame(train_pca, columns=col_names)
    tr[ID_COL] = train_persona[ID_COL].values
    te = pd.DataFrame(test_pca, columns=col_names)
    te[ID_COL] = test_persona[ID_COL].values
    return tr, te, pca.explained_variance_ratio_.sum()


emb_train_list, emb_test_list = [], []
for col in TEXT_COLS:
    tr, te, ev = create_embedding_features(train_persona, test_persona, col)
    logger.info(f"  {col}: Embedding PCA累積寄与率={ev:.3f}")
    emb_train_list.append(tr)
    emb_test_list.append(te)

# ============================================================
# NEW: キーワード辞書ベースの感情/極性スコア（73_と同一の語彙を再利用）
# ============================================================
logger.info("=" * 60)
logger.info("[NEW-3] キーワード辞書ベースの感情/極性スコアを生成中...")
NEG_WORDS = ["遅刻", "課題", "不足", "苦手", "難しい", "抱え込", "確認漏れ", "手戻り",
             "受け身", "指示待ち", "戸惑", "偏り", "限定的"]
POS_WORDS = ["期待", "優秀", "リーダー", "主体的", "自律", "率先", "丁寧", "着実",
             "信頼", "貢献", "柔軟", "前向き", "素地"]


def create_sentiment_features(persona):
    out = pd.DataFrame({ID_COL: persona[ID_COL].values})
    for col in TEXT_COLS:
        t = persona[col].fillna("").astype(str)
        tag = TEXT_TAG[col]
        out[f"KW_{tag}_neg数"] = sum(t.str.contains(w, regex=False).astype(int) for w in NEG_WORDS).to_numpy()
        out[f"KW_{tag}_pos数"] = sum(t.str.contains(w, regex=False).astype(int) for w in POS_WORDS).to_numpy()
        out[f"KW_{tag}_極性差"] = out[f"KW_{tag}_pos数"] - out[f"KW_{tag}_neg数"]
    return out


train_sentiment = create_sentiment_features(train_persona)
test_sentiment = create_sentiment_features(test_persona)
logger.info(f"  感情特徴量: Train {train_sentiment.shape}, Test {test_sentiment.shape}")

NEW_BLOCK_COLS = (
    [c for df in bow_train_list for c in df.columns if c != ID_COL]
    + [c for df in emb_train_list for c in df.columns if c != ID_COL]
    + [c for c in train_sentiment.columns if c != ID_COL]
)
logger.info(f"新規テキスト特徴量ブロック 合計 {len(NEW_BLOCK_COLS)} 列 "
            f"(BoW+ngram {sum(len(df.columns)-1 for df in bow_train_list)}, "
            f"Embedding {sum(len(df.columns)-1 for df in emb_train_list)}, "
            f"Sentiment {len(train_sentiment.columns)-1})")


# ============================================================
# split非依存の基本特徴量を生成（54_と同一）
# ============================================================
logger.info("=" * 60)
logger.info("split非依存の基本特徴量を生成中...")

train_monthly_agg = create_monthly_aggregation_features(train_monthly, train_ids)
test_monthly_agg = create_monthly_aggregation_features(test_monthly, test_ids)
train_cat_change = create_monthly_categorical_change_features(train_monthly, train_ids)
test_cat_change = create_monthly_categorical_change_features(test_monthly, test_ids)
train_missing = create_missing_value_features(train_monthly, train_ids)
test_missing = create_missing_value_features(test_monthly, test_ids)
train_domain = create_domain_knowledge_features(train_monthly, train_ids)
test_domain = create_domain_knowledge_features(test_monthly, test_ids)
train_advanced_stats = create_advanced_statistical_features(train_monthly, train_ids)
test_advanced_stats = create_advanced_statistical_features(test_monthly, test_ids)
train_cluster = create_cluster_features(train_monthly, train_ids, n_clusters=5, seed=SEED)
test_cluster = create_cluster_features(test_monthly, test_ids, n_clusters=5, seed=SEED)
train_eda_feats = create_eda_driven_features(train_monthly, train_ids)
test_eda_feats = create_eda_driven_features(test_monthly, test_ids)
train_mgr = create_manager_team_size_features(train_monthly, train_ids)
test_mgr = create_manager_team_size_features(test_monthly, test_ids)

TEXT_COLS_ = TEXT_COLS
logger.info("テキストTF-IDF+SVD特徴量(A_v1)を生成中...")
tfidf_train_list, tfidf_test_list = [], []
for col in TEXT_COLS_:
    tr, te, ev = create_tfidf_svd_features(train_persona, test_persona, col, max_features=300, n_components=15, min_df=3, seed=SEED)
    tfidf_train_list.append(tr)
    tfidf_test_list.append(te)

D_EXPANDED_METRICS = [
    "残業時間", "有給取得日数", "欠勤日数", "研修時間",
    "上司との面談実施回数", "情報共有件数", "在宅勤務日数",
    "360度評価_親和度", "360度評価_信頼度", "360度評価_主体度",
    "360度評価_学習度", "360度評価_共有貢献度", "360度評価者数",
    "顧客満足度評価", "担当プロジェクト数", "月例給与_円",
]
train_quarterly_exp = create_quarterly_features(train_monthly, train_ids, D_EXPANDED_METRICS, suffix="_exp")
test_quarterly_exp = create_quarterly_features(test_monthly, test_ids, D_EXPANDED_METRICS, suffix="_exp")

train_persona["入社日"] = pd.to_datetime(train_persona["入社日"])
test_persona["入社日"] = pd.to_datetime(test_persona["入社日"])
for col in TEXT_COLS_:
    train_persona[f"{col}_len"] = train_persona[col].fillna("").astype(str).apply(len)
    test_persona[f"{col}_len"] = test_persona[col].fillna("").astype(str).apply(len)
train_persona["text_total_chars"] = train_persona[TEXT_COLS_].fillna("").apply(lambda x: sum(len(str(v)) for v in x), axis=1)
test_persona["text_total_chars"] = test_persona[TEXT_COLS_].fillna("").apply(lambda x: sum(len(str(v)) for v in x), axis=1)
train_persona["入社年"] = train_persona["入社日"].dt.year
train_persona["入社月"] = train_persona["入社日"].dt.month
train_persona["入社四半期"] = train_persona["入社日"].dt.quarter
test_persona["入社年"] = test_persona["入社日"].dt.year
test_persona["入社月"] = test_persona["入社日"].dt.month
test_persona["入社四半期"] = test_persona["入社日"].dt.quarter
train_persona["年齢_x_前職経験"] = train_persona["入社時年齢"] * train_persona["前職経験月数"]
test_persona["年齢_x_前職経験"] = test_persona["入社時年齢"] * test_persona["前職経験月数"]
grade_map = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}
train_persona["初期等級_num"] = train_persona["初期等級"].map(grade_map)
test_persona["初期等級_num"] = test_persona["初期等級"].map(grade_map)
train_persona["初任給_x_等級"] = train_persona["初任給_円"] * train_persona["初期等級_num"]
test_persona["初任給_x_等級"] = test_persona["初任給_円"] * test_persona["初期等級_num"]
train_persona["is_Q2_新卒"] = ((train_persona["入社四半期"] == 2) & (train_persona["入社区分"] == "新卒")).astype(int)
test_persona["is_Q2_新卒"] = ((test_persona["入社四半期"] == 2) & (test_persona["入社区分"] == "新卒")).astype(int)

logger.info("転居×勤務地マッチ交互作用特徴量(ブロックL v2)を生成中...")
train_reloc_v2 = create_relocation_mismatch_features(train_persona, extract_desired_location_v2, "転居x勤務地_状態_v2", "転居x勤務地_ダブル悪条件_v2")
test_reloc_v2 = create_relocation_mismatch_features(test_persona, extract_desired_location_v2, "転居x勤務地_状態_v2", "転居x勤務地_ダブル悪条件_v2")
train_l2m = create_l2_m_interaction_features(train_persona, train_reloc_v2)
test_l2m = create_l2_m_interaction_features(test_persona, test_reloc_v2)
logger.info("split非依存の基本特徴量生成完了")


# ============================================================
# prepare_split（54_と同一 + 新規3ブロックのmergeを追加）
# ============================================================
def prepare_split_extended(split_ratio, add_new_blocks=True, exclude_early_from_val=True):
    sorted_persona = train_persona.sort_values("入社日")
    split_point = int(len(sorted_persona) * split_ratio)
    train_period_ids = set(sorted_persona.iloc[:split_point][ID_COL])

    train_dept_te, test_dept_te = create_department_target_encoding(
        train_persona, test_persona, y_train, fit_ids=train_period_ids, seed=SEED, n_splits=5, smoothing=10
    )

    train_persona_features = train_persona.drop(columns=[TARGET_COL])
    tf = train_persona_features.merge(train_monthly_agg, on=ID_COL, how="left")
    tf = tf.merge(train_cat_change, on=ID_COL, how="left")
    tf = tf.merge(train_missing, on=ID_COL, how="left")
    tf = tf.merge(train_domain, on=ID_COL, how="left")
    tf = tf.merge(train_advanced_stats, on=ID_COL, how="left")
    tf = tf.merge(train_cluster, on=ID_COL, how="left")
    tf = tf.merge(train_dept_te, on=ID_COL, how="left")
    tf = tf.merge(train_eda_feats, on=ID_COL, how="left")
    tf = tf.merge(train_mgr, on=ID_COL, how="left")
    tf = tf.merge(train_quarterly_exp, on=ID_COL, how="left")
    for trdf in tfidf_train_list:
        tf = tf.merge(trdf, on=ID_COL, how="left")

    ttf = test_persona.merge(test_monthly_agg, on=ID_COL, how="left")
    ttf = ttf.merge(test_cat_change, on=ID_COL, how="left")
    ttf = ttf.merge(test_missing, on=ID_COL, how="left")
    ttf = ttf.merge(test_domain, on=ID_COL, how="left")
    ttf = ttf.merge(test_advanced_stats, on=ID_COL, how="left")
    ttf = ttf.merge(test_cluster, on=ID_COL, how="left")
    ttf = ttf.merge(test_dept_te, on=ID_COL, how="left")
    ttf = ttf.merge(test_eda_feats, on=ID_COL, how="left")
    ttf = ttf.merge(test_mgr, on=ID_COL, how="left")
    ttf = ttf.merge(test_quarterly_exp, on=ID_COL, how="left")
    for tedf in tfidf_test_list:
        ttf = ttf.merge(tedf, on=ID_COL, how="left")

    # L2 + LM は常に採用（R0_memofix_plus_LM = 54_現行チャンピオンに固定）
    tf = tf.merge(train_reloc_v2, on=ID_COL, how="left")
    tf = tf.merge(train_l2m, on=ID_COL, how="left")
    ttf = ttf.merge(test_reloc_v2, on=ID_COL, how="left")
    ttf = ttf.merge(test_l2m, on=ID_COL, how="left")

    if add_new_blocks:
        for trdf in bow_train_list:
            tf = tf.merge(trdf, on=ID_COL, how="left")
        for tedf in bow_test_list:
            ttf = ttf.merge(tedf, on=ID_COL, how="left")
        for trdf in emb_train_list:
            tf = tf.merge(trdf, on=ID_COL, how="left")
        for tedf in emb_test_list:
            ttf = ttf.merge(tedf, on=ID_COL, how="left")
        tf = tf.merge(train_sentiment, on=ID_COL, how="left")
        ttf = ttf.merge(test_sentiment, on=ID_COL, how="left")

    _train_period_features = tf[tf[ID_COL].isin(train_period_ids)]
    job_dev_metrics = ["残業時間_mean", "研修時間_mean", "360度評価_親和度_mean"]
    job_means = {m: _train_period_features.groupby("初期職種")[m].mean().to_dict() for m in job_dev_metrics}
    category_means_train = {m: _train_period_features.groupby("入社区分")[m].mean().to_dict() for m in job_dev_metrics}
    grade_salary_mean = _train_period_features.groupby("初期等級")["初任給_円"].mean().to_dict()
    category_salary_mean = _train_period_features.groupby("入社区分")["初任給_円"].mean().to_dict()
    grade_monthly_salary_mean = _train_period_features.groupby("初期等級")["月例給与_円_mean"].mean().to_dict()

    for df_ in [tf, ttf]:
        for m in job_dev_metrics:
            df_[f"{m}_job_deviation"] = df_[m] - df_["初期職種"].map(job_means[m])
        df_["研修時間_職種比"] = df_["研修時間_mean"] / df_["初期職種"].map(job_means["研修時間_mean"]).replace(0, np.nan)
        df_["研修時間_区分比"] = df_["研修時間_mean"] / df_["入社区分"].map(category_means_train["研修時間_mean"]).replace(0, np.nan)
        df_["初任給_等級内偏差"] = df_["初任給_円"] - df_["初期等級"].map(grade_salary_mean)
        df_["初任給_区分内偏差"] = df_["初任給_円"] - df_["入社区分"].map(category_salary_mean)
        df_["月例給与_等級内偏差"] = df_["月例給与_円_mean"] - df_["初期等級"].map(grade_monthly_salary_mean)

    drop_cols = ["入社時メモ", "上司からのフィードバック", "同僚からのフィードバック",
                 "初期部署ID", "初期等級", "最終学歴", "前職職種"]
    tf = tf.drop(columns=[c for c in drop_cols if c in tf.columns]).set_index(ID_COL)
    ttf = ttf.drop(columns=[c for c in drop_cols if c in ttf.columns]).set_index(ID_COL)

    target_series = train_persona.set_index(ID_COL)[TARGET_COL]
    tf_sorted = tf.sort_values("入社日")
    y_sorted = target_series.loc[tf_sorted.index]

    ag_train = tf_sorted.iloc[:split_point].copy()
    ag_tuning = tf_sorted.iloc[split_point:].copy()
    ag_train[TARGET_COL] = y_sorted.iloc[:split_point].values
    ag_tuning[TARGET_COL] = y_sorted.iloc[split_point:].values

    if exclude_early_from_val and len(ag_tuning) > 0:
        ag_tuning = ag_tuning[~ag_tuning.index.isin(EARLY_LEAVER_IDS)]

    return ag_train, ag_tuning, ttf


def _feature_cols(df):
    return [c for c in df.columns if c not in ["入社日", TARGET_COL]]


# ============================================================
# 全件(FULL) / 80-20ホールドアウト(生存者のみ) を構築
# ============================================================
logger.info("=" * 60)
logger.info("特徴量セットを構築中(80/20ホールドアウト + 全件学習)...")
ag_train_80, ag_val_surv, _ = prepare_split_extended(0.8, add_new_blocks=True, exclude_early_from_val=True)
ag_full, _, test_features_full = prepare_split_extended(1.0, add_new_blocks=True, exclude_early_from_val=True)

FULL_FEATURE_COLS = _feature_cols(ag_full)
BASE_COLS = [c for c in FULL_FEATURE_COLS if c not in NEW_BLOCK_COLS]
NEW_COLS_PRESENT = [c for c in FULL_FEATURE_COLS if c in NEW_BLOCK_COLS]
logger.info(f"FULL特徴量数: {len(FULL_FEATURE_COLS)} (既存ベース {len(BASE_COLS)} + 新規 {len(NEW_COLS_PRESENT)})")

A_PARAMS = {
    "depth": 4,
    "learning_rate": 0.03518359458951149,
    "l2_leaf_reg": 2.217690447016724,
    "border_count": 218,
    "bagging_temperature": 0.6787467566574921,
    "random_strength": 1.438494697238285,
}
ITER_HOLDOUT = 560
ITER_FULL = 560
SEEDS_SUB = [42, 2024, 7, 1234, 99]


def _fit_one_classifier(X_tr, y_tr, obj_cols, params, n_iter, seed):
    model = cb.CatBoostClassifier(**params, iterations=int(n_iter), random_seed=seed,
                                   verbose=False, cat_features=obj_cols, task_type="CPU")
    model.fit(X_tr, y_tr)
    return model


def fit_holdout_fixed(ag_train, ag_val, feature_cols, params, n_iter, seeds):
    obj_cols = [c for c in feature_cols if ag_train[c].dtype == "object"]
    X_tr, y_tr = ag_train[feature_cols].fillna(-999), ag_train[TARGET_COL]
    X_va, y_va = ag_val[feature_cols].fillna(-999), ag_val[TARGET_COL]
    val_preds = []
    for seed in seeds:
        model = _fit_one_classifier(X_tr, y_tr, obj_cols, params, n_iter, seed)
        val_preds.append(model.predict_proba(X_va)[:, 1])
    val_preds = np.array(val_preds)
    return {
        "val_seedavg": float(log_loss(y_va, val_preds.mean(axis=0))),
        "val_preds": val_preds,
        "y_val": y_va.values,
        "last_model": model,
    }


def fit_full_fixed_classifier(ag_full_, test_feats, feature_cols, params, n_iter, seeds):
    obj_cols = [c for c in feature_cols if ag_full_[c].dtype == "object"]
    X_tr, y_tr = ag_full_[feature_cols].fillna(-999), ag_full_[TARGET_COL]
    X_test = test_feats[feature_cols].fillna(-999)
    test_preds = []
    for seed in seeds:
        model = _fit_one_classifier(X_tr, y_tr, obj_cols, params, n_iter, seed)
        test_preds.append(model.predict_proba(X_test)[:, 1])
        logger.info(f"    [classifier] seed={seed}: 完了")
    return np.array(test_preds)


def save_submission(test_index, preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_index, TARGET_COL: preds}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}）")
    return str(path)


# ============================================================
# [1] FULL: 既存444列 + 新規99列 すべて使う分類器
# ============================================================
logger.info("=" * 60)
logger.info("[FULL] 診断: 80/20ホールドアウト(生存者)で検証中...")
hold_full = fit_holdout_fixed(ag_train_80, ag_val_surv, FULL_FEATURE_COLS, A_PARAMS, ITER_HOLDOUT, SEEDS_SUB)
logger.info(f"[FULL] val_seedavg={hold_full['val_seedavg']:.6f}（54_のR0_memofix_plus_LM基準: 0.505477)")

logger.info("[FULL] Train全件学習 → Test予測...")
test_preds_full = fit_full_fixed_classifier(ag_full, test_features_full, FULL_FEATURE_COLS, A_PARAMS, ITER_FULL, SEEDS_SUB)
pred_full_cls = test_preds_full.mean(axis=0)
path1 = save_submission(test_features_full.index, pred_full_cls, "full_classifier")

# ============================================================
# 特徴量選択: FULLモデルの重要度から、新規列のうち重要度>0のものだけ残す
# ============================================================
importances = hold_full["last_model"].get_feature_importance()
imp_series = pd.Series(importances, index=FULL_FEATURE_COLS)
new_importances = imp_series.loc[NEW_COLS_PRESENT].sort_values(ascending=False)
surviving_new_cols = new_importances[new_importances > 0].index.tolist()
logger.info(f"[NARROWED] 新規{len(NEW_COLS_PRESENT)}列のうちCatBoost重要度>0で生き残ったのは{len(surviving_new_cols)}列")
logger.info(f"  重要度トップ10(新規列のみ):\n{new_importances.head(10).to_string()}")

NARROWED_FEATURE_COLS = BASE_COLS + surviving_new_cols
logger.info(f"[NARROWED] 特徴量数: {len(NARROWED_FEATURE_COLS)} (既存ベース{len(BASE_COLS)} + 新規生存{len(surviving_new_cols)})")

# ============================================================
# [2] NARROWED: 既存444列 + 新規列のうち重要度>0のものだけ
# ============================================================
logger.info("=" * 60)
logger.info("[NARROWED] 診断: 80/20ホールドアウト(生存者)で検証中...")
hold_narrow = fit_holdout_fixed(ag_train_80, ag_val_surv, NARROWED_FEATURE_COLS, A_PARAMS, ITER_HOLDOUT, SEEDS_SUB)
logger.info(f"[NARROWED] val_seedavg={hold_narrow['val_seedavg']:.6f}")

logger.info("[NARROWED] Train全件学習 → Test予測...")
test_preds_narrow = fit_full_fixed_classifier(ag_full, test_features_full, NARROWED_FEATURE_COLS, A_PARAMS, ITER_FULL, SEEDS_SUB)
pred_narrow_cls = test_preds_narrow.mean(axis=0)
path2 = save_submission(test_features_full.index, pred_narrow_cls, "narrowed_classifier")


# ============================================================
# 在籍月数（TENURE）: xxxx_v4 / 74_ と同一定義
# ============================================================
logger.info("=" * 60)
logger.info("在籍月数(TENURE)ラベルを算出中...")
TENURE = (train_monthly_full.sort_values("経過月数").groupby(ID_COL)["経過月数"].last())
TENURE_full = TENURE.reindex(ag_full.index).to_numpy().astype(float)
assert not np.isnan(TENURE_full).any()
logger.info(f"TENURE: 平均={TENURE_full.mean():.1f} 最小={TENURE_full.min():.0f} 最大={TENURE_full.max():.0f}")

IS_SURV_FULL = (train_monthly.groupby(ID_COL).size().reindex(ag_full.index).to_numpy()) == 24
logger.info(f"24か月在籍(生存者): {IS_SURV_FULL.sum()} / {len(IS_SURV_FULL)}")


def _fit_one_regressor(X_tr, y_tr, obj_cols, params, n_iter, seed):
    model = cb.CatBoostRegressor(**params, loss_function="RMSE", eval_metric="RMSE",
                                  iterations=int(n_iter), random_seed=seed,
                                  verbose=False, cat_features=obj_cols, task_type="CPU")
    model.fit(X_tr, y_tr)
    return model


def _platt_fit(score, label):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(np.asarray(score).reshape(-1, 1), label)
    return lr


def _platt_apply(lr, score):
    return lr.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]


def _to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


MIX_RATIO = 0.85


def build_regression_blend(label, feature_cols, cls_test_pred, ag_train_80_, ag_val_surv_, ag_full_, test_feats_):
    """xxxx_v4 / 74_expD と同一設計(全件学習+ホールドアウトPlatt較正)の回帰ブレンド。"""
    logger.info("=" * 60)
    logger.info(f"[{label}] 回帰ブレンド(在籍月数回帰+Platt較正, 85:15)を構築中...")

    obj_cols_80 = [c for c in feature_cols if ag_train_80_[c].dtype == "object"]
    X_tr80 = ag_train_80_[feature_cols].fillna(-999)
    y_tr80_cls = ag_train_80_[TARGET_COL]
    X_va = ag_val_surv_[feature_cols].fillna(-999)
    y_va = ag_val_surv_[TARGET_COL].to_numpy()
    y_tenure_tr80 = TENURE.reindex(ag_train_80_.index).to_numpy().astype(float)
    assert not np.isnan(y_tenure_tr80).any()

    cls_val_preds, reg_val_preds = [], []
    for seed in SEEDS_SUB:
        m_cls = _fit_one_classifier(X_tr80, y_tr80_cls, obj_cols_80, A_PARAMS, ITER_HOLDOUT, seed)
        cls_val_preds.append(m_cls.predict_proba(X_va)[:, 1])
        m_reg = _fit_one_regressor(X_tr80, y_tenure_tr80, obj_cols_80, A_PARAMS, ITER_HOLDOUT, seed)
        reg_val_preds.append(m_reg.predict(X_va))
    cls_val = np.mean(cls_val_preds, axis=0)
    reg_val = np.mean(reg_val_preds, axis=0)

    reg_platt = _platt_fit(reg_val, y_va)
    reg_prob_val = _platt_apply(reg_platt, reg_val)

    obj_cols_full = [c for c in feature_cols if ag_full_[c].dtype == "object"]
    X_full = ag_full_[feature_cols].fillna(-999)
    X_test = test_feats_[feature_cols].fillna(-999)
    reg_test_preds = []
    for seed in SEEDS_SUB:
        m_reg_full = _fit_one_regressor(X_full, TENURE.reindex(ag_full_.index).to_numpy().astype(float),
                                         obj_cols_full, A_PARAMS, ITER_FULL, seed)
        reg_test_preds.append(m_reg_full.predict(X_test))
        logger.info(f"    [regressor-full] seed={seed}: 完了")
    reg_test = np.mean(reg_test_preds, axis=0)
    reg_prob_test = _platt_apply(reg_platt, reg_test)

    blend_val = MIX_RATIO * cls_val + (1 - MIX_RATIO) * reg_prob_val
    blend_test = MIX_RATIO * cls_test_pred + (1 - MIX_RATIO) * reg_prob_test

    z_val = _to_logit(blend_val)
    calibrated_val = np.zeros(len(z_val))
    for tri, vai in KFold(n_splits=5, shuffle=True, random_state=27).split(z_val):
        cal = _platt_fit(z_val[tri], y_va[tri])
        calibrated_val[vai] = _platt_apply(cal, z_val[vai])
    logger.info(f"[{label}] 分類器単体val={log_loss(y_va, cls_val):.6f} / "
                f"回帰→確率単体val={log_loss(y_va, reg_prob_val):.6f} / "
                f"85:15ブレンドval={log_loss(y_va, blend_val):.6f} / "
                f"最終Platt較正val={log_loss(y_va, calibrated_val):.6f}")

    final_cal = _platt_fit(z_val, y_va)
    calibrated_test = _platt_apply(final_cal, _to_logit(blend_test))
    return calibrated_test


pred_full_blend = build_regression_blend(
    "FULL", FULL_FEATURE_COLS, pred_full_cls, ag_train_80, ag_val_surv, ag_full, test_features_full)
path3 = save_submission(test_features_full.index, pred_full_blend, "full_regression_blend")

pred_narrow_blend = build_regression_blend(
    "NARROWED", NARROWED_FEATURE_COLS, pred_narrow_cls, ag_train_80, ag_val_surv, ag_full, test_features_full)
path4 = save_submission(test_features_full.index, pred_narrow_blend, "narrowed_regression_blend")

# ============================================================
# [3] TOP20PCT: FULLモデルの重要度上位20%の特徴量だけを使う分類器・回帰ブレンド
#   NARROWED(重要度>0)は544→537列とほとんど絞れなかったため、
#   ユーザー指示により「上位20%だけ残す」というより強い削減を追加検証する。
# ============================================================
logger.info("=" * 60)
n_top20 = max(1, round(len(FULL_FEATURE_COLS) * 0.2))
TOP20_FEATURE_COLS = imp_series.sort_values(ascending=False).head(n_top20).index.tolist()
n_new_in_top20 = sum(1 for c in TOP20_FEATURE_COLS if c in NEW_BLOCK_COLS)
logger.info(f"[TOP20PCT] FULLモデルの重要度上位20%を採用: {len(TOP20_FEATURE_COLS)}列 "
            f"(全{len(FULL_FEATURE_COLS)}列中、うち新規ブロック由来{n_new_in_top20}列)")

logger.info("[TOP20PCT] 診断: 80/20ホールドアウト(生存者)で検証中...")
hold_top20 = fit_holdout_fixed(ag_train_80, ag_val_surv, TOP20_FEATURE_COLS, A_PARAMS, ITER_HOLDOUT, SEEDS_SUB)
logger.info(f"[TOP20PCT] val_seedavg={hold_top20['val_seedavg']:.6f}")

logger.info("[TOP20PCT] Train全件学習 → Test予測...")
test_preds_top20 = fit_full_fixed_classifier(ag_full, test_features_full, TOP20_FEATURE_COLS, A_PARAMS, ITER_FULL, SEEDS_SUB)
pred_top20_cls = test_preds_top20.mean(axis=0)
path5 = save_submission(test_features_full.index, pred_top20_cls, "top20pct_classifier")

pred_top20_blend = build_regression_blend(
    "TOP20PCT", TOP20_FEATURE_COLS, pred_top20_cls, ag_train_80, ag_val_surv, ag_full, test_features_full)
path6 = save_submission(test_features_full.index, pred_top20_blend, "top20pct_regression_blend")

logger.info("=" * 60)
logger.info("=== 全6ファイル出力完了 ===")
for p in [path1, path2, path3, path4, path5, path6]:
    logger.info(f"  {p}")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
