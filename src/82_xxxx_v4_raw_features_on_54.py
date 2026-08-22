"""82_xxxx_v4_raw_features_on_54

ベースは54_の444列そのものではなく、`77_last_month_category_on_54`でPublic確認済みの
447列（444 + 最終月の職種/勤務地/部署ID、Public 0.511547、-0.003483改善確定）。
この上に、xxxx_v4.ipynb(著者オリジナルパイプライン)が使っていて54_/77_には無かった、
単純な生特徴量だけを追加する単一の事前登録済み構成。学習時間合計/学習テーマ数等の
自己学習ブロックは`73_`で既に不採用確定、CatBoost native text_featuresは`69_`で既に不採用確定
のため、この2つは対象外（重複検証しない）。

追加する8列（すべてxxxx_v4に実在し、77_の447列には無い単純な生特徴量。BoW/embedding等の
高次元ブロックは含まない）:
  - 希望勤務地      : 54_の高カバレッジ抽出(extract_desired_location_v2)を流用した生カテゴリ
  - 在宅希望        : xxxx_v4の在宅可否パース(-1/0/1)をそのまま追加
  - キャリア志向    : xxxx_v4の「・キャリア志向：」節パース(専門職/管理職/安定/限定なし/不明)
  - 最終月に不在籍  : 最終月の月末在籍状態が「在籍」でないか(0/1)
  - 昇級あり        : 初月→最終月で等級が上がったか(0/1)
  - 給与の伸び率    : (最終月給与-初月給与)/初月給与
  - 初期部署ID_頻度 : 初期部署IDの出現頻度(train+test)
  - 最終月の部署ID_頻度: 最終月の部署IDの出現頻度(train+test)

`81_`で見つかった「特徴量選択のリーク」を繰り返さないため、重要度による絞り込みは一切行わない
（FULL構成1本のみ、選択ステップなし）。検証は80_/81_と同じリークなしKFold OOF。

出力2ファイル: xxxxv4feat_classifier / xxxxv4feat_regression_blend
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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold
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

SCRIPT_NAME = "82_xxxx_v4_raw_features_on_54"
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
    return train_out, test_out


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
    }), desired  # desiredも返す(希望勤務地の生カテゴリとして使う)


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
# 77_で既にPublic確認済みの最終月の職種/勤務地/部署ID（3列、ベースに含める）
# ============================================================
def create_last_month_category_features(monthly_df, employee_ids):
    features_list = []
    for employee_id in employee_ids:
        emp_data = monthly_df[monthly_df["社員ID"] == employee_id].sort_values("経過月数").reset_index(drop=True)
        last = emp_data.iloc[-1]
        features_list.append({
            "社員ID": employee_id,
            "最終月の職種": last["職種"],
            "最終月の勤務地": last["勤務地"],
            "最終月の部署ID": last["部署ID"],
        })
    return pd.DataFrame(features_list)


# ============================================================
# NEW: xxxx_v4にあり54_/77_には無い単純な生特徴量（8列）
# ============================================================
在宅の否定 = r"必須条件と(?:は)?していない|不要"


def parse_telework(section):
    if section is None:
        return -1
    在宅の句 = re.search(r"(在宅[^、。]*)", section)
    句 = 在宅の句.group(1) if 在宅の句 else ""
    if not 句:
        return -1
    if re.search(在宅の否定, 句):
        return 0
    if "希望" in 句:
        return 1
    return -1


def parse_career_orientation(memo_text):
    if pd.isna(memo_text):
        return "不明"
    m = re.search(r"・キャリア志向：(.+?)(?:\n|$)", memo_text)
    s = m.group(1) if m else ""
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


def create_xxxx_v4_raw_features(persona_df, monthly_df, employee_ids, ws_section_map, desired_loc_map, dept_freq_map, last_dept_freq_map):
    rows = []
    for employee_id in employee_ids:
        emp_data = monthly_df[monthly_df["社員ID"] == employee_id].sort_values("経過月数").reset_index(drop=True)
        初月, 最終月 = emp_data.iloc[0], emp_data.iloc[-1]
        等級の数値 = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}
        初月等級 = 等級の数値.get(初月["等級"])
        最終月等級 = 等級の数値.get(最終月["等級"])
        row = {
            "社員ID": employee_id,
            "最終月に不在籍": int(最終月["月末在籍状態"] != "在籍"),
            "昇級あり": int(最終月等級 > 初月等級) if (初月等級 is not None and 最終月等級 is not None) else 0,
            "給与の伸び率": (最終月["月例給与_円"] - 初月["月例給与_円"]) / 初月["月例給与_円"] if 初月["月例給与_円"] else np.nan,
            "希望勤務地": desired_loc_map.get(employee_id, "不明") or "不明",
            "在宅希望": parse_telework(ws_section_map.get(employee_id)),
            "初期部署ID_頻度": dept_freq_map.get(persona_df.loc[persona_df[ID_COL] == employee_id, "初期部署ID"].values[0], 0)
                if len(persona_df.loc[persona_df[ID_COL] == employee_id]) > 0 else 0,
            "最終月の部署ID_頻度": last_dept_freq_map.get(最終月["部署ID"], 0),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    memo_map = persona_df.set_index(ID_COL)["入社時メモ"]
    out["キャリア志向"] = out[ID_COL].map(memo_map).apply(parse_career_orientation)
    return out


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

tfidf_train_list, tfidf_test_list = [], []
for col in TEXT_COLS:
    tr, te = create_tfidf_svd_features(train_persona, test_persona, col, max_features=300, n_components=15, min_df=3, seed=SEED)
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
for col in TEXT_COLS:
    train_persona[f"{col}_len"] = train_persona[col].fillna("").astype(str).apply(len)
    test_persona[f"{col}_len"] = test_persona[col].fillna("").astype(str).apply(len)
train_persona["text_total_chars"] = train_persona[TEXT_COLS].fillna("").apply(lambda x: sum(len(str(v)) for v in x), axis=1)
test_persona["text_total_chars"] = test_persona[TEXT_COLS].fillna("").apply(lambda x: sum(len(str(v)) for v in x), axis=1)
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

train_reloc_v2, train_desired = create_relocation_mismatch_features(train_persona, extract_desired_location_v2, "転居x勤務地_状態_v2", "転居x勤務地_ダブル悪条件_v2")
test_reloc_v2, test_desired = create_relocation_mismatch_features(test_persona, extract_desired_location_v2, "転居x勤務地_状態_v2", "転居x勤務地_ダブル悪条件_v2")
train_l2m = create_l2_m_interaction_features(train_persona, train_reloc_v2)
test_l2m = create_l2_m_interaction_features(test_persona, test_reloc_v2)
logger.info("split非依存の基本特徴量生成完了")

logger.info("[NEW] xxxx_v4由来の生特徴量(8列)を生成中...")
train_ws_map = train_persona.set_index(ID_COL)["入社時メモ"].apply(extract_workstyle_section).to_dict()
test_ws_map = test_persona.set_index(ID_COL)["入社時メモ"].apply(extract_workstyle_section).to_dict()
train_desired_map = dict(zip(train_persona[ID_COL], train_desired.values))
test_desired_map = dict(zip(test_persona[ID_COL], test_desired.values))

dept_freq = pd.concat([train_persona["初期部署ID"], test_persona["初期部署ID"]]).value_counts().to_dict()
last_dept_all = pd.concat([
    train_monthly.sort_values("経過月数").groupby(ID_COL)["部署ID"].last(),
    test_monthly.sort_values("経過月数").groupby(ID_COL)["部署ID"].last(),
])
last_dept_freq = last_dept_all.value_counts().to_dict()

train_xv4 = create_xxxx_v4_raw_features(train_persona, train_monthly, train_ids, train_ws_map, train_desired_map, dept_freq, last_dept_freq)
test_xv4 = create_xxxx_v4_raw_features(test_persona, test_monthly, test_ids, test_ws_map, test_desired_map, dept_freq, last_dept_freq)
NEW_BLOCK_COLS = [c for c in train_xv4.columns if c != ID_COL]
logger.info(f"新規列(xxxx_v4由来, 8列): {NEW_BLOCK_COLS}")
for c in ["希望勤務地", "在宅希望", "キャリア志向"]:
    logger.info(f"  {c} 分布(train): {dict(train_xv4[c].value_counts())}")

logger.info("77_で確認済みの最終月の職種/勤務地/部署ID(3列)をベースに生成中...")
train_lastmonth_cat = create_last_month_category_features(train_monthly, train_ids)
test_lastmonth_cat = create_last_month_category_features(test_monthly, test_ids)


# ============================================================
# build_features(train_id_subset): 80_/81_と同一パターン
# ============================================================
def build_features(train_id_subset):
    train_id_subset = set(train_id_subset)
    train_dept_te, test_dept_te = create_department_target_encoding(
        train_persona, test_persona, y_train, fit_ids=train_id_subset, seed=SEED, n_splits=5, smoothing=10
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
    tf = tf.merge(train_reloc_v2, on=ID_COL, how="left")
    tf = tf.merge(train_l2m, on=ID_COL, how="left")
    tf = tf.merge(train_lastmonth_cat, on=ID_COL, how="left")
    tf = tf.merge(train_xv4, on=ID_COL, how="left")

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
    ttf = ttf.merge(test_reloc_v2, on=ID_COL, how="left")
    ttf = ttf.merge(test_l2m, on=ID_COL, how="left")
    ttf = ttf.merge(test_lastmonth_cat, on=ID_COL, how="left")
    ttf = ttf.merge(test_xv4, on=ID_COL, how="left")

    _train_period_features = tf[tf[ID_COL].isin(train_id_subset)]
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
    tf[TARGET_COL] = train_persona.set_index(ID_COL).loc[tf.index, TARGET_COL].values
    return tf, ttf


def _feature_cols(df):
    return [c for c in df.columns if c not in ["入社日", TARGET_COL]]


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

surv_mask = np.array([tid not in EARLY_LEAVER_IDS for tid in train_ids])
logger.info(f"生存者(24か月在籍): {surv_mask.sum()} / {len(surv_mask)}")


def _fit_one_classifier(X_tr, y_tr, obj_cols, seed):
    model = cb.CatBoostClassifier(**A_PARAMS, iterations=ITER, random_seed=seed,
                                   verbose=False, cat_features=obj_cols, task_type="CPU")
    model.fit(X_tr, y_tr)
    return model


def _fit_one_regressor(X_tr, y_tr, obj_cols, seed):
    model = cb.CatBoostRegressor(**A_PARAMS, loss_function="RMSE", eval_metric="RMSE",
                                  iterations=ITER, random_seed=seed,
                                  verbose=False, cat_features=obj_cols, task_type="CPU")
    model.fit(X_tr, y_tr)
    return model


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}）")
    return str(path)


TENURE = (train_monthly_full.sort_values("経過月数").groupby(ID_COL)["経過月数"].last())

# ============================================================
# Train全件のtf/ttf（提出用）
# ============================================================
logger.info("=" * 60)
logger.info("Train全件でのtf/ttfを構築中(提出用)...")
tf_full, ttf_full = build_features(train_ids.tolist())
FULL_FEATURE_COLS = _feature_cols(tf_full)
logger.info(f"tf_full 特徴量数: {len(FULL_FEATURE_COLS)}（444+3(77_)+8(xxxx_v4由来)=455のはず）")
assert len(FULL_FEATURE_COLS) == 455, f"想定外の列数: {len(FULL_FEATURE_COLS)}"

# ============================================================
# 対照群: 77_相当の447列(444+最終月カテゴリ3列、新規8列なし)を同じ5-foldでリークなしOOF
# ============================================================
logger.info("=" * 60)
logger.info(f"[baseline_447] 77_相当(新規8列なし)のリークなしKFold OOFを構築中...")
skf = StratifiedKFold(n_splits=OOF_N_SPLITS, shuffle=True, random_state=OOF_SEED)
base_oof = np.zeros(len(train_ids))
new_block_set = set(NEW_BLOCK_COLS)
for fold_i, (tr_pos, va_pos) in enumerate(skf.split(train_ids, y_train)):
    fold_train_ids = train_ids[tr_pos].tolist()
    fold_val_ids = train_ids[va_pos].tolist()
    tf_f, _ = build_features(fold_train_ids)
    feat_cols_f = [c for c in _feature_cols(tf_f) if c not in new_block_set]
    obj_cols_f = [c for c in feat_cols_f if tf_f[c].dtype == "object"]
    X_tr = tf_f.loc[fold_train_ids, feat_cols_f].fillna(-999)
    y_tr_cls = tf_f.loc[fold_train_ids, TARGET_COL]
    X_va = tf_f.loc[fold_val_ids, feat_cols_f].fillna(-999)
    m_base = _fit_one_classifier(X_tr, y_tr_cls, obj_cols_f, OOF_SEED)
    base_oof[va_pos] = m_base.predict_proba(X_va)[:, 1]
    logger.info(f"    [baseline_447] fold{fold_i}: 完了 (特徴量数={len(feat_cols_f)})")
base_score = log_loss(y_train.values[surv_mask], base_oof[surv_mask])
logger.info(f"[baseline_447] val(OOF, リークなし, n={surv_mask.sum()})={base_score:.6f}")

# ============================================================
# 本命: 77_相当447列 + xxxx_v4由来8列(455列)のリークなしKFold OOF（選択ステップなし）
# ============================================================
logger.info("=" * 60)
logger.info(f"[xxxxv4feat] リークなしKFold OOF(5-fold, seed={OOF_SEED})を構築中(選択ステップなし)...")
cls_oof = np.zeros(len(train_ids))
reg_oof = np.zeros(len(train_ids))

for fold_i, (tr_pos, va_pos) in enumerate(skf.split(train_ids, y_train)):
    fold_train_ids = train_ids[tr_pos].tolist()
    fold_val_ids = train_ids[va_pos].tolist()
    tf_f, _ = build_features(fold_train_ids)
    feat_cols_f = _feature_cols(tf_f)
    obj_cols_f = [c for c in feat_cols_f if tf_f[c].dtype == "object"]

    X_tr = tf_f.loc[fold_train_ids, feat_cols_f].fillna(-999)
    y_tr_cls = tf_f.loc[fold_train_ids, TARGET_COL]
    y_tr_reg = TENURE.reindex(fold_train_ids).to_numpy().astype(float)
    X_va = tf_f.loc[fold_val_ids, feat_cols_f].fillna(-999)

    m_cls = _fit_one_classifier(X_tr, y_tr_cls, obj_cols_f, OOF_SEED)
    cls_oof[va_pos] = m_cls.predict_proba(X_va)[:, 1]
    m_reg = _fit_one_regressor(X_tr, y_tr_reg, obj_cols_f, OOF_SEED)
    reg_oof[va_pos] = m_reg.predict(X_va)
    logger.info(f"    [xxxxv4feat] fold{fold_i}: 完了 (特徴量数={len(feat_cols_f)})")

cls_score = log_loss(y_train.values[surv_mask], cls_oof[surv_mask])
logger.info(f"[xxxxv4feat] 分類器val(OOF, リークなし, n={surv_mask.sum()})={cls_score:.6f}")
logger.info(f"=== 対照比較: baseline_447(新規8列なし)={base_score:.6f} → xxxxv4feat(455列)={cls_score:.6f} "
            f"(差={cls_score - base_score:+.6f}) ===")
logger.info("(参考: 80_の純粋な54_444列でのval = 0.520931(Colab)/0.522851(ローカル)、"
            "77_のPublic = 0.511547)")

# ============================================================
# 提出: 分類器
# ============================================================
logger.info("=" * 60)
logger.info("Train全件学習 → Test予測(5シード平均)...")
obj_cols_full = [c for c in FULL_FEATURE_COLS if tf_full[c].dtype == "object"]
X_tr_full = tf_full[FULL_FEATURE_COLS].fillna(-999)
y_tr_full = tf_full[TARGET_COL]
X_test_full = ttf_full[FULL_FEATURE_COLS].fillna(-999)
test_preds = []
for seed in SEEDS_SUB:
    m = _fit_one_classifier(X_tr_full, y_tr_full, obj_cols_full, seed)
    test_preds.append(m.predict_proba(X_test_full)[:, 1])
    logger.info(f"    [classifier-full] seed={seed}: 完了")
pred_cls = np.mean(test_preds, axis=0)
path1 = save_submission(pred_cls, "xxxxv4feat_classifier")

# ============================================================
# 回帰ブレンド（77_/81_と同一のOOF-safe設計）
# ============================================================
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
logger.info("=" * 60)
logger.info("回帰ブレンド(在籍月数回帰+Platt較正, 85:15)を構築中...")
y_full = y_train.values

reg_prob_oof = np.zeros(len(reg_oof))
for tri, vai in KFold(n_splits=5, shuffle=True, random_state=27).split(reg_oof):
    cal = _platt_fit(reg_oof[tri], y_full[tri])
    reg_prob_oof[vai] = _platt_apply(cal, reg_oof[vai])
reg_platt_full = _platt_fit(reg_oof, y_full)

tenure_full = TENURE.reindex(tf_full.index).to_numpy().astype(float)
reg_test_preds = []
for seed in SEEDS_SUB:
    m_reg = _fit_one_regressor(X_tr_full, tenure_full, obj_cols_full, seed)
    reg_test_preds.append(m_reg.predict(X_test_full))
    logger.info(f"    [regressor-full] seed={seed}: 完了")
reg_test = np.mean(reg_test_preds, axis=0)
reg_prob_test = _platt_apply(reg_platt_full, reg_test)

blend_oof = MIX_RATIO * cls_oof + (1 - MIX_RATIO) * reg_prob_oof
blend_test = MIX_RATIO * pred_cls + (1 - MIX_RATIO) * reg_prob_test

z_oof = _to_logit(blend_oof)
calibrated_oof = np.zeros(len(z_oof))
for tri, vai in KFold(n_splits=5, shuffle=True, random_state=27).split(z_oof):
    cal = _platt_fit(z_oof[tri], y_full[tri])
    calibrated_oof[vai] = _platt_apply(cal, z_oof[vai])

logger.info(f"分類器単体val(OOF, n={surv_mask.sum()})={log_loss(y_full[surv_mask], cls_oof[surv_mask]):.6f} / "
            f"回帰→確率単体val={log_loss(y_full[surv_mask], reg_prob_oof[surv_mask]):.6f} / "
            f"85:15ブレンドval={log_loss(y_full[surv_mask], blend_oof[surv_mask]):.6f} / "
            f"最終Platt較正val={log_loss(y_full[surv_mask], calibrated_oof[surv_mask]):.6f}")

final_cal = _platt_fit(z_oof, y_full)
calibrated_test = _platt_apply(final_cal, _to_logit(blend_test))
path2 = save_submission(calibrated_test, "xxxxv4feat_regression_blend")

logger.info("=" * 60)
logger.info("=== 全2ファイル出力完了 ===")
logger.info(f"  {path1}")
logger.info(f"  {path2}")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
