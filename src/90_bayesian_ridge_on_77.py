"""90_bayesian_ridge_on_77

ロジスティック回帰の系列（`87_`/`88_`/`89_`）では、CatBoost重要度による事前選択(134列)＋
ElasticNet正則化が最良（CB134比+0.016409）で、選択方法を変える／選択を外す方向は
すべて悪化した([[cpu-model-zoo-closed]]の追記参照)。本スクリプトは同じ「10年定着ラベル」
(0/1)をガウス尤度の線形回帰で直接回帰する——学習データが1fold当たり約2,200件と少なく、
自動で正則化強度を推定するベイズ的な縮小推定(エビデンス最大化)が効きやすいのではという
仮説を検証する。予測値は[eps, 1-eps]にクリップしてlog_lossで評価する（線形確率モデル）。

  - BayesianRidgeFull: 447列全部(One-Hot後約1105次元)にBayesianRidgeを適用
  - BayesianRidgeSelected: Fullの係数(|coef|)上位15%だけに絞って再学習
    （ロジスティック回帰のembedded selection(`89_`)と同型だが、BayesianRidgeは
    ElasticNetのように係数を厳密にゼロ化しないため、閾値ではなく上位%で足切りする。
    初回は上位30%で試しCB134比+0.076613と大幅悪化したため、絞りを強化して再検証する）

「全特徴量」と「絞り込み後」を同じリークなしKFold OOFで比較し、CB134(木ベース選択の
参照CatBoost)とのブレンド曲線argminも確認したうえで、Publicに提出して局所valとの
乖離を見る。

出力4ファイル: cb134_classifier / bayesridge_full_classifier / bayesridge_selected_classifier /
blend_argmin_w{N}pct
"""
import datetime
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import catboost as cb
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger
from common.utils.seed import seed_everything

SEED = 42
seed_everything(seed=SEED)

TARGET_COL = "10年定着ラベル"
ID_COL = "社員ID"

SCRIPT_NAME = "90_bayesian_ridge_on_77"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 実験開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 75_以降、学習済みモデルの永続化が抜け落ちていた(74_まではSAVED_MODELS_DIRを定義していたが
# CatBoost単体スクリプトでは実際に.save_model()を呼んだ形跡が一度もなかった)。
# 以後は必ず実際に保存する。
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models" / TODAY / SCRIPT_NAME
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Saved Models Directory: {SAVED_MODELS_DIR}")

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


# ============================================================
# 54_l2_m_interaction.ipynb と同一の特徴量関数（split非依存、84_/86_と同一）
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

train_reloc_v2 = create_relocation_mismatch_features(train_persona, extract_desired_location_v2, "転居x勤務地_状態_v2", "転居x勤務地_ダブル悪条件_v2")
test_reloc_v2 = create_relocation_mismatch_features(test_persona, extract_desired_location_v2, "転居x勤務地_状態_v2", "転居x勤務地_ダブル悪条件_v2")
train_l2m = create_l2_m_interaction_features(train_persona, train_reloc_v2)
test_l2m = create_l2_m_interaction_features(test_persona, test_reloc_v2)
logger.info("split非依存の基本特徴量生成完了")

logger.info("77_で確認済みの最終月の職種/勤務地/部署ID(3列)を生成中...")
train_lastmonth_cat = create_last_month_category_features(train_monthly, train_ids)
test_lastmonth_cat = create_last_month_category_features(test_monthly, test_ids)


# ============================================================
# build_features(train_id_subset): 80_/81_/82_/83_/84_/86_と同一パターン（純77_相当447列）
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
TOP_PCT_30 = 0.3
BLEND_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # w=CatBoost側の重み

surv_mask = np.array([tid not in EARLY_LEAVER_IDS for tid in train_ids])
logger.info(f"生存者(24か月在籍): {surv_mask.sum()} / {len(surv_mask)}")


def _fit_one_classifier(X_tr, y_tr, obj_cols, seed):
    model = cb.CatBoostClassifier(**A_PARAMS, iterations=ITER, random_seed=seed,
                                   verbose=False, cat_features=obj_cols, task_type="CPU")
    model.fit(X_tr, y_tr)
    return model


def select_top_pct(imp_series, pct):
    n_top = max(1, round(len(imp_series) * pct))
    return imp_series.sort_values(ascending=False).head(n_top).index.tolist()


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}）")
    return str(path)


# ============================================================
# 線形モデル用前処理（31_の数値パイプラインをそのまま踏襲、カテゴリはOne-Hot——
# 88_のiter1でTarget Encodingを試したが+0.0007悪化だったためOne-Hotに確定）
# 欠損フラグ→中央値補完→標準化→[-10,10]クリップ、カテゴリはOne-Hot。fitは学習データのみ。
# ============================================================
def prepare_linear_matrices(train_df, other_dfs, feat_cols):
    obj_cols = [c for c in feat_cols if train_df[c].dtype == "object"]
    num_cols = [c for c in feat_cols if c not in obj_cols]

    frames_raw = [train_df[feat_cols].copy()] + [d[feat_cols].copy() for d in other_dfs]

    missing_flag_cols = [c for c in num_cols if any(f[c].isna().any() for f in frames_raw)]
    for f in frames_raw:
        for c in missing_flag_cols:
            f[f"{c}_missing"] = f[c].isna().astype(float)

    medians = frames_raw[0][num_cols].median()
    num_scaled = []
    scaler = StandardScaler()
    tr_num = frames_raw[0][num_cols].fillna(medians).fillna(0.0)
    num_scaled.append(np.clip(scaler.fit_transform(tr_num), -10, 10))
    for f in frames_raw[1:]:
        num_scaled.append(np.clip(scaler.transform(f[num_cols].fillna(medians).fillna(0.0)), -10, 10))

    if obj_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        tr_cat = frames_raw[0][obj_cols].fillna("missing").astype(str)
        cat_enc = [ohe.fit_transform(tr_cat)]
        for f in frames_raw[1:]:
            cat_enc.append(ohe.transform(f[obj_cols].fillna("missing").astype(str)))
    else:
        cat_enc = [np.zeros((len(f), 0)) for f in frames_raw]

    mflag_arrs = []
    if missing_flag_cols:
        mflag_cols = [f"{c}_missing" for c in missing_flag_cols]
        for f in frames_raw:
            mflag_arrs.append(f[mflag_cols].values)
    else:
        mflag_arrs = [np.zeros((len(f), 0)) for f in frames_raw]

    out = [np.hstack([n, c, m]) for n, c, m in zip(num_scaled, cat_enc, mflag_arrs)]
    return out  # [train_matrix, *other_matrices]


def _to_prob(preds):
    return np.clip(preds, 1e-6, 1 - 1e-6)


def fit_bayesridge(X_tr, y_tr):
    # BayesianRidge: エビデンス最大化(EM)で正則化強度(alpha, lambda)を自動推定する。
    # ElasticNetのようなグリッドサーチは不要。0/1ラベルをガウス尤度で直接回帰する
    # (線形確率モデル)。予測は[eps, 1-eps]にクリップしてlog_lossで評価する。
    model = BayesianRidge(max_iter=500, tol=1e-4)
    model.fit(X_tr, y_tr)
    return model


TOP_PCT_COEF = 0.15  # BayesianRidgeは係数を厳密にゼロ化しないため、上位15%で足切りする（30%は悪化したため絞りを強化）


def fit_bayesridge_with_embedded_selection(X_tr, y_tr, X_va):
    m_full = fit_bayesridge(X_tr, y_tr)
    pred_full = _to_prob(m_full.predict(X_va))

    coef = m_full.coef_
    n_top = max(1, round(len(coef) * TOP_PCT_COEF))
    mask = np.zeros(len(coef), dtype=bool)
    top_idx = np.argsort(-np.abs(coef))[:n_top]
    mask[top_idx] = True
    n_selected = int(mask.sum())

    m_selected = fit_bayesridge(X_tr[:, mask], y_tr)
    pred_selected = _to_prob(m_selected.predict(X_va[:, mask]))
    return m_full, pred_full, m_selected, pred_selected, n_selected


# ============================================================
# Train全件のtf/ttf（提出用）
# ============================================================
logger.info("=" * 60)
logger.info("Train全件でのtf/ttfを構築中(提出用)...")
tf_full, ttf_full = build_features(train_ids.tolist())
FULL_FEATURE_COLS = _feature_cols(tf_full)
assert len(FULL_FEATURE_COLS) == 447, f"想定外の列数: {len(FULL_FEATURE_COLS)}"

# ============================================================
# リークなしKFold OOF: 447列全部でBayesianRidgeを学習し(選択なし)、
# 係数の上位30%だけに絞って再学習する(embedded selection)。
# 「全特徴量」と「絞り込み後」を両方OOFで比較する。
# ============================================================
logger.info("=" * 60)
logger.info("[BayesRidgeFull/BayesRidgeSelected/CB134] リークなしKFold OOFを構築中...")
skf = StratifiedKFold(n_splits=OOF_N_SPLITS, shuffle=True, random_state=OOF_SEED)

cb134_oof = np.zeros(len(train_ids))
bayesridge_full_oof = np.zeros(len(train_ids))
bayesridge_selected_oof = np.zeros(len(train_ids))
n_selected_per_fold = []

for fold_i, (tr_pos, va_pos) in enumerate(skf.split(train_ids, y_train)):
    fold_train_ids = train_ids[tr_pos].tolist()
    fold_val_ids = train_ids[va_pos].tolist()
    tf_f, _ = build_features(fold_train_ids)
    feat_cols_f = _feature_cols(tf_f)
    obj_cols_f = [c for c in feat_cols_f if tf_f[c].dtype == "object"]

    y_tr_cls = tf_f.loc[fold_train_ids, TARGET_COL]
    fold_train_df = tf_f.loc[fold_train_ids]
    fold_val_df = tf_f.loc[fold_val_ids]

    # CB134参照値(比較用、84_/87_と同じ木ベース選択パターン)
    X_tr447 = tf_f.loc[fold_train_ids, feat_cols_f].fillna(-999)
    m_cb447 = _fit_one_classifier(X_tr447, y_tr_cls, obj_cols_f, OOF_SEED)
    imp_series = pd.Series(m_cb447.get_feature_importance(), index=feat_cols_f)
    top30_cols = select_top_pct(imp_series, TOP_PCT_30)
    obj_top30 = [c for c in top30_cols if c in obj_cols_f]

    X_tr134 = tf_f.loc[fold_train_ids, top30_cols].fillna(-999)
    X_va134 = tf_f.loc[fold_val_ids, top30_cols].fillna(-999)
    m_cb134 = _fit_one_classifier(X_tr134, y_tr_cls, obj_top30, OOF_SEED)
    cb134_oof[va_pos] = m_cb134.predict_proba(X_va134)[:, 1]

    # BayesianRidge 447列全部(選択なし) → 係数上位30%でembedded selection → 再学習
    Xl_tr447, Xl_va447 = prepare_linear_matrices(fold_train_df, [fold_val_df], feat_cols_f)
    _, pred_full, _, pred_selected, n_sel = fit_bayesridge_with_embedded_selection(
        Xl_tr447, y_tr_cls.values, Xl_va447
    )
    bayesridge_full_oof[va_pos] = pred_full
    bayesridge_selected_oof[va_pos] = pred_selected
    n_selected_per_fold.append(n_sel)

    logger.info(f"    [fold{fold_i}] 完了 (BayesRidge全次元={Xl_tr447.shape[1]}, 上位30%絞り込み後={n_sel})")

cb134_score = log_loss(y_train.values[surv_mask], cb134_oof[surv_mask])
bayesridge_full_score = log_loss(y_train.values[surv_mask], bayesridge_full_oof[surv_mask])
bayesridge_selected_score = log_loss(y_train.values[surv_mask], bayesridge_selected_oof[surv_mask])

logger.info("=" * 60)
logger.info(f"[CB134]                val(OOF, n={surv_mask.sum()}) = {cb134_score:.6f} (参考: 87_と同一のはず=0.519260)")
logger.info(f"[BayesRidgeFull]       val(OOF, n={surv_mask.sum()}) = {bayesridge_full_score:.6f}  "
            f"(CB134比 {bayesridge_full_score - cb134_score:+.6f})")
logger.info(f"[BayesRidgeSelected]   val(OOF, n={surv_mask.sum()}) = {bayesridge_selected_score:.6f}  "
            f"(CB134比 {bayesridge_selected_score - cb134_score:+.6f}, "
            f"Full比 {bayesridge_selected_score - bayesridge_full_score:+.6f}, "
            f"fold毎の選択次元数={n_selected_per_fold})")
logger.info("(比較基準: 88_iter2 LogReg134(木ベース選択+ElasticNet) val=0.535668, CB134比+0.016409)")

# ============================================================
# ブレンド曲線(argmin)を局所OOFで確認: CB134 × 良い方のBayesRidge(Full/Selected)
# ============================================================
logger.info("=" * 60)
y_surv = y_train.values[surv_mask]
cb_s = cb134_oof[surv_mask]
USE_SELECTED = bayesridge_selected_score <= bayesridge_full_score
lr_s = bayesridge_selected_oof[surv_mask] if USE_SELECTED else bayesridge_full_oof[surv_mask]
logger.info(f"(ブレンド曲線は{'BayesRidgeSelected' if USE_SELECTED else 'BayesRidgeFull'}を採用: "
            f"val {bayesridge_selected_score:.6f} vs {bayesridge_full_score:.6f})")
rows = []
for w in BLEND_WEIGHTS:
    blend = np.clip(w * cb_s + (1 - w) * lr_s, 1e-9, 1 - 1e-9)
    ll = log_loss(y_surv, blend)
    rows.append((w, ll))
    logger.info(f"    w_cb={w:.1f}: val={ll:.6f}")
best_w, best_ll = min(rows, key=lambda r: r[1])
corr = float(np.corrcoef(cb_s, lr_s)[0, 1])
logger.info(f"[argmin] best_w_cb={best_w:.2f}, best_val={best_ll:.6f}, 相関={corr:.4f} "
            f"(CB134単体比 {best_ll - cb134_score:+.6f})")


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}）")
    return str(path)


# ============================================================
# 提出用: Train全件でCB134/BayesRidgeFull/BayesRidgeSelectedを学習してTest予測
# 学習済みモデルはsaved_models/に必ず永続化する
# ============================================================
logger.info("=" * 60)
logger.info("提出用: Train全件で選択・学習中...")
tf_full, ttf_full = build_features(train_ids.tolist())
FULL_FEATURE_COLS = _feature_cols(tf_full)
obj_cols_full = [c for c in FULL_FEATURE_COLS if tf_full[c].dtype == "object"]
y_tr_full = tf_full[TARGET_COL]

# CB134(比較用の参照モデル、木ベース選択)
m_selector_full = _fit_one_classifier(
    tf_full[FULL_FEATURE_COLS].fillna(-999), y_tr_full, obj_cols_full, OOF_SEED
)
imp_series_full = pd.Series(m_selector_full.get_feature_importance(), index=FULL_FEATURE_COLS)
top30_cols_full = select_top_pct(imp_series_full, TOP_PCT_30)
obj_top30_full = [c for c in top30_cols_full if c in obj_cols_full]
X_tr134_full = tf_full[top30_cols_full].fillna(-999)
X_test134_full = ttf_full[top30_cols_full].fillna(-999)

m_cb134_full = _fit_one_classifier(X_tr134_full, y_tr_full, obj_top30_full, OOF_SEED)
cb134_test = m_cb134_full.predict_proba(X_test134_full)[:, 1]
cb_model_path = SAVED_MODELS_DIR / "cb134_full.cbm"
m_cb134_full.save_model(str(cb_model_path))
logger.info(f"  モデル保存: {cb_model_path}")
path1 = save_submission(cb134_test, "cb134_classifier")

# BayesRidgeFull(447列全部) → BayesRidgeSelected(係数上位30%のみ再学習)
Xl_tr447_full, Xl_test447_full = prepare_linear_matrices(tf_full, [ttf_full], FULL_FEATURE_COLS)
m_brfull, brfull_test, m_brsel, brsel_test, n_sel_full = fit_bayesridge_with_embedded_selection(
    Xl_tr447_full, y_tr_full.values, Xl_test447_full
)
logger.info(f"[提出用] BayesRidge全次元={Xl_tr447_full.shape[1]} → 上位30%絞り込み後={n_sel_full}")

brfull_model_path = SAVED_MODELS_DIR / "bayesridge_full.joblib"
joblib.dump(m_brfull, brfull_model_path)
logger.info(f"  モデル保存: {brfull_model_path}")
path2 = save_submission(brfull_test, "bayesridge_full_classifier")

brsel_model_path = SAVED_MODELS_DIR / "bayesridge_selected.joblib"
joblib.dump(m_brsel, brsel_model_path)
logger.info(f"  モデル保存: {brsel_model_path}")
path3 = save_submission(brsel_test, "bayesridge_selected_classifier")

br_test = brsel_test if USE_SELECTED else brfull_test
blend_test = best_w * cb134_test + (1 - best_w) * br_test
path4 = save_submission(blend_test, f"blend_argmin_w{round(best_w * 100):d}pct")

np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_cb134_oofpreds.npy", cb134_oof)
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_bayesridge_full_oofpreds.npy", bayesridge_full_oof)
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_bayesridge_selected_oofpreds.npy", bayesridge_selected_oof)

logger.info("=" * 60)
logger.info("=== 全4ファイル出力完了(+ モデル3件・OOF3件を保存) ===")
logger.info(f"  {path1}")
logger.info(f"  {path2}")
logger.info(f"  {path3}")
logger.info(f"  {path4}")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
