"""102_autogluon_on_77_features

現最良(94_add_w25、Public 0.495570)の内部では、AutoGluonの**プールC**（441列・古い特徴量
パイプライン・8回学習平均、Public 0.514050）が展開後の最終ウェイトで**22.3%**を占める
（[[best-submission-status]]参照）。プールCは`44_`〜`51_`世代の特徴量（L2xM・最終月カテゴリ
列・xxxx_v4統合前）で作られており、その後`54_`〜`77_`で積み上げた特徴量の進化を一切
反映していない。

`56_`でL2xMブロックをAutoGluonに統合した結果は当初「2/2で悪化」と判定されたが、
その後`62_`でAutoGluonの**単一実行のシード間ノイズがweighted MAD 0.0238・best_single
MAD 0.0372**もあると判明し、「LMブロックはAutoGluonでは効かない」という結論は
**自己訂正され「未決」のまま**になっていた（[[l2-m-risk-count-confirmed]]）。
つまり新特徴量統合をAutoGluonで正しく検証したことは一度もない。

本スクリプトは`77_last_month_category_on_54`が確立した**現行447列**の特徴量パイプライン
（L2xM・最終月カテゴリ3列・TF-IDF/SVD・部署ターゲットエンコーディング等、単層CatBoostの
現最良`84_TOP30PCT`と同じ土台）を、`51_autogluon_catboost_bias`が確立したAutoGluon設定
（`presets=best_quality`, `excluded_model_types=["FASTAI","NN_TORCH","KNN"]`,
`num_bag_folds=8`, `num_stack_levels=1`, `dynamic_stacking=False`, `time_limit=7200秒/fit`）
にそのまま投入し、ノイズ床を超えるために**3シード([42, 2024, 7])を独立実行して平均**する。

新規変更点（51_からの差分）:
  - `num_gpus=1` を`predictor.fit()`に渡し、CatBoost/LightGBM/XGBoostのGPU学習を有効化
    （NN_TORCH/FASTAI/KNNは従来通り除外のまま。データ規模(Train約2,700行)ではGPUの恩恵は
    限定的な可能性が高いが、time_limit予算内でより多くのモデル/foldを完走させる狙い）
  - 3シード独立実行 + 平均（[[l2-m-risk-count-confirmed]]の自己訂正「最低3シードずつ回して
    平均で比較する必要がある」に対応）
  - `fit_autogluon`のtag別チェックポイント読み込みは51_と同様に維持——Colab切断時も
    完了済みシードは失われず、中断したシードだけ再実行すればよい

置き換え/追加どちらのブレンドパターンを採るかは、実際の結果（内部val・相関）を見てから
別スクリプトで決める（ユーザー判断: 「どちらのパターンも用意する、ブレンドだけなので
コストが低い」）。

出力: シードごとに weighted / best_single の2種、+ 3シード平均版2種（計8ファイル）
"""
import datetime
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("mode.chained_assignment", None)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger
from common.utils.seed import seed_everything

SEED = 42
seed_everything(seed=SEED)

TARGET_COL = "10年定着ラベル"
ID_COL = "社員ID"

SCRIPT_NAME = "102_autogluon_on_77_features"
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

y_train = train_persona[TARGET_COL]
train_ids = train_persona[ID_COL].values
test_ids = test_persona[ID_COL].values
logger.info(f"Train Persona: {train_persona.shape}, Test Persona: {test_persona.shape}")

EARLY_LEAVER_IDS = set(train_monthly.loc[train_monthly["月末在籍状態"] == "退職", ID_COL].unique())
_test_early = set(test_monthly.loc[test_monthly["月末在籍状態"] == "退職", ID_COL].unique())
assert len(_test_early) == 0

TEXT_COLS = ["入社時メモ", "上司からのフィードバック", "同僚からのフィードバック"]


# ============================================================
# 77_last_month_category_on_54 と同一の特徴量関数（447列、split非依存）
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
logger.info("split非依存の基本特徴量を生成中(77_と同一、447列)...")
train_lastmonth_cat = create_last_month_category_features(train_monthly, train_ids)
test_lastmonth_cat = create_last_month_category_features(test_monthly, test_ids)

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


# ============================================================
# prepare_split_v77（77_と同一。split_ratio=1.0でTrain全件をAutoGluonに渡す）
# ============================================================
def prepare_split_v77(split_ratio, add_lastmonth=True, exclude_early_from_val=True):
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

    tf = tf.merge(train_reloc_v2, on=ID_COL, how="left")
    tf = tf.merge(train_l2m, on=ID_COL, how="left")
    ttf = ttf.merge(test_reloc_v2, on=ID_COL, how="left")
    ttf = ttf.merge(test_l2m, on=ID_COL, how="left")

    if add_lastmonth:
        tf = tf.merge(train_lastmonth_cat, on=ID_COL, how="left")
        ttf = ttf.merge(test_lastmonth_cat, on=ID_COL, how="left")

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
# Train全件でのAutoGluon投入用フレームを構築(447列)
# ============================================================
logger.info("=" * 60)
logger.info("Train全件での447列フレームを構築中(AutoGluon投入用)...")
ag_full, _, test_features_full = prepare_split_v77(1.0, add_lastmonth=True, exclude_early_from_val=True)
FEATURE_COLS = _feature_cols(ag_full)
logger.info(f"特徴量数: {len(FEATURE_COLS)}（77_と同じ447列のはず）")
assert len(FEATURE_COLS) == 447, f"想定外の列数: {len(FEATURE_COLS)}"


# ============================================================
# AutoGluon の学習関数（51_からの移植 + GPU対応）
# ============================================================
logger.info("=" * 60)
logger.info("AutoGluon (autogluon.tabular) を読み込み中...")
from autogluon.tabular import TabularPredictor

try:
    import ray  # noqa: F401
    logger.warning("⚠️ ray が入っている。ParallelLocalFoldFittingStrategy経由でfold全滅の危険。"
                    "ランタイムを初期化しrayを入れずにやり直すこと。")
except ImportError:
    logger.info("✅ ray は入っていない（正常）。foldは逐次学習される。")

import torch
HAS_GPU = torch.cuda.is_available()
logger.info(f"GPU利用可能: {HAS_GPU}")
if not HAS_GPU:
    logger.warning("GPUが無い。ColabのランタイムをGPUに変更すること（num_gpus=1を渡す前提のコード）。")

PRESETS = "best_quality"
TIME_LIMIT = 7200            # 秒/fit(51_と同一)
AG_METRIC = "log_loss"
EXCLUDED_MODELS = ["FASTAI", "NN_TORCH", "KNN"]   # 51_と同一
DYNAMIC_STACKING = False
NUM_STACK_LEVELS = 1
NUM_BAG_FOLDS = 8
NUM_GPUS = 1 if HAS_GPU else 0   # 51_からの変更点: GPU学習を有効化(CatBoost/LightGBM/XGBoost)
SEEDS = [42, 2024, 7]             # 62_のノイズ床(weighted MAD 0.0238)を超えるため3シード平均

AG_DIR = PROJECT_ROOT / "saved_models" / TODAY / SCRIPT_NAME
AG_DIR.mkdir(parents=True, exist_ok=True)


def _frame(df, feats, with_label=True):
    cols = list(feats) + ([TARGET_COL] if with_label and TARGET_COL in df.columns else [])
    out = df[cols].copy()
    out.reset_index(drop=True, inplace=True)
    out._is_copy = None
    return out


def fit_autogluon(tag, train_df, feats, time_limit=TIME_LIMIT, num_gpus=NUM_GPUS):
    """AutoGluon を1回 fit する。保存済みなら読み込むだけ（Colab切断に備える）"""
    path = AG_DIR / tag
    if (path / "predictor.pkl").exists():
        logger.info(f"[{tag}] 保存済みpredictorを読み込む（再学習しない）")
        return TabularPredictor.load(str(path))

    logger.info("=" * 60)
    logger.info(f"[{tag}] AutoGluon fit: n={len(train_df)}, 特徴量={len(feats)}, "
                f"time_limit={time_limit}, excluded={EXCLUDED_MODELS}, num_gpus={num_gpus}")
    kw = dict(presets=PRESETS, time_limit=time_limit,
              excluded_model_types=EXCLUDED_MODELS,
              num_bag_folds=NUM_BAG_FOLDS, num_stack_levels=NUM_STACK_LEVELS,
              dynamic_stacking=DYNAMIC_STACKING, num_gpus=num_gpus)
    p = TabularPredictor(label=TARGET_COL, eval_metric=AG_METRIC, path=str(path),
                         problem_type="binary")
    p.fit(_frame(train_df, feats), **kw)
    return p


def leaderboard(predictor, data=None):
    try:
        return predictor.leaderboard(data, silent=True) if data is not None \
            else predictor.leaderboard(silent=True)
    except TypeError:
        return predictor.leaderboard(data) if data is not None else predictor.leaderboard()


def positive_proba_model(predictor, X, model=None):
    pp = predictor.predict_proba(X, model=model)
    pos = predictor.positive_class if hasattr(predictor, "positive_class") else None
    if pos is None or pos not in pp.columns:
        pos = 1 if 1 in pp.columns else pp.columns[-1]
    return pp[pos].values


def report_failures(predictor, tag):
    lb = leaderboard(predictor)
    names = list(lb["model"])
    n_xgb = sum(1 for m in names if m.startswith("XGBoost"))
    n_fastai = sum(1 for m in names if "FastAI" in m)
    logger.info(f"  [{tag}] 学習できたモデル {len(names)}件 / うち XGBoost {n_xgb}件")
    if n_fastai:
        logger.warning(f"  ⚠️ FASTAIが{n_fastai}件混ざっている（除外できていない）")
    if n_xgb == 0:
        logger.warning("  ⚠️ XGBoostが1件も無い")
    return lb


def save_submission(preds, config_label):
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_{config_label}_submission.csv"
    pd.DataFrame({ID_COL: test_features_full.index, TARGET_COL: preds}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={preds.mean():.4f}）")
    return str(path)


# ============================================================
# 3シード実行: seedごとにweighted/best_singleを抽出
# ============================================================
logger.info("=" * 60)
logger.info(f"[3シード実行] SEEDS={SEEDS} で独立にAutoGluonを学習...")
Xte = _frame(test_features_full, FEATURE_COLS, with_label=False)

weighted_preds_by_seed = {}
best_single_preds_by_seed = {}
seed_val_logloss = {}

for seed in SEEDS:
    seed_everything(seed=seed)
    tag = f"full447_seed{seed}"
    p = fit_autogluon(tag, ag_full, FEATURE_COLS)
    lb = report_failures(p, tag)
    logger.info(f"\n===== [{tag}] leaderboard(AutoGluon内部検証) =====\n"
                f"{lb[['model', 'score_val', 'fit_time']].head(12).to_string(index=False)}")

    weighted_names = lb[lb["model"].str.startswith("WeightedEnsemble")]["model"].tolist()
    single_names = lb[~lb["model"].str.startswith("WeightedEnsemble")]["model"].tolist()

    if weighted_names:
        mdl = weighted_names[0]
        preds = positive_proba_model(p, Xte, mdl)
        weighted_preds_by_seed[seed] = preds
        seed_val_logloss[(seed, "weighted")] = float(-lb.set_index("model").loc[mdl, "score_val"])
        save_submission(preds, f"seed{seed}_weighted")
        np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_seed{seed}_weighted_testpreds.npy", preds)

    if single_names:
        mdl = single_names[0]
        preds = positive_proba_model(p, Xte, mdl)
        best_single_preds_by_seed[seed] = preds
        seed_val_logloss[(seed, "best_single")] = float(-lb.set_index("model").loc[mdl, "score_val"])
        save_submission(preds, f"seed{seed}_best_single")
        np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_seed{seed}_best_single_testpreds.npy", preds)

    logger.info(f"[{tag}] 完了: weighted val={seed_val_logloss.get((seed, 'weighted')):.6f} / "
                f"best_single val={seed_val_logloss.get((seed, 'best_single')):.6f}")

# ============================================================
# 3シード平均
# ============================================================
logger.info("=" * 60)
logger.info("3シード平均を計算中...")
weighted_avg = np.mean(list(weighted_preds_by_seed.values()), axis=0)
best_single_avg = np.mean(list(best_single_preds_by_seed.values()), axis=0)
path_w = save_submission(weighted_avg, "weighted_avg3seed")
path_b = save_submission(best_single_avg, "best_single_avg3seed")
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_weighted_avg3seed_testpreds.npy", weighted_avg)
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_best_single_avg3seed_testpreds.npy", best_single_avg)

# ============================================================
# 参考診断: 現最良・旧プールCとの相関(置き換え/追加の判断材料、本スクリプトでは採否判定しない)
# ============================================================
logger.info("=" * 60)
logger.info("[参考diagnostics] 現最良・旧プールCとの相関を計算中...")


def _load_pred(path):
    df = pd.read_csv(path, header=None, names=[ID_COL, "p"])
    return df.set_index(ID_COL)["p"].reindex(test_features_full.index)


try:
    poolc_old = _load_pred(PROJECT_ROOT / "data" / "output" / "20260816" / "20260816_pool_poolC_weighted.csv")
    current_best = _load_pred(PROJECT_ROOT / "data" / "output" / "20260823" /
                               "20260823_94_blend_tabpfn89_into_best_add_w25_submission.csv")
    for label, preds in [("weighted_avg3seed", weighted_avg), ("best_single_avg3seed", best_single_avg)]:
        s = pd.Series(preds, index=test_features_full.index)
        corr_old = float(np.corrcoef(s.values, poolc_old.values)[0, 1])
        corr_best = float(np.corrcoef(s.values, current_best.values)[0, 1])
        logger.info(f"  {label}: corr(旧プールC)={corr_old:.4f} / corr(現最良94_add_w25)={corr_best:.4f}")
except FileNotFoundError as e:
    logger.warning(f"  参考ファイルが見つからず相関計算をスキップ: {e}")

logger.info("=" * 60)
logger.info("=== シード別val一覧 ===")
for k, v in seed_val_logloss.items():
    logger.info(f"  {k}: {v:.6f}")
logger.info(f"  weighted_avg3seed 提出ファイル: {path_w}")
logger.info(f"  best_single_avg3seed 提出ファイル: {path_b}")
logger.info("(置き換え/追加どちらのブレンドパターンにするかは、この結果を見てから別スクリプトで決める)")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
