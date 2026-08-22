# 独立プロセスとして実行するスクリプト（メモリを使い切ったらプロセスごと終了させ、
# OSにメモリを返す設計。Driveのマウントは呼び出し元(run_all.ipynb)が既に済ませている前提）。
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "catboost", "optuna"], check=True)

import multiprocessing
print(f"CPUコア数: {multiprocessing.cpu_count()}（今回はCPUで学習するため、GPUランタイムは不要）")

from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/jaggle_2026")
assert PROJECT_ROOT.exists(), (
    f"{PROJECT_ROOT} が見つからない。run_all.ipynb で先に drive.mount() を実行してから"
    " このスクリプトをサブプロセスとして呼び出すこと。"
)
sys.path.append(str(PROJECT_ROOT))

import datetime
import json
import re
import warnings

import numpy as np
import pandas as pd
import catboost as cb
import optuna
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common.utils.logger import get_logger
from common.utils.metrics import calculate_logloss
from common.utils.seed import seed_everything

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
seed_everything(seed=SEED)

TARGET_COL = "10年定着ラベル"
ID_COL = "社員ID"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

SCRIPT_NAME = "75_00_build_features"
TODAY = datetime.datetime.now().strftime("%Y%m%d")

LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 実験開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models" / TODAY / SCRIPT_NAME
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# チェックポイント（日付非依存の固定パス。セッションをまたいで再開できるようにする）
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "output" / "_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / f"{SCRIPT_NAME}_checkpoint.csv"

RESET_CHECKPOINT = True  # Trueにすると既存チェックポイントを削除して最初から再計算する
if RESET_CHECKPOINT and CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()
    print("チェックポイントを削除しました（全構成を再計算します）")

logger.info(f"Output Directory: {OUTPUT_DIR}")
logger.info(f"Checkpoint Path: {CHECKPOINT_PATH}")
if CHECKPOINT_PATH.exists():
    logger.info(f"既存のチェックポイントを発見: {len(pd.read_csv(CHECKPOINT_PATH))}件の結果が記録済み")
else:
    logger.info("チェックポイントは未作成（新規実行）")


INPUT_DIR = PROJECT_ROOT / "data" / "input"

train_persona = pd.read_csv(INPUT_DIR / "employee_persona_train.csv")
test_persona = pd.read_csv(INPUT_DIR / "employee_persona_test.csv")
train_monthly = pd.read_csv(INPUT_DIR / "employee_monthly_train.csv")
test_monthly = pd.read_csv(INPUT_DIR / "employee_monthly_test.csv")

logger.info(f"Train Persona Shape: {train_persona.shape}, Test Persona Shape: {test_persona.shape}")
logger.info(f"Train Monthly Shape: {train_monthly.shape}, Test Monthly Shape: {test_monthly.shape}")

y_train = train_persona[TARGET_COL]
train_ids = train_persona[ID_COL].values
test_ids = test_persona[ID_COL].values

logger.info(f"定着率: {y_train.mean():.4f}")
logger.info(f"Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")

EARLY_LEAVER_IDS = set(train_monthly.loc[train_monthly["月末在籍状態"] == "退職", ID_COL].unique())
_test_early = set(test_monthly.loc[test_monthly["月末在籍状態"] == "退職", ID_COL].unique())

logger.info(f"Train 早期退職者: {len(EARLY_LEAVER_IDS)}名 / {len(train_ids)}名 ({len(EARLY_LEAVER_IDS)/len(train_ids):.1%})")
logger.info(f"Test  早期退職者: {len(_test_early)}名 / {len(test_ids)}名")
_y_idx = train_persona.set_index(ID_COL)[TARGET_COL]
logger.info(f"早期退職者のラベル平均: {_y_idx.loc[list(EARLY_LEAVER_IDS)].mean():.4f}（0.0のはず）")
logger.info(f"定着率: 全体 {y_train.mean():.4f} / 早期退職者を除く {_y_idx[~_y_idx.index.isin(EARLY_LEAVER_IDS)].mean():.4f}")
assert len(_test_early) == 0, "Testに早期退職者が存在する。EDA v6の前提が崩れているので調査すること"

def create_monthly_aggregation_features(monthly_df, employee_ids):
    """月次データから集約特徴量を生成（12_〜18_と同一ロジック）"""
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
            changes = sum(1 for i in range(1, len(values)) if pd.notna(values[i]) and pd.notna(values[i-1]) and values[i] != values[i-1])
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
    """統計的特徴量：歪度、尖度、パーセンタイル"""
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
    """クラスター特徴量：月次データの平均をKMeansクラスタリング"""
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
    """欠勤日数パターン・360度評価タイミング・月次ボラティリティ・比率特徴量（12_〜18_と同一ロジック）"""
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
    """初期（経過月数=0）時点で同じ上司IDを持つ社員数（12_〜18_と同一ロジック）"""
    month0 = monthly_df[monthly_df["経過月数"] == 0].copy()
    month0["初期上司_部下数"] = month0.groupby("上司ID")["社員ID"].transform("count")
    out = month0[["社員ID", "初期上司_部下数"]]
    return out[out["社員ID"].isin(employee_ids)].reset_index(drop=True)

print("✅ split非依存の基本特徴量関数定義完了")

logger.info("-" * 60)
logger.info("split非依存の基本特徴量を生成中...")
logger.info("-" * 60)

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

logger.info("split非依存の基本特徴量生成完了")

def create_tfidf_svd_features(train_persona, test_persona, col, max_features=300, n_components=15, min_df=3, seed=42):
    '''文字n-gram TF-IDF + TruncatedSVDでテキスト特徴量を生成（Trainのみでfit）'''
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

TEXT_COLS = ["入社時メモ", "上司からのフィードバック", "同僚からのフィードバック"]

logger.info("テキストTF-IDF+SVD特徴量(A_v1)を生成中...")
tfidf_train_list, tfidf_test_list = [], []
for col in TEXT_COLS:
    tr, te, explained_var = create_tfidf_svd_features(train_persona, test_persona, col, max_features=300, n_components=15, min_df=3, seed=SEED)
    logger.info(f"{col}: SVD累積寄与率={explained_var:.3f}")
    tfidf_train_list.append(tr)
    tfidf_test_list.append(te)

logger.info("テキストTF-IDF+SVD特徴量生成完了")

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

D_EXPANDED_METRICS = [
    "残業時間", "有給取得日数", "欠勤日数", "研修時間",
    "上司との面談実施回数", "情報共有件数", "在宅勤務日数",
    "360度評価_親和度", "360度評価_信頼度", "360度評価_主体度",
    "360度評価_学習度", "360度評価_共有貢献度", "360度評価者数",
    "顧客満足度評価", "担当プロジェクト数", "月例給与_円",
]

logger.info("四半期/加速度特徴量(D_expanded: 16指標)を生成中...")
train_quarterly_exp = create_quarterly_features(train_monthly, train_ids, D_EXPANDED_METRICS, suffix="_exp")
test_quarterly_exp = create_quarterly_features(test_monthly, test_ids, D_EXPANDED_METRICS, suffix="_exp")
logger.info(f"D_expanded: Train {train_quarterly_exp.shape}, Test {test_quarterly_exp.shape}")

logger.info("Persona単位の基本特徴量を生成中...")
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

logger.info("Persona単位の基本特徴量処理完了")

def extract_workstyle_section(text):
    if pd.isna(text):
        return None
    m = re.search(r"勤務地・働き方：(.+?)$", text, re.S)
    if m:
        return m.group(1).strip()
    # 49_: 見出しがない書式B（276件、5.24%）のフォールバック。
    # 「勤務地・転居・在宅勤務」に言及する行を拾い、疑似セクションとして返す。
    # 以降のclassify_reloc/extract_desired_location_v1/v2はre.searchで探すだけなので、
    # 複数行を連結してもそのまま動く。
    lines = [l for l in text.strip().splitlines() if re.search(r"勤務地|転居|在宅勤務", l)]
    return "".join(lines) if lines else None


def _report_ws_coverage_fix(train_persona, test_persona):
    """49_の修正がどれだけカバー率を回復させたかをログに残す（診断専用、学習には影響しない）"""
    def old_fn(text):
        if pd.isna(text):
            return None
        m = re.search(r"勤務地・働き方：(.+?)$", text, re.S)
        return m.group(1).strip() if m else None

    all_persona = pd.concat([train_persona[["入社時メモ"]], test_persona[["入社時メモ"]]], ignore_index=True)
    ws_old = all_persona["入社時メモ"].apply(old_fn)
    ws_new = all_persona["入社時メモ"].apply(extract_workstyle_section)
    n = len(all_persona)
    logger.info(f"[49_診断] 見出し欠落 修正前 {ws_old.isna().sum()}件({ws_old.isna().sum()/n:.2%}) "
                f"→ 修正後 {ws_new.isna().sum()}件({ws_new.isna().sum()/n:.2%})")


_report_ws_coverage_fix(train_persona, test_persona)


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


def extract_desired_location_v1(s):
    '''27_・25_・EDA v3/v4/v5と同一（Public 0.529672で確認済み、カバー率88.6%/train）'''
    if s is None:
        return None
    m = re.search(r"(?:勤務地は|希望勤務地は)(.+?)(?:を希望|。)", s)
    if m:
        return m.group(1)
    m2 = re.search(r"(.+?)を希望勤務地", s)
    return m2.group(1) if m2 else None


def extract_desired_location_v2(s):
    '''v1に「◯◯(勤務|での勤務)?を希望。」パターンを追加した拡張版（カバー率94.1%/train）'''
    if s is None:
        return None
    loc = extract_desired_location_v1(s)
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

    # reloc_ok_rawはobject dtype(True/False/None混在)のため、~演算子は使わず
    # 明示的な等価比較でTrue/False/欠損を扱う（欠損に対する~はTypeErrorになる）
    reloc_true = reloc_ok_raw == True
    reloc_false = reloc_ok_raw == False
    valid = desired.notna() & reloc_ok_raw.notna()

    # 4値カテゴリ（決定木が交互作用を直接学習しやすいよう明示的にエンコード）
    state = pd.Series("unknown", index=persona_df.index)
    state[valid & reloc_true & match] = "許容_一致"
    state[valid & reloc_true & ~match] = "許容_不一致"
    state[valid & reloc_false & match] = "非許容_一致"
    state[valid & reloc_false & ~match] = "非許容_不一致"

    # ダブル悪条件フラグ（EDA v5で確認した最も強いシグナル: 転居許容せず AND 勤務地不一致）
    double_bad = (valid & reloc_false & ~match).astype(int)

    return pd.DataFrame({
        "社員ID": persona_df["社員ID"].values,
        state_col: state.values,
        flag_col: double_bad.values,
    })

logger.info("転居×勤務地マッチ交互作用特徴量(ブロックL v1/v2)を生成中...")
train_reloc_v1 = create_relocation_mismatch_features(train_persona, extract_desired_location_v1, "転居x勤務地_状態_v1", "転居x勤務地_ダブル悪条件_v1")
test_reloc_v1 = create_relocation_mismatch_features(test_persona, extract_desired_location_v1, "転居x勤務地_状態_v1", "転居x勤務地_ダブル悪条件_v1")
train_reloc_v2 = create_relocation_mismatch_features(train_persona, extract_desired_location_v2, "転居x勤務地_状態_v2", "転居x勤務地_ダブル悪条件_v2")
test_reloc_v2 = create_relocation_mismatch_features(test_persona, extract_desired_location_v2, "転居x勤務地_状態_v2", "転居x勤務地_ダブル悪条件_v2")

logger.info(f"L_v1: Train {train_reloc_v1.shape}, Test {test_reloc_v1.shape}")
logger.info(f"L_v2: Train {train_reloc_v2.shape}, Test {test_reloc_v2.shape}")
print("L_v1 ダブル悪条件:")
print(train_reloc_v1["転居x勤務地_ダブル悪条件_v1"].value_counts())
print("\nL_v2 ダブル悪条件:")
print(train_reloc_v2["転居x勤務地_ダブル悪条件_v2"].value_counts())


# ============================================================
# 54_: L2×Mのリスク要因数（Job Embeddedness理論: 複数の埋め込み不足の重なり）
#   ブロックM（専攻職種の分析的ミスマッチ）は29_で単体却下済み（GBDT redundancy）。
#   L2（転居x勤務地_状態_v2 = "非許容_不一致"）との組み合わせをフラグ化する。
#
#   ローカルEDAでの生の効果量（Cochran-Armitage傾向検定, p≈0・機械精度限界）:
#     リスク要因0個(n=1885): 定着率67.0%
#     リスク要因1個(n=653) : 定着率28.8%
#     リスク要因2個(n=47)  : 定着率 2.1%
#   単なるAND(2個該当)だけでなく、0→1→2ときれいな段階的用量反応があったため、
#   二値フラグに加えて順序尺度の"risk_count"も特徴量として渡す。
#   ただし該当セルのp値は採用基準(p<1e-20)を厳密には満たさないセルもあり、
#   単一の事前登録済み検証として扱う（閾値をチューニングしない）。
# ============================================================

_ANALYTICAL_MAJOR = {"情報", "理工学"}
_ANALYTICAL_JOB = {"IT・エンジニアリング", "データ・商品企画・コンサルティング"}


def create_l2_m_interaction_features(persona_df, reloc_v2_df):
    is_analytical_major = persona_df["専攻分野"].isin(_ANALYTICAL_MAJOR)
    is_analytical_job = persona_df["初期職種"].isin(_ANALYTICAL_JOB)
    m_bad = (~is_analytical_major & is_analytical_job).astype(int)

    state = reloc_v2_df.set_index("社員ID").loc[persona_df["社員ID"], "転居x勤務地_状態_v2"].values
    l2_bad = (state == "非許容_不一致").astype(int)

    both_bad = (l2_bad & m_bad)
    risk_count = l2_bad + m_bad  # 0/1/2の順序尺度（用量反応をそのまま渡す）

    return pd.DataFrame({
        "社員ID": persona_df["社員ID"].values,
        "M_不適合": m_bad,
        "L2xM_ダブル不適合": both_bad,
        "L2xM_リスク要因数": risk_count,
    })


train_l2m = create_l2_m_interaction_features(train_persona, train_reloc_v2)
test_l2m = create_l2_m_interaction_features(test_persona, test_reloc_v2)
logger.info(f"L2xMリスク特徴量: Train {train_l2m.shape}, Test {test_l2m.shape}")
print(train_l2m["L2xM_リスク要因数"].value_counts().sort_index())


def create_department_target_encoding(train_persona, test_persona, y_train, fit_ids, seed=42, n_splits=5, smoothing=10):
    '''初期部署IDのKFold + スムージング付きTarget Encoding（15_〜18_の修正版と同一ロジック）'''
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


def prepare_split(split_ratio, extra_blocks=None, exclude_early_from_val=True):
    '''指定した分割比率で特徴量を組み立てる。

    28_ からの変更点は2つだけ:
      - split_ratio=1.0 を許容（Train全件学習用。ag_tuningは空になる）
      - exclude_early_from_val=True のとき、検証セットから早期退職者を除く（改善3）
    特徴量の作り方そのものは 28_ と完全に同一。
    '''
    extra_blocks = extra_blocks or set()
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

    if "L1" in extra_blocks:
        tf = tf.merge(train_reloc_v1, on=ID_COL, how="left")
        ttf = ttf.merge(test_reloc_v1, on=ID_COL, how="left")

    if "L2" in extra_blocks:
        tf = tf.merge(train_reloc_v2, on=ID_COL, how="left")
        tf = tf.merge(train_l2m, on=ID_COL, how="left")
        ttf = ttf.merge(test_reloc_v2, on=ID_COL, how="left")
        ttf = ttf.merge(test_l2m, on=ID_COL, how="left")

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

    # --- 改善3: 検証セットから早期退職者を除く（学習側からは除かない） ---
    if exclude_early_from_val and len(ag_tuning) > 0:
        n_before = len(ag_tuning)
        ag_tuning = ag_tuning[~ag_tuning.index.isin(EARLY_LEAVER_IDS)]
        logger.info(f"  検証セット: {n_before} → {len(ag_tuning)}件（早期退職者{n_before - len(ag_tuning)}名を除外）")

    return ag_train, ag_tuning, ttf

print("✅ 部署Target Encoding・prepare_split関数定義完了（37_版: 全件学習・検証セット補正に対応）")

import time                      # 54_ の import セルには入っていないのでここで追加
from sklearn.metrics import log_loss

BLOCK = {"L2"}    # 54_ と同一（L_v2 + L2xM が入る）

logger.info("[全件] 特徴量を組み立て中...")
ag_full, _empty, test_features_full = prepare_split(1.0, extra_blocks=BLOCK, exclude_early_from_val=True)
assert len(_empty) == 0

def _feature_cols(df):
    return [c for c in df.columns if c not in ["入社日", TARGET_COL]]

BASE_FEATS = _feature_cols(ag_full)
y_all = ag_full[TARGET_COL].values.astype(int)
print(f"ベースライン {len(BASE_FEATS)} 列 / 学習 {len(ag_full)}名 / Test {len(test_features_full)}名")

# 生存者マスク（採点対象）。Test には早期退職者が0名なので、検証も生存者だけで採点する
IS_SURV = ~ag_full.index.isin(EARLY_LEAVER_IDS)
print(f"生存者 {IS_SURV.sum()}名 / 早期退職者 {(~IS_SURV).sum()}名（学習には使うが採点からは外す）")


A_PARAMS_ORIGINAL = {   # 54_の元のチューニング済み値（N_TRIALS=25・単一ホールドアウト）。比較用
    "depth": 4, "learning_rate": 0.03518359458951149, "l2_leaf_reg": 2.217690447016724,
    "border_count": 218, "bagging_temperature": 0.6787467566574921, "random_strength": 1.438494697238285,
}

obj_cols = [c for c in BASE_FEATS if ag_full[c].dtype == "object"]
X_full = ag_full[BASE_FEATS].fillna(-999)
y_full = ag_full[TARGET_COL].astype(int)
X_test = test_features_full[BASE_FEATS].fillna(-999)
print(f"obj_cols: {len(obj_cols)}列 / X_full: {X_full.shape} / X_test: {X_test.shape}")

# ============================================================
# キャッシュ保存（01_tune_*.py が再利用する。ここで一度だけ計算する）
# ============================================================
import pickle

CACHE_DIR = PROJECT_ROOT / "data" / "output" / "_checkpoints" / "75_tuned_ensemble_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

X_full.to_pickle(CACHE_DIR / "X_full.pkl")
X_test.to_pickle(CACHE_DIR / "X_test.pkl")
np.save(CACHE_DIR / "y_full.npy", y_full.to_numpy())
np.save(CACHE_DIR / "IS_SURV.npy", IS_SURV)
np.save(CACHE_DIR / "test_ids.npy", test_ids)
with open(CACHE_DIR / "obj_cols.json", "w", encoding="utf-8") as f:
    json.dump(obj_cols, f, ensure_ascii=False)
with open(CACHE_DIR / "A_PARAMS_ORIGINAL.json", "w", encoding="utf-8") as f:
    json.dump(A_PARAMS_ORIGINAL, f, ensure_ascii=False)

logger.info(f"特徴量キャッシュを保存: {CACHE_DIR}")
print(f"\n✅ 特徴量キャッシュを保存しました: {CACHE_DIR}")
print(f"   X_full={X_full.shape}  X_test={X_test.shape}  IS_SURV.sum()={IS_SURV.sum()}")
print("このプロセスをここで終了する（メモリを解放してから次のスクリプトに進む）。")


# ============================================================
# 基盤モデル(TabPFN/TabICL)用: 441列基盤(L2xM無し, 63_/70_と同一)のキャッシュ
#   54_/73_の444列パイプラインから L2xM の3列を落とすだけで63_/70_と同一の441列になる
#   （72_で確認済み: 63_/70_は51_と同一パイプラインでL2xM導入(54_)より前のバージョン）。
#   追加で 80/20ホールドアウト分割・CatBoost重要度(top150)・hire_fixed(78列)も
#   63_/70_と同一の手順で計算してキャッシュする。
# ============================================================
logger.info("-" * 60)
logger.info("基盤モデル用441列キャッシュを構築中...")

L2XM_COLS = ["M_不適合", "L2xM_ダブル不適合", "L2xM_リスク要因数"]
FEATS_441 = [c for c in BASE_FEATS if c not in L2XM_COLS]
assert len(FEATS_441) == 441, f"{len(FEATS_441)}列（441列のはず）"

# 63_/70_と同一の80/20ホールドアウト分割（prepare_splitは決定的なので同一分割が得られる）。
# 第3戻り値はttf(テスト特徴量)で、split_ratioに関わらず常に非空（空になるのはsplit_ratio=1.0
# のときの第2戻り値だけ）。ここでは使わないので破棄する。
ag_train_80b, ag_val_surv, _ttf_dup80 = prepare_split(0.8, extra_blocks=BLOCK, exclude_early_from_val=True)
assert len(ag_val_surv) == 535, f"{len(ag_val_surv)}名（535名のはず）"

# --- CatBoost重要度でtop150を選ぶ（63_と同一: A_PARAMS・ITER_FIXED=560・8シード） ---
SEEDS_VAL_FM = [42, 2024, 7, 1234, 99, 555, 31337, 2718]
_obj80 = [c for c in FEATS_441 if ag_train_80b[c].dtype == "object"]
_Xtr80, _ytr80 = ag_train_80b[FEATS_441].fillna(-999), ag_train_80b[TARGET_COL]
_Xva80 = ag_val_surv[FEATS_441].fillna(-999)

_imp_models = []
for _s in SEEDS_VAL_FM:
    _m = cb.CatBoostClassifier(**A_PARAMS_ORIGINAL, iterations=560, random_seed=_s, verbose=False,
                                cat_features=_obj80, task_type="CPU")
    _m.fit(_Xtr80, _ytr80)
    _imp_models.append(_m)
    logger.info(f"  [重要度算出] seed={_s}: 完了")

_imp = pd.Series(np.mean([m.get_feature_importance() for m in _imp_models], axis=0),
                  index=FEATS_441).sort_values(ascending=False)
FEATS_TOP150 = _imp.head(150).index.tolist()
print(f"top150（CatBoost重要度上位）: {len(FEATS_TOP150)}列")

# --- hire_fixed（70_と同一: HIRE_GROUPS = persona/deptte/derived/L2/tfidf） ---
DERIVED_COLS = [
    "残業時間_mean_job_deviation", "研修時間_mean_job_deviation", "360度評価_親和度_mean_job_deviation",
    "研修時間_職種比", "研修時間_区分比",
    "初任給_等級内偏差", "初任給_区分内偏差", "月例給与_等級内偏差",
]
DEPT_TE_COLS = ["dept_target_enc", "dept_size"]


def _cols_of(df):
    return [c for c in df.columns if c != ID_COL]


_ALL_FEATS_SET = set(FEATS_441)
_RAW_GROUPS = {
    "persona":   [c for c in train_persona.columns if c not in (ID_COL, TARGET_COL)],
    "agg":       _cols_of(train_monthly_agg),
    "catchange": _cols_of(train_cat_change),
    "missing":   _cols_of(train_missing),
    "domain":    _cols_of(train_domain),
    "advstats":  _cols_of(train_advanced_stats),
    "cluster":   _cols_of(train_cluster),
    "deptte":    DEPT_TE_COLS,
    "edafeat":   _cols_of(train_eda_feats),
    "mgr":       _cols_of(train_mgr),
    "quarterly": _cols_of(train_quarterly_exp),
    "tfidf":     [c for _df in tfidf_train_list for c in _cols_of(_df)],
    "L2":        _cols_of(train_reloc_v2),
    "derived":   DERIVED_COLS,
}
FEATURE_GROUPS_FM = {g: [c for c in cols if c in _ALL_FEATS_SET] for g, cols in _RAW_GROUPS.items()}

HIRE_GROUPS = ["persona", "deptte", "derived", "L2", "tfidf"]
_hire_cols = {c for g in HIRE_GROUPS if g in FEATURE_GROUPS_FM for c in FEATURE_GROUPS_FM[g]}
FEATS_HIRE_FIXED = [c for c in FEATS_441 if c in _hire_cols]
print(f"hire_fixed（persona/deptte/derived/L2/tfidfのみ）: {len(FEATS_HIRE_FIXED)}列")

# --- キャッシュ保存（RAW, NaN未補完 — TabPFNはNaNをネイティブに扱えるので埋めない） ---
FM_CACHE_DIR = PROJECT_ROOT / "data" / "output" / "_checkpoints" / "75_tuned_ensemble_cache_fm"
FM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ag_train_80b[FEATS_441 + [TARGET_COL]].to_pickle(FM_CACHE_DIR / "ag_train_80b.pkl")
ag_val_surv[FEATS_441 + [TARGET_COL]].to_pickle(FM_CACHE_DIR / "ag_val_surv.pkl")
ag_full[FEATS_441 + [TARGET_COL]].to_pickle(FM_CACHE_DIR / "ag_full.pkl")
test_features_full[FEATS_441].to_pickle(FM_CACHE_DIR / "test_features_full.pkl")
np.save(FM_CACHE_DIR / "test_ids.npy", test_ids)

with open(FM_CACHE_DIR / "feature_sets.json", "w", encoding="utf-8") as f:
    json.dump({"full441": FEATS_441, "top150": FEATS_TOP150, "hire_fixed": FEATS_HIRE_FIXED},
               f, ensure_ascii=False)

logger.info(f"基盤モデル用キャッシュを保存: {FM_CACHE_DIR}")
print(f"\n✅ 基盤モデル用441列キャッシュを保存しました: {FM_CACHE_DIR}")
print(f"   ag_train_80b={ag_train_80b.shape}  ag_val_surv={ag_val_surv.shape}  "
      f"ag_full={ag_full.shape}  test_features_full={test_features_full.shape}")
