"""96_ft_transformer_on_84_top30pct

イテレーション1（実験の性質上、複数回の試行錯誤を前提とする——初回のスコアが悪くても
即座に切り捨てず、`88_`のロジスティック回帰と同じ要領でこのファイル自体を書き換えて
何度か試す）。

これまでこのプロジェクトで試したニューラルネット系はいずれも「表形式データに特化した
設計」ではなかった: `55_`のGRU(月次系列用RNN、絶対性能+0.136で致命的)、AutoGluonの
NN_TORCH(汎用全結合層、除外した方がスコア改善)、`71_`のMLP(sklearn標準の単純な
多層パーセプトロン、val 0.566216で閉じた、[[cpu-model-zoo-closed]])。GBDT(CatBoost)・
表形式基盤モデル(TabPFN)とは異なる第3の系統として、**FT-Transformer**
（Feature Tokenizer + Transformer、Gorishniy et al. "Revisiting Deep Learning Models
for Tabular Data"、`rtdl_revisiting_models`パッケージ）を初めて試す。
（旧`rtdl`パッケージは依存関係が古く固定されており、現行Colab環境ではpip解決不能
 [ResolutionImpossible]で導入できない。著者公式の後継パッケージに切り替えた。）

84_のTOP30PCT_nestedの列選択(CatBoostの重要度上位30%)は各foldの学習データだけで
やり直す(選択リークを避ける、[[validation-asymmetry]]のHow to apply参照)。選択した134列に
対してCatBoostとFT-Transformerの両方をfitし、リークなしKFold OOFで両者を比較・
ブレンド曲線を確認する([[blend-curve-beats-val-margin-gate]])。

FT-Transformerの実装メモ:
  - 数値列: 欠損フラグ→中央値補完→標準化（88_のprepare_linear_matricesと同じ数値パイプライン）
  - カテゴリ列: 序数コード化(0..k-1)し、列ごとの埋め込みテーブルに渡す（TabPFNの
    序数コード化とは違い、列ごとに別々の埋め込み次元を学習する）
  - 学習: `FTTransformer(**FTTransformer.get_default_kwargs())` + AdamW(`model.make_parameter_groups()`)
    + BCEWithLogitsLoss、fold学習データの10%を内部検証にしてearly stopping
  - GPU推奨だが必須ではない(小データなのでCPUでも数分程度で終わる見込み)

列数の足切りは84_/86_(30%)・93_(20%)の前例に加え、より絞った15%も同一実行内で比較する
（TOP_PCTS=[0.30, 0.20, 0.15]）。FT-Transformerは1fitが数分程度で済むため、TabPFNの
ように別スクリプトに分けず1回のColab実行で3水準ぶんの結果が得られる設計にした。

出力: 各水準(top30pct/top20pct/top15pct)ごとに cb_classifier / ft_classifier /
blend_argmin の3ファイル、計9ファイル
（+ OOF・Test予測を.npyで保存し、後日プール/xxxx_v4とのブレンドに再利用できるようにする）

イテレーション2（2026-08-24）: イテレーション1はCPU実行で約4時間かかり、結果は
CatBoost比+0.031〜+0.035（TOP30/20/15PCTいずれも）でargminブレンドの重みw_cb=0.90
——FT側の寄与が分解能未満([[validation-asymmetry]])という「明確に劣るが致命的でない」
判定だった（詳細は[[ft-transformer-96-action-plan]]の2026-08-24追記）。GPU+TOP15PCT
単独+FT_SEEDS=1に絞り、FT_EPOCHS 100→200・FT_PATIENCE 10→15を試したが、
FT val が0.546026→0.569670に悪化(+0.0236)、CB比ギャップも+0.032→+0.0498に拡大
——「早期停止が早すぎて未収束」という仮説は誤りで、むしろ内部検証10%という小さく
ノイジーな基準に対して過学習方向に進んだと解釈。

イテレーション3（2026-08-24）: FT_EPOCHS/FT_PATIENCEはイテレーション1の値(100/10)に
戻し、代わりにモデル容量を絞る方向を試す——`FTTransformer.get_default_kwargs(n_blocks=1)`
（既定n_blocks=3・d_block=192から、パッケージ公式のn_blocks別スケーリング表に基づき
n_blocks=1・d_block=96・attention_dropout=0.1・ffn_dropout=0.0に縮小、手動の
アドホックな値ではなく著者が用意したn_blocks別デフォルト表を使用）。TOP15PCT単独・
FT_SEEDS=1のままGPUで高速反復（ユーザー指示: GPUなら処理時間短いので試行継続でよい）。
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

SCRIPT_NAME = "96_ft_transformer_on_84_top30pct"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 実験開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# 54_l2_m_interaction.ipynb と同一の特徴量関数（split非依存、84_と同一）
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
# build_features(train_id_subset): 80_/81_/82_/83_/84_と同一パターン（純77_相当447列）
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
FT_SEEDS = [42]  # イテレーション2: GPU+高速反復のため1シードに削減(良化したら3シードに戻す)
OOF_SEED = 42
OOF_N_SPLITS = 5
# イテレーション2: イテレーション1(3水準)で最良だったTOP15PCTのみに絞り高速反復
TOP_PCTS = [0.15]
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
# FT-Transformer 統合（rtdl_revisiting_modelsパッケージ、GBDT/TabPFNとは異なる第3の系統）
# ============================================================
logger.info("=" * 60)
logger.info("FT-Transformer (rtdl_revisiting_models) を読み込み中...")
import torch
import torch.nn as nn
from rtdl_revisiting_models import FTTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"device = {DEVICE}")
if DEVICE == "cpu":
    logger.warning("GPUが無い。小データなのでCPUでも動くはずだが、遅い場合はColabのランタイムをGPUに変更すること。")

FT_EPOCHS = 100  # イテレーション3: イテレーション2(200)は過学習方向で悪化、100に戻す
FT_BATCH_SIZE = 128
FT_LR = 1e-4
FT_WEIGHT_DECAY = 1e-5
FT_PATIENCE = 10  # イテレーション3: イテレーション2(15)は悪化、10に戻す
FT_VAL_FRAC = 0.1  # fold学習データのうちearly stopping用に取り分ける割合
FT_N_BLOCKS = 1  # イテレーション3: 既定3→1(モデル容量を縮小、小データでの過学習対策)


def to_ft_matrix(train_df, other_dfs, feats):
    """FT-Transformer用の行列を作る。数値列は標準化、カテゴリ列は列ごとに
    序数コード化(0..k-1、未知カテゴリはk番目)して別々の埋め込みテーブルに渡す。"""
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
        cardinalities.append(len(vals) + 1)  # +1: 未知カテゴリ用の予備インデックス
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
    # イテレーション3: n_blocks=1(既定3)——著者公式のn_blocks別スケーリング表により
    # d_block=96/attention_dropout=0.1/ffn_dropout=0.0も連動して縮小される
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

    # cardinalitiesはto_ft_matrixがtrain_df全体のカテゴリから作った値を使う
    # (fold内のtr/val分割の偶然による過小カウントを避けるため、部分集合のmaxは使わない)
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
    """FT-Transformerをシード平均で学習して予測を返す"""
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
# Train全件のtf/ttf（提出用）
# ============================================================
logger.info("=" * 60)
logger.info("Train全件でのtf/ttfを構築中(提出用)...")
tf_full, ttf_full = build_features(train_ids.tolist())
FULL_FEATURE_COLS = _feature_cols(tf_full)
assert len(FULL_FEATURE_COLS) == 447, f"想定外の列数: {len(FULL_FEATURE_COLS)}"

# ============================================================
# リークなしKFold OOF: 各foldでCatBoost選択→30%/20%/15%の3水準でCatBoost/FT-Transformerを評価
# ============================================================
logger.info("=" * 60)
logger.info(f"[CB/FT × {TOP_PCTS}] ネストされた特徴量選択+CatBoost/FT-Transformer比較のOOFを構築中...")
skf = StratifiedKFold(n_splits=OOF_N_SPLITS, shuffle=True, random_state=OOF_SEED)

cb_oof = {pct: np.zeros(len(train_ids)) for pct in TOP_PCTS}
ft_oof = {pct: np.zeros(len(train_ids)) for pct in TOP_PCTS}
ncols_per_fold = {pct: [] for pct in TOP_PCTS}

for fold_i, (tr_pos, va_pos) in enumerate(skf.split(train_ids, y_train)):
    fold_train_ids = train_ids[tr_pos].tolist()
    fold_val_ids = train_ids[va_pos].tolist()
    tf_f, _ = build_features(fold_train_ids)
    feat_cols_f = _feature_cols(tf_f)
    obj_cols_f = [c for c in feat_cols_f if tf_f[c].dtype == "object"]

    X_tr = tf_f.loc[fold_train_ids, feat_cols_f].fillna(-999)
    y_tr_cls = tf_f.loc[fold_train_ids, TARGET_COL]

    # 選択ステップ: このfoldの学習データだけで重要度を計算(84_と同一パターン、
    # 3水準とも同じ重要度モデルを使い回すので選択自体の計算は1回で済む)
    m_selector = _fit_one_classifier(X_tr, y_tr_cls, obj_cols_f, OOF_SEED)
    imp_series = pd.Series(m_selector.get_feature_importance(), index=feat_cols_f)

    fold_train_df = tf_f.loc[fold_train_ids]
    fold_val_df = tf_f.loc[fold_val_ids]

    for pct in TOP_PCTS:
        top_cols = select_top_pct(imp_series, pct)
        ncols_per_fold[pct].append(len(top_cols))
        obj_top = [c for c in top_cols if c in obj_cols_f]

        X_tr_p = tf_f.loc[fold_train_ids, top_cols].fillna(-999)
        X_va_p = tf_f.loc[fold_val_ids, top_cols].fillna(-999)
        m_cb = _fit_one_classifier(X_tr_p, y_tr_cls, obj_top, OOF_SEED)
        cb_oof[pct][va_pos] = m_cb.predict_proba(X_va_p)[:, 1]

        (ft_va,) = ft_fit_predict(fold_train_df, [fold_val_df], top_cols, FT_SEEDS)
        ft_oof[pct][va_pos] = ft_va

    logger.info(f"    [fold{fold_i}] 完了 (列数: {[(pct, ncols_per_fold[pct][-1]) for pct in TOP_PCTS]})")

cb_score, ft_score, corr_local = {}, {}, {}
for pct in TOP_PCTS:
    cb_score[pct] = log_loss(y_train.values[surv_mask], cb_oof[pct][surv_mask])
    ft_score[pct] = log_loss(y_train.values[surv_mask], ft_oof[pct][surv_mask])
    corr_local[pct] = float(np.corrcoef(cb_oof[pct], ft_oof[pct])[0, 1])
    logger.info(f"[TOP{int(pct*100)}PCT] CB val={cb_score[pct]:.6f} / FT val={ft_score[pct]:.6f} / "
                f"相関={corr_local[pct]:.4f} (fold毎列数: {ncols_per_fold[pct]})")

# ============================================================
# ブレンド曲線(argmin)を局所OOFで確認 -- 3水準それぞれで走査する。
# ラベル付きデータが両モデルとも揃っているので、argmin重みをそのまま採用してよい
# ============================================================
logger.info("=" * 60)
logger.info("[ブレンド曲線] 3水準それぞれでw=CatBoost側の重みを0〜1で走査...")
y_surv = y_train.values[surv_mask]
best_w, best_ll = {}, {}
for pct in TOP_PCTS:
    cb_s = cb_oof[pct][surv_mask]
    ft_s = ft_oof[pct][surv_mask]
    scan_rows = []
    for w in BLEND_WEIGHTS:
        blend = np.clip(w * cb_s + (1 - w) * ft_s, 1e-9, 1 - 1e-9)
        ll = log_loss(y_surv, blend)
        scan_rows.append((w, ll))
    best_w[pct], best_ll[pct] = min(scan_rows, key=lambda r: r[1])
    logger.info(f"[TOP{int(pct*100)}PCT argmin] best_w_cb={best_w[pct]:.2f}, best_val={best_ll[pct]:.6f} "
                f"(CB単体比 {best_ll[pct] - cb_score[pct]:+.6f} / FT単体比 {best_ll[pct] - ft_score[pct]:+.6f})")

logger.info("=" * 60)
logger.info("[3水準まとめ] pct | 列数(fold0) | CB val | FT val | 相関 | argmin(w_cb, val)")
for pct in TOP_PCTS:
    logger.info(f"  {int(pct*100)}%: 列数={ncols_per_fold[pct][0]} / CB={cb_score[pct]:.6f} / "
                f"FT={ft_score[pct]:.6f} / 相関={corr_local[pct]:.4f} / "
                f"argmin=(w_cb={best_w[pct]:.2f}, val={best_ll[pct]:.6f})")

# ============================================================
# OOFの保存(後日プール/xxxx_v4とのブレンドに再利用するため)
# ============================================================
for pct in TOP_PCTS:
    np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_cb_top{int(pct*100)}pct_oofpreds.npy", cb_oof[pct])
    np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_ft_top{int(pct*100)}pct_oofpreds.npy", ft_oof[pct])
logger.info("  OOF予測(.npy)を保存しました")

# ============================================================
# 提出: Train全件で選択→3水準×(CatBoost5シード・FT-Transformer3シード・argminブレンド)
# ============================================================
logger.info("=" * 60)
logger.info("提出用: Train全件で選択ステップを実行中(Testはラベルを持たないためリークなし)...")
obj_cols_full = [c for c in FULL_FEATURE_COLS if tf_full[c].dtype == "object"]
X_tr_full = tf_full[FULL_FEATURE_COLS].fillna(-999)
y_tr_full = tf_full[TARGET_COL]

m_selector_full = _fit_one_classifier(X_tr_full, y_tr_full, obj_cols_full, OOF_SEED)
imp_series_full = pd.Series(m_selector_full.get_feature_importance(), index=FULL_FEATURE_COLS)

all_paths = []
for pct in TOP_PCTS:
    tag = f"top{int(pct*100)}pct"
    logger.info("=" * 60)
    top_cols_full = select_top_pct(imp_series_full, pct)
    logger.info(f"[提出用 {tag}] {len(top_cols_full)}列")
    obj_top_full = [c for c in top_cols_full if c in obj_cols_full]

    X_tr_p_full = tf_full[top_cols_full].fillna(-999)
    X_test_p_full = ttf_full[top_cols_full].fillna(-999)

    cb_test_preds = []
    last_cb_model = None
    for seed in SEEDS_SUB:
        last_cb_model = _fit_one_classifier(X_tr_p_full, y_tr_full, obj_top_full, seed)
        cb_test_preds.append(last_cb_model.predict_proba(X_test_p_full)[:, 1])
    cb_test = np.mean(cb_test_preds, axis=0)
    cb_model_path = SAVED_MODELS_DIR / f"cb_{tag}_full_lastseed.cbm"
    last_cb_model.save_model(str(cb_model_path))
    logger.info(f"  モデル保存: {cb_model_path}")
    p1 = save_submission(cb_test, f"cb_{tag}_classifier")

    logger.info(f"[FT-full {tag}] シード平均で学習・予測中...")
    (ft_test,) = ft_fit_predict(tf_full, [ttf_full], top_cols_full, FT_SEEDS)
    p2 = save_submission(ft_test, f"ft_{tag}_classifier")

    _num_arrs, _cat_arrs, _cardinalities = to_ft_matrix(tf_full, [], top_cols_full)
    last_ft_model = _ft_train_one(_num_arrs[0], _cat_arrs[0], tf_full[TARGET_COL].values.astype(np.float32),
                                   _cardinalities, FT_SEEDS[0])
    ft_model_path = SAVED_MODELS_DIR / f"ft_{tag}_full_seed42.pt"
    torch.save(last_ft_model.state_dict(), ft_model_path)
    logger.info(f"  モデル保存: {ft_model_path}")

    blend_test = best_w[pct] * cb_test + (1 - best_w[pct]) * ft_test
    p3 = save_submission(blend_test, f"{tag}_blend_argmin_w{round(best_w[pct] * 100):d}pct")

    np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_cb_{tag}_testpreds.npy", cb_test)
    np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_ft_{tag}_testpreds.npy", ft_test)
    all_paths.extend([p1, p2, p3])
    logger.info(f"[{tag}] 局所OOF: CB={cb_score[pct]:.6f} / FT={ft_score[pct]:.6f} / "
                f"argminブレンド(w_cb={best_w[pct]:.2f})={best_ll[pct]:.6f} / 相関={corr_local[pct]:.4f}")

logger.info("=" * 60)
logger.info(f"=== 全{len(all_paths)}ファイル出力完了 ===")
for p in all_paths:
    logger.info(f"  {p}")
logger.info("(参考: 84_のTOP30PCT_nested Public=0.509500。93_のTOP20PCT(89列)Public=0.504065(TabPFN単体)。"
            "CB系はこれらとほぼ同じ構成のはず)")
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
