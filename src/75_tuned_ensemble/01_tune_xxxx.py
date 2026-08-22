# 独立プロセスとして実行するスクリプト。xxxx_v4.ipynb（reference著者パイプライン、104列・
# text_featuresネイティブ・StratifiedKFold）の特徴量を自前で再構築し、分類器のハイパー
# パラメータだけをOptunaで強化探索する。回帰器は xxxx_v4 の元の設定のまま変えない
# （一度に複数要因を変えないため。[[modeling-levers-beat-new-features]]参照）。
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "catboost", "japanize-matplotlib", "janome", "optuna"], check=True)

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from janome.tokenizer import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path("/content/drive/MyDrive/jaggle_2026")
assert PROJECT_ROOT.exists(), (
    f"{PROJECT_ROOT} が見つからない。run_all.ipynb で先に drive.mount() を実行してから"
    " このスクリプトをサブプロセスとして呼び出すこと。"
)
sys.path.append(str(PROJECT_ROOT))

from common.utils.logger import get_logger
from common.utils.seed import seed_everything

import datetime

SCRIPT_NAME = "75_01_tune_xxxx"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
LOG_DIR = PROJECT_ROOT / "logs"
logger = get_logger(SCRIPT_NAME, log_dir=str(LOG_DIR))
logger.info(f"=== [{SCRIPT_NAME}] 開始 ===")

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / TODAY
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
seed_everything(seed=SEED)

データ置き場 = PROJECT_ROOT / "data" / "input"

# ============================================================
# 1. データの読み込み
# ============================================================
logger.info("-" * 60)
logger.info("データを読み込み中...")
入社時データ_学習 = pd.read_csv(データ置き場 / "employee_persona_train.csv")
入社時データ_予測 = pd.read_csv(データ置き場 / "employee_persona_test.csv")
月次データ_学習 = pd.read_csv(データ置き場 / "employee_monthly_train.csv")
月次データ_予測 = pd.read_csv(データ置き場 / "employee_monthly_test.csv")
月次データ_学習_全期間 = pd.read_csv(データ置き場 / "employee_monthly_train_full.csv")

logger.info(f"入社時データ_学習: {入社時データ_学習.shape}")
logger.info(f"入社時データ_予測: {入社時データ_予測.shape}")
logger.info(f"月次データ_学習 : {月次データ_学習.shape}")
logger.info(f"月次データ_予測 : {月次データ_予測.shape}")
logger.info(f"月次データ_学習_全期間: {月次データ_学習_全期間.shape}")

# ============================================================
# 2. 月次集約特徴量の作成
# ============================================================
def 学習時間の合計(text):
    if pd.isna(text) or text == "受講なし":
        return 0.0
    return sum(float(x) for x in re.findall(r"([\d.]+)時間", text))

def 学習テーマの一覧(text):
    if pd.isna(text) or text == "受講なし":
        return []
    return [n.strip().replace(" ", "") for n in re.findall(r"([^｜：]+)：[\d.]+時間", text)]

def 傾き(値の列):
    有効 = ~np.isnan(値の列)
    if 有効.sum() < 2:
        return 0.0
    return np.polyfit(np.arange(len(値の列))[有効], 値の列[有効], 1)[0]

等級の数値 = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}
役割の数値 = {"メンバー": 1, "シニア": 2, "エキスパート": 3, "リード": 4,
          "シニアエキスパート": 4, "マネージャー": 5}
行動列 = ["残業時間", "有給取得日数", "欠勤日数", "研修時間",
        "上司との面談実施回数", "情報共有件数", "在宅勤務日数"]
評価列 = ["360度評価_親和度", "360度評価_信頼度", "360度評価_主体度",
        "360度評価_学習度", "360度評価_共有貢献度"]
評価系列 = [*評価列, "顧客満足度評価", "担当プロジェクト数", "360度評価者数"]

def 月次集約(月次データ: pd.DataFrame) -> pd.DataFrame:
    月次データ = 月次データ.sort_values(["社員ID", "経過月数"]).copy()
    月次データ["学習時間"] = 月次データ["自己学習（詳細）"].apply(学習時間の合計)
    月次データ["学習テーマ"] = 月次データ["自己学習（詳細）"].apply(学習テーマの一覧)
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
            "学習時間合計": g["学習時間"].sum(),
        }
        テーマ = [t for 一覧 in g["学習テーマ"] for t in 一覧]
        行["学習txt"] = " ".join(テーマ) if テーマ else "なし"
        行["学習テーマ数"] = len(set(テーマ))
        行["学習月数"] = int((g["学習時間"] > 0).sum())

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

logger.info("-" * 60)
logger.info("月次集約を実行中...")
月次集約_学習 = 月次集約(月次データ_学習)
月次集約_予測 = 月次集約(月次データ_予測)
logger.info(f"月次集約_学習: {月次集約_学習.shape}, 月次集約_予測: {月次集約_予測.shape}")

# ============================================================
# 3. 入社時特徴量と文書分割
# ============================================================
テキスト列 = ["入社時メモ", "上司からのフィードバック", "同僚からのフィードバック"]

def 入社時特徴量(入社時データ: pd.DataFrame) -> pd.DataFrame:
    d = 入社時データ.copy()
    d["入社年"] = pd.to_datetime(d["入社日"]).dt.year
    d["前職職種"] = d["前職職種"].fillna("前職なし")
    d["初期等級_num"] = d["初期等級"].map(等級の数値)
    d["初期役割_num"] = d["初期役割"].map(役割の数値)
    for 列 in テキスト列:
        d[f"{列}_文字数"] = d[列].str.len()
    return d

入社時_学習 = 入社時特徴量(入社時データ_学習).reset_index(drop=True)
入社時_予測 = 入社時特徴量(入社時データ_予測).reset_index(drop=True)

初期部署の頻度 = pd.concat([入社時_学習["初期部署ID"], 入社時_予測["初期部署ID"]]).value_counts()
入社時_学習["初期部署ID_頻度"] = 入社時_学習["初期部署ID"].map(初期部署の頻度)
入社時_予測["初期部署ID_頻度"] = 入社時_予測["初期部署ID"].map(初期部署の頻度)

最終部署の頻度 = pd.concat([月次集約_学習["最終月の部署ID"], 月次集約_予測["最終月の部署ID"]]).value_counts()
月次集約_学習["最終月の部署ID_頻度"] = 月次集約_学習["最終月の部署ID"].map(最終部署の頻度)
月次集約_予測["最終月の部署ID_頻度"] = 月次集約_予測["最終月の部署ID"].map(最終部署の頻度)

形態素解析器 = Tokenizer()
残す品詞 = {"名詞", "動詞", "形容詞"}

def 文書分割(text):
    if pd.isna(text):
        return ""
    return " ".join(語.base_form for 語 in 形態素解析器.tokenize(text)
                    if 語.part_of_speech.split(",")[0] in 残す品詞)

logger.info("-" * 60)
logger.info("入社時メモの文書分割を実行中...")
入社時_学習["メモtxt"] = 入社時_学習["入社時メモ"].apply(文書分割)
入社時_予測["メモtxt"] = 入社時_予測["入社時メモ"].apply(文書分割)
logger.info(f"文書分割サンプル: {入社時_学習['メモtxt'].iloc[0][:100]}")

# ============================================================
# 4. メモのパース
# ============================================================
勤務地一覧 = ["東京", "大阪", "愛知", "福岡", "仙台", "北海道", "その他"]
転居の否定 = r"許容せず|許容しておらず|許容していない|許容しない|希望しておらず|希望せず|希望していない"
在宅の否定 = r"必須条件と(?:は)?していない|不要"

def メモをパース(メモ: pd.Series) -> pd.DataFrame:
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

logger.info("-" * 60)
logger.info("メモのパースを実行中...")
for データ in (入社時_学習, 入社時_予測):
    パース結果 = メモをパース(データ["入社時メモ"])
    for 列 in ["転居許容", "在宅希望", "希望勤務地", "キャリア志向"]:
        データ[列] = パース結果[列].to_numpy()

for 列 in ["転居許容", "在宅希望", "キャリア志向"]:
    count_dict = dict(pd.concat([入社時_学習[列], 入社時_予測[列]]).value_counts())
    logger.info(f"{列} の分布: {count_dict}")

# ============================================================
# 5. 特徴量の結合
# ============================================================
logger.info("-" * 60)
logger.info("データを結合し、CatBoostに渡す特徴量を準備中...")

目的変数 = "10年定着ラベル"

学習データ = 入社時_学習.merge(月次集約_学習, on="社員ID", how="left")
予測データ = 入社時_予測.merge(月次集約_予測, on="社員ID", how="left")

カテゴリ列 = ["入社区分", "最終学歴", "専攻分野", "前職職種", "採用経路", "性別",
         "初期職種", "初期勤務地", "初期等級", "初期役割",
         "最終月の勤務地", "最終月の職種", "最終月の部署ID",
         "希望勤務地", "キャリア志向"]
text列 = ["メモtxt", "学習txt"]
除外列 = ["社員ID", "入社日", "初期部署ID", *テキスト列, 目的変数]

正解ラベル = 学習データ[目的変数].astype(int)
特徴量 = 学習データ.drop(columns=[c for c in 除外列 if c in 学習データ.columns])
予測用特徴量 = 予測データ.drop(columns=[c for c in 除外列 if c in 予測データ.columns])

for 列 in カテゴリ列:
    特徴量[列] = 特徴量[列].astype(str).fillna("missing")
    予測用特徴量[列] = 予測用特徴量[列].astype(str).fillna("missing")

logger.info(f"学習データ: {学習データ.shape}, 予測データ: {予測データ.shape}")
logger.info(f"特徴量 {特徴量.shape[1]} 列(内カテゴリ{len(カテゴリ列)}・テキスト{len(text列)})")

# ============================================================
# 6. 在籍月数の算出（全期間データから）
# ============================================================
logger.info("-" * 60)
logger.info("全期間データから在籍月数を算出中...")

在籍月数 = (月次データ_学習_全期間.sort_values("経過月数")
        .groupby("社員ID")["経過月数"].last()
        .reindex(学習データ["社員ID"]).to_numpy().astype(float))
logger.info(f"在籍月数: 平均={在籍月数.mean():.1f} 最小={在籍月数.min():.0f} 最大={在籍月数.max():.0f}")

生存者 = (月次データ_学習.groupby("社員ID").size()
       .reindex(学習データ["社員ID"]).to_numpy()) == 24
logger.info(f"24か月在籍: {生存者.sum()} / {len(生存者)} 人(予測データは全員24か月在籍)")

# ============================================================


# ============================================================
# 7. 強化Optuna探索（分類器のみ。3-fold・単一シードで高速化）
#    回帰器は xxxx_v4 の元の設定のまま変えない（一度に複数要因を変えない）
# ============================================================
logger.info("-" * 60)
logger.info("分類器のOptuna探索を開始（3-fold・単一シード）")

N_TRIALS_TUNE = 40      # xxxx_v4はtext_features処理があり54_より重いので150より少なめにする
K_FOLDS_TUNE = 3
MAX_ITER_TUNE = 1500

_skf_tune = StratifiedKFold(n_splits=K_FOLDS_TUNE, shuffle=True, random_state=SEED)


def _objective(trial):
    params = {
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "loss_function": "Logloss", "eval_metric": "Logloss",
        "iterations": MAX_ITER_TUNE, "random_seed": SEED, "verbose": False,
        "early_stopping_rounds": 50, "allow_writing_files": False,
    }
    oof = np.zeros(len(特徴量))
    for 学習行, 検証行 in _skf_tune.split(特徴量, 正解ラベル):
        学習プール = Pool(特徴量.iloc[学習行], 正解ラベル.iloc[学習行],
                     cat_features=カテゴリ列, text_features=text列)
        検証プール = Pool(特徴量.iloc[検証行], 正解ラベル.iloc[検証行],
                     cat_features=カテゴリ列, text_features=text列)
        m = CatBoostClassifier(**params)
        m.fit(学習プール, eval_set=検証プール)
        oof[検証行] = m.predict_proba(検証プール)[:, 1]
    return log_loss(正解ラベル, oof)


t0 = time.time()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(_objective, n_trials=N_TRIALS_TUNE, show_progress_bar=False)
logger.info(f"Optuna完了 ({time.time()-t0:.0f}秒): best_value={study.best_value:.6f}")
logger.info(f"best_params={study.best_params}")
print(f"best_value(3-fold OOF logloss) = {study.best_value:.6f}")
print(f"best_params = {study.best_params}")
print("参考: xxxx_v4の元の設定(depth=6, learning_rate=0.03、他はCatBoost既定値)")

TUNED_CLS_PARAMS = {
    **study.best_params,
    "loss_function": "Logloss", "eval_metric": "Logloss",
    "early_stopping_rounds": 100, "verbose": 0, "allow_writing_files": False,
}

import json
with open(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_tuned_params.json", "w", encoding="utf-8") as f:
    json.dump({"params": study.best_params, "cv_oof_logloss": study.best_value},
               f, ensure_ascii=False, indent=2)

# ============================================================
# 8. 最終学習（xxxx_v4と同一アーキテクチャ: 7シード×6-fold bagging）
#    分類器はチューニング済みパラメータ、回帰器は元の設定のまま
# ============================================================
logger.info("-" * 60)
logger.info("最終学習を開始（分類器=チューニング済み、回帰器=xxxx_v4の元の設定）")

回帰設定 = dict(
    iterations=3000, learning_rate=0.03, depth=6,
    loss_function="RMSE", eval_metric="RMSE",
    early_stopping_rounds=100, verbose=0, allow_writing_files=False,
)

シード一覧 = [42, 2024, 7, 1234, 99, 1111, 512]   # xxxx_v4と同一の7シード
fold数 = 6                                        # xxxx_v4と同一

予測プール = Pool(予測用特徴量, cat_features=カテゴリ列, text_features=text列)

検証予測_シード別, 提出予測_シード別 = [], []
検証回帰_シード別, 提出回帰_シード別 = [], []

for シード in シード一覧:
    分割器 = StratifiedKFold(n_splits=fold数, shuffle=True, random_state=シード)
    検証予測 = np.zeros(len(特徴量))
    提出予測 = np.zeros(len(予測用特徴量))
    検証回帰 = np.zeros(len(特徴量))
    提出回帰 = np.zeros(len(予測用特徴量))

    for 学習行, 検証行 in 分割器.split(特徴量, 正解ラベル):
        学習プール = Pool(特徴量.iloc[学習行], 正解ラベル.iloc[学習行],
                     cat_features=カテゴリ列, text_features=text列)
        検証プール = Pool(特徴量.iloc[検証行], 正解ラベル.iloc[検証行],
                     cat_features=カテゴリ列, text_features=text列)
        分類器 = CatBoostClassifier(random_seed=シード, **TUNED_CLS_PARAMS)
        分類器.fit(学習プール, eval_set=検証プール)
        検証予測[検証行] = 分類器.predict_proba(検証プール)[:, 1]
        提出予測 += 分類器.predict_proba(予測プール)[:, 1] / fold数

        学習プール回帰 = Pool(特徴量.iloc[学習行], 在籍月数[学習行],
                       cat_features=カテゴリ列, text_features=text列)
        検証プール回帰 = Pool(特徴量.iloc[検証行], 在籍月数[検証行],
                       cat_features=カテゴリ列, text_features=text列)
        回帰器 = CatBoostRegressor(random_seed=シード, **回帰設定)
        回帰器.fit(学習プール回帰, eval_set=検証プール回帰)
        検証回帰[検証行] = 回帰器.predict(検証プール回帰)
        提出回帰 += 回帰器.predict(予測プール) / fold数

    検証予測_シード別.append(検証予測)
    提出予測_シード別.append(提出予測)
    検証回帰_シード別.append(検証回帰)
    提出回帰_シード別.append(提出回帰)
    logger.info(f"シード{シード}: 分類器単体 logloss={log_loss(正解ラベル, 検証予測):.5f}")

検証予測_平均 = np.mean(検証予測_シード別, axis=0)
提出予測_平均 = np.mean(提出予測_シード別, axis=0)
検証回帰_平均 = np.mean(検証回帰_シード別, axis=0)
提出回帰_平均 = np.mean(提出回帰_シード別, axis=0)
logger.info(f"分類器 {len(シード一覧)}シード平均: {log_loss(正解ラベル, 検証予測_平均):.5f}")
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_classifier_only_testpreds.npy", 提出予測_平均)

# ============================================================
# 9. Platt変換と確率ブレンド（xxxx_v4と同一手順）
# ============================================================
logger.info("-" * 60)
logger.info("Platt変換による確率化とブレンドを実行中...")


def Platt変換(学習用スコア, 学習用ラベル, 適用先スコア):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(np.asarray(学習用スコア).reshape(-1, 1), 学習用ラベル)
    return lr.predict_proba(np.asarray(適用先スコア).reshape(-1, 1))[:, 1]


def ロジット(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


回帰確率_検証 = Platt変換(検証回帰_平均, 正解ラベル, 検証回帰_平均)
回帰確率_提出 = Platt変換(検証回帰_平均, 正解ラベル, 提出回帰_平均)
logger.info(f"回帰→確率 単体: {log_loss(正解ラベル, 回帰確率_検証):.5f}")
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_regression_platt_testpreds.npy", 回帰確率_提出)

混合比 = 0.85
ブレンド_検証 = 混合比 * 検証予測_平均 + (1 - 混合比) * 回帰確率_検証
ブレンド_提出 = 混合比 * 提出予測_平均 + (1 - 混合比) * 回帰確率_提出
logger.info(f"ブレンド後: {log_loss(正解ラベル, ブレンド_検証):.5f}")

z_検証 = ロジット(ブレンド_検証)
較正_検証 = np.zeros(len(z_検証))
for 学習行, 検証行 in KFold(n_splits=5, shuffle=True, random_state=27).split(z_検証):
    学習行_生存者 = 学習行[生存者[学習行]]
    較正_検証[検証行] = Platt変換(z_検証[学習行_生存者], 正解ラベル.iloc[学習行_生存者],
                          z_検証[検証行])
logger.info(f"最終Platt較正: {log_loss(正解ラベル, 較正_検証):.5f}")

最終較正器 = LogisticRegression(max_iter=1000)
最終較正器.fit(z_検証[生存者].reshape(-1, 1), 正解ラベル.iloc[生存者])
最終予測 = 最終較正器.predict_proba(ロジット(ブレンド_提出).reshape(-1, 1))[:, 1]
logger.info(f"提出予測の分布: min={最終予測.min():.4f} mean={最終予測.mean():.4f} max={最終予測.max():.4f}")
print(f"\n参考: xxxx_v4の元の最終Platt較正後 val = 0.47011（分類器チューニング前）")
print(f"本スクリプトの最終Platt較正後 val = {log_loss(正解ラベル, 較正_検証):.5f}")

# ============================================================
# 10. 提出ファイルの出力
# ============================================================
logger.info("-" * 60)
logger.info("提出ファイルを作成中...")

提出見本 = pd.read_csv(データ置き場 / "sample_submission.csv", header=None)
提出見本.columns = ["社員ID", "見本値"]

提出データ = 提出見本[["社員ID"]].merge(
    pd.DataFrame({"社員ID": 予測データ["社員ID"], "定着確率": 最終予測}),
    on="社員ID", how="left")
assert 提出データ["定着確率"].notna().all(), "IDが不一致"

output_path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_xxxx_tuned_submission.csv"
提出データ.to_csv(output_path, index=False, header=False)
np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_xxxx_tuned_testpreds.npy", 最終予測)

logger.info(f"提出ファイルを保存しました: {output_path}")
print(f"\n保存: {output_path}")
print(提出データ.head())
print(f"\n✅ [{SCRIPT_NAME}] 完了。このプロセスをここで終了する（メモリを解放する）。")
