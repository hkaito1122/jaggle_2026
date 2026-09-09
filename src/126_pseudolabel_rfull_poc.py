"""126_pseudolabel_rfull_poc

ユーザー承認の「候補1→候補2の順で実施」のうち候補2。`123_`(CB89)・`125_`(lgbm134)に
続く3つ目の疑似ラベリング対象として、`72_reference_pipeline_standalone`のR_full
(115_の8モデルスタックの中で全fold一貫して最大の係数(約+0.8)を持つ、事実上の主力モデル)
を選ぶ。R_fullは84_/93_/114_系(447列パイプライン+CatBoost重要度によるnested選択)とは
完全に独立した別の特徴量パイプライン(janome形態素解析+CatBoostネイティブtext_features、
depth=6・iterations=3000)であり、既存モデルとの相関が比較的低い(0.75〜0.81)。もし
自己学習の増分がスタックに乗るとすれば、係数が最大のモデルへの適用が最も直接的に
効くはずという仮説を検証する。

R_fullのパイプラインは72_の特徴量構築コード(月次集約・入社時特徴量・janomeメモ解析・
勤務地/転居/キャリア志向パース・カテゴリ列/text列の確定)をそのまま移植する。84_/93_/114_
と違い、部署ターゲットエンコーディングなどfold依存の処理が一切無いため特徴量は
Train全件・Test全件で一度だけ構築すればよく、per-foldでのbuild_features()呼び直しは
不要——123_/125_より設計がシンプルになる。

72_のオリジナルは3シード([42,2024,7])×5-foldの平均だが、本POCでは9段階の閾値スキャンの
コストを抑えるため単一seed(42)の5-fold StratifiedKFold OOFのみで比較する
(ベースラインとの比較は同一seed・同一fold割当で公正に行うため問題ない)。学習設定も
72_と同一(iterations=3000, depth=6, early_stopping_rounds=100)——fold実際の検証データを
CatBoostのeval_setとして早期打ち切りに使う設計自体が72_(reference再現)のオリジナル
仕様であり、疑似ラベリングはこれに追加の学習データ(Test由来の高信頼度行)を訓練側に
足すだけで、早期打ち切りの検証データには一切触れない。

判定基準: [[validation-asymmetry]]で確立した分解能±0.0043を正規にクリアした場合のみ
「疑似ラベリングは有効」と判断する。123_/125_と同じ2段階検証フロー
(単体で分解能クリア→115_スタックのr_fullを置換して入れ子CVで再評価)。

出力: 分解能を超えて改善した閾値が見つかった場合のみ
rfull_pseudo_best_oofpreds.npy / rfull_pseudo_best_testpreds.npy を保存。
"""
import datetime, json, re, sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
from janome.tokenizer import Tokenizer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from common.utils.logger import get_logger
from common.utils.seed import seed_everything

warnings.filterwarnings("ignore")
SEED = 42; seed_everything(seed=SEED)
TARGET_COL, ID_COL = "10年定着ラベル", "社員ID"
pd.set_option("display.max_columns", None)

SCRIPT_NAME = "126_pseudolabel_rfull_poc"
TODAY = datetime.datetime.now().strftime("%Y%m%d")
logger = get_logger(SCRIPT_NAME, log_dir=str(PROJECT_ROOT/"logs"))
OUTPUT_DIR = PROJECT_ROOT/"data"/"output"/TODAY; OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"=== [{SCRIPT_NAME}] 開始 ===")

INPUT = PROJECT_ROOT/"data"/"input"
入社時データ_学習 = pd.read_csv(INPUT/"employee_persona_train.csv")
入社時データ_予測 = pd.read_csv(INPUT/"employee_persona_test.csv")
月次データ_学習   = pd.read_csv(INPUT/"employee_monthly_train.csv")
月次データ_予測   = pd.read_csv(INPUT/"employee_monthly_test.csv")
print(入社時データ_学習.shape, 入社時データ_予測.shape)
def 学習時間の合計(text):
    if pd.isna(text) or text == "受講なし": return 0.0
    return sum(float(x) for x in re.findall(r"([\d.]+)時間", text))

def 学習テーマの一覧(text):
    if pd.isna(text) or text == "受講なし": return []
    return [n.strip().replace(" ", "") for n in re.findall(r"([^｜：]+)：[\d.]+時間", text)]

def 傾き(値の列):
    有効 = ~np.isnan(値の列)
    if 有効.sum() < 2: return 0.0
    return np.polyfit(np.arange(len(値の列))[有効], 値の列[有効], 1)[0]

等級の数値 = {"G1":1,"G2":2,"G3":3,"G4":4,"G5":5}
役割の数値 = {"メンバー":1,"シニア":2,"エキスパート":3,"リード":4,"シニアエキスパート":4,"マネージャー":5}
行動列 = ["残業時間","有給取得日数","欠勤日数","研修時間","上司との面談実施回数","情報共有件数","在宅勤務日数"]
評価列 = ["360度評価_親和度","360度評価_信頼度","360度評価_主体度","360度評価_学習度","360度評価_共有貢献度"]
評価系列 = [*評価列, "顧客満足度評価","担当プロジェクト数","360度評価者数"]

# 生存指示子（Trainのみ非定数。第4節で分離して検証する）
生存指示子 = ["観測月数", "退職済み", "最終月に不在籍"]

def 月次集約(月次データ):
    月次データ = 月次データ.sort_values(["社員ID","経過月数"]).copy()
    月次データ["学習時間"]   = 月次データ["自己学習（詳細）"].apply(学習時間の合計)
    月次データ["学習テーマ"] = 月次データ["自己学習（詳細）"].apply(学習テーマの一覧)
    月次データ["等級_num"]  = 月次データ["等級"].map(等級の数値)
    月次データ["役割_num"]  = 月次データ["役割"].map(役割の数値)
    行一覧=[]
    for 社員, g in 月次データ.groupby("社員ID", sort=False):
        g = g.reset_index(drop=True); 初月, 最終月 = g.iloc[0], g.iloc[-1]
        行 = {"社員ID":社員,
              "観測月数":len(g),
              "最終月に不在籍":int(最終月["月末在籍状態"]!="在籍"),
              "休職あり":int((g["月末在籍状態"]=="休職").any()),
              "退職済み":int((g["月末在籍状態"]=="退職").any()),
              "最終月の勤務地":最終月["勤務地"], "最終月の職種":最終月["職種"],
              "最終月の部署ID":最終月["部署ID"],
              "部署異動回数":g["部署ID"].nunique()-1, "上司交代回数":g["上司ID"].nunique()-1,
              "役割変化回数":g["役割"].nunique()-1,
              "初月の等級":初月["等級_num"], "最終月の等級":最終月["等級_num"],
              "昇級あり":int(最終月["等級_num"]>初月["等級_num"]),
              "初月の役割":初月["役割_num"], "最終月の役割":最終月["役割_num"],
              "初月給与":初月["月例給与_円"], "最終月給与":最終月["月例給与_円"],
              "給与の伸び率":(最終月["月例給与_円"]-初月["月例給与_円"])/初月["月例給与_円"],
              "学習時間合計":g["学習時間"].sum()}
        テーマ=[t for 一覧 in g["学習テーマ"] for t in 一覧]
        行["学習txt"]="  ".join(テーマ) if テーマ else "なし"
        行["学習テーマ数"]=len(set(テーマ)); 行["学習月数"]=int((g["学習時間"]>0).sum())
        for 列 in 行動列:
            行[f"{列}_平均"]=g[列].mean(); 行[f"{列}_標準偏差"]=g[列].std(); 行[f"{列}_最終月"]=g[列].iloc[-1]
        行["残業時間_傾き"]=傾き(g["残業時間"].to_numpy(dtype=float))
        for 列 in 評価系列:
            行[f"{列}_平均"]=g[列].mean()
            行[f"{列}_直近"]=g[列].dropna().iloc[-1] if g[列].notna().any() else np.nan
            行[f"{列}_欠損率"]=g[列].isna().mean()
        行["360評価_傾き"]=傾き(g[評価列].mean(axis=1).to_numpy(dtype=float))
        行["360評価_更新回数"]=g["360度評価更新フラグ"].sum()
        行["360評価_更新率"]=g["360度評価更新フラグ"].mean()
        前半, 後半 = g[g["経過月数"]<12], g[g["経過月数"]>=12]
        for 列 in ["残業時間","有給取得日数","欠勤日数","情報共有件数","研修時間","月例給与_円"]:
            行[f"{列}_後半前半差"]=後半[列].mean()-前半[列].mean()
        行["360評価_後半前半差"]=(後半[評価列].mean(axis=1).mean()-前半[評価列].mean(axis=1).mean())
        行一覧.append(行)
    return pd.DataFrame(行一覧)

t0=time.time()
月次集約_学習 = 月次集約(月次データ_学習); 月次集約_予測 = 月次集約(月次データ_予測)
print(月次集約_学習.shape, 月次集約_予測.shape, f"({time.time()-t0:.0f}秒)")
テキスト列 = ["入社時メモ","上司からのフィードバック","同僚からのフィードバック"]

def 入社時特徴量(入社時データ):
    d = 入社時データ.copy()
    d["入社年"] = pd.to_datetime(d["入社日"]).dt.year
    d["前職職種"] = d["前職職種"].fillna("前職なし")
    d["初期等級_num"] = d["初期等級"].map(等級の数値)
    d["初期役割_num"] = d["初期役割"].map(役割の数値)
    for 列 in テキスト列: d[f"{列}_文字数"] = d[列].str.len()
    return d

入社時_学習 = 入社時特徴量(入社時データ_学習).reset_index(drop=True)
入社時_予測 = 入社時特徴量(入社時データ_予測).reset_index(drop=True)

初期部署の頻度 = pd.concat([入社時_学習["初期部署ID"], 入社時_予測["初期部署ID"]]).value_counts()
入社時_学習["初期部署ID_頻度"] = 入社時_学習["初期部署ID"].map(初期部署の頻度)
入社時_予測["初期部署ID_頻度"] = 入社時_予測["初期部署ID"].map(初期部署の頻度)
最終部署の頻度 = pd.concat([月次集約_学習["最終月の部署ID"], 月次集約_予測["最終月の部署ID"]]).value_counts()
月次集約_学習["最終月の部署ID_頻度"] = 月次集約_学習["最終月の部署ID"].map(最終部署の頻度)
月次集約_予測["最終月の部署ID_頻度"] = 月次集約_予測["最終月の部署ID"].map(最終部署の頻度)

形態素解析器 = Tokenizer(); 残す品詞 = {"名詞","動詞","形容詞"}
def 文書分割(text):
    if pd.isna(text): return ""
    return " ".join(語.base_form for 語 in 形態素解析器.tokenize(text)
                    if 語.part_of_speech.split(",")[0] in 残す品詞)
t0=time.time()
入社時_学習["メモtxt"] = 入社時_学習["入社時メモ"].apply(文書分割)
入社時_予測["メモtxt"] = 入社時_予測["入社時メモ"].apply(文書分割)
print(f"janome分割 完了 ({time.time()-t0:.0f}秒)"); print(入社時_学習["メモtxt"].iloc[0][:90])
勤務地一覧 = ["東京","大阪","愛知","福岡","仙台","北海道","その他"]
転居の否定 = r"許容せず|許容しておらず|許容していない|許容しない|希望しておらず|希望せず|希望していない"
在宅の否定 = r"必須条件と(?:は)?していない|不要"

def メモをパース(メモ):
    メモ = メモ.fillna("")
    働き方の節 = メモ.str.extract(r"・勤務地・働き方：(.+?)(?:\n|$)")[0].fillna("")
    志向の節   = メモ.str.extract(r"・キャリア志向：(.+?)(?:\n|$)")[0].fillna("")
    結果 = pd.DataFrame(index=メモ.index)
    def 最初の勤務地(s):
        候補, 位置 = "不明", 10**9
        for 勤務地 in 勤務地一覧:
            i = s.find(勤務地)
            if 0 <= i < 位置: 候補, 位置 = 勤務地, i
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
        if not s.strip(): return "不明"
        if ("限定していない" in s or "方向を限定" in s or "限定せず" in s or "特定していない" in s
            or "方向は特定" in s or "限定した志向は確認されていない" in s): return "限定なし"
        for k in ("専門職","管理職","安定"):
            if k in s: return k
        return "不明"
    結果["キャリア志向"] = 志向の節.map(志向の分類)
    return 結果

for データ in (入社時_学習, 入社時_予測):
    p = メモをパース(データ["入社時メモ"])
    for 列 in ["転居許容","在宅希望","希望勤務地","キャリア志向"]: データ[列] = p[列].to_numpy()
for 列 in ["転居許容","在宅希望","キャリア志向"]:
    print(列, dict(pd.concat([入社時_学習[列], 入社時_予測[列]]).value_counts()))
学習データ = 入社時_学習.merge(月次集約_学習, on=ID_COL, how="left")
予測データ = 入社時_予測.merge(月次集約_予測, on=ID_COL, how="left")

カテゴリ列 = ["入社区分","最終学歴","専攻分野","前職職種","採用経路","性別","初期職種","初期勤務地",
          "初期等級","初期役割","最終月の勤務地","最終月の職種","最終月の部署ID","希望勤務地","キャリア志向"]
text列 = ["メモtxt","学習txt"]
除外列 = [ID_COL, "入社日", "初期部署ID", *テキスト列, TARGET_COL]

正解ラベル = 学習データ[TARGET_COL].astype(int)
特徴量     = 学習データ.drop(columns=[c for c in 除外列 if c in 学習データ.columns])
予測用特徴量 = 予測データ.drop(columns=[c for c in 除外列 if c in 予測データ.columns])
for 列 in カテゴリ列:
    特徴量[列] = 特徴量[列].astype(str).fillna("missing")
    予測用特徴量[列] = 予測用特徴量[列].astype(str).fillna("missing")
特徴量 = 特徴量[予測用特徴量.columns]      # 列順を揃える
logger.info(f"特徴量 {特徴量.shape[1]} 列（カテゴリ{len(カテゴリ列)}・テキスト{len(text列)}）")
assert list(特徴量.columns) == list(予測用特徴量.columns)

EARLY_LEAVER_IDS = set(月次データ_学習.loc[月次データ_学習["月末在籍状態"] == "退職", ID_COL].unique())
train_ids = 入社時データ_学習[ID_COL].values
test_ids = 入社時データ_予測[ID_COL].values
surv_mask = np.array([tid not in EARLY_LEAVER_IDS for tid in train_ids])
logger.info(f"生存者(24か月在籍): {surv_mask.sum()} / {len(surv_mask)}")

学習設定 = dict(iterations=3000, learning_rate=0.03, depth=6,
              loss_function="Logloss", eval_metric="Logloss",
              early_stopping_rounds=100, verbose=0, allow_writing_files=False)
OOF_SEED = 42  # 72_オリジナルは[42,2024,7]の3シード平均だが、閾値スキャンのコスト抑制のため単一seedで比較する
OOF_N_SPLITS = 5
THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# ============================================================
# 疑似ラベル源: 115_の現Public最良提出(8モデルスタッキング、0.493891)のTest予測
# ============================================================
logger.info("=" * 60)
BEST_SUBMISSION_PATH = (
    PROJECT_ROOT / "data" / "output" / "20260905"
    / "20260905_115_stacking_with_lightgbm_stacker8_lgbm_full_survivor_submission.csv"
)
best_sub_df = pd.read_csv(BEST_SUBMISSION_PATH, header=None, names=[ID_COL, "pred"])
best_sub = best_sub_df.set_index(ID_COL).loc[test_ids, "pred"].values
assert len(best_sub) == len(test_ids)
logger.info(f"疑似ラベル源(115_ Test予測): shape={best_sub.shape}, "
            f"mean={best_sub.mean():.4f}, min={best_sub.min():.4f}, max={best_sub.max():.4f}")
for t in THRESHOLDS:
    n_low = (best_sub < t).sum()
    n_high = (best_sub > 1 - t).sum()
    logger.info(f"  閾値t={t:.2f}: 低信頼={n_low}行(pred<{t}) / 高信頼={n_high}行(pred>{1-t:.2f}) "
                f"/ 合計{n_low + n_high}行(全{len(test_ids)}行中)")

# ============================================================
# 72_の「(a) reference式」と同一設計(全2761名StratifiedKFold OOF)を単一seedで再現し、
# 各fold・各閾値で「疑似ラベル追加あり」vs「追加なし(素のR_full)」を同一条件で比較する。
# 84_/93_/114_と異なり部署ターゲットエンコーディング等fold依存の処理が無いため、
# 特徴量は上で構築した全件版(特徴量/予測用特徴量)をそのまま使い回せる。
# ============================================================
logger.info("=" * 60)
logger.info("[R_full vs +疑似ラベル] StratifiedKFold(単一seed)+閾値スキャンのOOFを構築中...")
skf = StratifiedKFold(n_splits=OOF_N_SPLITS, shuffle=True, random_state=OOF_SEED)
rfull_baseline_oof = np.zeros(len(特徴量))
rfull_pseudo_oof = {t: np.zeros(len(特徴量)) for t in THRESHOLDS}

for fold_i, (tr_idx, va_idx) in enumerate(skf.split(特徴量, 正解ラベル)):
    X_tr, y_tr = 特徴量.iloc[tr_idx], 正解ラベル.iloc[tr_idx]
    X_va, y_va = 特徴量.iloc[va_idx], 正解ラベル.iloc[va_idx]

    # ベースライン: 疑似ラベルなしの素のR_full(72_と同一、val foldをeval_setに使う設計も踏襲)
    m_baseline = CatBoostClassifier(random_seed=OOF_SEED, **学習設定)
    m_baseline.fit(Pool(X_tr, y_tr, cat_features=カテゴリ列, text_features=text列),
                    eval_set=Pool(X_va, y_va, cat_features=カテゴリ列, text_features=text列))
    rfull_baseline_oof[va_idx] = m_baseline.predict_proba(
        Pool(X_va, cat_features=カテゴリ列, text_features=text列))[:, 1]

    for t in THRESHOLDS:
        pseudo_mask = (best_sub < t) | (best_sub > 1 - t)
        pseudo_X = 予測用特徴量[pseudo_mask]
        pseudo_y = pd.Series((best_sub[pseudo_mask] > 0.5).astype(int), index=pseudo_X.index)
        X_aug = pd.concat([X_tr, pseudo_X], axis=0)
        y_aug = pd.concat([y_tr, pseudo_y], axis=0)
        m_pseudo = CatBoostClassifier(random_seed=OOF_SEED, **学習設定)
        m_pseudo.fit(Pool(X_aug, y_aug, cat_features=カテゴリ列, text_features=text列),
                      eval_set=Pool(X_va, y_va, cat_features=カテゴリ列, text_features=text列))
        rfull_pseudo_oof[t][va_idx] = m_pseudo.predict_proba(
            Pool(X_va, cat_features=カテゴリ列, text_features=text列))[:, 1]

    logger.info(f"    [fold{fold_i}] 完了")

rfull_baseline_score = log_loss(正解ラベル.values[surv_mask], rfull_baseline_oof[surv_mask])
logger.info(f"[R_fullベースライン(疑似ラベルなし、単一seed)] val(OOF, 生存者のみ)={rfull_baseline_score:.6f} "
            f"(72_の3シード平均参考値0.503144と近い値になるはず、単一seedのためズレはあり得る)")

logger.info("=" * 60)
logger.info("[閾値スキャン結果]")
results = {}
for t in THRESHOLDS:
    score = log_loss(正解ラベル.values[surv_mask], rfull_pseudo_oof[t][surv_mask])
    diff = rfull_baseline_score - score
    results[t] = (score, diff)
    logger.info(f"  閾値t={t:.2f}: val={score:.6f} (対ベースライン{diff:+.6f})")

RESOLUTION = 0.0043
best_t, (best_score, best_diff) = max(results.items(), key=lambda kv: kv[1][1])
logger.info("=" * 60)
logger.info(f"[最良閾値] t={best_t:.2f}: val={best_score:.6f} (対ベースライン{best_diff:+.6f})")

# ============================================================
# 判定: 分解能を超えて改善した場合のみ、最良閾値でOOF/Test予測を保存
# (72_と同じくTrain全件+疑似ラベルで最終モデルを学習、eval_setは無し(全件学習のため))
# ============================================================
if best_diff > RESOLUTION:
    logger.info(f"[判定] 分解能を超えて改善({best_diff:+.6f} > {RESOLUTION})——"
                f"疑似ラベリング(t={best_t:.2f})は有効。全データで最終R_full_pseudoを学習しOOF/Test予測を保存する")
    pseudo_mask_final = (best_sub < best_t) | (best_sub > 1 - best_t)
    pseudo_X_final = 予測用特徴量[pseudo_mask_final]
    pseudo_y_final = pd.Series((best_sub[pseudo_mask_final] > 0.5).astype(int), index=pseudo_X_final.index)
    X_aug_final = pd.concat([特徴量, pseudo_X_final], axis=0)
    y_aug_final = pd.concat([正解ラベル, pseudo_y_final], axis=0)
    # 全件学習のため内部early stopping用にランダムな最終年少数を切り出す(72_の全件提出時と同じ発想:
    # iterations固定ではなくeval_setで打ち切るため、ここでは学習データの一部を内部検証に回す)
    rng = np.random.RandomState(OOF_SEED)
    n_aug = len(y_aug_final)
    perm = rng.permutation(n_aug)
    n_val_final = max(1, int(n_aug * 0.1))
    val_pos, fit_pos = perm[:n_val_final], perm[n_val_final:]
    X_fit_final, X_val_final = X_aug_final.iloc[fit_pos], X_aug_final.iloc[val_pos]
    y_fit_final, y_val_final = y_aug_final.iloc[fit_pos], y_aug_final.iloc[val_pos]

    m_final = CatBoostClassifier(random_seed=OOF_SEED, **学習設定)
    m_final.fit(Pool(X_fit_final, y_fit_final, cat_features=カテゴリ列, text_features=text列),
                eval_set=Pool(X_val_final, y_val_final, cat_features=カテゴリ列, text_features=text列))
    test_pred_final = m_final.predict_proba(Pool(予測用特徴量, cat_features=カテゴリ列, text_features=text列))[:, 1]

    np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_rfull_pseudo_best_oofpreds.npy", rfull_pseudo_oof[best_t])
    np.save(OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_rfull_pseudo_best_testpreds.npy", test_pred_final)
    logger.info("  OOF/Test予測(.npy)を保存しました(後日スタッキングへの組み込み候補)")
else:
    logger.info(f"[判定] 分解能未満(最良でも{best_diff:+.6f} <= {RESOLUTION})——"
                f"疑似ラベリングは不採用。R_fullベースライン(72_)を維持する")

logger.info("=" * 60)
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
