"""105_logit_blend_and_stacking

コンペ残り2週間、現在10位（9位との差0.00004、1位との差0.0125）を受けて、ユーザー承認済みの
2方向を検証する。いずれも新規モデル学習を伴わない、既存の保存済みOOF/Test予測のみを使った
純ローカル・低コストの実験。

## タスクA: ログオッズ(logit)空間でのブレンド vs 確率空間でのブレンド

これまでの全ブレンド(72_/85_/92_/93_/94_/95_/98_等)は確率のまま加重平均していた。
複数の強いモデルを混ぜる場合、対数損失に対してはlogit空間（log(p/(1-p))で平均してから
sigmoidに戻す）の方が理論的に有利なことがある。同一fold割当(StratifiedKFold n_splits=5,
random_state=42)を持つ3ペア（84_TOP30PCTプール由来のCB134×TabPFN134(86_)、
84_TOP20PCTプール由来のCB89×TabPFN89(93_、現最良に統合済み)、xxxx_v4独自144列プール由来の
CB144×TabPFN144(97_/99_)）について、確率空間argminとlogit空間argminのOOF logloss を比較する。

## タスクB: 正則化+入れ子CV付きのスタッキング（メタモデル）

[[ensemble-oof-overfitting]]の「重み学習は過学習する」は、プロジェクト初期の生モデルに対する
素朴な重み学習（探索的に複数重みを試してOOFで最良を選ぶ）を指す。現在は同一fold割当を持つ
6つのOOF（CB89/TabPFN89/CB134/TabPFN134/CB144/TabPFN144）が揃っており、これらをlogit変換して
特徴量とし、L2正則化ロジスティック回帰のメタモデルを**入れ子CV**（外側5-foldで
メタモデル自体の汎化性能を測る、内側でCの選択にLogisticRegressionCVを使う二重の入れ子）
で評価する。単純な等重み平均・既存の事前登録済みargminブレンドと比較し、分解能±0.0043を
超えて安定的に上回る場合のみ提出候補とする。

出力（タスクAは提出ファイルなし、ログのみ）:
  タスクB: stacker_full_survivor_submission.csv（過学習の兆候が無い場合のみ生成、
  入れ子CVでの評価が事前登録済みの唯一の判断材料——単一の重み探索結果を見て決めない）
"""
import datetime
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
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

SCRIPT_NAME = "105_logit_blend_and_stacking"
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

y_train = train_persona[TARGET_COL].values
train_ids = train_persona[ID_COL].values
test_ids = test_persona[ID_COL].values

EARLY_LEAVER_IDS = set(train_monthly.loc[train_monthly["月末在籍状態"] == "退職", ID_COL].unique())
surv_mask = np.array([tid not in EARLY_LEAVER_IDS for tid in train_ids])
logger.info(f"生存者(24か月在籍): {surv_mask.sum()} / {len(surv_mask)}")

EPS = 1e-9


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ============================================================
# 保存済みOOF/Test予測の読み込み(全て同一fold割当: StratifiedKFold n_splits=5, seed=42)
# ============================================================
OOF_DIR = PROJECT_ROOT / "data" / "output"
BASE_MODELS = {
    "cb134": ("20260822/20260822_86_tabpfn_on_84_top30pct_cb134_oofpreds.npy",
              "20260822/20260822_86_tabpfn_on_84_top30pct_cb134_testpreds.npy"),
    "tabpfn134": ("20260822/20260822_86_tabpfn_on_84_top30pct_tabpfn134_oofpreds.npy",
                  "20260822/20260822_86_tabpfn_on_84_top30pct_tabpfn134_testpreds.npy"),
    "cb89": ("20260822/20260822_93_tabpfn_top20pct_on_77_cb89_oofpreds.npy",
             "20260822/20260822_93_tabpfn_top20pct_on_77_cb89_testpreds.npy"),
    "tabpfn89": ("20260822/20260822_93_tabpfn_top20pct_on_77_tabpfn89_oofpreds.npy",
                 "20260822/20260822_93_tabpfn_top20pct_on_77_tabpfn89_testpreds.npy"),
    "cb144": ("20260824/20260824_97_ft_transformer_on_xxxxv4_features_cb_oofpreds.npy",
              "20260824/20260824_97_ft_transformer_on_xxxxv4_features_cb_testpreds.npy"),
    "tabpfn144": ("20260824/20260824_99_tabpfn_on_xxxxv4_features_tabpfn_oofpreds.npy",
                  "20260824/20260824_99_tabpfn_on_xxxxv4_features_tabpfn_testpreds.npy"),
}

oof = {}
test = {}
for name, (oof_rel, test_rel) in BASE_MODELS.items():
    oof[name] = np.load(OOF_DIR / oof_rel)
    test[name] = np.load(OOF_DIR / test_rel)
    assert len(oof[name]) == len(train_ids)
    assert len(test[name]) == len(test_ids)
    val = log_loss(y_train[surv_mask], oof[name][surv_mask])
    logger.info(f"[単体val] {name}: {val:.6f}")

# ============================================================
# タスクA: logit空間 vs 確率空間のブレンドargmin比較（3ペア）
# ============================================================
logger.info("=" * 60)
logger.info("[タスクA] logit空間 vs 確率空間のブレンドargmin比較")
BLEND_WEIGHTS = np.linspace(0.0, 1.0, 21)
PAIRS = [("cb134", "tabpfn134"), ("cb89", "tabpfn89"), ("cb144", "tabpfn144")]

for cb_name, other_name in PAIRS:
    y_s = y_train[surv_mask]
    p_cb = oof[cb_name][surv_mask]
    p_other = oof[other_name][surv_mask]
    z_cb = logit(p_cb)
    z_other = logit(p_other)

    best_prob = min(
        ((w, log_loss(y_s, np.clip(w * p_cb + (1 - w) * p_other, EPS, 1 - EPS))) for w in BLEND_WEIGHTS),
        key=lambda r: r[1],
    )
    best_logit = min(
        ((w, log_loss(y_s, sigmoid(w * z_cb + (1 - w) * z_other))) for w in BLEND_WEIGHTS),
        key=lambda r: r[1],
    )
    logger.info(f"  [{cb_name} x {other_name}] 確率空間argmin: w_cb={best_prob[0]:.2f}, val={best_prob[1]:.6f} / "
                f"logit空間argmin: w_cb={best_logit[0]:.2f}, val={best_logit[1]:.6f} "
                f"(差 {best_logit[1] - best_prob[1]:+.6f}, 分解能±0.0043)")

# ============================================================
# タスクB: logit変換した6モデルOOFでL2ロジスティック回帰スタッキング（入れ子CV）
# ============================================================
logger.info("=" * 60)
logger.info("[タスクB] logit変換6モデルでのスタッキング(入れ子CV)")
STACK_MODELS = ["cb134", "tabpfn134", "cb89", "tabpfn89", "cb144", "tabpfn144"]

surv_idx = np.where(surv_mask)[0]
y_s = y_train[surv_idx]
X_s = np.column_stack([logit(oof[m][surv_idx]) for m in STACK_MODELS])
logger.info(f"  スタッキング用行列: {X_s.shape} (行=生存者, 列={STACK_MODELS})")

# 等重み平均(logit空間)をベースラインとして比較
equal_weight_pred = sigmoid(X_s.mean(axis=1))
equal_weight_val = log_loss(y_s, equal_weight_pred)
logger.info(f"  [ベースライン] logit空間等重み平均: val={equal_weight_val:.6f}")

# 事前登録済みargminブレンド(93_のcb89xtabpfn89, w_cb=0.2)を参考値として記録
cb89_s, tabpfn89_s = oof["cb89"][surv_idx], oof["tabpfn89"][surv_idx]
ref_blend_val = log_loss(y_s, np.clip(0.2 * cb89_s + 0.8 * tabpfn89_s, EPS, 1 - EPS))
logger.info(f"  [参考] 93_の事前登録済みargminブレンド(cb89×0.2+tabpfn89×0.8): val={ref_blend_val:.6f}")

# 外側5-fold(93_等とは別のrandom_stateで、メタモデル自身の入れ子CVとして独立させる)
OUTER_SEED = 12345
outer_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
meta_oof = np.zeros(len(y_s))
fold_coefs = []

for fold_i, (tr_idx, va_idx) in enumerate(outer_skf.split(X_s, y_s)):
    # 内側CVでCを選択(LogisticRegressionCV)、外側foldの検証データには一切触れない
    clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 15), cv=5, penalty="l2", scoring="neg_log_loss",
        max_iter=5000, random_state=SEED,
    )
    clf.fit(X_s[tr_idx], y_s[tr_idx])
    meta_oof[va_idx] = clf.predict_proba(X_s[va_idx])[:, 1]
    fold_coefs.append(clf.coef_[0])
    logger.info(f"    [外側fold{fold_i}] 選択されたC={clf.C_[0]:.4f}, 係数={dict(zip(STACK_MODELS, clf.coef_[0].round(3)))}")

stack_val = log_loss(y_s, meta_oof)
logger.info(f"  [スタッキング(入れ子CV)] val={stack_val:.6f}")
logger.info(f"  対等重み平均: {stack_val - equal_weight_val:+.6f} / 対93_参考ブレンド: {stack_val - ref_blend_val:+.6f}")
logger.info(f"  (分解能±0.0043を超えて改善しているかを判断基準とする。超えていなければ採用しない)")

mean_coefs = np.mean(fold_coefs, axis=0)
logger.info(f"  fold平均係数: {dict(zip(STACK_MODELS, mean_coefs.round(3)))}")

# ============================================================
# 判定: 入れ子CVでの改善が分解能を超えた場合のみ、全データで再学習してTest予測を生成
# ============================================================
RESOLUTION = 0.0043
improvement_vs_equal = equal_weight_val - stack_val
improvement_vs_ref = ref_blend_val - stack_val
logger.info("=" * 60)
if improvement_vs_equal > RESOLUTION and improvement_vs_ref > RESOLUTION:
    logger.info(f"[判定] 分解能を超えて改善(対等重み{improvement_vs_equal:+.6f} / 対参考ブレンド{improvement_vs_ref:+.6f})"
                f"——全データで最終スタッカーを学習しTest予測を生成する")
    final_clf = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 15), cv=5, penalty="l2", scoring="neg_log_loss",
        max_iter=5000, random_state=SEED,
    )
    final_clf.fit(X_s, y_s)
    logger.info(f"  最終係数: {dict(zip(STACK_MODELS, final_clf.coef_[0].round(3)))} / C={final_clf.C_[0]:.4f}")

    X_test = np.column_stack([logit(test[m]) for m in STACK_MODELS])
    test_pred = final_clf.predict_proba(X_test)[:, 1]
    path = OUTPUT_DIR / f"{TODAY}_{SCRIPT_NAME}_stacker_full_survivor_submission.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_pred}).to_csv(path, index=False, header=False)
    logger.info(f"  提出ファイル保存: {path.name}（予測平均={test_pred.mean():.4f}）")
else:
    logger.info(f"[判定] 分解能未満(対等重み{improvement_vs_equal:+.6f} / 対参考ブレンド{improvement_vs_ref:+.6f})"
                f"——スタッキングは不採用。事前登録済みの入れ子CV基準に従い、Test予測は生成しない")

logger.info("=" * 60)
logger.info(f"=== [{SCRIPT_NAME}] 実験終了 ===")
