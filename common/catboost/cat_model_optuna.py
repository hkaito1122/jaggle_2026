# common/catboost/cat_model_optuna.py
import copy
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_cat_optuna(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    X_train = data.get("X_train")
    y_train = data.get("y_train")

    if X_train is None or y_train is None:
        raise ValueError(
            "data には 'X_train' と 'y_train' が含まれている必要があります。"
        )

    # 1. 制御用パラメータの取り出し
    base_params = copy.deepcopy(params)
    n_trials = base_params.pop("n_trials", 20)
    n_splits = base_params.pop("n_splits", 5)
    seed = base_params.pop("seed", 42)
    stopping_rounds = base_params.pop("early_stopping_rounds", 50)
    direction = base_params.pop("direction", "minimize")

    # CatBoostに直接渡さない非モデル引数を掃除
    base_params.pop("save_dir", None)
    base_params.pop("verbose", None)

    # タスク判定（分類 vs 回帰）
    is_classification = base_params.get("loss_function", "").lower() in [
        "logloss",
        "cross_entropy",
    ] or base_params.get("eval_metric", "").lower() in [
        "logloss",
        "auc",
        "accuracy",
    ]

    # 2. カテゴリ変数の処理
    cat_cols = list(
        X_train.select_dtypes(include=["object", "category"]).columns
    )
    X_train_proc = X_train.copy()
    for col in cat_cols:
        X_train_proc[col] = X_train_proc[col].astype(str)

    # 3. Optuna 目的関数
    def objective(trial: optuna.Trial) -> float:
        # common/catboost/cat_model_optuna.py 内の trial_params 提案
        trial_params = {
            # 学習率と木の構造
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            # 正則化と構造制御
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-9, 10.0, log=True
            ),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            # 特徴量の量子化（数値特徴量の分割数）
            "border_count": trial.suggest_int("border_count", 32, 255),
        }

        current_params = base_params.copy()
        current_params.update(trial_params)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(X_train_proc))

        for train_idx, val_idx in kf.split(X_train_proc, y_train):
            X_tr, y_tr = X_train_proc.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train_proc.iloc[val_idx], y_train.iloc[val_idx]

            train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
            val_pool = Pool(X_va, y_va, cat_features=cat_cols)

            if is_classification:
                model = CatBoostClassifier(
                    **current_params,
                    random_seed=seed,
                    early_stopping_rounds=stopping_rounds,
                    verbose=False,
                )
                model.fit(train_pool, eval_set=val_pool)
                oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
            else:
                model = CatBoostRegressor(
                    **current_params,
                    random_seed=seed,
                    early_stopping_rounds=stopping_rounds,
                    verbose=False,
                )
                model.fit(train_pool, eval_set=val_pool)
                oof_preds[val_idx] = model.predict(val_pool)

        if is_classification:
            score = log_loss(y_train, oof_preds)
        else:
            score = np.sqrt(mean_squared_error(y_train, oof_preds))

        return score

    # 4. チューニング実行
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # 5. 最良パラメータ返却
    best_params = copy.deepcopy(params)
    best_params.update(study.best_params)

    result_data = {
        "best_score": study.best_value,
        "study": study,
    }

    return result_data, best_params