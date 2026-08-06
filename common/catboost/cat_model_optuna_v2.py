import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _preprocess_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """object 型の列を CatBoost 用に文字列/カテゴリ型へ変換し、カテゴリ列名リストを返す内部関数"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype(str)
    return df, cat_cols


def run_cat_optuna(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Optunaを用いてCatBoostのハイパーパラメータ探索を交差検証（CV）ベースで実行する関数（v2: TimeSeriesSplit対応版）"""
    X_train = data.get("X_train")
    y_train = data.get("y_train")
    sort_col = data.get("sort_col", None)

    if X_train is None or y_train is None:
        raise ValueError(
            "data には 'X_train' と 'y_train' が含まれている必要があります。"
        )

    # 1. 制御用パラメータの取り出し
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits", 5)
    seed = params_exec.pop("seed", 42)
    n_trials = params_exec.pop("n_trials", 20)
    early_stopping_rounds = params_exec.pop("early_stopping_rounds", 50)
    cv_strategy = params_exec.pop("cv_strategy", "stratified")
    
    params_exec.pop("save_dir", None)
    params_exec.pop("verbose", None)

    # 二元分類タスクかの自動判定
    loss_function = params_exec.get(
        "loss_function", params_exec.get("eval_metric", "Logloss")
    )
    is_binary = loss_function.lower() in ["logloss", "binary"]

    # 2. カテゴリ変数の前処理とリスト取得
    X_train_proc, cat_cols = _preprocess_categorical(X_train)

    if cv_strategy == "timeseries":
        if sort_col is None:
            raise ValueError(
                "TimeSeriesSplit を使用する場合は、data に 'sort_col' を指定してください。"
            )
        sort_indices = X_train_proc[sort_col].argsort()
        X_train_proc = X_train_proc.iloc[sort_indices].reset_index(drop=True)
        y_train = y_train.iloc[sort_indices].reset_index(drop=True)
        X_train_proc = X_train_proc.drop(columns=[sort_col])
        if sort_col in cat_cols:
            cat_cols.remove(sort_col)

    # 共通固定設定
    params_exec["random_seed"] = seed
    params_exec["early_stopping_rounds"] = early_stopping_rounds
    params_exec["verbose"] = False

    # 3. Optuna の目的関数（Objective）を定義
    def objective_func(trial: optuna.Trial) -> float:
        # 探索するハイパーパラメータ範囲の設定
        trial_params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-9, 10.0, log=True
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 1.0
            ),
        }

        # 固定パラメータと合体
        current_params = {**params_exec, **trial_params}

        # 交差検証
        if cv_strategy == "timeseries":
            cv = TimeSeriesSplit(n_splits=n_splits)
            splits = list(cv.split(X_train_proc))
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(cv.split(X_train_proc, y_train))
        
        oof_preds = np.zeros(len(X_train_proc))

        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X_train_proc.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train_proc.iloc[val_idx], y_train.iloc[val_idx]

            trn_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
            val_pool = Pool(X_va, y_va, cat_features=cat_cols)

            if is_binary:
                model = CatBoostClassifier(**current_params)
            else:
                model = CatBoostRegressor(**current_params)

            model.fit(
                trn_pool,
                eval_set=val_pool,
                use_best_model=True,
            )

            # 確率値の取り出しとクリッピング
            if is_binary:
                val_preds = model.predict_proba(val_pool)[:, 1]
                val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
            else:
                val_preds = model.predict(val_pool)

            oof_preds[val_idx] = val_preds

        # CV評価スコアの計算 (Log Loss)
        if is_binary:
            from sklearn.metrics import log_loss

            if cv_strategy == "timeseries":
                valid_mask = oof_preds > 0
                score = log_loss(y_train[valid_mask], oof_preds[valid_mask])
            else:
                score = log_loss(y_train, oof_preds)
        else:
            from sklearn.metrics import mean_squared_error

            score = np.sqrt(mean_squared_error(y_train, oof_preds))

        return score

    # 4. Optuna Study の作成と最適化実行
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective_func, n_trials=n_trials)

    # 5. ベストパラメータの合成
    best_trial_params = study.best_params
    best_params = {
        **params,
        **best_trial_params,
    }

    result_data = {
        "best_score": study.best_value,
        "best_params": best_trial_params,
        "study": study,
    }

    return result_data, best_params
