import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _preprocess_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """object 型の列を category 型に変換する内部関数"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


def run_histgb_optuna(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Optunaを用いてHistGradientBoostingのハイパーパラメータ探索を交差検証（CV）ベースで実行する関数（v2: TimeSeriesSplit対応版）"""
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
    cv_strategy = params_exec.pop("cv_strategy", "stratified")
    params_exec.pop("save_dir", None)

    # 2. カテゴリ変数の型変換
    X_train_proc = _preprocess_categorical(X_train)

    if cv_strategy == "timeseries":
        if sort_col is None:
            raise ValueError(
                "TimeSeriesSplit を使用する場合は、data に 'sort_col' を指定してください。"
            )
        sort_indices = X_train_proc[sort_col].argsort()
        X_train_proc = X_train_proc.iloc[sort_indices].reset_index(drop=True)
        y_train = y_train.iloc[sort_indices].reset_index(drop=True)
        X_train_proc = X_train_proc.drop(columns=[sort_col])

    # 3. Optuna の目的関数を定義
    def objective_func(trial: optuna.Trial) -> float:
        # 探索するハイパーパラメータ範囲の設定
        trial_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
            "max_bins": trial.suggest_int("max_bins", 32, 255),
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

            model = HistGradientBoostingClassifier(
                random_state=seed + fold,
                **current_params
            )

            model.fit(X_tr, y_tr)

            val_preds = model.predict_proba(X_va)[:, 1]
            val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
            oof_preds[val_idx] = val_preds

        # CV評価スコアの計算 (Log Loss)
        from sklearn.metrics import log_loss
        if cv_strategy == "timeseries":
            valid_mask = oof_preds > 0
            score = log_loss(y_train[valid_mask], oof_preds[valid_mask])
        else:
            score = log_loss(y_train, oof_preds)

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
