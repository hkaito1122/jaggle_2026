import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _preprocess_categorical(df: pd.DataFrame, label_encoders: Dict = None) -> Tuple:
    """カテゴリ変数をLabel Encodeする内部関数"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if label_encoders is None:
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        for col in cat_cols:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col].astype(str))
    
    cat_idxs = [df.columns.get_loc(col) for col in cat_cols]
    cat_dims = [int(df[col].max()) + 1 for col in cat_cols]
    
    return df, label_encoders, cat_idxs, cat_dims


def run_tabnet_optuna(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Optunaを用いてTabNetのハイパーパラメータ探索を交差検証（CV）ベースで実行する関数（v2: TimeSeriesSplit対応版）"""
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

    if cv_strategy == "timeseries":
        if sort_col is None:
            raise ValueError(
                "TimeSeriesSplit を使用する場合は、data に 'sort_col' を指定してください。"
            )
        sort_indices = X_train[sort_col].argsort()
        X_train = X_train.iloc[sort_indices].reset_index(drop=True)
        y_train = y_train.iloc[sort_indices].reset_index(drop=True)
        X_train = X_train.drop(columns=[sort_col])

    # 2. Optuna の目的関数を定義
    def objective_func(trial: optuna.Trial) -> float:
        # 探索するハイパーパラメータ範囲の設定
        trial_params = {
            "n_d": trial.suggest_int("n_d", 8, 64),
            "n_a": trial.suggest_int("n_a", 8, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
            "max_epochs": trial.suggest_int("max_epochs", 50, 200),
            "patience": trial.suggest_int("patience", 10, 30),
        }

        # 固定パラメータと合体
        current_params = {**params_exec, **trial_params}

        # 交差検証
        if cv_strategy == "timeseries":
            cv = TimeSeriesSplit(n_splits=n_splits)
            splits = list(cv.split(X_train))
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(cv.split(X_train, y_train))
        
        oof_preds = np.zeros(len(X_train))

        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

            X_tr_proc, label_encoders, cat_idxs, cat_dims = _preprocess_categorical(X_tr)
            X_va_proc, _, _, _ = _preprocess_categorical(X_va, label_encoders)

            # fit()専用パラメータを除外
            fit_only_params = {"max_epochs", "patience", "batch_size", "virtual_batch_size"}
            constructor_params = {k: v for k, v in current_params.items() 
                                 if k not in trial_params and k not in fit_only_params}
            
            model = TabNetClassifier(
                seed=seed + fold,
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                n_d=trial_params["n_d"],
                n_a=trial_params["n_a"],
                n_steps=trial_params["n_steps"],
                gamma=trial_params["gamma"],
                lambda_sparse=trial_params["lambda_sparse"],
                **constructor_params
            )

            model.fit(
                X_tr_proc.values, y_tr.values,
                eval_set=[(X_va_proc.values, y_va.values)],
                eval_metric=["logloss"],
                max_epochs=trial_params["max_epochs"],
                patience=trial_params["patience"],
                batch_size=current_params.get("batch_size", 256),
                virtual_batch_size=current_params.get("virtual_batch_size", 128),
            )

            val_preds = model.predict_proba(X_va_proc.values)[:, 1]
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

    # 3. Optuna Study の作成と最適化実行
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective_func, n_trials=n_trials)

    # 4. ベストパラメータの合成
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
