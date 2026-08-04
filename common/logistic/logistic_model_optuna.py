import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ログを静かにする設定
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _preprocess_data(X_train: pd.DataFrame, X_val: pd.DataFrame = None) -> Tuple:
    """検証用の前処理関数"""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, columns=cat_cols, drop_first=True) if X_val is not None else None
    
    if X_val_encoded is not None:
        X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    scaler = StandardScaler()
    num_cols_encoded = [col for col in X_train_encoded.columns if any(col.startswith(nc) for nc in num_cols)]
    
    if len(num_cols_encoded) > 0:
        X_train_encoded[num_cols_encoded] = scaler.fit_transform(X_train_encoded[num_cols_encoded])
        if X_val_encoded is not None:
            X_val_encoded[num_cols_encoded] = scaler.transform(X_val_encoded[num_cols_encoded])
    
    return X_train_encoded, X_val_encoded


def run_logistic_optuna(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Optunaを用いてLogistic Regressionのハイパーパラメータ探索を交差検証（CV）ベースで実行する関数

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        'X_train': 学習用特徴量
        'y_train': 学習用ターゲット
    params : Dict[str, Any]
        固定パラメータおよび探索用制御パラメータ ('n_splits', 'seed', 'n_trials' など)

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        1. result_data  : 'best_score', 'best_params', 'study' オブジェクトを含む辞書
        2. best_params  : 最適化された全パラメータ
    """
    X_train = data.get("X_train")
    y_train = data.get("y_train")

    if X_train is None or y_train is None:
        raise ValueError(
            "data には 'X_train' と 'y_train' が含まれている必要があります。"
        )

    # 1. 制御用パラメータの取り出し
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits", 5)
    seed = params_exec.pop("seed", 42)
    n_trials = params_exec.pop("n_trials", 20)
    params_exec.pop("save_dir", None)  # LogisticRegressionに渡さないパラメータを除外

    # 2. Optuna の目的関数を定義
    def objective_func(trial: optuna.Trial) -> float:
        # 探索するハイパーパラメータ範囲の設定
        trial_params = {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "saga",  # l1とl2両方をサポート
        }

        # 固定パラメータと合体
        current_params = {**params_exec, **trial_params}

        # Stratified K-Fold による交差検証
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(X_train))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

            X_tr_proc, X_va_proc = _preprocess_data(X_tr, X_va)

            model = LogisticRegression(
                random_state=seed + fold,
                max_iter=1000,
                **current_params
            )

            model.fit(X_tr_proc, y_tr)

            val_preds = model.predict_proba(X_va_proc)[:, 1]
            val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
            oof_preds[val_idx] = val_preds

        # CV評価スコアの計算 (Log Loss)
        from sklearn.metrics import log_loss
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