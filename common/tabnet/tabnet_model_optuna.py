import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pytorch_tabnet.tab_model import TabNetClassifier

# ログを静かにする設定
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _preprocess_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """検証用の前処理関数"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_indices = []
    
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
        cat_indices.append(df.columns.get_loc(col))
    
    return df, cat_indices


def run_tabnet_optuna(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Optunaを用いてTabNetのハイパーパラメータ探索を交差検証（CV）ベースで実行する関数

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
    max_epochs = params_exec.pop("max_epochs", 100)
    patience = params_exec.pop("patience", 15)
    batch_size = params_exec.pop("batch_size", 1024)

    # 2. カテゴリ変数の型変換
    X_train_proc, cat_indices = _preprocess_categorical(X_train)
    cat_dims = [int(X_train_proc.iloc[:, idx].max() + 1) for idx in cat_indices]

    # 3. Optuna の目的関数を定義
    def objective_func(trial: optuna.Trial) -> float:
        # 探索するハイパーパラメータ範囲の設定
        trial_params = {
            "n_d": trial.suggest_int("n_d", 8, 64),
            "n_a": trial.suggest_int("n_a", 8, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
            "optimizer_params": {
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            },
            "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        }

        # Stratified K-Fold による交差検証
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(X_train_proc))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_proc, y_train)):
            X_tr, y_tr = X_train_proc.iloc[train_idx].values, y_train.iloc[train_idx].values
            X_va, y_va = X_train_proc.iloc[val_idx].values, y_train.iloc[val_idx].values

            model = TabNetClassifier(
                cat_idxs=cat_indices,
                cat_dims=cat_dims,
                seed=seed + fold,
                verbose=0,
                **trial_params
            )

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                eval_metric=['logloss']
            )

            val_preds = model.predict_proba(X_va)[:, 1]
            val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
            oof_preds[val_idx] = val_preds

        # CV評価スコアの計算 (Log Loss)
        from sklearn.metrics import log_loss
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