import os
from pathlib import Path
from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold  # 分類用に StratifiedKFold を使用

# ログを静かにする設定
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _preprocess_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """object 型の列を category 型に変換する内部関数"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


def run_lgb_optuna(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Optunaを用いてLightGBMのハイパーパラメータ探索を交差検証（CV）ベースで実行する関数

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
        2. best_params  : 最適化された全パラメータ（固定パラメータ + ベスト探索パラメータ）
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
    early_stopping_rounds = params_exec.pop("early_stopping_rounds", 50)

    # 二元分類タスクかの自動判定
    objective = params_exec.get("objective", "binary")
    is_binary = objective in ["binary", "binary_logloss"]

    # 2. カテゴリ変数の型変換
    X_train_proc = _preprocess_categorical(X_train)

    # 3. Optuna の目的関数（Objective）を定義
    def objective_func(trial: optuna.Trial) -> float:
        # 探索するハイパーパラメータ範囲の設定
        trial_params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True
            ),
        }

        # 固定パラメータと合体
        current_params = {**params_exec, **trial_params}

        # Stratified K-Fold による交差検証
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        oof_preds = np.zeros(len(X_train_proc))

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X_train_proc, y_train)
        ):
            X_tr, y_tr = X_train_proc.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train_proc.iloc[val_idx], y_train.iloc[val_idx]

            trn_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

            model = lgb.train(
                params=current_params,
                train_set=trn_data,
                valid_sets=[trn_data, val_data],
                callbacks=[
                    lgb.early_stopping(
                        early_stopping_rounds, verbose=False
                    ),
                    lgb.log_evaluation(period=0),
                ],
            )

            val_preds = model.predict(X_va)

            # 確率値の判定とクリッピング
            if is_binary:
                val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)

            oof_preds[val_idx] = val_preds

        # CV評価スコアの計算 (Log Loss)
        if is_binary:
            from sklearn.metrics import log_loss

            score = log_loss(y_train, oof_preds)
        else:
            from sklearn.metrics import mean_squared_error

            score = np.sqrt(mean_squared_error(y_train, oof_preds))

        return score

    # 4. Optuna Study の作成と最適化実行 (Log Loss 最小化のため direction="minimize")
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective_func, n_trials=n_trials)

    # 5. ベストパラメータの合成（固定設定パラメータを復元）
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