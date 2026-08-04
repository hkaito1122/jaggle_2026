# common/xgboost/xgb_model_optuna.py
import copy
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Optunaのログ出力を制御
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _preprocess_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """object 型の列を category 型に変換する内部関数"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


def run_xgb_optuna(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Optunaを用いてXGBoostのハイパーパラメータ探索を実行する関数

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        'X_train': 学習用特徴量
        'y_train': 学習用ターゲット
    params : Dict[str, Any]
        固定パラメータおよびOptuna制御用パラメータ

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        1. result_data : Optunaの探索結果を含む辞書 ('best_score', 'study')
        2. best_params : 最適化されたパラメータ（xgb_model.py の run_xgb にそのまま渡せます）
    """
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

    # 2. カテゴリ変数の前処理
    X_train_proc = _preprocess_categorical(X_train)

    # 3. Optuna 目的関数の定義
    def objective(trial: optuna.Trial) -> float:
        # common/xgboost/xgb_model_optuna.py 内の trial_params 提案
        trial_params = {
            # 学習率と木の構成
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 30.0, log=True
            ),
            # サンプリング（過学習防止）
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            # 木の分割・正則化
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        }

        current_params = base_params.copy()
        current_params.update(trial_params)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(X_train_proc))

        for train_idx, val_idx in kf.split(X_train_proc, y_train):
            X_tr, y_tr = X_train_proc.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train_proc.iloc[val_idx], y_train.iloc[val_idx]

            dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
            dval = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)

            evals = [(dtrain, "train"), (dval, "val")]

            model = xgb.train(
                params=current_params,
                dtrain=dtrain,
                evals=evals,
                early_stopping_rounds=stopping_rounds,
                verbose_eval=False,
            )

            oof_preds[val_idx] = model.predict(dval)

        score = np.sqrt(mean_squared_error(y_train, oof_preds))
        return score

    # 4. チューニングの実行
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # 5. 最良パラメータの作成
    best_params = copy.deepcopy(params)
    best_params.update(study.best_params)

    # 6. 戻り値
    result_data = {
        "best_score": study.best_value,
        "study": study,
    }

    return result_data, best_params