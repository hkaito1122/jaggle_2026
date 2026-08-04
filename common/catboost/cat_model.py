# common/catboost/cat_model.py
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import KFold


def run_cat(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """CatBoostの学習・交差検証（CV）および予測を機械的に実行する関数"""
    X_train = data.get("X_train")
    y_train = data.get("y_train")
    X_test = data.get("X_test", None)

    if X_train is None or y_train is None:
        raise ValueError("data には 'X_train' と 'y_train' が含まれている必要があります。")

    # 1. 制御用パラメータの取り出し
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits")
    seed = params_exec.pop("seed")
    save_dir = params_exec.pop("save_dir", None)
    stopping_rounds = params_exec.pop("early_stopping_rounds", 50)
    verbose_eval = params_exec.pop("verbose_eval", False)

    # CatBoostの引数競合を防ぐため非モデル設定を削除
    params_exec.pop("n_trials", None)
    params_exec.pop("direction", None)
    params_exec.pop("verbose", None)

    # タスク判定（分類 vs 回帰）
    is_classification = params_exec.get("loss_function", "").lower() in [
        "logloss",
        "cross_entropy",
    ] or params_exec.get("eval_metric", "").lower() in [
        "logloss",
        "auc",
        "accuracy",
    ]

    # 2. カテゴリ変数の列名を自動抽出して string 型へ変換
    cat_cols = list(
        X_train.select_dtypes(include=["object", "category"]).columns
    )
    X_train_proc = X_train.copy()
    for col in cat_cols:
        X_train_proc[col] = X_train_proc[col].astype(str)

    X_test_proc = X_test.copy() if X_test is not None else None
    if X_test_proc is not None:
        for col in cat_cols:
            X_test_proc[col] = X_test_proc[col].astype(str)

    # 3. 配列の初期化
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X_train_proc))
    test_preds = np.zeros(len(X_test_proc)) if X_test_proc is not None else None
    models = []

    test_pool = (
        Pool(X_test_proc, cat_features=cat_cols)
        if X_test_proc is not None
        else None
    )

    # 4. K-Fold 交差検証ループ
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y_train)):
        X_tr, y_tr = X_train_proc.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train_proc.iloc[val_idx], y_train.iloc[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_va, y_va, cat_features=cat_cols)

        if is_classification:
            model = CatBoostClassifier(
                **params_exec,
                random_seed=seed,
                early_stopping_rounds=stopping_rounds,
                verbose=100 if verbose_eval else False,
            )
            model.fit(train_pool, eval_set=val_pool)
            
            # 分類の場合は確率（クラス1）を出力
            oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
            if test_pool is not None:
                test_preds += model.predict_proba(test_pool)[:, 1] / n_splits
        else:
            model = CatBoostRegressor(
                **params_exec,
                random_seed=seed,
                early_stopping_rounds=stopping_rounds,
                verbose=100 if verbose_eval else False,
            )
            model.fit(train_pool, eval_set=val_pool)
            
            oof_preds[val_idx] = model.predict(val_pool)
            if test_pool is not None:
                test_preds += model.predict(test_pool) / n_splits

        models.append(model)

    # 5. モデル保存処理
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for fold, model in enumerate(models):
            model.save_model(str(save_path / f"cat_model_fold{fold}.cbm"))

    # 6. 戻り値
    result_data = {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "models": models,
    }

    return result_data, params