import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold


def _preprocess_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """object 型の列を CatBoost 用に文字列/カテゴリ型へ変換し、カテゴリ列名リストを返す内部関数"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype(str)
    return df, cat_cols


def run_cat(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """CatBoostの学習・交差検証（CV）および予測を機械的に実行する関数"""
    X_train = data.get("X_train")
    y_train = data.get("y_train")
    X_test = data.get("X_test", None)

    if X_train is None or y_train is None:
        raise ValueError(
            "data には 'X_train' と 'y_train' が含まれている必要があります。"
        )

    # 1. 制御用パラメータの取り出し (CatBoost本体に渡さないものを除外)
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits", 5)
    seed = params_exec.pop("seed", 42)
    save_dir = params_exec.pop("save_dir", None)
    stopping_rounds = params_exec.pop("early_stopping_rounds", 50)
    verbose_eval = params_exec.pop("verbose_eval", False)

    # ★重要: Optuna探索から引き継がれた不要な制御用引数を除外
    params_exec.pop("n_trials", None)

    # 二元分類かどうかの自動判定
    loss_function = params_exec.get(
        "loss_function", params_exec.get("eval_metric", "Logloss")
    )
    is_binary = loss_function.lower() in ["logloss", "binary"]

    # 2. カテゴリ変数の前処理とリスト取得
    X_train_proc, cat_cols = _preprocess_categorical(X_train)
    X_test_proc = (
        _preprocess_categorical(X_test)[0] if X_test is not None else None
    )

    # 3. 配列の初期化（StratifiedKFold）
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X_train_proc))
    test_preds = np.zeros(len(X_test_proc)) if X_test_proc is not None else None
    models = []

    # 共通パラメータの設定
    params_exec["random_seed"] = seed
    params_exec["early_stopping_rounds"] = stopping_rounds
    params_exec["verbose"] = 100 if verbose_eval else False

    # 4. Stratified K-Fold 交差検証ループ
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_proc, y_train)):
        X_tr, y_tr = X_train_proc.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train_proc.iloc[val_idx], y_train.iloc[val_idx]

        trn_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_va, y_va, cat_features=cat_cols)

        # 二元分類か回帰かでモデルクラスを切り替え
        if is_binary:
            model = CatBoostClassifier(**params_exec)
        else:
            model = CatBoostRegressor(**params_exec)

        model.fit(
            trn_pool,
            eval_set=val_pool,
            use_best_model=True,
        )

        # --- 予測処理 (二元分類時は predict_proba から 1 である確率を抽出) ---
        if is_binary:
            val_preds = model.predict_proba(val_pool)[:, 1]
            val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
        else:
            val_preds = model.predict(val_pool)

        oof_preds[val_idx] = val_preds

        # テストデータの予測
        if X_test_proc is not None:
            t_pool = Pool(X_test_proc, cat_features=cat_cols)
            if is_binary:
                t_preds = model.predict_proba(t_pool)[:, 1]
                t_preds = np.clip(t_preds, 1e-15, 1.0 - 1e-15)
            else:
                t_preds = model.predict(t_pool)
            test_preds += t_preds / n_splits

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