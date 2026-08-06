import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit


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
    """CatBoostの学習・交差検証（CV）および予測を機械的に実行する関数（v2: TimeSeriesSplit対応版）"""
    X_train = data.get("X_train")
    y_train = data.get("y_train")
    X_test = data.get("X_test", None)
    sort_col = data.get("sort_col", None)

    if X_train is None or y_train is None:
        raise ValueError(
            "data には 'X_train' と 'y_train' が含まれている必要があります。"
        )

    # 1. 制御用パラメータの取り出し
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits", 5)
    seed = params_exec.pop("seed", 42)
    save_dir = params_exec.pop("save_dir", None)
    stopping_rounds = params_exec.pop("early_stopping_rounds", 50)
    verbose_eval = params_exec.pop("verbose_eval", False)
    cv_strategy = params_exec.pop("cv_strategy", "stratified")

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

    # 3. CV戦略の選択
    if cv_strategy == "timeseries":
        if sort_col is None:
            raise ValueError(
                "TimeSeriesSplit を使用する場合は、data に 'sort_col' を指定してください。"
            )
        sort_indices = X_train_proc[sort_col].argsort()
        X_train_proc = X_train_proc.iloc[sort_indices].reset_index(drop=True)
        y_train = y_train.iloc[sort_indices].reset_index(drop=True)
        cv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(cv.split(X_train_proc))
        X_train_proc = X_train_proc.drop(columns=[sort_col])
        # カテゴリ列リストを更新
        if sort_col in cat_cols:
            cat_cols.remove(sort_col)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X_train_proc, y_train))

    # 4. 配列の初期化
    oof_preds = np.zeros(len(X_train_proc))
    test_preds = np.zeros(len(X_test_proc)) if X_test_proc is not None else None
    models = []
    scores = []

    # 共通パラメータの設定
    params_exec["random_seed"] = seed
    params_exec["early_stopping_rounds"] = stopping_rounds
    params_exec["verbose"] = 100 if verbose_eval else False

    # 5. K-Fold 交差検証ループ
    for fold, (train_idx, val_idx) in enumerate(splits):
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

        # バリデーションスコアを計算
        if is_binary:
            from sklearn.metrics import log_loss
            fold_score = log_loss(y_va, val_preds)
            scores.append(fold_score)

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

    # 6. OOFスコアの計算
    if is_binary:
        from sklearn.metrics import log_loss
        if cv_strategy == "timeseries":
            valid_mask = oof_preds > 0
            oof_score = log_loss(y_train[valid_mask], oof_preds[valid_mask])
        else:
            oof_score = log_loss(y_train, oof_preds)
    else:
        oof_score = None

    # 7. モデル保存処理
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for fold, model in enumerate(models):
            model.save_model(str(save_path / f"cat_model_fold{fold}.cbm"))

    # 8. 戻り値
    result_data = {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "models": models,
        "scores": scores,
        "oof_score": oof_score,
    }

    return result_data, params
