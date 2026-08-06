import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def _preprocess_data(X_train: pd.DataFrame, X_val: pd.DataFrame = None, X_test: pd.DataFrame = None) -> Tuple:
    """数値変数を標準化し、カテゴリ変数をOne-Hotエンコードする内部関数"""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, columns=cat_cols, drop_first=True) if X_val is not None else None
    X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True) if X_test is not None else None
    
    if X_val_encoded is not None:
        X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    if X_test_encoded is not None:
        X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    scaler = StandardScaler()
    num_cols_encoded = [col for col in X_train_encoded.columns if any(col.startswith(nc) for nc in num_cols)]
    
    if len(num_cols_encoded) > 0:
        X_train_encoded[num_cols_encoded] = scaler.fit_transform(X_train_encoded[num_cols_encoded])
        if X_val_encoded is not None:
            X_val_encoded[num_cols_encoded] = scaler.transform(X_val_encoded[num_cols_encoded])
        if X_test_encoded is not None:
            X_test_encoded[num_cols_encoded] = scaler.transform(X_test_encoded[num_cols_encoded])
    
    return X_train_encoded, X_val_encoded, X_test_encoded, scaler


def run_logistic(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Logistic Regressionの学習・交差検証（CV）および予測を機械的に実行する関数（v2: TimeSeriesSplit対応版）"""
    X_train = data.get("X_train")
    y_train = data.get("y_train")
    X_test = data.get("X_test", None)
    sort_col = data.get("sort_col", None)

    if X_train is None or y_train is None:
        raise ValueError(
            "data には 'X_train' と 'y_train' が含まれている必要があります。"
        )

    # 1. 制御用パラメータを params から取り出し
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits")
    seed = params_exec.pop("seed")
    save_dir = params_exec.pop("save_dir", None)
    cv_strategy = params_exec.pop("cv_strategy", "stratified")

    # 2. CV戦略の選択（特徴量エンジニアリング前に実行）
    if cv_strategy == "timeseries":
        if sort_col is None:
            raise ValueError(
                "TimeSeriesSplit を使用する場合は、data に 'sort_col' を指定してください。"
            )
        sort_indices = X_train[sort_col].argsort()
        X_train = X_train.iloc[sort_indices].reset_index(drop=True)
        y_train = y_train.iloc[sort_indices].reset_index(drop=True)
        cv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(cv.split(X_train))
        X_train = X_train.drop(columns=[sort_col])
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X_train, y_train))

    # 3. 配列の初期化
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    models = []
    scalers = []
    scores = []

    # 4. K-Fold 交差検証ループ
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # 前処理（標準化、One-Hotエンコード）
        X_tr_proc, X_va_proc, X_test_proc, scaler = _preprocess_data(X_tr, X_va, X_test)

        # Logistic Regressionモデルの作成
        model = LogisticRegression(
            random_state=seed + fold,
            **params_exec
        )

        # 学習
        model.fit(X_tr_proc, y_tr)

        # 予測（確率）
        val_preds = model.predict_proba(X_va_proc)[:, 1]
        val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
        oof_preds[val_idx] = val_preds

        # バリデーションスコアを計算
        from sklearn.metrics import log_loss
        fold_score = log_loss(y_va, val_preds)
        scores.append(fold_score)

        # テストデータの予測
        if X_test is not None:
            t_preds = model.predict_proba(X_test_proc)[:, 1]
            t_preds = np.clip(t_preds, 1e-15, 1.0 - 1e-15)
            test_preds += t_preds / n_splits

        models.append(model)
        scalers.append(scaler)

    # 5. OOFスコアの計算
    from sklearn.metrics import log_loss
    if cv_strategy == "timeseries":
        valid_mask = oof_preds > 0
        oof_score = log_loss(y_train[valid_mask], oof_preds[valid_mask])
    else:
        oof_score = log_loss(y_train, oof_preds)

    # 6. モデル保存処理
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for fold, (model, scaler) in enumerate(zip(models, scalers)):
            with open(save_path / f"logistic_model_fold{fold}.pkl", "wb") as f:
                pickle.dump({"model": model, "scaler": scaler}, f)

    # 7. 戻り値
    result_data = {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "models": models,
        "scalers": scalers,
        "scores": scores,
        "oof_score": oof_score,
    }

    return result_data, params
