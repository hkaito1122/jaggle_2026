import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit


def _preprocess_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """object 型の列を category 型に変換する内部関数"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


def run_histgb(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """HistGradientBoostingの学習・交差検証（CV）および予測を機械的に実行する関数（v2: TimeSeriesSplit対応版）"""
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

    # 2. カテゴリ変数の型変換
    X_train_proc = _preprocess_categorical(X_train)
    X_test_proc = (
        _preprocess_categorical(X_test) if X_test is not None else None
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
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X_train_proc, y_train))

    # 4. 配列の初期化
    oof_preds = np.zeros(len(X_train_proc))
    test_preds = np.zeros(len(X_test_proc)) if X_test_proc is not None else None
    models = []
    scores = []

    # 5. K-Fold 交差検証ループ
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X_train_proc.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train_proc.iloc[val_idx], y_train.iloc[val_idx]

        # HistGradientBoostingモデルの作成
        model = HistGradientBoostingClassifier(
            random_state=seed + fold,
            **params_exec
        )

        # 学習
        model.fit(X_tr, y_tr)

        # 予測（確率）
        val_preds = model.predict_proba(X_va)[:, 1]
        val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
        oof_preds[val_idx] = val_preds

        # バリデーションスコアを計算
        from sklearn.metrics import log_loss
        fold_score = log_loss(y_va, val_preds)
        scores.append(fold_score)

        # テストデータの予測
        if X_test_proc is not None:
            t_preds = model.predict_proba(X_test_proc)[:, 1]
            t_preds = np.clip(t_preds, 1e-15, 1.0 - 1e-15)
            test_preds += t_preds / n_splits

        models.append(model)

    # 6. OOFスコアの計算
    from sklearn.metrics import log_loss
    if cv_strategy == "timeseries":
        valid_mask = oof_preds > 0
        oof_score = log_loss(y_train[valid_mask], oof_preds[valid_mask])
    else:
        oof_score = log_loss(y_train, oof_preds)

    # 7. モデル保存処理
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for fold, model in enumerate(models):
            with open(save_path / f"histgb_model_fold{fold}.pkl", "wb") as f:
                pickle.dump(model, f)

    # 8. 戻り値
    result_data = {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "models": models,
        "scores": scores,
        "oof_score": oof_score,
    }

    return result_data, params
