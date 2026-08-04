import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold  # 分類用に StratifiedKFold を使用


def _preprocess_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """object 型の列を category 型に変換する内部関数"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


def run_xgb(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """XGBoostの学習・交差検証（CV）および予測を機械的に実行する関数

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        'X_train': 学習用特徴量
        'y_train': 学習用ターゲット
        'X_test' : (任意) テスト用特徴量
    params : Dict[str, Any]
        XGBoostのハイパーパラメータおよび制御用パラメータ
        必須・推奨キー: 'n_splits', 'seed', 'save_dir', 'early_stopping_rounds' など

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        1. result_data : 予測結果データを含む辞書 ('oof_preds', 'test_preds', 'models')
        2. params      : 入力されたパラメータ（そのまま返却）
    """
    X_train = data.get("X_train")
    y_train = data.get("y_train")
    X_test = data.get("X_test", None)

    if X_train is None or y_train is None:
        raise ValueError(
            "data には 'X_train' と 'y_train' が含まれている必要があります。"
        )

    # 1. 制御用パラメータの取り出し
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits")
    seed = params_exec.pop("seed")
    save_dir = params_exec.pop("save_dir", None)
    stopping_rounds = params_exec.pop("early_stopping_rounds", 50)
    verbose_eval = params_exec.pop("verbose_eval", False)

    # 二元分類（binary）かどうかの自動判定
    objective = params_exec.get("objective", "binary:logistic")
    is_binary = objective in ["binary:logistic", "binary:logitraw", "binary:hinge"]

    # 2. カテゴリ変数の型変換
    X_train_proc = _preprocess_categorical(X_train)
    X_test_proc = (
        _preprocess_categorical(X_test) if X_test is not None else None
    )

    # 3. 配列の初期化（分類用に StratifiedKFold を使用）
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X_train_proc))
    test_preds = np.zeros(len(X_test_proc)) if X_test_proc is not None else None
    models = []

    # DMatrix 用のカテゴリ有効化フラグ
    params_exec["enable_categorical"] = True

    # 4. Stratified K-Fold 交差検証ループ
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_proc, y_train)):
        X_tr, y_tr = X_train_proc.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train_proc.iloc[val_idx], y_train.iloc[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)

        evals = [(dtrain, "train"), (dval, "val")]

        model = xgb.train(
            params=params_exec,
            dtrain=dtrain,
            evals=evals,
            early_stopping_rounds=stopping_rounds,
            verbose_eval=100 if verbose_eval else False,
        )

        # --- 予測処理 ---
        val_preds = model.predict(dval)

        # 二元分類（binary）の場合、確率を 1e-15 〜 1-1e-15 にクリッピング
        if is_binary:
            val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)

        oof_preds[val_idx] = val_preds

        # テストデータの予測（前処理済みの X_test_proc を使用）
        if X_test_proc is not None:
            dtest = xgb.DMatrix(X_test_proc, enable_categorical=True)
            t_preds = model.predict(dtest)
            if is_binary:
                t_preds = np.clip(t_preds, 1e-15, 1.0 - 1e-15)
            test_preds += t_preds / n_splits

        models.append(model)

    # 5. モデル保存処理
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for fold, model in enumerate(models):
            model.save_model(str(save_path / f"xgb_model_fold{fold}.json"))

    # 6. 戻り値
    result_data = {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "models": models,
    }

    return result_data, params