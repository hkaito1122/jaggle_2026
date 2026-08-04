import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pytorch_tabnet.tab_model import TabNetClassifier


def _preprocess_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """object 型の列を数値エンコードし、カテゴリカルインデックスを返す内部関数"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_indices = []
    
    for col in cat_cols:
        # Label Encoding
        df[col] = df[col].astype('category').cat.codes
        cat_indices.append(df.columns.get_loc(col))
    
    return df, cat_indices


def run_tabnet(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """TabNetの学習・交差検証(CV)および予測を機械的に実行する関数

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        'X_train': 学習用特徴量
        'y_train': 学習用ターゲット
        'X_test' : (任意) テスト用特徴量
    params : Dict[str, Any]
        TabNetのハイパーパラメータおよび制御用パラメータ
        必須・推奨キー: 'n_splits', 'seed', 'save_dir', 'max_epochs', 'patience' など

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        1. result_data : 予測結果データを含む辞書 ('oof_preds', 'test_preds', 'models')
        2. params      : 入力されたパラメータ(そのまま返却)
    """
    X_train = data.get("X_train")
    y_train = data.get("y_train")
    X_test = data.get("X_test", None)

    if X_train is None or y_train is None:
        raise ValueError(
            "data には 'X_train' と 'y_train' が含まれている必要があります。"
        )

    # 1. 制御用パラメータを params から取り出し
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits")
    seed = params_exec.pop("seed")
    save_dir = params_exec.pop("save_dir", None)
    max_epochs = params_exec.pop("max_epochs", 100)
    patience = params_exec.pop("patience", 15)
    batch_size = params_exec.pop("batch_size", 1024)
    verbose = params_exec.pop("verbose", 0)

    # 2. カテゴリ変数の前処理
    X_train_proc, cat_indices = _preprocess_categorical(X_train)
    X_test_proc = None
    if X_test is not None:
        X_test_proc, _ = _preprocess_categorical(X_test)

    # 3. 配列の初期化
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X_train_proc))
    test_preds = np.zeros(len(X_test_proc)) if X_test_proc is not None else None
    models = []

    # 4. Stratified K-Fold 交差検証ループ
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_proc, y_train)):
        X_tr, y_tr = X_train_proc.iloc[train_idx].values, y_train.iloc[train_idx].values
        X_va, y_va = X_train_proc.iloc[val_idx].values, y_train.iloc[val_idx].values

        # TabNetモデルの作成
        model = TabNetClassifier(
            cat_idxs=cat_indices,
            cat_dims=[int(X_train_proc.iloc[:, idx].max() + 1) for idx in cat_indices],
            seed=seed + fold,
            verbose=verbose,
            **params_exec
        )

        # 学習
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            eval_metric=['logloss']
        )

        # 予測(確率)
        val_preds = model.predict_proba(X_va)[:, 1]
        val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
        oof_preds[val_idx] = val_preds

        # テストデータの予測
        if X_test_proc is not None:
            t_preds = model.predict_proba(X_test_proc.values)[:, 1]
            t_preds = np.clip(t_preds, 1e-15, 1.0 - 1e-15)
            test_preds += t_preds / n_splits

        models.append(model)

    # 5. モデル保存処理
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for fold, model in enumerate(models):
            model.save_model(str(save_path / f"tabnet_model_fold{fold}"))

    # 6. 戻り値
    result_data = {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "models": models,
    }

    return result_data, params