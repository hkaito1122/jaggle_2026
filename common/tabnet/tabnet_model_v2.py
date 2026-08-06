import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def _preprocess_tabnet_data(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    X_test: pd.DataFrame = None, 
    cat_feature_names: list = []
) -> Tuple:
    """TabNet用に数値変数のスケーリング・欠損値補完、およびカテゴリ変数のエンコーディングを行う"""
    X_train_proc = X_train.copy()
    X_val_proc = X_val.copy()
    X_test_proc = X_test.copy() if X_test is not None else None
    
    # 1. 欠損値補完と標準化（ニューラルネットワーク用）
    # 元々の -999 は欠損値に戻してから平均値/中央値で補完する
    for df in [X_train_proc, X_val_proc, X_test_proc]:
        if df is not None:
            df.replace(-999, np.nan, inplace=True)
            
    num_cols = [col for col in X_train.columns if col not in cat_feature_names]
    
    if len(num_cols) > 0:
        # 中央値で補完
        imputer = SimpleImputer(strategy="median")
        X_train_proc[num_cols] = imputer.fit_transform(X_train_proc[num_cols])
        X_val_proc[num_cols] = imputer.transform(X_val_proc[num_cols])
        if X_test_proc is not None:
            X_test_proc[num_cols] = imputer.transform(X_test_proc[num_cols])
            
        # 標準化
        scaler = StandardScaler()
        X_train_proc[num_cols] = scaler.fit_transform(X_train_proc[num_cols])
        X_val_proc[num_cols] = scaler.transform(X_val_proc[num_cols])
        if X_test_proc is not None:
            X_test_proc[num_cols] = scaler.transform(X_test_proc[num_cols])

    # 2. カテゴリ変数のエンコーディング（未知のカテゴリに対応するため、LabelEncoderを改良）
    # 未知のカテゴリは「0」にマッピングし、既知のカテゴリは 1 から連番にする
    cat_dims = []
    cat_idxs = []
    
    for col in cat_feature_names:
        if col not in X_train_proc.columns:
            continue
        
        idx = X_train_proc.columns.get_loc(col)
        cat_idxs.append(idx)
        
        # 学習データで LabelEncoder を学習
        le = LabelEncoder()
        # 未知値用のプレースホルダーを含める
        unique_vals = X_train_proc[col].astype(str).unique()
        le.fit(unique_vals)
        
        # 変換マッピングの作成
        train_encoded = le.transform(X_train_proc[col].astype(str)) + 1 # 0は未知値用に空ける
        X_train_proc.iloc[:, idx] = train_encoded
        
        # 検証データ・テストデータの処理（未知カテゴリは 0 に置換）
        val_encoded = []
        for val in X_val_proc[col].astype(str):
            if val in le.classes_:
                val_encoded.append(le.transform([val])[0] + 1)
            else:
                val_encoded.append(0) # 未知カテゴリ
        X_val_proc.iloc[:, idx] = val_encoded
        
        if X_test_proc is not None:
            test_encoded = []
            for val in X_test_proc[col].astype(str):
                if val in le.classes_:
                    test_encoded.append(le.transform([val])[0] + 1)
                else:
                    test_encoded.append(0)
            X_test_proc.iloc[:, idx] = test_encoded
            
        # Embedding次元は (ユニーク数 + 未知カテゴリの1個 + 安全マージン1個)
        cat_dims.append(int(len(le.classes_) + 2))
        
    return X_train_proc, X_val_proc, X_test_proc, cat_idxs, cat_dims


def run_tabnet(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """TabNetの学習・交差検証（CV）および予測を実行する関数 (修正・ロバスト版)"""
    X_train = data.get("X_train")
    y_train = data.get("y_train")
    X_test = data.get("X_test", None)
    sort_col = data.get("sort_col", None)

    if X_train is None or y_train is None:
        raise ValueError("data には 'X_train' と 'y_train' が含まれている必要があります。")

    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits")
    seed = params_exec.pop("seed")
    save_dir = params_exec.pop("save_dir", None)
    cv_strategy = params_exec.pop("cv_strategy", "stratified")
    
    max_epochs = params_exec.pop("max_epochs", 100)
    patience = params_exec.pop("patience", 10)
    batch_size = params_exec.pop("batch_size", 256)
    virtual_batch_size = params_exec.pop("virtual_batch_size", 128)
    
    # カテゴリカル変数の列名を取得
    cat_feature_names = params_exec.pop("cat_feature_names", [])

    # CV戦略の選択
    if cv_strategy == "timeseries":
        sort_indices = X_train[sort_col].argsort()
        X_train = X_train.iloc[sort_indices].reset_index(drop=True)
        y_train = y_train.iloc[sort_indices].reset_index(drop=True)
        cv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(cv.split(X_train))
        X_train = X_train.drop(columns=[sort_col])
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X_train, y_train))

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    models = []
    scores = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # 提案した堅牢な前処理を適用
        X_tr_proc, X_va_proc, X_test_proc, cat_idxs, cat_dims = _preprocess_tabnet_data(
            X_tr, X_va, X_test, cat_feature_names
        )

        # 自動でカテゴリエンコード後の次元を反映させるため
        params_exec["cat_idxs"] = cat_idxs
        params_exec["cat_dims"] = cat_dims
        
        # cat_emb_dim を自動決定させるため指定を削除するか、適切なサイズを設定
        # パラメータで指定されていない、または 1 の場合は自動設計(各次元ごとにリストで管理)に切り替えるのが安全
        if "cat_emb_dim" in params_exec and params_exec["cat_emb_dim"] == 1:
            params_exec.pop("cat_emb_dim")

        model = TabNetClassifier(
            seed=seed + fold,
            **params_exec
        )

        model.fit(
            X_tr_proc.values, y_tr.values,
            eval_set=[(X_va_proc.values, y_va.values)],
            eval_metric=["logloss"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
        )

        # 予測
        val_preds = model.predict_proba(X_va_proc.values)[:, 1]
        val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
        oof_preds[val_idx] = val_preds

        from sklearn.metrics import log_loss
        fold_score = log_loss(y_va, val_preds)
        scores.append(fold_score)

        if X_test is not None:
            t_preds = model.predict_proba(X_test_proc.values)[:, 1]
            t_preds = np.clip(t_preds, 1e-15, 1.0 - 1e-15)
            test_preds += t_preds / n_splits

        models.append(model)

    if cv_strategy == "timeseries":
        valid_mask = oof_preds > 0
        oof_score = log_loss(y_train[valid_mask], oof_preds[valid_mask])
    else:
        oof_score = log_loss(y_train, oof_preds)

    result_data = {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "models": models,
        "scores": scores,
        "oof_score": oof_score,
    }

    return result_data, params
