import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _preprocess_data(X_train: pd.DataFrame, X_val: pd.DataFrame = None) -> Tuple:
    """検証用の前処理関数"""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, columns=cat_cols, drop_first=True) if X_val is not None else None
    
    if X_val_encoded is not None:
        X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    scaler = StandardScaler()
    num_cols_encoded = [col for col in X_train_encoded.columns if any(col.startswith(nc) for nc in num_cols)]
    
    if len(num_cols_encoded) > 0:
        X_train_encoded[num_cols_encoded] = scaler.fit_transform(X_train_encoded[num_cols_encoded])
        if X_val_encoded is not None:
            X_val_encoded[num_cols_encoded] = scaler.transform(X_val_encoded[num_cols_encoded])
    
    return X_train_encoded, X_val_encoded


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x


def run_nn_optuna(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Optunaを用いてNNのハイパーパラメータ探索を交差検証（CV）ベースで実行する関数（v2: TimeSeriesSplit対応版）"""
    X_train = data.get("X_train")
    y_train = data.get("y_train")
    sort_col = data.get("sort_col", None)

    if X_train is None or y_train is None:
        raise ValueError(
            "data には 'X_train' と 'y_train' が含まれている必要があります。"
        )

    # 1. 制御用パラメータの取り出し
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits", 5)
    seed = params_exec.pop("seed", 42)
    n_trials = params_exec.pop("n_trials", 20)
    cv_strategy = params_exec.pop("cv_strategy", "stratified")
    params_exec.pop("save_dir", None)

    if cv_strategy == "timeseries":
        if sort_col is None:
            raise ValueError(
                "TimeSeriesSplit を使用する場合は、data に 'sort_col' を指定してください。"
            )
        sort_indices = X_train[sort_col].argsort()
        X_train = X_train.iloc[sort_indices].reset_index(drop=True)
        y_train = y_train.iloc[sort_indices].reset_index(drop=True)
        X_train = X_train.drop(columns=[sort_col])

    # 2. Optuna の目的関数を定義
    def objective_func(trial: optuna.Trial) -> float:
        # 探索するハイパーパラメータ範囲の設定
        trial_params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 32, 256),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "epochs": trial.suggest_int("epochs", 30, 100),
        }

        # 交差検証
        if cv_strategy == "timeseries":
            cv = TimeSeriesSplit(n_splits=n_splits)
            splits = list(cv.split(X_train))
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(cv.split(X_train, y_train))
        
        oof_preds = np.zeros(len(X_train))

        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

            X_tr_proc, X_va_proc = _preprocess_data(X_tr, X_va)

            train_dataset = TensorDataset(
                torch.FloatTensor(X_tr_proc.values),
                torch.FloatTensor(y_tr.values.reshape(-1, 1))
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_va_proc.values),
                torch.FloatTensor(y_va.values.reshape(-1, 1))
            )
            
            train_loader = DataLoader(train_dataset, batch_size=trial_params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=trial_params["batch_size"], shuffle=False)

            input_dim = X_tr_proc.shape[1]
            model = SimpleNN(
                input_dim=input_dim,
                hidden_dim=trial_params["hidden_dim"],
                dropout=trial_params["dropout"]
            )
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=trial_params["learning_rate"])

            model.train()
            for epoch in range(trial_params["epochs"]):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            model.eval()
            val_preds = []
            with torch.no_grad():
                for batch_X, _ in val_loader:
                    outputs = model(batch_X)
                    val_preds.append(outputs.numpy())
            
            val_preds = np.concatenate(val_preds).flatten()
            val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
            oof_preds[val_idx] = val_preds

        # CV評価スコアの計算 (Log Loss)
        from sklearn.metrics import log_loss
        if cv_strategy == "timeseries":
            valid_mask = oof_preds > 0
            score = log_loss(y_train[valid_mask], oof_preds[valid_mask])
        else:
            score = log_loss(y_train, oof_preds)

        return score

    # 3. Optuna Study の作成と最適化実行
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective_func, n_trials=n_trials)

    # 4. ベストパラメータの合成
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
