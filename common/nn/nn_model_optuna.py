import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ログを静かにする設定
optuna.logging.set_verbosity(optuna.logging.WARNING)


class SimpleNN(nn.Module):
    """シンプルなニューラルネットワークモデル"""
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super(SimpleNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


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


def run_nn_optuna(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Optunaを用いてNeural Networkのハイパーパラメータ探索を交差検証（CV）ベースで実行する関数

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        'X_train': 学習用特徴量
        'y_train': 学習用ターゲット
    params : Dict[str, Any]
        固定パラメータおよび探索用制御パラメータ ('n_splits', 'seed', 'n_trials' など)

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        1. result_data  : 'best_score', 'best_params', 'study' オブジェクトを含む辞書
        2. best_params  : 最適化された全パラメータ
    """
    X_train = data.get("X_train")
    y_train = data.get("y_train")

    if X_train is None or y_train is None:
        raise ValueError(
            "data には 'X_train' と 'y_train' が含まれている必要があります。"
        )

    # 1. 制御用パラメータの取り出し
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits", 5)
    seed = params_exec.pop("seed", 42)
    n_trials = params_exec.pop("n_trials", 20)
    epochs = params_exec.pop("epochs", 50)
    batch_size = params_exec.pop("batch_size", 256)
    patience = params_exec.pop("patience", 10)
    params_exec.pop("save_dir", None)  # Neural Networkに渡さないパラメータを除外

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Optuna の目的関数を定義
    def objective_func(trial: optuna.Trial) -> float:
        # 探索するハイパーパラメータ範囲の設定
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        n_layers = trial.suggest_int("n_layers", 2, 4)
        
        hidden_dims = []
        for i in range(n_layers):
            hidden_dims.append(trial.suggest_int(f"hidden_dim_{i}", 64, 512, step=64))

        # Stratified K-Fold による交差検証
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(X_train))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

            X_tr_proc, X_va_proc = _preprocess_data(X_tr, X_va)

            X_tr_tensor = torch.FloatTensor(X_tr_proc.values).to(device)
            y_tr_tensor = torch.FloatTensor(y_tr.values).to(device)
            X_va_tensor = torch.FloatTensor(X_va_proc.values).to(device)

            train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            input_dim = X_tr_proc.shape[1]
            model = SimpleNN(input_dim, hidden_dims, dropout).to(device)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_va_tensor).squeeze()
                    val_loss = criterion(val_outputs, torch.FloatTensor(y_va.values).to(device))

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            model.eval()
            with torch.no_grad():
                val_preds = model(X_va_tensor).squeeze().cpu().numpy()
                val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
                oof_preds[val_idx] = val_preds

        # CV評価スコアの計算 (Log Loss)
        from sklearn.metrics import log_loss
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