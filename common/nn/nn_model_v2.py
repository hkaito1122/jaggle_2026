import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


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


class SimpleNN(nn.Module):
    """シンプルな3層ニューラルネットワーク"""
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


def run_nn(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Neural Networkの学習・交差検証（CV）および予測を機械的に実行する関数（v2: TimeSeriesSplit対応版）"""
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
    
    batch_size = params_exec.pop("batch_size", 128)
    epochs = params_exec.pop("epochs", 50)
    lr = params_exec.pop("learning_rate", 0.001)
    hidden_dim = params_exec.pop("hidden_dim", 64)
    dropout = params_exec.pop("dropout", 0.3)

    # 2. CV戦略の選択
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

        # 前処理
        X_tr_proc, X_va_proc, X_test_proc, scaler = _preprocess_data(X_tr, X_va, X_test)

        # PyTorchデータセット作成
        train_dataset = TensorDataset(
            torch.FloatTensor(X_tr_proc.values),
            torch.FloatTensor(y_tr.values.reshape(-1, 1))
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_va_proc.values),
            torch.FloatTensor(y_va.values.reshape(-1, 1))
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # モデル作成
        input_dim = X_tr_proc.shape[1]
        model = SimpleNN(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 学習（Early Stopping付き）
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    # print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 最良のモデルを復元
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # 予測
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                outputs = model(batch_X)
                val_preds.append(outputs.numpy())
        
        val_preds = np.concatenate(val_preds).flatten()
        val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
        oof_preds[val_idx] = val_preds

        # スコア計算
        from sklearn.metrics import log_loss
        fold_score = log_loss(y_va, val_preds)
        scores.append(fold_score)

        # テストデータの予測
        if X_test is not None:
            test_dataset = TensorDataset(torch.FloatTensor(X_test_proc.values))
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            t_preds = []
            with torch.no_grad():
                for batch_X, in test_loader:
                    outputs = model(batch_X)
                    t_preds.append(outputs.numpy())
            
            t_preds = np.concatenate(t_preds).flatten()
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
            torch.save(model.state_dict(), str(save_path / f"nn_model_fold{fold}.pth"))
            with open(save_path / f"scaler_fold{fold}.pkl", "wb") as f:
                pickle.dump(scaler, f)

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
