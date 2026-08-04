import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetClassifier(nn.Module):
    """シンプルなフィードフォワードニューラルネットワーク"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super(NeuralNetClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 出力層（2クラス分類）
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def _preprocess_data(X_train: pd.DataFrame, X_val: pd.DataFrame = None, X_test: pd.DataFrame = None) -> Tuple:
    """数値変数を標準化し、カテゴリ変数をOne-Hotエンコードする内部関数"""
    # カテゴリ変数と数値変数を分離
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # One-Hot Encoding
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, columns=cat_cols, drop_first=True) if X_val is not None else None
    X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True) if X_test is not None else None
    
    # カラムを揃える
    if X_val_encoded is not None:
        X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    if X_test_encoded is not None:
        X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    # 数値変数の標準化
    scaler = StandardScaler()
    X_train_encoded = pd.DataFrame(
        scaler.fit_transform(X_train_encoded),
        columns=X_train_encoded.columns,
        index=X_train_encoded.index
    )
    if X_val_encoded is not None:
        X_val_encoded = pd.DataFrame(
            scaler.transform(X_val_encoded),
            columns=X_val_encoded.columns,
            index=X_val_encoded.index
        )
    if X_test_encoded is not None:
        X_test_encoded = pd.DataFrame(
            scaler.transform(X_test_encoded),
            columns=X_test_encoded.columns,
            index=X_test_encoded.index
        )
    
    return X_train_encoded, X_val_encoded, X_test_encoded, scaler


def run_nn(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Neural Networkの学習・交差検証（CV）および予測を機械的に実行する関数
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        'X_train': 学習用特徴量
        'y_train': 学習用ターゲット
        'X_test' : (任意) テスト用特徴量
    params : Dict[str, Any]
        Neural Networkのハイパーパラメータおよび制御用パラメータ
        必須・推奨キー: 'n_splits', 'seed', 'save_dir', 'hidden_dims', 'dropout',
                       'learning_rate', 'epochs', 'batch_size', 'patience' など
    
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
    
    # 1. 制御用パラメータを params から取り出し
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits")
    seed = params_exec.pop("seed")
    save_dir = params_exec.pop("save_dir", None)
    hidden_dims = params_exec.pop("hidden_dims", [256, 128, 64])
    dropout = params_exec.pop("dropout", 0.3)
    learning_rate = params_exec.pop("learning_rate", 0.001)
    epochs = params_exec.pop("epochs", 50)
    batch_size = params_exec.pop("batch_size", 256)
    patience = params_exec.pop("patience", 10)
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 配列の初期化
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    models = []
    scalers = []
    
    # 3. Stratified K-Fold 交差検証ループ
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        # シード設定
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)
        
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # 前処理（標準化、One-Hotエンコード）
        X_tr_proc, X_va_proc, X_test_proc, scaler = _preprocess_data(X_tr, X_va, X_test)
        
        # Tensorに変換
        X_tr_tensor = torch.FloatTensor(X_tr_proc.values).to(device)
        y_tr_tensor = torch.FloatTensor(y_tr.values).to(device)
        X_va_tensor = torch.FloatTensor(X_va_proc.values).to(device)
        y_va_tensor = torch.FloatTensor(y_va.values).to(device)
        
        # DataLoaderの作成
        train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # モデルの作成
        input_dim = X_tr_proc.shape[1]
        model = NeuralNetClassifier(input_dim, hidden_dims, dropout).to(device)
        
        # 損失関数と最適化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Early Stoppingの準備
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # 学習ループ
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_va_tensor).squeeze()
                val_loss = criterion(val_outputs, y_va_tensor).item()
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Best モデルをロード
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 予測（確率）
        model.eval()
        with torch.no_grad():
            val_preds = model(X_va_tensor).squeeze().cpu().numpy()
            val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)
            oof_preds[val_idx] = val_preds
            
            # テストデータの予測
            if X_test is not None:
                X_test_tensor = torch.FloatTensor(X_test_proc.values).to(device)
                t_preds = model(X_test_tensor).squeeze().cpu().numpy()
                t_preds = np.clip(t_preds, 1e-15, 1.0 - 1e-15)
                test_preds += t_preds / n_splits
        
        # モデルとスケーラーを保存
        model = model.cpu()
        models.append(model)
        scalers.append(scaler)
    
    # 4. モデル保存処理
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for fold, (model, scaler) in enumerate(zip(models, scalers)):
            torch.save(model.state_dict(), save_path / f"nn_model_fold{fold}.pth")
            with open(save_path / f"nn_scaler_fold{fold}.pkl", "wb") as f:
                pickle.dump(scaler, f)
    
    # 5. 戻り値
    result_data = {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "models": models,
        "scalers": scalers,
    }
    
    return result_data, params