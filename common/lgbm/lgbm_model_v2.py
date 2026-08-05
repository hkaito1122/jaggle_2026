import os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit


def _preprocess_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """object 型の列を category 型に変換する内部関数"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


def run_lgb(
    data: Dict[str, pd.DataFrame], params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """LightGBMの学習・交差検証（CV）および予測を機械的に実行する関数（v2: TimeSeriesSplit対応版）

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        'X_train': 学習用特徴量
        'y_train': 学習用ターゲット
        'X_test' : (任意) テスト用特徴量
        'sort_col': (任意) TimeSeriesSplitを使用する場合のソート列（入社日など）
    params : Dict[str, Any]
        LightGBMのハイパーパラメータおよび制御用パラメータ
        必須・推奨キー: 
        - 'n_splits': CV分割数
        - 'seed': 乱数シード
        - 'save_dir': モデル保存ディレクトリ
        - 'early_stopping_rounds': Early Stopping
        - 'cv_strategy': CV戦略 ('stratified' or 'timeseries')

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        1. result_data : 予測結果データを含む辞書 ('oof_preds', 'test_preds', 'models', 'scores')
        2. params      : 入力されたパラメータ（そのまま返却）
    """
    X_train = data.get("X_train")
    y_train = data.get("y_train")
    X_test = data.get("X_test", None)
    sort_col = data.get("sort_col", None)

    if X_train is None or y_train is None:
        raise ValueError(
            "data には 'X_train' と 'y_train' が含まれている必要があります。"
        )

    # 1. 制御用パラメータを params から取り出し (LightGBM本体に渡さないため pop)
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits")
    seed = params_exec.pop("seed")
    save_dir = params_exec.pop("save_dir", None)
    stopping_rounds = params_exec.pop("early_stopping_rounds", 50)
    verbose_eval = params_exec.pop("verbose_eval", False)
    cv_strategy = params_exec.pop("cv_strategy", "stratified")  # 'stratified' or 'timeseries'

    # ★追加: 二元分類（binary）かどうかの自動判定フラグ
    objective = params_exec.get("objective", "binary")
    is_binary = objective in ["binary", "binary_logloss"]

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
        # 入社日でソート
        sort_indices = X_train_proc[sort_col].argsort()
        X_train_proc = X_train_proc.iloc[sort_indices].reset_index(drop=True)
        y_train = y_train.iloc[sort_indices].reset_index(drop=True)
        
        # TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(cv.split(X_train_proc))
        # sort_colは特徴量から除外
        X_train_proc = X_train_proc.drop(columns=[sort_col])
    else:
        # StratifiedKFold
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X_train_proc, y_train))

    # 4. 配列の初期化
    oof_preds = np.zeros(len(X_train_proc))
    test_preds = np.zeros(len(X_test_proc)) if X_test_proc is not None else None
    models = []
    scores = []  # 各foldのバリデーションスコアを記録

    # 5. K-Fold 交差検証ループ
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X_train_proc.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train_proc.iloc[val_idx], y_train.iloc[val_idx]

        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

        # 残った params_exec をそのまま LightGBM に渡す
        model = lgb.train(
            params=params_exec,
            train_set=trn_data,
            valid_sets=[trn_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100 if verbose_eval else 0),
            ],
        )

        # --- 予測処理 ---
        val_preds = model.predict(X_va)

        # 二元分類（binary）の場合、確率を 1e-15 〜 1-1e-15 にクリッピング
        if is_binary:
            val_preds = np.clip(val_preds, 1e-15, 1.0 - 1e-15)

        oof_preds[val_idx] = val_preds

        # バリデーションスコアを計算（Log Loss）
        from sklearn.metrics import log_loss
        fold_score = log_loss(y_va, val_preds)
        scores.append(fold_score)
        
        if verbose_eval:
            print(f"Fold {fold+1}/{n_splits} - Log Loss: {fold_score:.6f}")

        # テストデータの予測（前処理済みの X_test_proc を使用）
        if X_test_proc is not None:
            t_preds = model.predict(X_test_proc)
            if is_binary:
                t_preds = np.clip(t_preds, 1e-15, 1.0 - 1e-15)
            test_preds += t_preds / n_splits

        models.append(model)

    # 6. OOFスコアの計算
    # TimeSeriesSplitではバリデーションに含まれないサンプル（初期のトレーニング専用データ）があるため
    # oof_predsがゼロのままの部分を除外してスコアを計算
    if cv_strategy == "timeseries":
        # バリデーションに含まれたサンプルのみでスコア計算
        valid_mask = oof_preds > 0
        oof_score = log_loss(y_train[valid_mask], oof_preds[valid_mask])
    else:
        # StratifiedKFoldでは全サンプルがバリデーションに含まれる
        oof_score = log_loss(y_train, oof_preds)
    
    if verbose_eval:
        print(f"\n{'='*50}")
        print(f"Overall OOF Log Loss: {oof_score:.6f}")
        print(f"Fold scores: {[f'{s:.6f}' for s in scores]}")
        print(f"Mean ± Std: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
        if cv_strategy == "timeseries":
            uncovered = (oof_preds == 0).sum()
            print(f"Note: {uncovered} samples not in validation (TimeSeriesSplit)")
        print(f"{'='*50}\n")

    # 7. モデル保存処理
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for fold, model in enumerate(models):
            model.save_model(str(save_path / f"lgb_model_fold{fold}.txt"))

    # 8. 戻り値
    result_data = {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "models": models,
        "scores": scores,
        "oof_score": oof_score,
    }

    return result_data, params