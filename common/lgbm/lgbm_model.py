import os
from pathlib import Path
from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold  # 分類用に StratifiedKFold へ変更


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
    """LightGBMの学習・交差検証（CV）および予測を機械的に実行する関数

    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        'X_train': 学習用特徴量
        'y_train': 学習用ターゲット
        'X_test' : (任意) テスト用特徴量
    params : Dict[str, Any]
        LightGBMのハイパーパラメータおよび制御用パラメータ
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

    # 1. 制御用パラメータを params から取り出し (LightGBM本体に渡さないため pop)
    params_exec = params.copy()
    n_splits = params_exec.pop("n_splits")
    seed = params_exec.pop("seed")
    save_dir = params_exec.pop("save_dir", None)
    stopping_rounds = params_exec.pop("early_stopping_rounds", 50)
    verbose_eval = params_exec.pop("verbose_eval", False)

    # ★追加: 二元分類（binary）かどうかの自動判定フラグ
    objective = params_exec.get("objective", "binary")
    is_binary = objective in ["binary", "binary_logloss"]

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

    # 4. Stratified K-Fold 交差検証ループ
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_proc, y_train)):
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

        # テストデータの予測（前処理済みの X_test_proc を使用）
        if X_test_proc is not None:
            t_preds = model.predict(X_test_proc)
            if is_binary:
                t_preds = np.clip(t_preds, 1e-15, 1.0 - 1e-15)
            test_preds += t_preds / n_splits

        models.append(model)

    # 5. モデル保存処理
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for fold, model in enumerate(models):
            model.save_model(str(save_path / f"lgb_model_fold{fold}.txt"))

    # 6. 戻り値
    result_data = {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "models": models,
    }

    return result_data, params