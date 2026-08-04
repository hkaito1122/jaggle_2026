# common/utils/metrics.py
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score


def calculate_logloss(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15
) -> float:
    """Log Loss（対数損失）を計算する関数"""
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return float(log_loss(y_true, y_pred_clipped))


def calculate_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """AUC (Area Under the ROC Curve) を計算"""
    return float(roc_auc_score(y_true, y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE (Root Mean Squared Error) を計算"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# -------------------------------------------------------------------
# 各モデル用のカスタム評価指標 (Custom Eval Metrics)
# -------------------------------------------------------------------


def lgb_logloss_eval(
    preds: np.ndarray, train_data
) -> Tuple[str, float, bool]:
    """LightGBM用 Log Loss カスタム評価関数"""
    labels = train_data.get_label()
    score = calculate_logloss(labels, preds)
    return "custom_logloss", score, False  # 小さいほど良いので False


def xgb_logloss_eval(
    preds: np.ndarray, dtrain
) -> Tuple[str, float]:
    """XGBoost用 Log Loss カスタム評価関数"""
    labels = dtrain.get_label()
    score = calculate_logloss(labels, preds)
    return "custom_logloss", score


class CatBoostLoglossMetric(object):
    """CatBoost用 Log Loss カスタム評価指標クラス"""

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        # Log Loss は値が「小さいほど良い」ため False を返す
        return False

    def evaluate(self, approxes, target, weight):
        """CatBoostの内部評価ループ関数"""
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]
        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            # シグモイド関数で確率値 (0~1) に変換
            p = 1.0 / (1.0 + np.exp(-approx[i]))
            p = np.clip(p, 1e-15, 1.0 - 1e-15)
            w = 1.0 if weight is None else weight[i]

            y = target[i]
            # Log Loss 計算
            loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

            error_sum += loss * w
            weight_sum += w

        return error_sum, weight_sum