# JAGGLE_2026

データ分析・コンペティション用のリポジトリです。  
実行プログラム（`src/`）と提出ファイル（`data/output/`）を1対1で対応させて管理し、実験の再現性を担保する構成になっています。

---

## 📁 ディレクトリ構造

JAGGLE_2026/
├── common/                     # 学習・予測・共通ユーティリティ
│   ├── catboost/
│   │   ├── cat_model.py        # CatBoost用実行モジュール（通常学習）
│   │   └── cat_model_optuna.py # CatBoost用実行モジュール（Optunaチューニング）
│   ├── lgbm/
│   │   ├── lgbm_model.py       # LightGBM用実行モジュール（通常学習）
│   │   └── lgbm_model_optuna.py# LightGBM用実行モジュール（Optunaチューニング）
│   ├── utils/                  # 共通ユーティリティ
│   │   ├── logger.py           # 実行ログ出力・保存処理
│   │   ├── metrics.py          # カスタム評価指標・スコア計算
│   │   └── seed.py             # 乱数シード一括固定処理
│   └── xgboost/
│       ├── xgb_model.py        # XGBoost用実行モジュール（通常学習）
│       └── xgb_model_optuna.py # XGBoost用実行モジュール（Optunaチューニング）
│
├── data/                       # データ格納ディレクトリ
│   ├── input/                  # 元データ（train.csv, test.csv など ※Git対象外）
│   └── output/
│       └── yyyymmdd/           # 日付ごとの提出用ファイル格納先
│           └── yyyymmdd_vX_XXXXXX.csv
│
├── logs/                       # 実験実行ログの保存先（*.log）
│
├── notebooks/                  # 探索的データ分析（EDA）用
│   └── eda/
│       └── eda.ipynb
│
├── saved_models/               # 学習済みモデルの保存先（.pkl / .bin など）
│
├── src/                        # 実験用実行スクリプト（1対1管理のコア）
│   ├── 01_baseline.py
│   └── 02_lgbm_optuna.py
│
└── README.md                   # 本ファイル（プロジェクト概要・管理ノート）