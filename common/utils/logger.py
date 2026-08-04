# common/utils/logger.py
import logging
import sys
from pathlib import Path


def get_logger(script_name: str, log_dir: str = "../logs") -> logging.Logger:
    """コンソールとログファイル（logs/配下）の両方に同時に出力するロガーを作成"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        log_path / f"{script_name}.log", encoding="utf-8"
    )
    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # 二重出力防止のための重複チェック
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger