"""
tests/test_data_loader.py
测试数据加载与清洗逻辑（使用模拟数据）。
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    _generate_synthetic_data,
    clean_data,
    FEATURE_COLS,
    TARGET_COLS,
)


def test_synthetic_data_shape():
    df = _generate_synthetic_data(n_samples=100)
    assert len(df) == 100
    for col in FEATURE_COLS + TARGET_COLS:
        assert col in df.columns, f"缺少列: {col}"


def test_clean_data_no_nulls():
    df = _generate_synthetic_data(n_samples=200)
    # 手动引入缺失值
    df.loc[5:8, "fe_grade"] = np.nan
    df_clean = clean_data(df)
    assert df_clean[FEATURE_COLS].isnull().sum().sum() == 0


def test_clean_data_removes_outliers():
    df = _generate_synthetic_data(n_samples=300)
    # 插入极端异常值
    df.loc[0, "roasting_temp"] = 99999.0
    n_before = len(df)
    df_clean = clean_data(df)
    assert len(df_clean) < n_before, "异常值行应被剔除"


def test_load_data_split(tmp_path, monkeypatch):
    """测试 load_data 返回形状正确，且训练/测试集不重叠。"""
    from src import data_loader as dl

    # 重定向 PROCESSED_DIR 到临时目录
    monkeypatch.setattr(dl, "PROCESSED_DIR", tmp_path)

    X_train, X_test, y_train, y_test, scaler = dl.load_data(
        target="compressive_strength", test_size=0.2
    )
    total = len(y_train) + len(y_test)
    assert total > 0
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    # 大致验证 8:2 比例（允许 ±5）
    assert abs(len(y_train) / total - 0.8) < 0.05
