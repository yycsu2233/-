"""
features.py
-----------
特征工程模块：在原始过程变量基础上衍生新特征，提升模型精度。

主要操作：
  - 滑动窗口统计特征（均值、标准差）
  - 比值/交互特征
  - 多项式特征（可选）
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def add_rolling_features(
    df: pd.DataFrame,
    cols: list,
    windows: list = (3, 6),
) -> pd.DataFrame:
    """
    为指定列添加滑动窗口均值和标准差特征。

    Parameters
    ----------
    df : pd.DataFrame
        输入 DataFrame（须已按时间排序）
    cols : list of str
        需要计算滑动统计的列名
    windows : tuple of int
        滑动窗口大小（单位：行）

    Returns
    -------
    pd.DataFrame
        添加新列后的 DataFrame（原列不删除）
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_mean{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_std{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加领域知识驱动的交互/比值特征。
    """
    df = df.copy()
    if "roasting_temp" in df.columns and "roasting_time" in df.columns:
        # 热输入量近似（温度 × 时间）
        df["heat_input"] = df["roasting_temp"] * df["roasting_time"]
    if "grate_temp_1" in df.columns and "grate_temp_3" in df.columns:
        # 链篦机温度梯度
        df["grate_temp_diff"] = df["grate_temp_1"] - df["grate_temp_3"]
    if "fe_grade" in df.columns and "fineness" in df.columns:
        # 铁矿品质综合指数
        df["ore_quality"] = df["fe_grade"] * df["fineness"] / 100.0
    return df


def add_polynomial_features(
    X: np.ndarray, degree: int = 2, interaction_only: bool = True
) -> np.ndarray:
    """
    生成多项式交互特征（默认只生成交叉项，不生成高次项，避免维度爆炸）。

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    degree : int
    interaction_only : bool
        True 时只保留交叉项

    Returns
    -------
    np.ndarray
        扩展后的特征矩阵
    """
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    return poly.fit_transform(X)


def build_features(df: pd.DataFrame, use_rolling: bool = True, use_interaction: bool = True) -> pd.DataFrame:
    """
    完整特征工程流水线。

    Parameters
    ----------
    df : pd.DataFrame
        清洗后的原始 DataFrame
    use_rolling : bool
        是否添加滑动窗口特征
    use_interaction : bool
        是否添加交互特征

    Returns
    -------
    pd.DataFrame
        含所有新特征的 DataFrame
    """
    from src.data_loader import FEATURE_COLS  # 避免循环导入

    if use_rolling:
        df = add_rolling_features(df, cols=FEATURE_COLS)
    if use_interaction:
        df = add_interaction_features(df)
    return df
