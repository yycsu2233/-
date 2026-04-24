"""
data_loader.py
--------------
加载并清洗球团矿工艺数据。

期望的原始 CSV 格式（data/raw/*.csv）：
  timestamp, fe_grade, fineness, bentonite_ratio, basicity,
  roasting_temp, roasting_time, grate_temp_1, grate_temp_2,
  grate_temp_3, kiln_speed, coal_rate,
  compressive_strength, tumbling_index, reduction_index

使用示例：
  from src.data_loader import load_data
  X_train, X_test, y_train, y_test = load_data(target="compressive_strength")
"""

import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 列名定义
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "fe_grade",        # 铁精矿品位 (%)
    "fineness",        # 细度，-0.074 mm 占比 (%)
    "bentonite_ratio", # 膨润土配比 (%)
    "basicity",        # 碱度 CaO/SiO₂
    "roasting_temp",   # 焙烧温度 (°C)
    "roasting_time",   # 焙烧时间 (min)
    "grate_temp_1",    # 链篦机预热段温度 (°C)
    "grate_temp_2",    # 链篦机干燥段温度 (°C)
    "grate_temp_3",    # 链篦机抽风干燥段温度 (°C)
    "kiln_speed",      # 回转窑转速 (r/min)
    "coal_rate",       # 煤粉量 (t/h)
]

TARGET_COLS = [
    "compressive_strength",  # 抗压强度 (N/个)
    "tumbling_index",        # 转鼓指数 (%)
    "reduction_index",       # 还原度指数 (%)
]

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def load_raw_csv() -> pd.DataFrame:
    """
    从 data/raw/ 读取所有 CSV，合并为一个 DataFrame。
    如果目录下没有真实数据，则生成模拟数据用于开发调试。
    """
    csv_files = list(RAW_DIR.glob("*.csv"))
    if csv_files:
        frames = [pd.read_csv(f, parse_dates=["timestamp"]) for f in csv_files]
        df = pd.concat(frames, ignore_index=True)
        logger.info("已加载 %d 条原始记录，来自 %d 个文件。", len(df), len(csv_files))
    else:
        logger.warning("data/raw/ 下未找到 CSV，使用模拟数据（仅供开发调试）。")
        df = _generate_synthetic_data()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：
    1. 时间排序
    2. 删除全空列
    3. 线性插值填补少量缺失
    4. 基于 IQR 的异常值剔除（仅对特征列）
    """
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    df = df.dropna(axis=1, how="all")

    # 插值填补缺失（不超过 5 个连续缺失点）
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear", limit=5, limit_direction="both"
    )
    df = df.dropna(subset=[c for c in TARGET_COLS if c in df.columns])

    # IQR 剔除异常值行
    feature_cols_present = [c for c in FEATURE_COLS if c in df.columns]
    for col in feature_cols_present:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[df[col].between(q1 - 3 * iqr, q3 + 3 * iqr)]

    df = df.reset_index(drop=True)
    logger.info("清洗后剩余 %d 条记录。", len(df))
    return df


def load_data(
    target: str = "compressive_strength",
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> tuple:
    """
    完整数据加载流程，返回 (X_train, X_test, y_train, y_test, scaler)。

    Parameters
    ----------
    target : str
        目标变量名，可选 'compressive_strength' / 'tumbling_index' / 'reduction_index'
    test_size : float
        测试集比例
    random_state : int
        随机种子，保证可复现
    scale : bool
        是否对特征做 Z-score 标准化

    Returns
    -------
    X_train, X_test : np.ndarray
    y_train, y_test : np.ndarray
    scaler : StandardScaler 或 None
    """
    if target not in TARGET_COLS:
        raise ValueError(f"target 须为 {TARGET_COLS} 之一，当前为 '{target}'")

    df = load_raw_csv()
    df = clean_data(df)

    feature_cols_present = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols_present].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    return X_train, X_test, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# 模拟数据生成（无真实数据时供开发调试）
# ---------------------------------------------------------------------------

def _generate_synthetic_data(n_samples: int = 500, seed: int = 0) -> pd.DataFrame:
    """
    生成符合球团矿工艺参数范围的模拟数据。
    目标变量由线性组合加噪声合成，仅用于框架调试，不代表真实工艺规律。
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")

    fe_grade = rng.uniform(63, 68, n)
    fineness = rng.uniform(70, 90, n)
    bentonite_ratio = rng.uniform(1.5, 2.5, n)
    basicity = rng.uniform(0.8, 1.2, n)
    roasting_temp = rng.uniform(1260, 1320, n)
    roasting_time = rng.uniform(15, 25, n)
    grate_temp_1 = rng.uniform(900, 1050, n)
    grate_temp_2 = rng.uniform(800, 950, n)
    grate_temp_3 = rng.uniform(700, 850, n)
    kiln_speed = rng.uniform(1.0, 1.8, n)
    coal_rate = rng.uniform(8, 14, n)

    # 合成目标变量（仅模拟线性关系 + 噪声）
    compressive_strength = (
        2500
        + 30 * (fe_grade - 65)
        + 5 * (fineness - 80)
        - 80 * (bentonite_ratio - 2.0)
        + 20 * (roasting_temp - 1290) / 10
        + rng.normal(0, 50, n)
    )
    tumbling_index = (
        92
        + 0.1 * (fe_grade - 65)
        + 0.05 * (fineness - 80)
        + rng.normal(0, 0.5, n)
    )
    reduction_index = (
        65
        - 0.5 * (basicity - 1.0)
        + 0.2 * (roasting_temp - 1290) / 10
        + rng.normal(0, 1.0, n)
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "fe_grade": fe_grade,
            "fineness": fineness,
            "bentonite_ratio": bentonite_ratio,
            "basicity": basicity,
            "roasting_temp": roasting_temp,
            "roasting_time": roasting_time,
            "grate_temp_1": grate_temp_1,
            "grate_temp_2": grate_temp_2,
            "grate_temp_3": grate_temp_3,
            "kiln_speed": kiln_speed,
            "coal_rate": coal_rate,
            "compressive_strength": compressive_strength,
            "tumbling_index": tumbling_index,
            "reduction_index": reduction_index,
        }
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    X_train, X_test, y_train, y_test, scaler = load_data()
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
