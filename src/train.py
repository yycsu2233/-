"""
train.py
--------
训练入口脚本。

使用示例：
  python src/train.py --model svr --target compressive_strength
  python src/train.py --model pls --target tumbling_index --pls-n-components 8
  python src/train.py --model lstm --target compressive_strength --lstm-epochs 100
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 保证从项目根目录运行时可正确导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data
from src.models import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 评估函数
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算常用回归指标。"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R²": r2}


def plot_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """绘制真实值 vs 预测值散点图及趋势对比图，并保存。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot (真实值 vs 预测值)
    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.2, label="Ideal")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"{title} — Scatter")
    axes[0].legend()

    # Trend comparison (趋势对比)
    idx = np.arange(len(y_true))
    axes[1].plot(idx, y_true, label="Actual", linewidth=1.2)
    axes[1].plot(idx, y_pred, label="Predicted", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Value")
    axes[1].set_title(f"{title} — Trend")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("预测图已保存至 %s", save_path)


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    logger.info("=== 开始训练 | 模型: %s | 目标变量: %s ===", args.model, args.target)

    # 1. 加载数据
    X_train, X_test, y_train, y_test, scaler = load_data(
        target=args.target,
        test_size=args.test_size,
        random_state=args.seed,
    )
    logger.info("训练集: %s  测试集: %s", X_train.shape, X_test.shape)

    # 2. 构建模型
    model_kwargs: dict = {}
    if args.model == "pls":
        model_kwargs["n_components"] = args.pls_n_components
    elif args.model == "lstm":
        model_kwargs.update(
            {
                "input_size": X_train.shape[1],
                "hidden_size": args.lstm_hidden,
                "num_layers": args.lstm_layers,
                "seq_len": args.lstm_seq_len,
                "lr": args.lstm_lr,
                "epochs": args.lstm_epochs,
                "batch_size": args.lstm_batch,
            }
        )

    model = get_model(args.model, **model_kwargs)

    # 3. 训练
    model.fit(X_train, y_train)

    # 4. 预测 & 评估
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    if args.model == "lstm":
        # LSTM 输出长度比输入短 seq_len 个
        y_train = y_train[args.lstm_seq_len :]
        y_test = y_test[args.lstm_seq_len :]

    train_metrics = evaluate(y_train, y_pred_train)
    test_metrics = evaluate(y_test, y_pred_test)

    logger.info("--- 训练集指标 ---")
    for k, v in train_metrics.items():
        logger.info("  %s: %.4f", k, v)
    logger.info("--- 测试集指标 ---")
    for k, v in test_metrics.items():
        logger.info("  %s: %.4f", k, v)

    # 5. 保存模型
    model_path = LOGS_DIR / f"{args.model}_{args.target}.pkl"
    if args.model == "lstm":
        model_path = LOGS_DIR / f"{args.model}_{args.target}.pt"
    model.save(model_path)

    # 6. 保存预测图
    plot_prediction(
        y_test,
        y_pred_test,
        title=f"{args.model.upper()} — {args.target}",
        save_path=LOGS_DIR / f"{args.model}_{args.target}_pred.png",
    )

    logger.info("=== 训练完成 ===")


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="球团矿软测量模型训练脚本")
    parser.add_argument("--model", choices=["pls", "svr", "lstm"], default="svr")
    parser.add_argument(
        "--target",
        choices=["compressive_strength", "tumbling_index", "reduction_index"],
        default="compressive_strength",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # PLS 参数
    parser.add_argument("--pls-n-components", type=int, default=6)

    # LSTM 参数
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-seq-len", type=int, default=10)
    parser.add_argument("--lstm-lr", type=float, default=1e-3)
    parser.add_argument("--lstm-epochs", type=int, default=50)
    parser.add_argument("--lstm-batch", type=int, default=32)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
