"""
models.py
---------
球团矿软测量模型集合。

提供三种模型：
  1. PLSModel     — 偏最小二乘回归（适合多重共线性场景，可解释性强）
  2. SVRModel     — 支持向量回归（小样本非线性建模）
  3. LSTMModel    — 长短时记忆网络（时序特征建模）

每种模型都封装了 fit / predict / save / load 接口，供 train.py 统一调用。
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "logs"


# ---------------------------------------------------------------------------
# 1. PLS 模型
# ---------------------------------------------------------------------------

class PLSModel:
    """偏最小二乘回归软测量模型。"""

    def __init__(self, n_components: int = 6):
        self.n_components = n_components
        self.model = PLSRegression(n_components=n_components, scale=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSModel":
        self.model.fit(X, y)
        logger.info("PLS 模型训练完成，潜变量数 = %d", self.n_components)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).ravel()

    def save(self, path: Path | None = None) -> Path:
        path = path or (MODEL_DIR / "pls_model.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("PLS 模型已保存至 %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "PLSModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# 2. SVR 模型
# ---------------------------------------------------------------------------

class SVRModel:
    """
    支持向量回归软测量模型。
    内部包含 StandardScaler，可直接接收未标准化的特征。
    """

    def __init__(self, kernel: str = "rbf", C: float = 10.0, epsilon: float = 0.1, gamma: str = "scale"):
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVRModel":
        self.pipeline.fit(X, y)
        logger.info("SVR 模型训练完成。")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def save(self, path: Path | None = None) -> Path:
        path = path or (MODEL_DIR / "svr_model.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("SVR 模型已保存至 %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "SVRModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# 3. LSTM 模型（PyTorch）
# ---------------------------------------------------------------------------

class LSTMModel:
    """
    基于 PyTorch 的 LSTM 软测量模型。
    将时序窗口内的过程变量序列映射到当前时刻的质量指标。

    Parameters
    ----------
    input_size  : 输入特征维度
    hidden_size : LSTM 隐藏单元数
    num_layers  : LSTM 层数
    seq_len     : 时序窗口长度（历史步数）
    lr          : 学习率
    epochs      : 训练轮数
    batch_size  : 小批量大小
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64,
        num_layers: int = 2,
        seq_len: int = 10,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._net = None

    # ------------------------------------------------------------------
    # 内部 PyTorch 网络定义（延迟导入，避免无 GPU 环境报错）
    # ------------------------------------------------------------------
    def _build_net(self):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                lstm_dropout = 0.2 if num_layers > 1 else 0.0
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self._net = _Net(self.input_size, self.hidden_size, self.num_layers)
        return self._net

    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        """将 (T, F) 矩阵切成 (N, seq_len, F) 的窗口序列。"""
        xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            xs.append(X[i - self.seq_len : i])
            ys.append(y[i])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMModel":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        net = self._build_net()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)

        xs, ys = self._make_sequences(X, y)
        dataset = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        net.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            avg_loss = epoch_loss / len(dataset)
            if epoch % 10 == 0:
                logger.info("LSTM Epoch %3d / %d  Loss: %.4f", epoch, self.epochs, avg_loss)

        self._net = net
        self._device = device
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")
        xs, _ = self._make_sequences(X, np.zeros(len(X)))
        self._net.eval()
        with torch.no_grad():
            preds = self._net(torch.from_numpy(xs).to(self._device)).cpu().numpy()
        return preds

    def save(self, path: Path | None = None) -> Path:
        import torch

        path = path or (MODEL_DIR / "lstm_model.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self._net.state_dict(), "config": self.__dict__},
            path,
        )
        logger.info("LSTM 模型已保存至 %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "LSTMModel":
        import torch

        checkpoint = torch.load(path, map_location="cpu")
        cfg = {k: v for k, v in checkpoint["config"].items() if not k.startswith("_")}
        obj = cls(**cfg)
        obj._build_net()
        obj._net.load_state_dict(checkpoint["state_dict"])
        obj._device = torch.device("cpu")
        return obj


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def get_model(name: str, **kwargs):
    """
    根据名称返回模型实例。

    Parameters
    ----------
    name : str
        'pls' | 'svr' | 'lstm'
    **kwargs
        传递给对应模型构造器的参数

    Returns
    -------
    PLSModel | SVRModel | LSTMModel
    """
    mapping = {
        "pls": PLSModel,
        "svr": SVRModel,
        "lstm": LSTMModel,
    }
    name = name.lower()
    if name not in mapping:
        raise ValueError(f"未知模型名称 '{name}'，可选: {list(mapping.keys())}")
    return mapping[name](**kwargs)
