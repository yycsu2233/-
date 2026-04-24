# 球团矿软测量模型 (Pellet Ore Soft Sensor Model)

> 大二大创项目 · 矿物加工专业 · 2025–2026

## 项目简介

球团矿是钢铁冶金的重要原料，其质量指标（如抗压强度、转鼓指数等）直接影响高炉冶炼效果。
传统检测方法滞后性强，本项目基于工艺过程数据构建**软测量模型**，实时预测球团矿关键质量指标，
为生产过程的在线优化提供数据支撑。

## 主要质量指标

| 指标 | 说明 |
|------|------|
| 抗压强度 (Compressive Strength) | 成品球团的机械强度，单位 N/个 |
| 转鼓指数 (Tumbling Index, TI) | 球团耐磨强度，单位 % |
| 还原度指数 (Reduction Index, RI) | 球团在高炉中的还原性能，单位 % |

## 主要输入变量 (过程参数)

- 铁精矿品位、细度（-0.074 mm 占比）
- 膨润土配比
- 碱度（CaO/SiO₂）
- 焙烧温度、焙烧时间
- 链篦机各段温度、风量
- 回转窑转速、煤粉量

## 技术路线

```
原始数据  →  数据清洗  →  特征工程  →  模型训练  →  在线预测
               ↑                          ↑
          异常值剔除              PLS / SVR / LSTM 对比
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 数据预处理（将原始 CSV 放入 data/raw/）
python src/data_loader.py

# 3. 训练模型
python src/train.py --model svr --target compressive_strength

# 4. 评估 & 可视化（Jupyter Notebook）
jupyter notebook notebooks/01_exploration.ipynb
```

## 项目结构

```
.
├── data/
│   ├── raw/          # 原始工艺数据（不上传至 Git）
│   └── processed/    # 预处理后数据
├── notebooks/        # 探索性分析
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   └── train.py
├── PROGRESS.md       # 每日进展记录
├── requirements.txt
└── README.md
```

## 进展日志

详见 [PROGRESS.md](PROGRESS.md)

## 参考资料

- 《球团矿生产技术》— 刘汉光
- Ljung, L. (1999). *System Identification: Theory for the User*
- Scikit-learn documentation: https://scikit-learn.org
