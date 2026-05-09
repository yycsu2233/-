import  numpy as np
import pandas as pd
import os
import sys
import math
from scipy import interpolate
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(CODE_DIR, "features_cleaned.pkl")  # 清洗后的特征数据
LABELS_PATH = os.path.join(CODE_DIR, "labels_cleaned.pkl")       # FeO标签数据
X_WINDOWS_PATH = os.path.join(CODE_DIR, "X_windows.npy")        # 窗口化特征（numpy格式）
Y_LABELS_PATH = os.path.join(CODE_DIR, "Y_labels.npy")          # FeO标签（numpy格式）
# 读取 预处理后的数据
def load_preprocessed_data():
    features_df = pd.read_pickle(FEATURES_PATH)  # 读取特征数据
    return features_df
# 定义温度测点列表，构建温度曲线的位置
#注意这个曲线采取的是注入温度，因为上下真空室的数据并不全，因此舍此前后温度差值采取注入温度，构建注入温度变化的温度曲线，其余温度测点当作全局变量输入模型，提供更多工艺信息
TEMP_COLUMNS =[
    #--------鼓干段--------
    '鼓干段烟罩温度',
    #--------预热段--------
    '烟罩预热温度西1', '烟罩预热温度西2', '烟罩预热温度西3', '烟罩预热温度西4',
    '烟罩预热温度西5', '烟罩预热温度西6', '烟罩预热温度西7',
    '烟罩预热温度东1', '烟罩预热温度东2', '烟罩预热温度东3', '烟罩预热温度东4',
    '烟罩预热温度东5', '烟罩预热温度东6', '烟罩预热温度东7',
    #--------焙烧段--------
    '焙烧烟罩温度1', '焙烧烟罩温度2', '焙烧烟罩温度3', '焙烧烟罩温度5',
    #--------冷却段--------
    '一冷段温度1', '一冷段温度2', '一冷段温度3',
    '二冷段烟罩温度（左148）', '二冷段烟罩温度（右152）',
]
TEMP_POSITIONS = {
    # ===== 鼓风干燥段 =====
    # 位置 0.02：刚进料就开始鼓风干燥
    '鼓干段烟罩温度':                      0.02,

    # ===== 预热段（西侧7个） =====
    # 位置从 0.04 到 0.46，均匀分布
    # 数值越来越大 → 越靠近焙烧段，位置值越大
    '烟罩预热温度西1':                      0.04,
    '烟罩预热温度西2':                      0.10,
    '烟罩预热温度西3':                      0.16,
    '烟罩预热温度西4':                      0.22,
    '烟罩预热温度西5':                      0.28,
    '烟罩预热温度西6':                      0.34,
    '烟罩预热温度西7':                      0.40,

    # ===== 预热段（东侧7个） =====
    # 和西侧对称，相同位置 → 插值时会取平均
    '烟罩预热温度东1':                      0.04,
    '烟罩预热温度东2':                      0.10,
    '烟罩预热温度东3':                      0.16,
    '烟罩预热温度东4':                      0.22,
    '烟罩预热温度东5':                      0.28,
    '烟罩预热温度东6':                      0.34,
    '烟罩预热温度东7':                      0.40,

    # ===== 焙烧段（核心工艺段） =====
    # 最高温区，位置 0.55~0.68
    '焙烧烟罩温度1':                        0.55,
    '焙烧烟罩温度2':                        0.58,
    '焙烧烟罩温度3':                        0.62,
    '焙烧烟罩温度5':                        0.68,

    # ===== 一冷段 =====
    # 快速冷却区，位置 0.88~0.92
    '一冷段温度1':                          0.88,   # 冷却入口（约1022°C）
    '一冷段温度2':                          0.90,   # 冷却中段（约917°C）
    '一冷段温度3':                          0.92,   # 冷却出口（约682°C）

    # ===== 二冷段 =====
    # 继续冷却，位置 0.93（左右两侧都在同一位置）
    '二冷段烟罩温度（左148）':              0.93,   # 二冷左侧
    '二冷段烟罩温度（右152）':              0.93,   # 二冷右侧}
}
def load_preprocessed_data():
    features_df = pd.read_pickle(FEATURES_PATH)  # 读取特征数据
    return features_df
def load_window_data():
    X_windows = np.load(X_WINDOWS_PATH)  # 加载窗口化特征
    Y_labels = np.load(Y_LABELS_PATH)    # 加载标签数据
    return X_windows, Y_labels
def build_tem_curve(row,n_grid=30,kind='linear'):
    # row 是DataFrame的一行，包含所有温度测点的值
    raw_positions = []
    raw_temps = []
    used_columns = []
    for col in TEMP_COLUMNS:
        if col not in row or pd.isna(row[col]):
            continue  # 跳过缺失的测点
        val = row[col]
        if pd.isna(val):
            continue  # 跳过NaN值
        pos = TEMP_POSITIONS.get(col,None)
        if pos is None:
            continue  # 跳过未定义位置的测点
        raw_positions.append(pos)
        raw_temps.append(float(val))
        used_columns.append(col)
    # 按位置排序
    sorted_data = sorted(zip(raw_positions, raw_temps, used_columns), key=lambda x: x[0])
    
    # 解包：把排序后的元组拆回三个列表
    raw_positions = [x[0] for x in sorted_data]   # 排序后的位置
    raw_temps = [x[1] for x in sorted_data]       # 排序后的温度
    used_columns = [x[2] for x in sorted_data]    # 排序后的列名
    # 去东西重
    n_raw_points = len(raw_temps)
    dedup_positions = []    # 去重后的位置
    dedup_temps = []        # 去重后的温度（重复位置取平均）
    i = 0
    while i < len(raw_positions):
        # 找出所有和 raw_positions[i] 相同位置的索引
        j = i
        while j < len(raw_positions) and raw_positions[j] == raw_positions[i]:
            j += 1
        # raw_temps[i:j] 是同一位置的所有温度值
        # np.mean() 取平均值作为这个位置的"代表温度"
        dedup_positions.append(raw_positions[i])
        dedup_temps.append(np.mean(raw_temps[i:j]))
        i = j   # 跳到下一个不同位置

    raw_positions = dedup_positions
    raw_temps = dedup_temps
    #插值
    grid_positions = np.linspace(0.02, 0.93, n_grid)

    if len(raw_temps) < 2:
        grid_temps = np.full(n_grid, np.nan)   # np.full 创建全为某个值的数组
    else:
        # interpolate.interp1d：一维插值函数
        # 给定 (raw_positions, raw_temps) 这些散点
        # 它就能算出来任意位置上的温度值
        f_interp = interpolate.interp1d(
            raw_positions,          # 已知点的位置（输入）
            raw_temps,              # 已知点的温度（输出）
            kind=kind,              
            bounds_error=False,     
            fill_value='extrapolate'  # 超出范围时 → 向外延伸（外推）
        )
        grid_temps = f_interp(grid_positions).astype(np.float32)

    return grid_positions, grid_temps, n_raw_points, len(raw_temps), used_columns
# grid_positions是曲线横坐标，grid_temps是曲线纵坐标，n_raw_points是原始测点数量，len(raw_temps)是去重后的测点数量，used_columns是参与插值的测点列名列表

def split_windows_to_model_inputs(X_windows, feature_columns, n_grid=30):

    if X_windows.ndim != 3:
        raise ValueError(f"X_windows 应该是三维数组 (N,6,F)，但现在是 {X_windows.shape}")

    N, T, F = X_windows.shape
    if len(feature_columns) != F:
        raise ValueError(
            f"列名数量和 X_windows 特征数不一致：len(feature_columns)={len(feature_columns)}, F={F}\n"
            f"请确认 X_windows.npy 是否由当前 features_cleaned.pkl 生成。"
        )

    # 找到真实存在于数据中的温度列
    temp_columns_existing = [c for c in TEMP_COLUMNS if c in feature_columns]
    temp_columns_set = set(temp_columns_existing)

    # 全局变量 = 所有列 - 温度曲线列
    global_columns = [c for c in feature_columns if c not in temp_columns_set]
    global_indices = [feature_columns.index(c) for c in global_columns]

    # 取出全局变量窗口
    global_windows = X_windows[:, :, global_indices].astype(np.float32)

    # 构建温度曲线窗口
    # temp_curves 的含义：每个样本有6小时，每小时1条曲线，每条曲线30个格点
    temp_curves = np.zeros((N, T, 1, n_grid), dtype=np.float32)

    print(f"[拆分] 温度测点列数: {len(temp_columns_existing)}")
    print(f"[拆分] 全局变量列数: {len(global_columns)}")
    print("[拆分] 正在把每个6小时窗口转换成温度曲线，这一步可能需要一点时间...")

    # 为了 build_tem_curve 能通过列名取值，这里把每个小时的数据临时转成 Series
    for i in range(N):
        for t in range(T):
            row = pd.Series(X_windows[i, t, :], index=feature_columns)
            _, grid_temps, _, _, _ = build_tem_curve(row, n_grid=n_grid)

            # 如果极少数情况下插值失败产生 NaN，这里用该曲线均值填补。
            # 如果整条曲线都是 NaN，就用 0 填补，避免神经网络报错。
            if np.isnan(grid_temps).any():
                if np.isnan(grid_temps).all():
                    grid_temps = np.zeros(n_grid, dtype=np.float32)
                else:
                    mean_val = np.nanmean(grid_temps)
                    grid_temps = np.nan_to_num(grid_temps, nan=mean_val).astype(np.float32)

            temp_curves[i, t, 0, :] = grid_temps

    print(f"[拆分] temp_curves 形状   : {temp_curves.shape}")
    print(f"[拆分] global_windows 形状: {global_windows.shape}")

    return temp_curves, global_windows, global_columns



try:
    import torch # 主库
    import torch.nn as nn # 神经网络模块，包含各种层和损失函数等
    import torch.optim as optim
    import torch.nn.functional as F # 常用函数库 （如Relu,SoftMax） 
    import math
    HAS_TORCH = True
except ImportError:

    HAS_TORCH = False
    # 这里定义一个空的，只是为了能通过语法检查
    class nn:
        class Module:
            pass    # pass 表示"啥也不做"
    class F:
        pass        # 同样是个空壳子
if HAS_TORCH:
    class TempCurveEncoder(nn.Module): 
        def __init__(self,channels=64, n_grid = 30,n_heads=4,dropout=0.3):
            super().__init__() # 调用父类的构造函数，初始化 nn.Module 的内部机制，规定写法
            self.conv1 = nn.Conv1d(1,16,kernel_size=3,padding=1)  # 输入通道数=1（单条曲线），输出通道数=16，卷积核大小=3，padding=1保持长度不变,补充左右两边一个零
            self.bn1 = nn.BatchNorm1d(16)  # 批归一化层，稳定训练 16是卷积层输出的通道数
            self.conv2 = nn.Conv1d(16,32,kernel_size=3,padding=1) # 第二层卷积，输入通道数=16，输出通道数=32，卷积核大小=3，padding=1
            self.bn2 = nn.BatchNorm1d(32)  # 第二层批归一化层
            self.conv3 = nn.Conv1d(32,channels,kernel_size=3,padding=1) # 第三层卷积，输入通道数=32，输出通道数=channels，卷积核大小=3，padding=1
            self.bn3 = nn.BatchNorm1d(channels)  # 第三层批归一化层
            #下面构建多头注意力机制 
            #（B，L,C） B是batch size，L是序列长度（格点数量），C是特征维度（channels）
            self.attention = nn.MultiheadAttention(
                ATTN_channels=channels,  # 每个格点的特征维度
                num_heads=n_heads,    # 注意力头数
                dropout=dropout,     # 注意力权重的随机失活率
                batch_first=True      # 输入输出的batch维在第一维
            )
            self.layer_norm = nn.LayerNorm(channels)  
            self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # 输出大小为1，表示把整个序列的特征压缩成一个固定长度的向量
            #构建全连接层，共64维特征
            self.fc1 = nn.Linear(channels,32)# 64维压缩32
            self.fc2 = nn.Linear(32,channels) # 最后输出64 维的嵌入向量，和卷积层输出的维度一致，方便后续拼接
            self.dropout = nn.Dropout(dropout) # 随机失活层，防止过拟合
            self._init_weights() # 初始化权重，调用下面定义的函数
        def _init_weights(self):
            #权重初始化函数，使用Kaiming正态分布初始化卷积层的权重，偏置初始化为0
            for m in self.modules(): # 遍历模型中的所有子模块
                if isinstance(m,nn.Conv1d): # 如果是卷积层
                    nn.init.kaiming_normal_(m.weight,nonlinearity='relu') # Kaiming正态分布初始化权重，适用于ReLU激活函数
                elif isinstance(m,nn.Linear): # 如果是全连接层
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu') # 同样使用Kaiming正态分布初始化全连接层的权重
                    if m.bias is not None:
                        nn.init.zeros_(m.bias) # 偏置初始化为0
        def forward(self,x):
            # x: (B,1,30)
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x) # 卷积层1 + 批归一化 + ReLU激活函数
            x = self.conv2(x)
            x= self.bn2(x)
            x = F.relu(x) # 卷积层2 + 批归一化 + ReLU激活函数
            x = self.conv3(x)   
            x = self.bn3(x)
            x = F.relu(x) # 卷积层3 + 批归一化 + ReLU激活函数
            #==== self-Attention机制 ====
            """
            卷积后的形状是（B,64,30）
            需要把它转成（B,30,64）才能送入多头注意力机制，因为多头注意力机制要求输入的形状是（B,L,C），其中L是序列长度（格点数量），C是特征维度（channels）
            可以用permute函数来交换维度，x.permute(0,2,1)表示把第0维（batch size）保持不变，把第1维（通道数）和第2维（格点数量）交换位置
            """
            x_attn_input = x.permute(0,2,1) # 转置成（B,30,64）
            x_attn,attn_weights = self.attention(
                x_attn_input, # query
                x_attn_input, # key
                x_attn_input,  # value
                need_weights=True,
                average_attn_weights=False  # ← 加这个才能拿到 (B, n_heads, L, L)
            )
            # attn_weights 的形状：(B, 4, 30, 30)
            # 4=注意力头数，30×30=位置之间的注意力得分
            x_attn += x.permute(0,2,1) # 残差连接，把卷积后的特征加回注意力输出
            x_attxzn = self.layer_norm(x_attn) # 层归一化，稳定训练 （需要在 __init__ 中定义 self.layer_norm = nn.LayerNorm(channels)）
            x = x_attn.permute(0,2,1) # 转回（B,64,30）

            #自适应权重
            x = self.adaptive_pool(x) # 输出形状（B,64,1）
            x = x.squeeze(-1) # 去掉最后一个维度，变成（B,64）

            #全连接层
            x = self.fc1(x) # 全连接层1，输出形状（B,32）
            x = F.relu(x) # ReLU激活函数
            x = self.dropout(x) # 随机失活，防止过拟合
            embed = self.fc2(x) # 全连接层2，输出形状（B,channels）

            return embed,attn_weights # 返回编码后的特征和注意力权重，注意力权重可以用来分析模型关注了哪些位置的温度变化
    def extract_window_stats_torch(x):
        # x: (B,1,30)
        mean = x.mean(dim = 1)
        std = x.std(dim = 1)
        max_val = x.max(dim = 1).values
        min_val = x.min(dim = 1).values
        last = x[:,-1.,:] # z最后一个格点的温度值            
        slope = x[:,-1 ,:]  - x[:,0,:] # 最后一个格点温度 - 第一个格点温度，表示整个曲线的斜率
        stat = torch.cat([mean,std,max_val,min_val,last,slope],dim=1) # 把这些统计特征拼接成一个向量，形状（B,6）
        return stat
    class FCResBlock(nn.Module):
        #这个是统计特征的全连接残差块，根据全局特征，由于工况是稳定的，所以这样可以保持模型的稳定性，避免过拟合
        def __init__(self,dim =128, dropout = 0.3):
            super().__init__() # 标准用法
            self.fc1 = nn.Linear(dim,dim) 
            self.bn1 = nn.BatchNorm1d(dim)
            self.fc2 = nn.Linear(dim,dim)
            self.bn2 = nn.BatchNorm1d(dim)
            self.dropout = nn.Dropout(dropout)
        def forward(self,x):
            idenfity = x # 残差连接的输入
            out = self.fc1(x)
            out = self.bn1(out)
            out = F.relu(out)
            out = self.fc2(out)
            out = self.bn2(out)
            out += idenfity # 残差连接，把输入加回输出
            out = F.relu(out)
            return out
    class GlobalStatBranch(nn.Module):
        # 全局变量统计分支    输入global_window(B,6,F_global)
        def __init__(self,global_dim,hidden_dim=128,out_dim=64,dropout=0.3):
            super().__init__()
            self.input_layer = nn.Sequential(
                nn.Linear(global_dim * 6,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            # 搭建残差快
            self.block1 = FCResBlock(dim=hidden_dim,dropout=dropout)
            self.block2 = FCResBlock(dim=hidden_dim,dropout=dropout)
            self.output_layer = nn.Linear(hidden_dim,out_dim)

        def forward(self,global_window):
            stat = extract_window_stats_torch(global_window) # 提取统计特征，形状（B,6）
            h = self.input_layer(stat) # 输入层，形状（B,hidden_dim）
            h = self.block1(h) # 残差块1，形状（B,hidden_dim）
            h = self.block2(h) # 残差块2，形状（B,hidden_dim）
            stat_embed = self.output_layer(h) # 输出层，形状（B,out_dim） 全局变量嵌入向量
            return stat_embed
    class TemporalResBlock(nn.Module):
        #这是6hour时间维度的残差快，输入是（B,channels），输出也是（B,channels），通过全连接层和残差连接来建模时间维度的特征交互
        def __init__(self,channels=64,dropout=0.3):
            super().__init__()
            self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(channels)
            self.dropout = nn.Dropout(dropout)
        def forward(self,x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += identity # 残差连接，把输入加回输出
            out = F.relu(out)
            return out
    class Temperature_Inject_TemporalBackbone(nn.Module):
        #温度注入全局温度残差块
        def __init__(self,global_dim, hidden_dim = 64,dropout =0.3):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Conv1d(global_dim,hidden_dim,kernel_size=3, padding=1), # 1x1卷积把全局变量的维度投影到hidden_dim，方便后续融合
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.block1 = TemporalResBlock(hidden_dim,dropout)
            self.block2 = TemporalResBlock(hidden_dim,dropout)
            self.block3 = TemporalResBlock(hidden_dim,dropout)
            
            self.alpha = nn.Parameter(torch.tensor(0.1)) # 可学习的温度注入权重，初始值0.1，训练过程中会自动调整
            self.pool = nn.AdaptiveAvgPool1d(1) # 全局平均池化，把时间维度压缩成1，得到全局特征         
            def forward(self,global_window, temp_seq):
                #global_window: (B,6,F_global) 全局变量窗口
                #temp_seq: (B,channels,6) 6小时的温度特征序列
                x = global_window.permute(0,2,1) # 转置成（B,F_global,6）
                h = self.input_proj(x) # 输入投影，形状（B,hidden_dim,6）
                h = self.block1(h) # 残差块1，形状（

                h = h + self.alpha * temp_seq # 温度注入，把温度特征乘以权重加到全局特征上，形状（B,hidden_dim,6）
                h = self.block2(h) # 残差块2，形状（B,hidden_dim,6）
                h = self.block3(h) # 残差块3，形状（B,hidden_dim,6）

                fused_temp_embed = self.pool(h).squeeze(-1) # 全局平均池化，得到融合后的温度嵌入向量，形状（B,hidden_dim）
                return fused_temp_embed
    class Temperature_Inject_DualBranchResNet(nn.Module):
        #这是最终的温度注入残差网络分支，输入是温度曲线特征和全局变量特征，输出是融合后的温度嵌入向量
        """
        输入 ：global_window(B,6,F_global) 和 temp_curves(B,1,30)
        输出 ：pred(B,1) FeO预测值
              attn 权重 (B, n_heads, 30, 30) 可以用来分析模型关注了哪些位置的温度变化
        模型结构:
            1. 先用 TempCurveEncoder 编码温度曲线，得到 temp_embed (B,channels) 和 attn_weights (B, n_heads, 30, 30)
            2. 用 GlobalStatBranch 编码全局变量，得到 stat_embed (B,64)
            3. Temperature_Inject_TemporalBackbone 温度注入主干
            4. PreHead 预测头
        """    
        def __init__(self,global_dim,embed_dim=64,n_grid = 30, dropout=0.3):
            super().__init__()
            self.temp_encoder = TempCurveEncoder(
                channels=embed_dim,
                n_grid=n_grid,
                n_heads=4,
                dropout=dropout
            )
            self.temporal_backbone = Temperature_Inject_TemporalBackbone(
                global_dim=global_dim,
                hidden_dim=embed_dim,
                dropout=dropout
            )
            self.stat_branch = GlobalStatBranch(
                global_dim=global_dim,
                hidden_dim=embed_dim,
                out_dim=embed_dim,
                dropout=dropout
                #(self,global_dim,hidden_dim=128,out_dim=64,dropout=0.3)
            )
            self.pre_head = nn.Sequential(
                nn.Linear(embed_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),# 我不确定可不可以再用一个dropout，反正试试，毕竟数据量不大，过拟合风险比较高
                nn.Linear(32, 1) # 最后输出一个FeO预测值
            )
        def forward(self,global_window,temp_curve):
            # global_window: (B,6,F_global) 全局变量窗口
            # temp_curve: (B,1,30) 温度曲线输入
            B,T,C,L = temp_curve.shape
            temp_input = temp_curve.reshape(B*T,C,L) # (B*6,1,30)
            temp_embed, attn_weights = self.temp_encoder(temp_input) # 编码温度曲线，得到嵌入向量和注意力权重，temp_embed (B*6,channels)

            # 还原成(B,6,channels)，方便后续处理
            temp_embed = temp_embed.reshape(B,T,-1).permute(0,2,1) # (B,channels,6)
            
            #温度注入主骨干路径输入
            fused_temp_embed = self.temporal_backbone(
                global_window=global_window,
                temp_seq=temp_embed
            )
            # 全局统计全局特征 
            stat_embed = self.stat_branch(global_window) # 编码全局变量，得到统计特征嵌入向量，形状（B,embed_dim）
            #两路融合
            combined = torch.cat([fused_temp_embed, stat_embed], dim=1) # 拼接温度嵌入和统计特征嵌入，形状（B,embed_dim*2）
            pred = self.pre_head(combined) # 预测头，输出FeO预测值，形状（B,1）
            return pred, attn_weights


else:
    # class TempCurveEncoder:
    #     def __init__(self, *args, **kwargs):
    #         raise ImportError("PyTorch is not installed. Please install PyTorch to use TempCurveEncoder.")
    #         return None

if __name__ == '__main__':