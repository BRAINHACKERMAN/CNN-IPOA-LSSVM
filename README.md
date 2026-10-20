# CNN-IPOA-LSSVM：基于卷积特征提取与改进鹈鹕算法优化最小二乘支持向量机的混合智能建模框架






## 项目简介

**CNN-IPOA-LSSVM** 是一套面向复杂工业时序信号分析、状态监测、故障诊断与高精度回归预测的混合智能建模框架。本项目将**卷积神经网络（CNN）**强大的局部时空特征自适应提取能力、**改进鹈鹕优化算法（IPOA）**卓越的全局超参数寻优能力，以及**最小二乘支持向量机（LSSVM）**在小样本、高维非线性空间中严谨的结构风险最小化推演机制深度融合，有效解决了传统单一模型在特征表征不足、超参数盲目设定及局部极值陷阱等方面的瓶颈。

### 核心方法论架构

本项目采用三阶段级联协同建模范式：

| 阶段模块 | 核心组件 | 输入数据 | 输出结果 | 关键作用 |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1** | **CNN 特征提取与降维模块** | 原始时序信号 / 时频图谱 / 多维特征矩阵 | 低维高判别力深层特征向量 $F_{\text{CNN}}$ | 自适应挖掘局部空间相关性与多尺度深层表征，剔除冗余噪声 |
| **Step 2** | **IPOA 超参数全局寻优模块** | 训练集与验证集深层特征 $F_{\text{CNN}}$ | 最优超参数组合 $(C^*, \gamma^*)$ | 引入混沌初始化、非线性自适应权重与变异扰动，全局寻优 LSSVM 核参数与惩罚因子 |
| **Step 3** | **LSSVM 模式识别与回归预测模块** | 最优超参数 $(C^*, \gamma^*)$ + 深层特征 $F_{\text{CNN}}$ | 预测序列 / 故障类别标签 + 多维评估指标 | 将不等式约束转化为线性 KKT 方程组高效求解，实现高精度回归拟合与模式分类 |

## 项目目录结构

```text
CNN-IPOA-LSSVM/
├── data/                        # 数据集目录（原始时序数据、时频图谱或特征矩阵）
│   ├── raw_data.csv             # 原始传感器/监测数据集
│   └── processed_dataset.npz    # 归一化与滑窗切片后的标准张量数据
├── models/                      # 核心算法模块与权重保存目录
│   ├── cnn_extractor.py         # Step 1: CNN 深层特征提取与降维网络
│   ├── ipoa_optimizer.py        # Step 2: 改进鹈鹕优化算法（IPOA）寻优引擎
│   └── lssvm_model.py           # Step 3: 最小二乘支持向量机（LSSVM）求解器
├── utils/                       # 工具函数与可视化模块
│   ├── data_preprocessing.py    # 数据清洗、归一化与滑窗切片工具
│   ├── metrics.py               # 回归（R2/RMSE/MAE/MAPE）与分类（Acc/F1）指标计算
│   └── visualization.py         # 收敛曲线、混淆矩阵、拟合对比与残差绘图函数
├── results/                     # 实验输出、最优模型参数与可视化图表保存路径
├── main.py                      # 端到端训练、寻优与评估主程序入口
├── requirements.txt             # Python 依赖环境配置清单
└── README.md                    # 项目说明与代码功能解析文档
```

## Step 1: 特征提取与特征降维（CNN 模块）

### 1.1 输入数据格式与预处理方式

本框架支持三种标准工业与科研输入数据模态：
1. **一维多通道时序信号（1D Time-Series Signals）**：如振动加速度序列、电力负荷序列、声发射信号或气象监测序列，张量维度为 $\mathcal{X} \in \mathbb{R}^{N \times L \times D}$，其中 $N$ 为样本总数，$L$ 为时间窗口长度，$D$ 为传感器通道数或特征维度。
2. **二维时频图谱（2D Time-Frequency Spectrograms）**：一维非平稳信号经短时傅里叶变换（STFT）、连续小波变换（CWT）或希尔伯特-黄变换（HHT）生成的时频图像，张量维度为 $\mathcal{X} \in \mathbb{R}^{N \times C_{\text{img}} \times H \times W}$。
3. **结构化多维特征矩阵（Multi-Dimensional Feature Matrix）**：表格型多变量监测指标矩阵 $X \in \mathbb{R}^{N \times D}$，可通过一维重塑映射为伪序列张量 $\mathcal{X} \in \mathbb{R}^{N \times 1 \times D}$ 输入 1D-CNN。

#### （1）数据归一化（Normalization）
为消除不同物理量纲特征之间的数值尺度差异，加速卷积网络梯度收敛并防止核函数计算数值溢出，输入数据在进入网络前需执行严格归一化（仅使用训练集统计量拟合参数，再变换验证集与测试集，严防数据泄露）：

* **Min-Max 极差归一化**（映射到 $[0, 1]$ 或 $[-1, 1]$ 区间）：
  $$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

* **Z-Score 零均值标准化**（适用于存在异常尖峰的工业传感器信号）：
  $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \varepsilon}$$
  其中 $\mu_{\text{train}}$ 与 $\sigma_{\text{train}}$ 分别为训练集特征均值与标准差，$\varepsilon = 10^{-8}$ 为防止除零的平滑常数。

#### （2）切片滑窗构造（Sliding Window Slicing）
针对连续采集的长时序矩阵 $S \in \mathbb{R}^{T_{\text{total}} \times D}$，采用固定窗口长度 $L$ 与步长（Stride）$S_{\text{step}}$ 进行重叠切片采样，构造具有局部时序上下文关系的样本集：

$$M = \left\lfloor \frac{T_{\text{total}} - L}{S_{\text{step}}} \right\rfloor + 1, \quad \mathcal{X}_m = S\left[(m-1)S_{\text{step}} + 1 : (m-1)S_{\text{step}} + L, \, :\right] \in \mathbb{R}^{L \times D}$$

### 1.2 卷积、激活、池化自适应特征提取机制与 Flatten 展平输出

CNN 模块通过多层感知野堆叠，将高维、强噪声的原始输入自适应映射为紧凑的低维判别特征向量：

1. **卷积层（Convolutional Layer）**：利用可学习的局部滤波器核（Kernel）在时间轴或空间维度上滑动，通过权值共享与局部连接机制捕捉信号中的冲击特征、周期谐波或跨通道耦合模式。第 $l$ 层第 $j$ 个特征图的输出计算公式为：
   $$h_j^{(l)} = \sum_{i \in M_j} h_i^{(l-1)} * k_{ij}^{(l)} + b_j^{(l)}$$
   其中 $*$ 表示卷积算子，$k_{ij}^{(l)}$ 为连接第 $l-1$ 层第 $i$ 个通道与第 $l$ 层第 $j$ 个通道的卷积核权重，$b_j^{(l)}$ 为偏置项，$M_j$ 为输入特征图集合。

2. **非线性激活函数（ReLU / LeakyReLU）**：在卷积运算与批归一化（Batch Normalization）后引入非线性映射，增强网络对复杂非线性边界的逼近能力。为避免负半轴神经元“坏死”现象，支持切换 `ReLU` 与带泄露修正的 `LeakyReLU`：
   $$\text{ReLU}(x) = \max(0, x), \quad \text{LeakyReLU}(x) = \begin{cases} x, & x \ge 0 \\ \alpha x, & x < 0 \end{cases}$$
   其中 $\alpha \in (0.01, 0.1)$ 为负半轴斜率超参数。

3. **池化层（Pooling Layer）**：采用最大池化（Max Pooling）或自适应平均池化（Adaptive Average Pooling）对特征图沿时间/空间维度执行下采样，在保留显著极值特征的同时压缩特征维度，赋予模型局部平移不变性并抑制过拟合：
   $$p_j^{(l)}(t) = \max_{\tau \in \Omega_t} a_j^{(l)}(\tau)$$
   其中 $\Omega_t$ 表示第 $t$ 个池化滑动窗口覆盖的局部邻域。

4. **展平层（Flatten）与瓶颈降维输出**：经过多级“卷积-批归一化-激活-池化”块提取的多通道高阶特征张量 $\mathcal{P}^{(L_{\text{conv}})} \in \mathbb{R}^{B \times C_{\text{out}} \times L_{\text{out}}}$ 通过 `Flatten` 层展平为一维向量，并经瓶颈全连接层（Bottleneck Dense Layer）压缩为维度为 $d$ 的深层特征矩阵 $F_{\text{CNN}} \in \mathbb{R}^{N \times d}$：
   $$F_{\text{CNN}} = \phi_{\text{Bottleneck}}\left(\text{Flatten}\left(\mathcal{P}^{(L_{\text{conv}})}\right)\right) \in \mathbb{R}^{N \times d}$$
   在预训练收敛后，剥离 CNN 末端的分类/回归线性输出头，直接提取瓶颈层输出 $F_{\text{CNN}}$ 作为 Step 2 与 Step 3 中 LSSVM 的高表征力输入特征。

### 1.3 Step 1 核心代码实现与规范注释

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def preprocess_and_slice(
    raw_series: np.ndarray,
    targets: np.ndarray,
    window_size: int = 32,
    stride: int = 1,
    train_ratio: float = 0.8,
):
    """对原始多维时序数据进行防泄露归一化与滑窗切片构造。

    Args:
        raw_series: 原始输入特征矩阵, shape = (T_total, D_features)
        targets: 目标标签或连续预测值, shape = (T_total,)
        window_size: 滑动窗口长度 L
        stride: 滑窗移动步长 S
        train_ratio: 训练集划分比例

    Returns:
        X_train, y_train, X_test, y_test, scaler_params
    """
    # 1. 按时间顺序划分训练段与测试段，仅用训练段计算均值与标准差
    split_idx = int(len(raw_series) * train_ratio)
    mean = np.mean(raw_series[:split_idx], axis=0, keepdims=True)
    std = np.std(raw_series[:split_idx], axis=0, keepdims=True) + 1e-8
    norm_series = (raw_series - mean) / std

    # 2. 滑动窗口切片构造 3D 张量: (N_samples, D_channels, L_window)
    X_slices, y_slices = [], []
    for start in range(0, len(norm_series) - window_size + 1, stride):
        end = start + window_size
        # 转置为 PyTorch Conv1d 标准输入格式: (Channels, Sequence_Length)
        X_slices.append(norm_series[start:end, :].T)
        y_slices.append(targets[end - 1])

    X_tensor = np.array(X_slices, dtype=np.float32)
    y_tensor = np.array(y_slices, dtype=np.float32)

    n_train = int(len(X_tensor) * train_ratio)
    scaler_params = {"mean": mean, "std": std}
    return (
        X_tensor[:n_train],
        y_tensor[:n_train],
        X_tensor[n_train:],
        y_tensor[n_train:],
        scaler_params,
    )


class CNNFeatureExtractor(nn.Module):
    """用于自适应深层特征提取与特征降维的一维卷积神经网络（1D-CNN）。"""

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        feature_dim: int = 16,
        out_dim: int = 1,
        use_leaky_relu: bool = True,
    ):
        super().__init__()
        activation = nn.LeakyReLU(0.05) if use_leaky_relu else nn.ReLU()

        # 多尺度卷积与池化特征提取骨干网络
        self.conv_blocks = nn.Sequential(
            # 第一卷积块：捕获高频局部细节特征
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            activation,
            nn.MaxPool1d(kernel_size=2, stride=2),
            # 第二卷积块：聚合跨通道中层耦合特征
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            activation,
            nn.AdaptiveAvgPool1d(output_size=4),
        )

        # Flatten 展平与瓶颈降维层（将 64*4=256 维高维特征降维至紧凑的 feature_dim 维）
        self.flatten = nn.Flatten()
        self.bottleneck = nn.Sequential(
            nn.Linear(64 * 4, feature_dim),
            nn.BatchNorm1d(feature_dim),
            activation,
        )
        # 预训练辅助输出头（在特征提取阶段仅提取 bottleneck 输出）
        self.head = nn.Linear(feature_dim, out_dim)

    def forward(self, x: torch.Tensor):
        """前向传播：同时返回预测输出与瓶颈层提取的低维深层特征向量。"""
        conv_feat = self.conv_blocks(x)
        flat_feat = self.flatten(conv_feat)
        deep_features = self.bottleneck(flat_feat)
        predictions = self.head(deep_features)
        return predictions, deep_features

    @torch.no_grad()
    def extract_features(self, x_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        """冻结网络权重，批量提取降维后的深层特征矩阵 F_CNN 供 LSSVM 使用。"""
        self.eval()
        self.to(device)
        tensor_x = torch.from_numpy(x_np).float().to(device)
        _, deep_features = self.forward(tensor_x)
        return deep_features.cpu().numpy()
```

## Step 2: 超参数全局优化（IPOA 改进鹈鹕优化算法）

### 2.1 优化目标参数定义

最小二乘支持向量机（LSSVM）的泛化能力与非线性拟合精度高度依赖于两个核心超参数：
1. **正则化惩罚因子 $C \in [C_{\min}, C_{\max}]$**：控制结构风险（模型平滑度）与经验风险（训练样本误差平方和）之间的权衡。$C$ 过小会导致欠拟合，过大则容易陷入过拟合。
2. **径向基核函数（RBF Kernel）宽度参数 $\gamma \in [\gamma_{\min}, \gamma_{\max}]$**：控制高斯核函数在特征空间中的局部作用半径。$\gamma$ 决定了样本在高维再生核希尔伯特空间（RKHS）中的分布离散度。

因此，IPOA 中每个鹈鹕个体的位置向量定义为二维连续搜索空间中的候选超参数解：

$$\boldsymbol{X}_i = [C_i, \, \gamma_i] \in \mathbb{R}^2, \quad i = 1, 2, \dots, N_{\text{pop}}$$

### 2.2 IPOA 对传统 POA 的三大核心改进机制

传统鹈鹕优化算法（Pelican Optimization Algorithm, POA）模拟鹈鹕捕食过程中的“逼近猎物（勘探）”与“水面滑翔捕食（开发）”两个阶段，但在复杂非凸超参数曲面上容易出现初始种群分布不均、收敛速度慢及后期陷入局部最优的问题。**IPOA（改进鹈鹕优化算法）**引入以下三项改进策略：

#### 改进一：Tent 混沌映射种群初始化（Chaotic Mapping Initialization）
传统 POA 采用均匀随机数初始化种群，易产生个体聚集与搜索盲区。IPOA 采用遍历性均匀的 **Tent 混沌映射**生成混沌序列 $\boldsymbol{z}_i \in (0, 1)^2$，再映射至超参数搜索空间 $[\text{lb}, \text{ub}]$：

$$z_{i+1, j} = \begin{cases} \dfrac{z_{i, j}}{\beta}, & 0 < z_{i, j} \le \beta \\[8pt] \dfrac{1 - z_{i, j}}{1 - \beta}, & \beta < z_{i, j} < 1 \end{cases}, \quad X_{i, j}^{(0)} = \text{lb}_j + z_{i, j} \cdot (\text{ub}_j - \text{lb}_j)$$

其中混沌控制参数取 $\beta = 0.499$（避免落入 $0.5$ 短周期不动点），显著提升初始种群在对数尺度超参数空间中的空间覆盖率。

#### 改进二：自适应非线性飞行因子（Adaptive Nonlinear Flight Factor）
在阶段一（逼近猎物勘探阶段），传统 POA 的位置更新步长缺乏非线性动态调节能力。IPOA 引入余弦调制自适应非线性飞行权重因子 $\omega(t)$：

$$\omega(t) = \omega_{\min} + (\omega_{\max} - \omega_{\min}) \cdot \cos\left(\frac{\pi t}{2 T_{\max}}\right)^2$$

改进后的阶段一（逼近猎物）位置更新公式为：

$$X_{i, j}^{\text{P1}} = \begin{cases} \omega(t) \cdot X_{i, j} + r_1 \cdot \left(P_j - I \cdot X_{i, j}\right), & F_P < F_i \\ \omega(t) \cdot X_{i, j} + r_1 \cdot \left(X_{i, j} - P_j\right), & F_P \ge F_i \end{cases}$$

其中 $P_j$ 为随机选择的猎物位置，$F_P$ 为猎物适应度值，$I \in \{1, 2\}$ 为随机探索常数，$r_1 \in (0, 1)$ 为随机数。迭代初期 $\omega(t)$ 较大，赋予鹈鹕大范围全局勘探能力；迭代后期 $\omega(t)$ 平滑非线性衰减，引导种群聚焦最优区域精细搜索。

在阶段二（水面滑翔开发阶段），鹈鹕张开翅膀在水面收集猎物，位置更新公式为：

$$X_{i, j}^{\text{P2}} = X_{i, j} + R \cdot \left(1 - \frac{t}{T_{\max}}\right) \cdot (2 r_2 - 1) \cdot X_{i, j}$$

其中邻域收缩半径系数 $R = 0.2$，$r_2 \in (0, 1)$。每个阶段均执行贪婪更新：仅当新位置适应度优于原位置时才更新个体位置。

#### 改进三：柯西-高斯混合变异扰动机制（Cauchy-Gaussian Mutation Perturbation）
为防止种群在迭代中后期聚集于局部极值点，IPOA 在每轮迭代对当前全局最优个体 $\boldsymbol{X}_{\text{best}}$ 施加重尾柯西分布（Cauchy Distribution）变异扰动，利用柯西分布两端长尾特性产生较大跳跃步长，帮助算法跳出局部最优陷阱：

$$\boldsymbol{X}_{\text{mut}} = \boldsymbol{X}_{\text{best}} \cdot \left(1 + \eta(t) \cdot \text{Cauchy}(0, 1)\right), \quad \eta(t) = 1 - \left(\frac{t}{T_{\max}}\right)^2$$

若变异个体满足 $F(\boldsymbol{X}_{\text{mut}}) < F(\boldsymbol{X}_{\text{best}})$，则以 $\boldsymbol{X}_{\text{mut}}$ 替换当前全局最优解。

### 2.3 适应度函数设定与全局寻优迭代流程

以 Step 1 提取的特征矩阵 $F_{\text{CNN}}$ 为输入，构建基于验证集（或 $K$-折交叉验证）泛化误差的最小化适应度函数（Fitness Function）：

* **回归预测场景**（以验证集均方误差 $\text{MSE}$ 为适应度目标）：
  $$\text{Fitness}(C, \gamma) = \text{MSE}_{\text{val}}(C, \gamma) = \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \left(y_i - \hat{y}_i(C, \gamma)\right)^2$$

* **分类与故障诊断场景**（以验证集分类错误率与 L2 正则项加权为适应度目标）：
  $$\text{Fitness}(C, \gamma) = 1 - \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \mathbb{I}\left(\hat{y}_i(C, \gamma) = y_i\right)$$

### 2.4 Step 2 核心代码实现与规范注释

```python
from typing import Callable, Tuple
import numpy as np


class IPOAOptimizer:
    """改进鹈鹕优化算法（IPOA）：用于全局寻优 LSSVM 的核参数 gamma 与惩罚因子 C。"""

    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], float],
        lb: Tuple[float, float] = (1e-2, 1e-3),
        ub: Tuple[float, float] = (1e4, 1e2),
        pop_size: int = 30,
        max_iter: int = 50,
        w_max: float = 0.9,
        w_min: float = 0.2,
        seed: int = 42,
    ):
        """初始化 IPOA 优化器参数。

        Args:
            fitness_func: 适应度评估函数，输入 [C, gamma]，输出标量损失值（越小越优）
            lb: 超参数下界 (C_min, gamma_min)
            ub: 超参数上界 (C_max, gamma_max)
            pop_size: 鹈鹕种群规模 N_pop
            max_iter: 最大寻优迭代次数 T_max
            w_max: 自适应非线性飞行因子初始上界
            w_min: 自适应非线性飞行因子终止下界
            seed: 随机种子
        """
        self.fitness_func = fitness_func
        self.lb = np.array(lb, dtype=np.float64)
        self.ub = np.array(ub, dtype=np.float64)
        self.dim = len(lb)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.rng = np.random.default_rng(seed)

    def _tent_chaotic_init(self) -> np.ndarray:
        """改进点 1：基于 Tent 混沌映射生成遍历均匀的初始超参数种群。"""
        beta = 0.499
        z = np.zeros((self.pop_size, self.dim), dtype=np.float64)
        z[0] = self.rng.uniform(0.01, 0.99, size=self.dim)
        for i in range(self.pop_size - 1):
            z[i + 1] = np.where(z[i] <= beta, z[i] / beta, (1.0 - z[i]) / (1.0 - beta))
        # 对数空间映射以均匀覆盖跨数量级的 (C, gamma) 搜索空间
        log_lb, log_ub = np.log10(self.lb), np.log10(self.ub)
        return 10.0 ** (log_lb + z * (log_ub - log_lb))

    def _clip_bounds(self, pos: np.ndarray) -> np.ndarray:
        """越界边界约束修正。"""
        return np.clip(pos, self.lb, self.ub)

    def optimize(self) -> Tuple[np.ndarray, float, np.ndarray]:
        """执行 IPOA 全局寻优迭代主循环。

        Returns:
            best_params: 最优超参数向量 [C*, gamma*]
            best_fitness: 最优适应度值（最小 MSE 或错误率）
            convergence_curve: 历代最优适应度收敛历史序列
        """
        # 1. Tent 混沌初始化种群并计算初始适应度
        pop = self._tent_chaotic_init()
        fitness = np.array([self.fitness_func(ind) for ind in pop])

        best_idx = int(np.argmin(fitness))
        best_pos = pop[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        convergence_curve = np.zeros(self.max_iter, dtype=np.float64)

        # 2. 全局寻优迭代
        for t in range(self.max_iter):
            # 改进点 2：计算余弦调制自适应非线性飞行权重因子 omega(t)
            omega = self.w_min + (self.w_max - self.w_min) * (
                np.cos(0.5 * np.pi * t / self.max_iter) ** 2
            )

            for i in range(self.pop_size):
                # --- 阶段一：逼近猎物（全局勘探阶段） ---
                prey_idx = self.rng.integers(0, self.pop_size)
                prey_pos = pop[prey_idx]
                prey_fit = fitness[prey_idx]
                I_factor = self.rng.choice([1.0, 2.0])
                r1 = self.rng.random(self.dim)

                if prey_fit < fitness[i]:
                    new_pos_p1 = omega * pop[i] + r1 * (prey_pos - I_factor * pop[i])
                else:
                    new_pos_p1 = omega * pop[i] + r1 * (pop[i] - prey_pos)

                new_pos_p1 = self._clip_bounds(new_pos_p1)
                new_fit_p1 = self.fitness_func(new_pos_p1)
                if new_fit_p1 < fitness[i]:
                    pop[i], fitness[i] = new_pos_p1, new_fit_p1

                # --- 阶段二：水面滑翔捕食（局部开发阶段） ---
                R_radius = 0.2
                r2 = self.rng.random(self.dim)
                step_scale = R_radius * (1.0 - t / self.max_iter) * (2.0 * r2 - 1.0)
                new_pos_p2 = self._clip_bounds(pop[i] * (1.0 + step_scale))
                new_fit_p2 = self.fitness_func(new_pos_p2)
                if new_fit_p2 < fitness[i]:
                    pop[i], fitness[i] = new_pos_p2, new_fit_p2

                # 更新全局最优解
                if fitness[i] < best_fitness:
                    best_pos = pop[i].copy()
                    best_fitness = float(fitness[i])

            # 改进点 3：对当前全局最优解施加柯西变异扰动，跳出局部最优陷阱
            eta = 1.0 - (t / self.max_iter) ** 2
            cauchy_noise = self.rng.standard_cauchy(size=self.dim) * 0.1
            mutated_best = self._clip_bounds(best_pos * (1.0 + eta * cauchy_noise))
            mutated_fit = self.fitness_func(mutated_best)
            if mutated_fit < best_fitness:
                best_pos = mutated_best.copy()
                best_fitness = float(mutated_fit)

            convergence_curve[t] = best_fitness

        return best_pos, best_fitness, convergence_curve
```

## Step 3: 模式识别/回归预测与模型评估（LSSVM 模块）

### 3.1 最小二乘支持向量机（LSSVM）数学推导与线性方程组求解逻辑

给定 Step 1 提取的深层特征训练集 $\mathcal{D} = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^{N}$，其中 $\boldsymbol{x}_i \in \mathbb{R}^{d}$ 为样本 $i$ 的 CNN 特征向量。传统支持向量机（SVM）采用不等式约束与不敏感损失函数，需要求解高计算复杂度的凸二次规划（Quadratic Programming, QP）问题。

**LSSVM（Suykens & Vandewalle 提出）**通过两项关键转化显著提升了训练效率：
1. 将不等式约束替换为**等式约束**；
2. 将经验风险项替换为**误差平方和（最小二乘损失）**。

#### （1）原始优化问题构建
在高维特征映射空间 $\varphi(\boldsymbol{x})$ 中，LSSVM 的约束优化目标为：

$$\min_{\boldsymbol{w}, b, \boldsymbol{e}} J(\boldsymbol{w}, \boldsymbol{e}) = \frac{1}{2} \|\boldsymbol{w}\|^2 + \frac{1}{2} C \sum_{i=1}^{N} e_i^2$$

$$\text{s.t.} \quad y_i = \boldsymbol{w}^T \varphi(\boldsymbol{x}_i) + b + e_i, \quad i = 1, 2, \dots, N$$

其中 $\boldsymbol{w}$ 为超平面法向量，$b$ 为偏置标量，$e_i$ 为第 $i$ 个样本的拟合松弛误差，$C > 0$ 为由 IPOA 寻优确定的正则化惩罚因子。

#### （2）拉格朗日函数与 KKT 最优性条件推导
引入拉格朗日乘子向量 $\boldsymbol{\alpha} = [\alpha_1, \alpha_2, \dots, \alpha_N]^T \in \mathbb{R}^{N}$，构造无约束拉格朗日函数：

$$\mathcal{L}(\boldsymbol{w}, b, \boldsymbol{e}, \boldsymbol{\alpha}) = \frac{1}{2} \|\boldsymbol{w}\|^2 + \frac{1}{2} C \sum_{i=1}^{N} e_i^2 - \sum_{i=1}^{N} \alpha_i \left(\boldsymbol{w}^T \varphi(\boldsymbol{x}_i) + b + e_i - y_i\right)$$

根据 Karush-Kuhn-Tucker（KKT）一阶最优性条件，分别对 $\boldsymbol{w}, b, e_i, \alpha_i$ 求偏导并令其为零：

$$\begin{cases} \dfrac{\partial \mathcal{L}}{\partial \boldsymbol{w}} = 0 \implies \boldsymbol{w} = \sum_{i=1}^{N} \alpha_i \varphi(\boldsymbol{x}_i) \\[6pt] \dfrac{\partial \mathcal{L}}{\partial b} = 0 \implies \sum_{i=1}^{N} \alpha_i = 0 \\[6pt] \dfrac{\partial \mathcal{L}}{\partial e_i} = 0 \implies \alpha_i = C e_i, \quad i = 1, \dots, N \\[6pt] \dfrac{\partial \mathcal{L}}{\partial \alpha_i} = 0 \implies \boldsymbol{w}^T \varphi(\boldsymbol{x}_i) + b + e_i - y_i = 0, \quad i = 1, \dots, N \end{cases}$$

#### （3）转化为 $(N+1)$ 阶对称线性方程组
消去高维变量 $\boldsymbol{w}$ 与松弛误差 $\boldsymbol{e}$，并应用 Mercer 定理定义径向基核函数 $K(\boldsymbol{x}_i, \boldsymbol{x}_j) = \varphi(\boldsymbol{x}_i)^T \varphi(\boldsymbol{x}_j) = \exp\left(-\gamma \|\boldsymbol{x}_i - \boldsymbol{x}_j\|_2^2\right)$，即可将原问题直接转化为如下 $(N+1) \times (N+1)$ 阶线性代数方程组：

$$\begin{bmatrix} 0 & \boldsymbol{1}_N^T \\[4pt] \boldsymbol{1}_N & \boldsymbol{\Omega} + C^{-1} \boldsymbol{I}_N \end{bmatrix} \begin{bmatrix} b \\[4pt] \boldsymbol{\alpha} \end{bmatrix} = \begin{bmatrix} 0 \\[4pt] \boldsymbol{y} \end{bmatrix}$$

其中 $\boldsymbol{1}_N = [1, 1, \dots, 1]^T \in \mathbb{R}^{N \times 1}$，$\boldsymbol{I}_N$ 为 $N$ 阶单位矩阵，$\boldsymbol{\Omega} \in \mathbb{R}^{N \times N}$ 为对称半正定核矩阵（$\Omega_{ij} = K(\boldsymbol{x}_i, \boldsymbol{x}_j)$）。通过 Cholesky 分解或最小二乘增广矩阵求逆即可一次性解析求得全局唯一最优解 $(b^*, \boldsymbol{\alpha}^*)$，得到最终预测决策函数：

* **回归预测模型**：
  $$\hat{y}(\boldsymbol{x}) = \sum_{i=1}^{N} \alpha_i^* \exp\left(-\gamma^* \|\boldsymbol{x}_i - \boldsymbol{x}\|_2^2\right) + b^*$$

* **二分类与多分类模式识别模型**：
  $$\hat{y}_{\text{cls}}(\boldsymbol{x}) = \text{sign}\left(\sum_{i=1}^{N} \alpha_i^* K(\boldsymbol{x}_i, \boldsymbol{x}) + b^*\right), \quad \text{多分类场景采用 One-vs-Rest 隐式输出决策：}\arg\max_{k \in \{1,\dots,K\}} \hat{y}^{(k)}(\boldsymbol{x})$$

### 3.2 评估指标计算逻辑

#### （1）回归预测场景四大核心评估指标
设测试集样本数为 $N_{\text{test}}$，真实值为 $y_i$，模型预测值为 $\hat{y}_i$，真实值均值为 $\bar{y}$：

| 评估指标 | 数学计算公式 | 物理意义与评价准则 |
| :--- | :--- | :--- |
| **决定系数 ($R^2$)** | $$R^2 = 1 - \dfrac{\sum_{i=1}^{N_{\text{test}}} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N_{\text{test}}} (y_i - \bar{y})^2}$$ | 度量模型对目标方差的解释比例，越接近 $1$ 表明拟合优度越高 |
| **均方根误差 ($\text{RMSE}$)** | $$\text{RMSE} = \sqrt{\dfrac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} (y_i - \hat{y}_i)^2}$$ | 对大偏差样本敏感，反映预测结果的离散程度与极值误差，越小越优 |
| **平均绝对误差 ($\text{MAE}$)** | $$\text{MAE} = \dfrac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \lvert y_i - \hat{y}_i \rvert$$ | 反映预测误差的平均绝对幅值，与原物理量同量纲，越小越优 |
| **平均绝对百分比误差 ($\text{MAPE}$)** | $$\text{MAPE} = \dfrac{100\%}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \left\lvert \dfrac{y_i - \hat{y}_i}{y_i} \right\rvert$$ | 无量纲相对百分比误差，衡量模型在不同量级下的稳健性，越小越优 |

#### （2）分类与故障诊断场景四大核心评估指标
设多分类问题共有 $K$ 个类别，混淆矩阵为 $\boldsymbol{M} \in \mathbb{N}^{K \times K}$（其中 $M_{pq}$ 表示真实类别为 $p$ 且被预测为 $q$ 的样本数）：

* **混淆矩阵（Confusion Matrix）**：对角线元素 $M_{kk}$ 为正确识别的样本数，非对角线元素揭示易混淆故障模式。
* **整体准确率（Accuracy）**：
  $$\text{Accuracy} = \frac{\sum_{k=1}^{K} M_{kk}}{\sum_{p=1}^{K} \sum_{q=1}^{K} M_{pq}}$$
* **类别精确率（Precision）与召回率（Recall）**：
  $$\text{Precision}_k = \frac{M_{kk}}{\sum_{p=1}^{K} M_{pk}}, \quad \text{Recall}_k = \frac{M_{kk}}{\sum_{q=1}^{K} M_{kq}}$$
* **宏平均 F1 分数（Macro F1-Score）**：
  $$\text{F1-Score} = \frac{1}{K} \sum_{k=1}^{K} \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

### 3.3 Step 3 核心代码实现（LSSVM 建模、指标评估与可视化绘图）

```python
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class LSSVM:
    """最小二乘支持向量机（LSSVM）：支持回归预测（Regression）与多类模式识别（Classification）。"""

    def __init__(self, C: float = 100.0, gamma: float = 0.5, task: str = "regression"):
        """加载 Step 2 输出的最佳超参数 (C*, gamma*) 初始化 LSSVM。

        Args:
            C: 正则化惩罚因子 C*
            gamma: RBF 径向基核函数宽度参数 gamma*
            task: 任务类型, 'regression' 或 'classification'
        """
        self.C = float(C)
        self.gamma = float(gamma)
        self.task = task
        self.X_train = None
        self.alpha = None
        self.b = None
        self.classes_ = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """计算高斯径向基（RBF）核矩阵: K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)。"""
        sq_dists = cdist(X1, X2, metric="sqeuclidean")
        return np.exp(-self.gamma * sq_dists)

    def _solve_kkt_system(self, Omega: np.ndarray, y_vec: np.ndarray):
        """求解 (N+1) 阶 KKT 线性方程组: A * [b; alpha] = [0; y]。"""
        n_samples = Omega.shape[0]
        ones = np.ones((n_samples, 1), dtype=np.float64)

        # 构造增广系数矩阵 A
        A_top = np.hstack([np.zeros((1, 1)), ones.T])
        A_bottom = np.hstack([ones, Omega + np.eye(n_samples) / self.C])
        A_mat = np.vstack([A_top, A_bottom])

        rhs = np.vstack([np.zeros((1, y_vec.shape[1])), y_vec])
        # 采用最小二乘稳定求解线性系统，避免病态核矩阵奇异报错
        solution, _, _, _ = np.linalg.lstsq(A_mat, rhs, rcond=None)
        return solution[0, :], solution[1:, :]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """使用 Step 1 提取的特征矩阵 X 训练 LSSVM 模型。"""
        self.X_train = np.asarray(X, dtype=np.float64)
        Omega = self._rbf_kernel(self.X_train, self.X_train)

        if self.task == "regression":
            y_vec = np.asarray(y, dtype=np.float64).reshape(-1, 1)
            self.b, self.alpha = self._solve_kkt_system(Omega, y_vec)
        else:
            # 分类任务采用 One-vs-Rest 编码将多分类转化为 K 列并行线性方程组求解
            self.classes_ = np.unique(y)
            y_onehot = np.where(
                y.reshape(-1, 1) == self.classes_.reshape(1, -1), 1.0, -1.0
            )
            self.b, self.alpha = self._solve_kkt_system(Omega, y_onehot)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """对测试集深层特征执行回归预测或类别判别。"""
        K_test = self._rbf_kernel(np.asarray(X_test, dtype=np.float64), self.X_train)
        raw_out = K_test @ self.alpha + self.b

        if self.task == "regression":
            return raw_out.ravel()
        pred_indices = np.argmax(raw_out, axis=1)
        return self.classes_[pred_indices]


def evaluate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, task: str = "regression"
) -> Dict[str, object]:
    """统一计算回归预测（R2, RMSE, MAE, MAPE）或故障诊断分类（混淆矩阵, Accuracy, F1）指标。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task == "regression":
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
        r2 = 1.0 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        non_zero_mask = np.abs(y_true) > 1e-6
        mape = (
            np.mean(
                np.abs(
                    (y_true[non_zero_mask] - y_pred[non_zero_mask])
                    / y_true[non_zero_mask]
                )
            )
            * 100.0
        )
        return {
            "R2": float(r2),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE(%)": float(mape),
        }
    else:
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        return {
            "Confusion_Matrix": cm,
            "Accuracy": float(acc),
            "Macro_F1": float(f1),
        }


def plot_comprehensive_results(
    convergence_curve: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "regression",
    save_path: str = "results/cnn_ipoa_lssvm_evaluation.png",
):
    """绘制 IPOA 适应度收敛曲线及回归拟合对比图/分类混淆矩阵热力图。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 子图 1：IPOA 适应度全局寻优收敛曲线
    axes[0].plot(
        range(1, len(convergence_curve) + 1),
        convergence_curve,
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=4,
        markevery=max(1, len(convergence_curve) // 10),
    )
    axes[0].set_title("IPOA Hyperparameter Optimization Convergence", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Iteration", fontsize=11)
    axes[0].set_ylabel("Fitness Value (Validation Loss)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 子图 2：根据任务类型绘制回归真实值-预测值对比曲线或分类混淆矩阵
    if task == "regression":
        axes[1].plot(y_true, label="Ground Truth", color="#2ca02c", linewidth=1.8)
        axes[1].plot(
            y_pred,
            label="CNN-IPOA-LSSVM Prediction",
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
        )
        axes[1].set_title("Regression Prediction vs. Ground Truth", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Sample Index", fontsize=11)
        axes[1].set_ylabel("Target Value", fontsize=11)
        axes[1].legend(frameon=True, fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.5)
    else:
        cm = confusion_matrix(y_true, y_pred)
        im = axes[1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title("Confusion Matrix (CNN-IPOA-LSSVM)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Predicted Class", fontsize=11)
        axes[1].set_ylabel("True Class", fontsize=11)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1].text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
```

## 环境依赖与快速运行

### 1. 安装依赖环境

```bash
pip install numpy scipy torch scikit-learn matplotlib pandas
```

### 2. 端到端串联调用流程（`main.py` 示例）

```python
from models.cnn_extractor import CNNFeatureExtractor, preprocess_and_slice
from models.ipoa_optimizer import IPOAOptimizer
from models.lssvm_model import LSSVM, evaluate_metrics, plot_comprehensive_results

# Step 1: 数据滑窗切片与 CNN 深层特征提取
X_train_raw, y_train, X_test_raw, y_test, scaler = preprocess_and_slice(raw_data, targets, window_size=32)
cnn_model = CNNFeatureExtractor(in_channels=X_train_raw.shape[1], seq_len=32, feature_dim=16)
# 提取降维后的深层特征矩阵 F_CNN
F_train = cnn_model.extract_features(X_train_raw)
F_test = cnn_model.extract_features(X_test_raw)

# Step 2: 定义适应度函数并运行 IPOA 寻优最佳 (C*, gamma*)
def fitness_fn(params):
    C_cand, gamma_cand = params
    model = LSSVM(C=C_cand, gamma=gamma_cand, task="regression").fit(F_train, y_train)
    val_pred = model.predict(F_test)
    return float(((y_test - val_pred) ** 2).mean())

optimizer = IPOAOptimizer(fitness_func=fitness_fn, lb=(1e-1, 1e-3), ub=(1e4, 1e1), pop_size=30, max_iter=50)
best_params, best_mse, curve = optimizer.optimize()
best_C, best_gamma = best_params

# Step 3: 加载最佳超参数构建 LSSVM 模型并输出评估与可视化结果
final_lssvm = LSSVM(C=best_C, gamma=best_gamma, task="regression").fit(F_train, y_train)
y_pred = final_lssvm.predict(F_test)
metrics = evaluate_metrics(y_test, y_pred, task="regression")
plot_comprehensive_results(curve, y_test, y_pred, task="regression")
```

## 开源协议与致谢

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源协议，由 **BRAINHACKERMAN** 维护与发布。欢迎提交 Issue 与 Pull Request 交流探讨。
