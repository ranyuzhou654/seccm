# SE-CCM: 基于替代数据的增强型收敛交叉映射

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[English Documentation / 英文文档](../README.md)**

SE-CCM 是一个用于**耦合动力系统因果推断**的 Python 框架。它在收敛交叉映射（CCM）算法（[Sugihara et al., 2012, *Science*](https://doi.org/10.1126/science.1227079)）基础上，引入替代数据假设检验，将原始交叉映射相关性转化为严格的统计决策。

---

## 核心思想

CCM 利用 **Takens 嵌入定理**检测因果关系：如果变量 Y 驱动变量 X，则 Y 的信息编码在 X 的重构吸引子中，从 X 构建的影子流形可以交叉预测 Y。

**问题：** 原始 CCM 相关性（ρ）容易因共享动力学、同步化或有限样本偏差产生假阳性。

**SE-CCM 的解决方案：** 在零假设（无因果影响）下生成候选因变量的替代时间序列，将观测 ρ 与替代分布进行比较以获得 p 值和 z 分数，然后对网络范围的检验应用 Benjamini-Hochberg FDR 校正。

## 主要特性

- **7 种耦合动力系统** — Logistic、Lorenz、Henon、Rossler、Hindmarsh-Rose、FitzHugh-Nagumo、Kuramoto
- **10 种单变量 + 2 种多变量替代数据方法 + 自适应选择** — FFT、AAFT、iAAFT、时移、随机重排、周期打乱、孪生替代、相位替代、小幅打乱、截断傅里叶；多变量 FFT/iAAFT（保留交叉相关）；`auto` 模式根据频谱集中度自动选择
- **自动嵌入参数选择** — 基于互信息和单纯形预测/FNN/Cao 方法的数据驱动 E 和 tau 选择；非均匀延迟嵌入支持多时间尺度系统
- **收敛性检验** — 交叉验证的 ρ(L) 收敛曲线 + Kendall-τ 评分
- **混沌预检测** — 0-1 混沌检测（Gottwald & Melbourne），支持 ODE 流的自动子采样
- **网络尺度检验** — 成对 CCM 配合 BH-FDR 校正、Theiler 窗口、自适应效应大小门槛
- **6 个实验模块** — 双变量验证、耦合强度扫描、噪声鲁棒性、拓扑比较、替代方法比较、鲁棒性消融研究
- **出版级输出** — 300 DPI 图像（PDF + PNG）、LaTeX 表格、CSV 数据

## 安装

```bash
git clone https://github.com/your-username/surrogate-ccm.git
cd surrogate-ccm
pip install -e .
```

或直接安装依赖：

```bash
pip install -r requirements.txt
```

**依赖：** Python >= 3.9, NumPy, SciPy, scikit-learn, NetworkX, matplotlib, seaborn, joblib, h5py, PyYAML, tqdm, statsmodels。

## 快速上手

### 最小示例

```python
from surrogate_ccm.generators import generate_network, create_system
from surrogate_ccm.testing import SECCM

# 创建 10 节点耦合 Lorenz 网络
adj = generate_network("ER", N=10, seed=42, p=0.3)
system = create_system("lorenz", adj, coupling=1.0)
data = system.generate(T=3000, transient=1000, seed=42)  # 形状: (3000, 10)

# 运行 SE-CCM
seccm = SECCM(surrogate_method="aaft", n_surrogates=99, alpha=0.05, fdr=True)
seccm.fit(data)

# 对照真实邻接矩阵评估
metrics = seccm.score(adj)
print(f"AUROC (替代数据): {metrics['AUC_ROC_surrogate']:.3f}")
print(f"AUROC (原始 rho): {metrics['AUC_ROC_rho']:.3f}")
print(f"Delta AUROC:      {metrics['AUC_ROC_delta']:+.3f}")
```

### 运行实验

```bash
# 使用默认配置运行所有实验
python run_experiments.py --experiment all --n-jobs 4

# 运行特定实验
python run_experiments.py --experiment robustness --n-jobs 16 --output-dir results/robustness

# 冒烟测试（快速，约 10 秒）
python run_experiments.py --experiment robustness --config configs/robustness_smoke.yaml
```

## 项目结构

```
surrogate-ccm/
├── run_experiments.py              # CLI 入口
├── configs/
│   ├── default.yaml                # 完整实验配置
│   └── robustness_smoke.yaml       # 快速冒烟测试配置
├── surrogate_ccm/
│   ├── ccm/                        # CCM 核心算法
│   │   ├── embedding.py            #   延迟嵌入、tau/E 选择、非均匀嵌入
│   │   ├── ccm_core.py             #   交叉映射预测 (ccm, ccm_convergence)
│   │   └── network_ccm.py          #   网络成对 CCM
│   ├── surrogate/                  # 替代数据生成方法
│   │   ├── fft_surrogate.py        #   FFT 相位随机化
│   │   ├── aaft_surrogate.py       #   幅度调整傅里叶变换
│   │   ├── iaaft_surrogate.py      #   迭代 AAFT
│   │   ├── timeshift_surrogate.py  #   循环时移
│   │   ├── random_reorder.py       #   随机排列
│   │   ├── cycle_shuffle_surrogate.py  #   周期打乱（窄带振荡系统）
│   │   ├── twin_surrogate.py       #   孪生替代（递推结构保持）
│   │   ├── phase_surrogate.py     #   相位替代（Hilbert 变换）
│   │   ├── small_shuffle_surrogate.py  #   小幅打乱替代
│   │   ├── truncated_fourier_surrogate.py  #   截断傅里叶替代
│   │   ├── adaptive.py            #   自适应方法选择
│   │   └── multivariate_surrogate.py  #   多变量 FFT/iAAFT 替代
│   ├── testing/                    # 统计检验
│   │   ├── hypothesis_test.py      #   p 值、z 分数、FDR 校正
│   │   └── se_ccm.py              #   SECCM 类（完整流水线）
│   ├── generators/                 # 耦合动力系统模拟器
│   │   ├── network.py              #   网络拓扑生成 (ER, WS, ring)
│   │   ├── logistic.py             #   耦合 Logistic 映射
│   │   ├── lorenz.py               #   耦合 Lorenz 振子
│   │   ├── henon.py                #   耦合 Henon 映射
│   │   ├── rossler.py              #   耦合 Rossler 振子
│   │   ├── hindmarsh_rose.py       #   Hindmarsh-Rose 神经网络
│   │   ├── fitzhugh_nagumo.py      #   FitzHugh-Nagumo 神经网络
│   │   └── kuramoto.py             #   Kuramoto 相位振子
│   ├── evaluation/                 # 检测指标 (TPR, FPR, AUROC)
│   ├── visualization/              # 绘图工具
│   ├── experiments/                # 实验模块
│   │   ├── exp_bivariate.py        #   双节点验证
│   │   ├── exp_coupling_strength.py#   耦合强度扫描
│   │   ├── exp_noise.py            #   噪声实验
│   │   ├── exp_network_topology.py #   拓扑比较
│   │   ├── exp_surrogate_comparison.py  #  替代方法比较
│   │   └── exp_surrogate_robustness.py  #  鲁棒性消融研究
│   └── utils/                      # I/O、并行执行
├── ccm.py                          # 高级独立 CCM（潜在 CCM、收敛评分）
└── system.py                       # 高级独立模拟器（RK4、过程噪声、BA/SF 图）
```

## 实验概览

| 实验 | 命令 | 描述 |
|---|---|---|
| `bivariate` | `--experiment bivariate` | 双节点验证：检验能否正确检测 X0 -> X1 |
| `coupling` | `--experiment coupling` | 扫描不同系统和拓扑下的耦合强度 |
| `noise` | `--experiment noise` | 观测噪声对检测精度的影响 |
| `topology` | `--experiment topology` | 比较 ER、WS、ring 拓扑在不同网络规模下的表现 |
| `surrogate` | `--experiment surrogate` | 比较 7 种替代方法在 7 个系统上的表现 |
| `robustness` | `--experiment robustness` | 消融实验：扫描 T、耦合、观测噪声、动态噪声 |

### 鲁棒性消融实验

鲁棒性实验包含 4 个子实验，每次扫描一个因素，其余固定：

| 子实验 | 扫描参数 | 取值 |
|---|---|---|
| Sub-A | 时间序列长度 T | 500, 1000, 2000, 3000, 5000 |
| Sub-B | 耦合强度 epsilon | 每个系统独立（各 6 个值） |
| Sub-C | 观测噪声 sigma_obs | 0.0, 0.01, 0.05, 0.1, 0.2 |
| Sub-D | 动态噪声 sigma_dyn | 0.0, 0.001, 0.005, 0.01, 0.05 |

**输出包括：**
- 带 SEM 误差棒的 AUROC 折线图（按系统、按方法）
- Delta-AUROC 热力图（系统 x 扫描值）
- 4 面板联合概览图
- 消融汇总表（CSV + LaTeX）
- 实验参数文档表

### 运行时间估算

| `--n-jobs` | 预计耗时 |
|---|---|
| 1 | ~3.7 小时 |
| 4 | ~1.5 小时 |
| 16 | ~25 分钟 |

## 动力系统

| 系统 | 类型 | 状态维度 | 观测变量 | 关键参数 |
|---|---|---|---|---|
| Logistic | 离散映射 | 1 | x | r = 3.9 |
| Henon | 离散映射 | 2 | x | a = 1.1, b = 0.3 |
| Lorenz | ODE（混沌） | 3 | x | sigma = 10, rho = 28, beta = 8/3 |
| Rossler | ODE（混沌） | 3 | x | a = 0.2, b = 0.2, c = 5.7 |
| Hindmarsh-Rose | ODE（神经元） | 3 | x（膜电位） | I_ext = 3.5, r = 0.01 |
| FitzHugh-Nagumo | ODE（神经元） | 2 | v（膜电位） | I_ext = 0.5, tau = 12.5 |
| Kuramoto | ODE（振子） | 1 | sin(theta) | omega ~ N(1.0, 0.2) |

所有系统支持：
- **观测噪声**（`noise_std`）：高斯噪声叠加到最终输出
- **动态噪声**（`dyn_noise_std`）：ODE 系统使用 Euler-Maruyama SDE 积分；映射系统在每步迭代中添加噪声
- **可配置耦合**和**网络拓扑**（ER、WS、ring）

## 替代数据方法

| 方法 | 保留 | 破坏 | 适用场景 | 速度 |
|---|---|---|---|---|
| FFT | 功率谱 | 幅度分布、非线性结构 | 宽频混沌 | 快 |
| AAFT | 幅度分布 + 近似功率谱 | 非线性结构 | 通用 | 快 |
| iAAFT | 幅度分布 + 精确功率谱 | 非线性结构 | 通用 | 中等 |
| 时移 | 所有局部结构 | 序列间时间对齐 | 快速基线 | 极快 |
| 随机重排 | 幅度分布 | 所有时间结构 | 独立性检验 | 极快 |
| **周期打乱** | 周期内波形、幅度分布 | 周期间相位耦合 | **窄带振荡**（Rossler） | 极快 |
| **孪生替代** | 吸引子拓扑（递推结构） | 系统间相位耦合 | **任意动力系统**（理论最优） | 中等 |
| **相位替代** | 幅度包络 | 相位耦合 | **相位耦合振荡**（Kuramoto、FHN） | 快 |
| **小幅打乱** | 大尺度趋势 | 精细时间顺序 | 趋势数据 | 极快 |
| **截断傅里叶** | 目标频段外结构 | 目标频段内相位 | 多时间尺度分析 | 快 |
| **多变量 FFT** | 交叉相关 + 功率谱 | 非线性交叉依赖 | 线性耦合系统 | 快 |
| **多变量 iAAFT** | 交叉相关 + 功率谱 + 幅度分布 | 非线性交叉依赖 | 线性耦合系统 | 中等 |

### 周期打乱替代（Cycle-Shuffle）

针对**窄带振荡系统**设计。在这类系统中，FFT/AAFT/iAAFT 保留了承载因果信号的周期结构，导致 z 分数被压低、ΔAUROC 为负。

**算法：** 通过均值交叉（上升沿）检测振荡周期 → 将信号切割为完整周期 → 随机打乱周期顺序 → 重新拼接。若检测到的完整周期少于 3 个，回退为时移替代。

### 孪生替代（Twin Surrogate）

生成保留原始时间序列**递推结构**（吸引子拓扑）的替代数据，是检验同步化和因果耦合理论上最严谨的替代方法之一。

**算法**（[Thiel et al., 2006, *EPL* 75(4)](https://doi.org/10.1209/epl/i2006-10147-0)）：
1. 通过延迟嵌入重构相空间
2. 使用 KDTree（Chebyshev 度量）构建递推邻域
3. 识别"孪生"状态——递推邻域完全相同的点对（哈希加速）
4. 沿吸引子行走构造替代轨迹，在孪生点处随机跳转到孪生后继

**性能优化：**
- KDTree + 稀疏邻居集合替代稠密 N×N 递推矩阵（快 6 倍，省 3 倍内存）
- 行哈希分桶的孪生检测（比逐行比较快 56 倍）
- 预计算的孪生结构在多条替代数据生成间缓存复用
- SECCM 管线中按因变量缓存替代数据（减少 9 倍生成调用）

## 约定

- **邻接矩阵：** `A[i, j] = 1` 表示节点 j 驱动节点 i（行 = 接收方）
- **CCM 方向：** `ccm(x, y)` 检验 "y 导致 x"（嵌入 x，预测 y）
- **替代目标：** 对*因变量*生成替代数据（而非果变量）
- **数据形状：** `(T, N)` — 行为时间步，列为节点
- **效应量门槛：** 统计显著性还要求原始 rho >= 0.3

## 配置

所有参数通过 YAML 配置文件控制。详见 [`configs/default.yaml`](../configs/default.yaml)。主要配置段：

```yaml
time_series:
  T: 3000           # 时间序列长度
  transient: 1000   # 暂态丢弃步数

surrogate:
  methods: [fft, aaft, iaaft, timeshift, random_reorder, cycle_shuffle, twin]
  n_surrogates: 100

surrogate_robustness:
  systems: [logistic, lorenz, henon, rossler, hindmarsh_rose, fitzhugh_nagumo, kuramoto]
  methods: [fft, aaft, iaaft, timeshift, cycle_shuffle]
  n_surrogates: 99
  N: 10
  n_reps: 10
  # ... 各子实验的扫描值
```

## API 参考

```python
# ── 数据生成 ────────────────────────────────────────────
from surrogate_ccm.generators import generate_network, create_system

adj = generate_network("ER", N=10, seed=0, p=0.3)    # -> (10, 10) 二值矩阵
system = create_system("lorenz", adj, coupling=1.0)
data = system.generate(T=3000, transient=1000, seed=0,
                       noise_std=0.0, dyn_noise_std=0.0)  # -> (3000, 10)

# ── CCM ─────────────────────────────────────────────────
from surrogate_ccm.ccm import delay_embed, select_parameters, ccm, compute_pairwise_ccm
from surrogate_ccm.ccm import convergence_score, select_E_fnn, select_E_cao
from surrogate_ccm.ccm import delay_embed_nonuniform, select_delays_nonuniform

E, tau = select_parameters(data[:, 0])                 # 自动选择 (E, tau)
E, tau = select_parameters(data[:, 0], E_method="fnn") # FNN 方法
E, tau = select_parameters(data[:, 0], E_method="cao") # Cao 方法
rho = ccm(data[:, 0], data[:, 1], E=3, tau=2)         # 检验 "节点 1 导致节点 0"
rho = ccm(data[:, 0], data[:, 1], E=3, tau=2, theiler_w=5)  # 带 Theiler 窗口
score, rho = convergence_score(data[:, 0], data[:, 1], E=3, tau=2,
                               cross_validate=True)    # 收敛性检验
ccm_matrix, params = compute_pairwise_ccm(data)        # 所有 N^2 对

# 非均匀嵌入（适用于多时间尺度系统如 Hindmarsh-Rose）
delays = select_delays_nonuniform(data[:, 0], E_max=5, tau_max=100)  # 如 [0, 5, 6, 17, 18]
M = delay_embed_nonuniform(data[:, 0], delays)         # 自定义延迟嵌入

# ── 替代数据 ────────────────────────────────────────────
from surrogate_ccm.surrogate import generate_surrogate, select_surrogate_method
from surrogate_ccm.surrogate import generate_multivariate_surrogate

surrogates = generate_surrogate(data[:, 0], method="aaft", n_surrogates=99)          # -> (99, T)
surrogates = generate_surrogate(data[:, 0], method="cycle_shuffle", n_surrogates=99) # 窄带振荡
surrogates = generate_surrogate(data[:, 0], method="twin", n_surrogates=99)          # 递推结构保持
method, profile = select_surrogate_method(data[:, 0])                               # 自适应选择

# 多变量替代（保留变量间交叉相关）
mv_surr = generate_multivariate_surrogate(data, method="multivariate_fft", n_surrogates=99)

# ── 完整流水线 ──────────────────────────────────────────
from surrogate_ccm.testing import SECCM

seccm = SECCM(surrogate_method="auto", n_surrogates=99, alpha=0.05, fdr=True)
# 默认值：theiler_w="auto"（中位 tau），adaptive_rho=True
seccm.fit(data)
metrics = seccm.score(adj)     # -> 字典：AUROC, TPR, FPR 等

# 启用收敛过滤（可选，更保守）
seccm = SECCM(convergence_filter=True, convergence_threshold=0.0)
seccm.fit(data)

# 访问内部结果
seccm.ccm_matrix_              # 原始 CCM rho 矩阵 (N, N)
seccm.pvalue_matrix_           # p 值矩阵 (N, N)
seccm.zscore_matrix_           # z 分数矩阵 (N, N)
seccm.detected_                # 二值检测矩阵 (N, N)
seccm.surrogate_methods_used_  # 每变量使用的方法（auto 模式）
seccm.min_rho_matrix_          # 每对自适应阈值
seccm.theiler_w_used_          # 实际使用的 Theiler 窗口值
seccm.convergence_matrix_      # 收敛分数（需启用 convergence_filter）
```

## 引用

如果您在研究中使用了本软件，请引用：

```bibtex
@software{seccm2025,
  title  = {SE-CCM: Surrogate-Enhanced Convergent Cross Mapping},
  year   = {2025},
  url    = {https://github.com/your-username/surrogate-ccm}
}
```

## 许可证

本项目采用 MIT 许可证。
