# SE-CCM 改进调研方案

## 一、项目背景

### 1.1 研究问题

**收敛交叉映射（CCM）** 是检测耦合动力系统间因果关系的核心方法，基于 Takens 嵌入定理：若 Y 因果驱动 X，则 X 的重构吸引子包含 Y 的信息，从 X 构建的影子流形可以交叉预测 Y。

**核心困难：** 原始 CCM 相关性（ρ）易受共享动力学、同步化和有限样本偏差影响产生假阳性。SE-CCM 通过对候选因变量生成替代数据（surrogate），比较观测 ρ 与替代分布来获得 p 值和 z 分数，再应用 FDR 校正实现网络尺度的因果推断。

### 1.2 当前框架

SE-CCM 框架包含以下模块：

| 模块 | 内容 |
|------|------|
| **动力系统** | 7 种：Logistic、Lorenz、Hénon、Rössler、Hindmarsh-Rose、FitzHugh-Nagumo、Kuramoto |
| **Surrogate 方法** | 7 种：FFT、AAFT、iAAFT、time-shift、random-reorder、cycle-shuffle、twin |
| **嵌入参数选择** | autocorrelation 1/e + MI first minimum 选 τ；simplex prediction 选 E |
| **统计检验** | rank-based p-value + z-score + BH-FDR 校正 |
| **实验设计** | 4 个子实验（T-sweep、coupling sweep、obs noise、dyn noise）× 7 系统 × 7 方法 × 10 reps |

### 1.3 已完成的工作

#### (a) Cycle-Shuffle Surrogate（新增）

**动机：** T-sweep 实验揭示窄带振荡系统（Rössler top-3 spectral concentration = 0.822, FHN = 0.901, Kuramoto = 0.846）对 FFT/AAFT/iAAFT 的 ΔAUROC 为负。根因是频谱替代保留了周期结构中承载的因果信号。

**算法：** 通过均值交叉（上升沿）检测振荡周期 → 切割为完整周期 → 随机打乱周期顺序 → 重新拼接。周期数 < 3 时回退到 time-shift。

#### (b) Twin Surrogate（新增）

**动机：** 需要一种理论上更严谨的替代方法，保留吸引子拓扑（递推结构）同时破坏系统间的相位耦合。

**算法：** （Thiel et al., 2006, EPL 75(4), 535-541）延迟嵌入重构相空间 → KDTree 构建递推邻域 → 哈希加速识别孪生状态 → 沿吸引子行走构造替代轨迹。

#### (c) 性能优化

| 优化项 | 加速效果 |
|--------|:--------:|
| iAAFT：单次 argsort + scatter 替代双重 argsort | **2×** |
| Twin：KDTree + L∞ 替代 dense N×N 矩阵 | **6×** |
| Twin：行哈希分桶 twin 检测 | **56×** |
| Twin：修复 `select_parameters` 重复调用 bug | **40-49×** |
| SECCM：按 cause 变量缓存 surrogate（10 次替代 90 次） | **2.4×** |
| **总计**（5 方法 → 7 方法含 twin）| 原始 35.2h → 当前 17.7h（8 核 ~2.2h） |

#### (d) 大规模实验

完成了完整的 robustness 实验：7 系统 × 7 方法 × 4 子实验 × 10 reps = 7,350 runs，总计 ~6,550 万条 surrogate 时间序列。

---

## 二、实验结果与问题诊断

### 2.1 总体结果

**方法排名（mean ΔAUROC_zscore，所有系统、所有子实验平均）：**

| 排名 | 方法 | ΔAUROC_zscore | 绝对 AUROC_zscore | "伤害率"（Δ < 0 的比例） |
|:----:|------|:---:|:---:|:---:|
| 1 | FFT | +0.006 | 0.676 | 43.5% |
| 2 | iAAFT | +0.003 | 0.673 | 45.4% |
| 3 | time-shift | -0.010 | 0.660 | 51.7% |
| 4 | cycle-shuffle | -0.013 | 0.657 | 52.4% |
| 5 | AAFT | -0.019 | 0.652 | 53.5% |
| 6 | twin | -0.023 | 0.647 | 53.1% |
| 7 | random-reorder | -0.041 | 0.629 | 61.1% |

**关键发现：即使最好的 FFT 方法，也在 43.5% 的条件下使 AUROC 下降。**

### 2.2 系统间差异巨大

**Raw CCM 基础能力（AUROC_rho）：**

| 系统 | AUROC_rho | FPR | 状态 |
|------|:---------:|:---:|:----:|
| Logistic | 0.908 | 0.079 | 正常 |
| Hénon | 0.725 | 0.386 | 正常 |
| Lorenz | 0.687 | 0.353 | 正常但 Δ 全负 |
| Kuramoto | 0.678 | 0.238 | 正常 |
| Rössler | 0.635 | **0.866** | FPR 异常高 + 30% 仿真失败 |
| FitzHugh-Nagumo | 0.560 | **0.678** | 接近随机 |
| Hindmarsh-Rose | **0.499** | 0.277 | **完全随机水平** |

### 2.3 按系统类型分析

**宽频混沌系统（Logistic, Hénon, Lorenz）：** mean ΔAUROC_zscore = **+0.002**
- FFT/iAAFT 有效（Hénon 上 ΔAUROC = +0.128）
- 因果信号在非线性相位关系中，频谱替代精确破坏了这些关系

**窄带振荡系统（Rössler, FHN, Kuramoto）：** mean ΔAUROC_zscore = **-0.030**
- 所有方法均为负值
- 频谱替代保留了周期结构中承载的因果信号，z-score 被压低

### 2.4 新方法的表现

| 系统 | cycle-shuffle | twin | 最优方法 | 说明 |
|------|:---:|:---:|:---:|------|
| **Rössler** | **-0.020** (第 1) | -0.071 | cycle-shuffle | 验证了设计假设（p < 0.0001 vs iAAFT） |
| **Kuramoto** | -0.041 (最末) | **+0.025** (第 1) | twin | twin 是唯一正向提升的方法（p < 0.0001） |
| **FHN** | -0.042 (最末) | -0.010 | FFT / twin | cycle-shuffle 假设不成立；twin 与 FFT 持平 |
| Hénon | +0.080 | -0.062 | FFT / iAAFT | twin 在宽频系统上失败 |
| Logistic | +0.001 | -0.002 | iAAFT | 差异极小 |
| Lorenz | -0.049 | -0.024 | AAFT | 所有方法 Δ 均为负 |
| HR | -0.020 | -0.017 | random-reorder | CCM 本身失败，surrogate 无关 |

### 2.5 问题总结

| 编号 | 问题 | 严重性 | 层面 |
|:----:|------|:------:|:----:|
| P1 | HR 系统 AUROC ≈ 0.5，CCM 完全失败 | **严重** | 系统生成 |
| P2 | FHN 系统 AUROC ≈ 0.56，接近随机 | **严重** | 系统生成 |
| P3 | Rössler 30% 仿真失败 + FPR = 0.87 | **严重** | 系统生成 |
| P4 | 所有 surrogate 在窄带系统上 ΔAUROC < 0 | **中等** | Surrogate 方法 |
| P5 | 即使 FFT 也有 43.5% 的伤害率 | **中等** | 检验方法 |
| P6 | Cycle-shuffle 仅在 Rössler 有效，Kuramoto/FHN 失败 | 中等 | Surrogate 方法 |
| P7 | Twin 在宽频系统上失败（Hénon -0.062） | 中等 | Surrogate 方法 |
| P8 | 无收敛性检验，library 和 prediction set 重叠 | 中等 | CCM 方法 |
| P9 | FPR 在 Rössler (0.87) 和 FHN (0.68) 上异常高 | 中等 | 检验方法 |

---

## 三、调研方案

### 3.1 CCM 方法层面

#### 3.1.1 嵌入参数选择改进

**现状：** `select_tau` 用 autocorrelation 1/e + MI first minimum；`select_E` 用 simplex 单步预测。对稳定的混沌吸引子有效，但对 bursting（HR）、excitable（FHN）和多时间尺度系统失败。

| 调研方向 | 具体内容 | 预期解决的问题 | 参考文献 |
|----------|----------|:---:|------|
| **False Nearest Neighbors (FNN)** | 用 FNN 比率判断嵌入维度是否充分展开吸引子；比 simplex prediction 对非混沌信号更鲁棒 | P1, P2 | Kennel, Brown & Abarbanel, Phys. Rev. A, 1992 |
| **Cao's Method** | 基于 E1/E2 统计量判断确定性结构是否存在；E2 ≈ 1 表示随机过程，E2 ≠ 1 表示有确定性成分 | P1, P2 | Cao, Physica D, 1997 |
| **非均匀嵌入** | 对多时间尺度系统（HR 的 fast-slow dynamics），选择不同 delay 的组合 (τ₁, τ₂, ...) 而非单一 τ | P1 | Pecora et al., Phys. Rev. E, 2007; Vlachos & Kugiumtzis, Phys. Rev. E, 2010 |
| **混沌性预检测** | CCM 前用 0-1 test for chaos 或 Lyapunov 指数检测信号是否混沌；对非混沌信号标记或跳过 | P1, P2 | Gottwald & Melbourne, Proc. R. Soc. A, 2004 |

#### 3.1.2 CCM 算法本身的改进

**现状：** `ccm()` 用前 L 个点做 library，library 和 prediction set 完全重叠（in-sample），不检查收敛性。

| 调研方向 | 具体内容 | 预期解决的问题 | 参考文献 |
|----------|----------|:---:|------|
| **Theiler Window** | 预测时排除当前点的时间邻近点（±w 步），避免时间自相关导致的虚假高 ρ | P5, P9 | Theiler, Phys. Rev. A, 1986 |
| **收敛性检验** | 不只看 ρ(L_max)，用 ρ(L) 随 L 递增的收敛斜率或 Kendall τ 作为检验统计量；真因果应单调递增，假因果应平坦或下降 | P5, P8 | Sugihara et al., Science, 2012（原文核心思想，当前未实现） |
| **Cross-validation** | Leave-one-out 或 k-fold 交叉验证替代 in-sample 评估，减少过拟合导致的 ρ 虚高 | P5, P9 | Luo et al., Nat. Commun., 2015 |
| **CCM 不对称性检验** | 同时计算 ccm(X→Y) 和 ccm(Y→X)，用不对称度 Δρ = ρ(X→Y) - ρ(Y→X) 作为因果方向判据 | P5 | Sugihara et al., Science, 2012 |
| **延迟因果 CCM (Extended CCM)** | 考虑因果传递的时间延迟 tp > 0，扫描不同 prediction horizon 找最优 tp | P4 | Ye et al., Sci. Rep., 2015 |

---

### 3.2 系统生成层面

#### 3.2.1 不可用系统修复

| 系统 | 问题 | 调研方向 | 具体方案 |
|------|------|----------|----------|
| **Hindmarsh-Rose** | AUROC ≈ 0.5，bursting dynamics 无 strange attractor | (a) 参数调整进入 chaotic bursting regime | 调研 r, I_ext 参数空间，寻找正 Lyapunov 指数区域（Innocenti et al., Chaos, 2007） |
| | | (b) 替换为混沌 Izhikevich 模型 | Izhikevich, IEEE Trans. Neural Netw., 2003 — 可产生 chaotic spiking |
| | | (c) 保留为负对照 | 明确标注"CCM 预期失败"，讨论 CCM 适用边界 |
| **FitzHugh-Nagumo** | AUROC ≈ 0.56，excitable (limit-cycle) 而非 chaotic | (a) 增大 I_ext 进入混沌区 | 调研 I_ext > 0.8 的分岔图，验证 strange attractor 存在性 |
| | | (b) 替换为 Rössler（已有）或 Chua 电路 | Chua's circuit 有丰富的混沌参数区域 |
| | | (c) 保留为负对照 | 同上 |
| **Rössler** | 30% 仿真失败，FPR = 0.87 | (a) 修复初始条件 | Z 分量初始化在吸引子附近 [0, 5] 而非 [-5, 5] |
| | | (b) 增强积分器 | 添加发散检测、clip；stiff 情况切换 Radau 方法 |
| | | (c) 耦合上限 | cap ε ≤ 1.0（当前到 2.0），减少发散 |

#### 3.2.2 耦合机制多样化

**现状：** 所有系统使用相同的扩散耦合 `ε(A·X - k_in·X)/k_in`。

| 调研方向 | 具体内容 | 动机 |
|----------|----------|------|
| **非线性耦合** | `ε·f(X_j - X_i)` 其中 f 为 sigmoid 或 threshold 函数 | 更接近真实神经元突触耦合 |
| **延迟耦合** | `ε·(X_j(t-d) - X_i(t))` 引入传播延迟 | 许多真实系统有传递延迟，CCM 需要能处理 |
| **脉冲耦合** | 仅在 X_j 超过阈值时触发耦合脉冲 | Kuramoto/HR 等振子系统的自然耦合方式 |

---

### 3.3 Surrogate 方法层面

#### 3.3.1 改进现有方法

| 方法 | 当前问题 | 调研方向 | 参考文献 |
|------|----------|----------|----------|
| **Cycle-shuffle** | 仅对 Rössler 有效；Kuramoto/FHN 失败 | (a) **Phase surrogate**：Hilbert 变换提取瞬时相位，随机化相位增量但保留幅度包络；对相位耦合系统（Kuramoto）应更有效 | Lancaster et al., Phys. Rev. E, 2018 |
| | | (b) **自适应方法选择**：根据频谱集中度自动选择 — 高集中度→cycle-shuffle；低集中度→FFT；相位主导→phase surrogate | 工程改进 |
| **Twin** | Hénon/Logistic 上失败（twin 太少） | (a) **Approximate twin**：放松"完全相同 row"要求，允许 Hamming 距离 ≤ k 的 rows 也作为 twin | Romano et al., Chaos, 2009 |
| | | (b) **自适应 ε**：根据找到的 twin 数量动态调整 recurrence threshold | pyunicorn 实现参考 |

#### 3.3.2 新增 Surrogate 方法

| 方向 | 核心思想 | 预期解决的问题 | 参考文献 |
|------|----------|:---:|------|
| **Multivariate surrogate** | 对多变量联合生成 surrogate，保留自协方差但破坏互协方差；避免共享动力学被 surrogate 保留 | P4, P9 | Prichard & Theiler, Phys. Rev. Lett., 1994; Schreiber & Schmitz, Phys. Rev. Lett., 2000 |
| **Conditional surrogate** | 只打乱 cause 对 effect 的"增量信息"，保留 effect 自身动力学；类似 conditional MI 的 surrogate 版本 | P4 | 受 PCMCI 启发 — Runge et al., Nat. Commun., 2019 |
| **Small-shuffle surrogate** | 对时间索引做小幅随机扰动（±Δt），破坏精细时间对齐但保留大尺度趋势 | P5 | Nakamura & Small, Phys. Rev. E, 2005 |
| **Truncated Fourier surrogate** | 只随机化特定频段的相位：(a) 低频随机化保留高频细节，或 (b) 高频随机化保留低频包络 | P4, P6 | Keylock, J. Hydrol., 2006; Lancaster et al., Phys. Rev. E, 2018 |
| **Block bootstrap surrogate** | 将时间序列切为随机长度的块再打乱，保留短程依赖但破坏长程因果传递 | P5 | Politis & Romano, J. Am. Stat. Assoc., 1994 |

#### 3.3.3 自适应 Surrogate 选择框架

**核心思想：** 不同系统适合不同的 surrogate 方法（已被实验证实）。设计一个预分析器，根据信号特征自动选择最优方法。

**可用特征：**

| 特征 | 计算方法 | 指示 |
|------|----------|------|
| 频谱集中度 | top-k FFT 系数能量占比 | 高 → 窄带振荡 → cycle-shuffle |
| 0-1 test 统计量 | Gottwald-Melbourne | 接近 0 → 周期/准周期 → 需特殊处理 |
| 自相关衰减速度 | 1/e 衰减时间 | 快 → 映射/宽频 → FFT；慢 → 流/窄带 → MI 选 τ |
| 递推率 (RR) | recurrence plot | 高 → 规则动力学 → twin 可能 twin 太多 |
| 相位一致性 (PLV) | Hilbert + 节点对相位差 | 高 → 相位耦合 → phase surrogate / twin |

---

### 3.4 统计检验层面

#### 3.4.1 降低假阳性率

**现状：** Rössler FPR = 0.87，FHN FPR = 0.68。即使 twin（最保守）在 Rössler 上 FPR 也有 0.58。

| 调研方向 | 具体内容 | 预期解决的问题 | 参考文献 |
|----------|----------|:---:|------|
| **自适应 min_rho 门槛** | 对每个系统根据 surrogate ρ 分布的上分位数（如 95th percentile）设 min_rho，而非固定 0.3 | P9 | 工程改进 |
| **Permutation-based FDR** | 不对 p-value 做 BH，而是用 permutation test 直接控制 FDR（MaxT procedure） | P9 | Westfall & Young, Resampling-Based Multiple Testing, 1993 |
| **Effect size + significance 联合判据** | 类似 volcano plot：同时要求 z-score > z_threshold 和 ρ > ρ_threshold；两个门槛联合控制 | P9 | 工程改进 |

#### 3.4.2 改进检验统计量

| 调研方向 | 具体内容 | 预期解决的问题 | 参考文献 |
|----------|----------|:---:|------|
| **收敛斜率统计量** | 用 ρ(L) 的收敛速率替代 ρ(L_max) 作为检验统计量；对 surrogate 也计算收敛斜率，比较斜率分布 | P5, P8 | Sugihara 2012 核心思想 |
| **非参数排名统计量** | 用 surrogate 分布的排名 rank/(n+1) 替代 z-score，不假设正态分布 | P5 | 工程改进 |
| **Bootstrap CI** | 对 observed ρ 做 bootstrap 置信区间，若 CI 下界 > surrogate ρ 上界则判显著 | P5 | Efron & Tibshirani, 1993 |

#### 3.4.3 考虑检验间依赖性

**现状：** 90 对中有 10 个 cause 变量，每个出现在 9 个 pair 中。这些 pair 的 surrogate 来自同一个 cause → p-value 之间不独立 → BH-FDR 假设被违反。

| 调研方向 | 具体内容 | 参考文献 |
|----------|----------|----------|
| **BY correction** | Benjamini-Yekutieli 校正，在 p-value 有正依赖时仍控制 FDR | Benjamini & Yekutieli, Ann. Stat., 2001 |
| **分层检验** | 先对每个 cause j 做整体检验（j 是否影响任何其他节点），再对具体 pair 做条件检验 | 工程改进 |

---

## 四、优先级与路线图

### Phase 1：基础修复（预期 1-2 周）

| 优先级 | 任务 | 预期收益 | 难度 |
|:------:|------|:--------:|:----:|
| **P0-1** | HR/FHN 混沌性预检测（0-1 test） | 消除 2/7 系统的基础性误判 | 低 |
| **P0-2** | Rössler 数值稳定性修复（初始条件 + 发散检测） | 消除 30% failed reps | 低 |
| **P0-3** | 自适应 min_rho 门槛 | 降低 Rössler/FHN 的异常高 FPR | 低 |

### Phase 2：核心方法改进（预期 2-4 周）

| 优先级 | 任务 | 预期收益 | 难度 |
|:------:|------|:--------:|:----:|
| **P1-1** | CCM 收敛性检验（ρ(L) 斜率统计量） | 对所有系统提升因果区分能力 | 中 |
| **P1-2** | Theiler window 排除时间邻近点 | 减少自相关导致的虚假高 ρ | 低 |
| **P1-3** | 自适应 surrogate 方法选择（频谱集中度特征） | 消除"一种方法对所有系统"困境 | 中 |
| **P1-4** | FNN / Cao's method 辅助嵌入 | 改善 E 选择对非标准动力学的鲁棒性 | 中 |

### Phase 3：新方法探索（预期 4-8 周）

| 优先级 | 任务 | 预期收益 | 难度 |
|:------:|------|:--------:|:----:|
| **P2-1** | Phase surrogate | 填补相位耦合系统的空白（Kuramoto） | 中 |
| **P2-2** | Multivariate / conditional surrogate | 解决共享动力学导致的高 FPR | 高 |
| **P2-3** | 非均匀嵌入 | 改善多时间尺度系统（HR） | 高 |
| **P2-4** | Small-shuffle / truncated Fourier | 提供更精细的零假设控制 | 中 |

### Phase 4：系统性验证（预期 2-3 周）

| 任务 | 内容 |
|------|------|
| 全量重跑 robustness 实验 | 纳入新方法和修复，对比 before/after |
| 消融实验 | 逐项关闭改进，量化每项的独立贡献 |
| 论文写作 | 结合诊断结果和改进效果，撰写方法论和实验分析 |

---

## 五、关键参考文献

### CCM 与因果推断
- Sugihara, G. et al. (2012). Detecting causality in complex ecosystems. *Science*, 338(6106), 496-500.
- Ye, H. et al. (2015). Distinguishing time-delayed causal interactions using convergent cross mapping. *Sci. Rep.*, 5, 14750.
- Luo, M. et al. (2015). Questionable dynamical evidence for causality between galactic cosmic rays and interannual variation in global temperature. *PNAS*, 112(34).
- Runge, J. et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Sci. Adv.*, 5(11).

### 嵌入理论
- Kennel, M.B., Brown, R. & Abarbanel, H.D.I. (1992). Determining embedding dimension for phase-space reconstruction. *Phys. Rev. A*, 45(6), 3403.
- Cao, L. (1997). Practical method for determining the minimum embedding dimension. *Physica D*, 110(1-2), 43-50.
- Pecora, L.M. et al. (2007). A unified approach to attractor reconstruction. *Chaos*, 17(1), 013110.

### Surrogate 方法
- Theiler, J. et al. (1992). Testing for nonlinearity in time series. *Physica D*, 58(1-4), 77-94.
- Schreiber, T. & Schmitz, A. (2000). Surrogate time series. *Physica D*, 142(3-4), 346-382.
- Thiel, M. et al. (2006). Twin surrogates to test for complex synchronisation. *EPL*, 75(4), 535.
- Romano, M.C. et al. (2009). Hypothesis test for synchronization: Twin surrogates revisited. *Chaos*, 19(1), 015108.
- Nakamura, T. & Small, M. (2005). Small-shuffle surrogate data. *Phys. Rev. E*, 72(5), 056216.
- Lancaster, G. et al. (2018). Surrogate data for hypothesis testing of physical systems. *Phys. Rep.*, 748, 1-60.

### 混沌检测
- Gottwald, G.A. & Melbourne, I. (2004). A new test for chaos in deterministic systems. *Proc. R. Soc. A*, 460(2042), 603-611.

### 统计检验
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *J. R. Stat. Soc. B*, 57(1), 289-300.
- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Ann. Stat.*, 29(4), 1165-1188.
- Westfall, P.H. & Young, S.S. (1993). *Resampling-Based Multiple Testing*. Wiley.

### 动力系统
- Innocenti, G. et al. (2007). Dynamical phases of the Hindmarsh-Rose neuronal model. *Phys. Rev. E*, 76(6), 061919.
- Izhikevich, E.M. (2003). Simple model of spiking neurons. *IEEE Trans. Neural Netw.*, 14(6), 1569-1572.
