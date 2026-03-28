# 诊断框架实验详解

本文档详细解释 `run_diagnostic_experiments.py` 中的四个实验（D1、D2、E5、E7）的原理、设计动机和实现细节。

---

## 目录

- [背景：为什么需要诊断框架？](#背景为什么需要诊断框架)
- [核心概念速查](#核心概念速查)
- [D1：综合诊断表（Diagnostic Table）](#d1综合诊断表diagnostic-table)
- [D2：体制边界分析（Regime Boundary Analysis）](#d2体制边界分析regime-boundary-analysis)
- [E5：收敛性验证（Convergence Behavior）](#e5收敛性验证convergence-behavior)
- [E7：噪声鲁棒性（Noise Robustness）](#e7噪声鲁棒性noise-robustness)
- [实验间的逻辑关系](#实验间的逻辑关系)

---

## 背景：为什么需要诊断框架？

### 问题起源

Convergent Cross Mapping（CCM）是一种基于 Takens 嵌入定理的因果推断方法。为了提高统计显著性，我们引入**替代数据检验（surrogate testing）**：生成一组在零假设下的替代时间序列，比较观测到的 CCM 相关系数 ρ 与替代分布，得到 z-score 和 p-value。

**关键发现：替代数据检验并不总是有效。** 在我们的前期实验中，我们发现：

1. **某些系统上替代数据有帮助**：如 Van der Pol 振荡器，使用替代数据后 AUROC 提升约 +0.13
2. **某些系统上替代数据有害**：如 Rossler 系统，替代数据无法破坏其动力学结构（"替代不可渗透"），导致 ΔAUROC 为负
3. **效果因替代方法而异**：不同的替代方法在不同系统上表现差异巨大

这引出了核心研究问题：**什么时候替代数据能帮助 CCM？我们能否事先诊断？**

### 诊断框架的核心思路

我们提出的诊断框架试图：

1. **量化替代数据的效用**：用 ΔAUROC = AUROC(z-score) − AUROC(raw ρ) 衡量替代检验带来的改善
2. **发现诊断指标**：寻找能预测 ΔAUROC 正负的先验可计算指标
3. **分类动力学体制**：将不同的系统-替代组合分类到不同的"体制"中，给使用者提供选择指南

---

## 核心概念速查

在阅读实验细节前，先理解以下核心指标：

| 指标 | 定义 | 直觉含义 |
|------|------|---------|
| **AUROC(ρ)** | 用原始 CCM 相关系数 ρ 作为排序分数，计算 ROC 曲线下面积 | 不使用替代数据时的检测能力 |
| **AUROC(z)** | 用替代检验 z-score 作为排序分数计算 AUROC | 使用替代数据后的检测能力 |
| **ΔAUROC** | AUROC(z) − AUROC(ρ) | **正值 = 替代数据有帮助，负值 = 替代数据有害** |
| **AUPRC** | 精确率-召回率曲线下面积 | 在类别不平衡时比 AUROC 更可靠的检测性能度量 |
| **rho_gap** | mean(ρ_observed) − mean(ρ_surrogate) | 观测值和替代值分布之间的"距离"；越大说明替代数据越有效地破坏了因果信息 |
| **surr_std** | 替代分布的标准差 | 替代分布的离散程度；越小说明替代数据越一致 |
| **null_overlap** | 替代ρ ≥ 观测ρ 的比例 | 零假设分布与观测值的重叠程度；= 0 表示"替代不可渗透" |
| **frac_converging** | 收敛性检验通过的边比例 | CCM 本身是否可靠：如果 ρ(L) 不随库长度增加而收敛，CCM 结论不可信 |
| **SSO** | Surrogate Spectral Overlap，原始与替代信号功率谱的 Jensen-Shannon 散度 | 替代数据在频域上改变了原始信号多少；高 SSO = 频谱被大幅改变 |
| **spectral_conc** | 前3大频率分量占总功率的比例 | 信号的"窄带程度"；接近 1 = 强周期性信号 |
| **acf_decay** | 自相关函数衰减到 1/e 的时间 | 信号的记忆长度；越长说明动力学越平滑 |
| **perm_entropy** | Bandt & Pompe 排列熵，归一化到 [0,1] | 时间序列的复杂度；1 = 完全随机，0 = 完全确定性 |

---

## D1：综合诊断表（Diagnostic Table）

**脚本位置：** `surrogate_ccm/experiments/exp_diagnostic_table.py`

### 这个实验在做什么？

D1 是整个诊断框架的**主表实验**。它对 **8 个动力学系统 × 12 种替代方法** 的所有组合进行全面评估，每个组合重复 20 次（不同随机种子），总计 **1920 次独立运行**。

对每一次运行，它计算：

- **检测性能**：AUROC(ρ)、AUROC(z-score)、ΔAUROC、AUPRC、TPR、FPR、F1
- **替代分布诊断**：rho_gap、surr_std、null_overlap
- **收敛性**：mean_convergence、frac_converging
- **频谱特性**：SSO、spectral_conc、acf_decay、perm_entropy

### 为什么这样设计？

#### 1. 全组合遍历的必要性

不同系统有截然不同的动力学特性：

| 系统 | 动力学类型 | 特征 |
|------|-----------|------|
| Logistic | 离散混沌映射 | 宽带频谱，对初始条件高度敏感 |
| Henon | 离散混沌映射 | 二维状态空间，类似 Logistic |
| Lorenz | 连续混沌系统 | 经典蝴蝶吸引子，宽带混沌 |
| Rossler | 连续混沌系统 | 频谱集中（窄带），有明显主频 |
| Kuramoto | 相位耦合振荡器 | 纯相位动力学，同步行为 |
| Van der Pol | 自激振荡器 | 极限环振荡，非线性弛豫 |
| FitzHugh-Nagumo | 神经元模型 | 兴奋-恢复动力学 |
| Hindmarsh-Rose | 神经元模型 | 多时间尺度，爆发式放电 |

不同的替代方法各有其保留和破坏的特性。例如：
- **FFT/iAAFT**：保留功率谱，破坏非线性结构 → 对宽带混沌系统有效
- **cycle_phase_A**：保留周期内波形，破坏周期间相位对齐 → 对振荡系统有效
- **random_reorder**：破坏一切时间结构 → 最激进的零假设

通过全组合遍历，我们才能回答："对于系统类型 X，最佳替代方法是什么？"

#### 2. 诊断指标的计算

这些指标的设计目的是**事先（不需要知道真实网络）判断替代检验是否有效**：

- **rho_gap** 大 → 替代数据成功地"降低"了因果信号 → 替代检验可能有效
- **null_overlap ≈ 0** → 替代分布完全不覆盖观测值 → 可能是"替代不可渗透"，z-score 过高反而损害区分度
- **frac_converging** 低 → CCM 本身不可靠 → 无论用什么替代方法都没用

#### 3. 体制分类（Regime Classification）

基于诊断指标，每个系统×替代组合被分类为：

```
if frac_converging < 0.5:
    → "ccm_unreliable"          # CCM 本身就不可靠
elif null_overlap < 0.02 and ΔAUROC < 0:
    → "surrogate_impermeable"   # 替代数据无法渗透动力学结构
elif ΔAUROC > 0.02:
    → "surrogate_helps"         # 替代检验有效
elif ΔAUROC < -0.02:
    → "surrogate_hurts"         # 替代检验反而有害
else:
    → "neutral"                 # 效果不显著
```

### 输出

| 文件 | 内容 |
|------|------|
| `full_diagnostics.csv` | 所有 1920 次运行的完整结果（每行一次运行） |
| `diagnostics_agg.csv` | 按 system × surrogate 聚合的均值 |
| `regime_classification.csv` | 每个组合的体制分类 |
| `regime_summary.json` | 分类器准确率和统计摘要 |
| `heatmap_delta_auroc.pdf` | ΔAUROC 热力图（系统 × 替代方法） |
| `regime_scatter.pdf` | rho_gap vs null_overlap 散点图（颜色编码 ΔAUROC 正负） |
| `surrogate_ranking.pdf` | 替代方法排名柱状图（跨系统平均 ΔAUROC） |
| `per_system_bars.pdf` | 按系统分面的每种替代方法 ΔAUROC 柱状图 |

---

## D2：体制边界分析（Regime Boundary Analysis）

**脚本位置：** `surrogate_ccm/experiments/exp_regime_boundaries.py`

### 这个实验在做什么？

D2 固定系统类型，**连续扫描耦合强度（coupling strength）**，观察替代效用（ΔAUROC）如何随耦合强度变化，从而找到体制转变的**边界**在哪里。

以 Rossler 系统为例，扫描 coupling ∈ {0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5}。

### 为什么这样设计？

#### 原理：耦合强度决定动力学体制

耦合强度直接影响系统的因果结构和可检测性：

- **弱耦合**：节点几乎独立，因果信号微弱，CCM 的 ρ 值低 → 替代检验可能帮助区分微弱信号和噪声
- **中等耦合**：因果信号清晰，ρ 值适中 → 替代检验效果取决于替代方法是否能有效破坏因果结构
- **强耦合**：节点高度同步，ρ 值极高 → 替代数据可能无法降低 ρ（替代不可渗透），或者产生过多假阳性

D1 只测试了固定耦合强度下的快照，而 D2 揭示的是**连续变化的全貌**。

#### 我们在寻找什么？

1. **临界耦合值**：ΔAUROC 从正变负的转折点 → 告诉使用者："对于 Rossler 系统，当耦合强度超过 X 时，不应该使用 iAAFT"
2. **替代不可渗透区域**：null_overlap 接近 0 的耦合范围 → 对应于 Rossler 的"光滑吸引子"使替代数据失效的区域
3. **不同替代方法的比较**：iAAFT 和 cycle_phase_A 在同一耦合扫描中的表现差异 → 识别哪种方法在哪个耦合区间更优

#### 具体测量

对每个 system × coupling × surrogate × replicate 组合，测量：
- ΔAUROC：替代检验的净增益
- rho_gap：观测值与替代分布的距离
- null_overlap：零假设覆盖率
- frac_converging：CCM 收敛比例

### 输出

| 文件 | 内容 |
|------|------|
| `regime_boundaries_raw.csv` | 所有运行的原始结果 |
| `regime_boundaries_agg.csv` | 按 system × coupling × surrogate 聚合的均值和标准差 |
| `boundaries_{system}.pdf` | 每个系统的四面板图（ΔAUROC, rho_gap, null_overlap, frac_converging vs coupling） |
| `phase_diagram.pdf` | 所有系统的 ΔAUROC vs coupling 叠加图（使用 iAAFT 作为参考） |

---

## E5：收敛性验证（Convergence Behavior）

**脚本位置：** `surrogate_ccm/experiments/exp_convergence.py`

### 这个实验在做什么？

E5 扫描**时间序列长度 T**（从 500 到 10000），验证替代增强 CCM 的检测性能是否随着数据量的增加而**单调改善**。

### 为什么这样设计？

#### 原理：CCM 收敛性是正确性的必要条件

CCM 的理论基础是 Takens 嵌入定理，该定理在极限意义下保证影子流形的忠实重构。实际中，这意味着：

- **数据越多 → 影子流形重构越准确 → ρ(L) 越接近真实因果强度**
- 一个合理的因果检测方法，其 AUROC 应该随 T 增大而单调增加（至少不恶化）

如果某个替代方法的 AUROC 在 T 增大时反而下降或剧烈波动，这说明该方法存在系统性偏差。

#### 具体验证内容

1. **AUROC(z-score) vs T 曲线**：应该单调递增
2. **ΔAUROC vs T 曲线**：替代数据的边际效用是否随数据量变化？
   - 如果 ΔAUROC 随 T 增大趋向 0 → 替代检验在大样本下"不必要但也不有害"
   - 如果 ΔAUROC 保持正值 → 即使数据充足，替代检验仍然有价值
3. **z-score 分离度**：真实因果边的平均 z-score（mean_z_true）和非因果边的平均 z-score（mean_z_false）之间的间距是否随 T 增大而扩大

#### 统计检验

对每个 system × surrogate 组合，计算 T 与 AUROC 之间的 **Spearman 秩相关系数**，并检测是否严格单调递增。

### 输出

| 文件 | 内容 |
|------|------|
| `convergence_raw.csv` | 所有运行的原始结果 |
| `convergence_check.json` | Spearman 相关系数、单调性检测结果 |
| `auroc_vs_T.pdf` | AUROC(z-score) vs T 曲线（每个系统一个面板） |
| `delta_auroc_vs_T.pdf` | ΔAUROC vs T 曲线 |
| `zscore_separation_vs_T.pdf` | 真/假因果边的 z-score 分离度 vs T |

---

## E7：噪声鲁棒性（Noise Robustness）

**脚本位置：** `surrogate_ccm/experiments/exp_noise_robustness.py`

### 这个实验在做什么？

E7 在观测数据上添加不同强度的**高斯观测噪声**（σ = 0 到 0.5，相对于信号标准差），测量：

1. **检测性能退化**：ΔAUROC 随噪声增大如何变化
2. **周期检测质量**：cycle_phase 替代方法依赖于正确识别振荡周期，噪声会如何影响周期检测？

### 为什么这样设计？

#### 原理：真实数据总是含噪的

实验室或观测数据不可能无噪声。一个实用的替代方法必须在合理噪声水平下保持有效。这个实验回答：

- **cycle_phase_A 的噪声容忍度是多少？** → 如果在 σ = 0.1 时就崩溃，那对真实数据没什么用
- **与 iAAFT 相比，哪个更抗噪？** → iAAFT 基于频域，对高斯噪声天然鲁棒；cycle_phase_A 基于时域周期检测，可能更脆弱

#### 周期检测质量的监控

cycle_phase 替代方法的第一步是**识别振荡周期**：通过 Hilbert 变换提取瞬时相位，在相位每经过 2π 时标记周期边界。噪声会导致：

- **虚假的相位跳变** → 错误地分割周期 → 产生大量短"伪周期"
- **相位不连续** → Hilbert 方法失败 → 退化到基于峰值检测的备用方案

E7 对每个变量监控：

| 指标 | 含义 |
|------|------|
| `mean_n_cycles` | 检测到的平均周期数；噪声增大时可能先增后减 |
| `min_n_cycles` | 最少检测到的周期数；< 3 时 cycle_phase 会退化为 timeshift |
| `cycle_len_cv` | 周期长度的变异系数（std/mean）；噪声增大时应增加 |
| `fallback_frac` | 退化到备用方案（周期不足 3 个）的变量比例；理想情况下应保持为 0 |

### 输出

| 文件 | 内容 |
|------|------|
| `noise_robustness_raw.csv` | 所有运行的原始结果 |
| `noise_summary.json` | 按 system × surrogate × noise_level 聚合的摘要 |
| `auroc_vs_noise.pdf` | ΔAUROC vs 噪声强度曲线 |
| `cycle_detection_quality.pdf` | 周期检测数量和退化比例 vs 噪声强度 |

---

## 实验间的逻辑关系

四个实验构成一个完整的论证链：

```
D1 (全面扫描)
│   回答：哪些系统×替代组合有效？诊断指标能否预测效用？
│
├──→ D2 (耦合强度扫描)
│       回答：体制转变发生在什么耦合强度？
│       依赖 D1 的发现来选择关注的系统和替代方法
│
├──→ E5 (收敛性)
│       回答：替代增强 CCM 是否满足统计一致性？
│       验证 D1 的结果不是小样本假象
│
└──→ E7 (噪声鲁棒性)
        回答：在现实噪声条件下，结论是否仍然成立？
        验证 cycle_phase 方法的实用性
```

### 对应论文的结构

| 实验 | 论文角色 |
|------|---------|
| D1 | **Table 1 / Figure 1**：主结果表 + ΔAUROC 热力图 |
| D2 | **Figure 2**：体制相图，展示耦合强度如何驱动体制转变 |
| E5 | **Figure 3**（补充）：收敛性曲线，证明方法的统计可靠性 |
| E7 | **Figure 4**（补充）：噪声鲁棒性曲线，证明方法的实际可用性 |

### 运行方式

```bash
# 完整运行（约 20 小时 CPU 时间，8 核并行）
python run_diagnostic_experiments.py --n-jobs 8

# 快速可行性测试（约 30 分钟）
python run_diagnostic_experiments.py --feasibility --n-jobs 5

# 只运行 D1 和 D2
python run_diagnostic_experiments.py --only D1 D2

# 自定义参数
python run_diagnostic_experiments.py --n-reps 10 --n-surrogates 50 --n-jobs 4
```

### Feasibility 模式的缩减

| 参数 | 完整模式 | Feasibility 模式 |
|------|---------|-----------------|
| 系统数 | 8 | 4（logistic, rossler, kuramoto, van_der_pol） |
| 替代方法数 | 12 | 4（iaaft, fft, cycle_phase_A, random_reorder） |
| 重复次数 | 20 | 3 |
| 替代数据数 | 100 | 30 |
| D2 耦合点数 | 7-9 | 3 |
| E5 T 值数 | 5 | 3 |
| E7 噪声级数 | 6 | 3 |
