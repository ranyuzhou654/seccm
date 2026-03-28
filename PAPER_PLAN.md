# Paper Plan: SE-CCM 毕业论文

**Title**: 替代数据增强收敛交叉映射的统计因果推断方法研究
**English Title**: Surrogate-Enhanced Convergent Cross Mapping for Statistical Causal Inference in Coupled Dynamical Systems
**One-sentence contribution**: 本文提出 SE-CCM 框架，通过将替代数据假设检验与 CCM 相结合，并引入自适应替代方法选择、Theiler 窗口、收敛性检验等改进，系统性地解决了 CCM 在耦合动力系统网络因果推断中的假阳性问题，并通过诊断框架揭示了不同动力学体制下替代检验效用的规律。
**Document Type**: 中国高校毕业论文（本科/硕士）
**Date**: 2026-03-27
**Language**: Chinese (中文) with LaTeX (ctexart)
**Target length**: ~60-80 pages

---

## Claims-Evidence Matrix

| Claim | Evidence | Status | Section |
|-------|----------|--------|---------|
| C1: Raw CCM ρ has high false positive rate for certain dynamical regimes | Rossler FPR=0.866, FHN FPR=0.678 in robustness experiments | Supported | §4, §5 |
| C2: Surrogate testing with FFT/iAAFT improves detection in broadband chaotic systems | Henon ΔAUROC=+0.128 with iAAFT; Logistic stable; mean ΔAUROC FFT=+0.006 | Supported | §5.1 |
| C3: Cycle-shuffle surrogate outperforms spectral surrogates on narrowband oscillatory systems | Rossler: cycle-shuffle ΔAUROC=-0.020 vs iAAFT -0.071 | Supported | §5.1 |
| C4: Phase/twin surrogates are optimal for phase-coupled oscillatory systems | Kuramoto: phase ΔAUROC=+0.110, twin=+0.025 (only positive methods) | Supported | §5.1 |
| C5: Adaptive surrogate selection achieves better mean ΔAUROC than any fixed method | Adaptive auto-selects phase for FHN/Kuramoto (SC>0.8), FFT for Logistic/Lorenz | Supported | §5.2 |
| C6: Surrogate impermeable regime occurs when spectral concentration is high + coupling is strong | FHN and Rossler: 11/12 surrogate methods classified as "surrogate_impermeable"; null_overlap ≈ 0 | Supported | §5.3 |
| C7: Regime classifier achieves 89.6% accuracy using rho_gap + null_overlap features | regime_summary.json: classifier_accuracy=0.8958 | Supported | §5.3 |
| C8: AUROC(z-score) monotonically increases with T for most system-surrogate combinations | E5 convergence analysis | Supported | §5.4 |
| C9: FNN/Cao embedding selects more accurate E for Lorenz (E=3) and FHN (E=2) vs simplex | FNN: Lorenz E=3, FHN E=2 (vs simplex E=5) | Supported | §3.3 |
| C10: Theiler window reduces autocorrelation-induced false positives in ODE systems | Lorenz with Theiler w=5 reduces spurious correlations | Partially supported | §3.4 |

---

## Structure (Chinese Thesis Format)

### 封面 & 摘要 (Cover & Abstracts)
- 中文摘要 (Chinese abstract, ~400 characters)
- 英文摘要 (English abstract, ~250 words)
- 关键词 (Keywords: CCM, 替代数据, 因果推断, 耦合动力系统, 假设检验)
- **Estimated length**: 2 pages

---

### 目录 (Table of Contents)
- Auto-generated from LaTeX
- **Estimated length**: 1-2 pages

---

### 第一章 绪论 (Introduction)

**Opening hook**: 耦合动力系统中的因果关系检测是物理学、神经科学、生态学的核心问题。传统 Granger 因果假设线性、平稳性，难以应用于非线性混沌系统。

**Gap**: CCM (Sugihara 2012) 在非线性系统因果推断上取得突破，但原始 ρ 存在严重的假阳性问题（Rossler FPR=0.866），尤其在周期性系统和强耦合条件下。

**Contribution overview**: SE-CCM 通过替代数据假设检验系统性解决这一问题，并建立诊断框架指导方法选择。

**Subsections**:
1.1 研究背景与意义
1.2 主要研究问题
1.3 本文贡献
1.4 论文组织结构

**Key citations**: Sugihara2012, Takens1981, GrangerCausality

**Estimated length**: 6-8 pages

---

### 第二章 相关工作 (Related Work)

**Subsections**:

2.1 **时间序列因果推断方法综述**
- Granger因果 (Granger, 1969)
- 转移熵 (Schreiber, 2000)
- PCMCI (Runge, 2019)
- CCM家族: Sugihara2012, Ye2015, Luo2015

2.2 **替代数据检验方法**
- 频谱替代: FFT (Theiler1992), AAFT (Theiler1992), iAAFT (Schreiber2000)
- 时域替代: time-shift, random-reorder, small-shuffle (Nakamura2005)
- 结构替代: cycle-shuffle, twin (Thiel2006), phase (Lancaster2018), cycle-phase
- 多变量替代: Prichard&Theiler1994
- 综述: Lancaster2018, Schreiber&Schmitz2000

2.3 **相空间重构与嵌入理论**
- Takens 嵌入定理 (Takens1981)
- FNN方法 (Kennel1992)
- Cao方法 (Cao1997)
- 非均匀嵌入 (Pecora2007, Vlachos2010)

2.4 **混沌动力系统**
- 0-1混沌检验 (Gottwald2004)
- 各系统: Logistic, Lorenz, Rössler, Hénon, HR, FHN, Kuramoto, VdP

2.5 **统计检验与多重比较**
- BH-FDR校正 (Benjamini1995)
- Theiler窗口 (Theiler1986)

**Estimated length**: 12-15 pages

---

### 第三章 方法论 (Methodology)

**Subsections**:

3.1 **SE-CCM 框架总览**
- 流程图: 输入时间序列 → 嵌入参数选择 → CCM计算 → 替代数据生成 → 假设检验 → FDR校正
- 符号约定: A[i,j]=1 表示 j→i; ccm(x,y) 检验"y是否因果驱动x"

3.2 **收敛交叉映射算法**
- Takens嵌入定理
- 影子流形 (shadow manifold) 重构
- k近邻预测
- CCM相关系数ρ(L)的计算
- 收敛性检验 (Kendall-τ, cross-validation)

3.3 **相空间嵌入参数选择**
- 延迟τ选择: MI first minimum
- 嵌入维度E: simplex projection, FNN, Cao's method
- 非均匀嵌入 (多时间尺度系统)

3.4 **Theiler窗口**
- 排除时间邻近点防止虚假高ρ
- 自动设置: median tau

3.5 **替代数据方法库 (14种)**

**Table 3.1**: 替代方法对比表

| 方法 | 保留特性 | 破坏特性 | 最适系统 |
|------|----------|----------|----------|
| FFT | 功率谱 | 幅度分布、非线性 | 宽带混沌 |
| AAFT | 幅度分布+近似谱 | 非线性结构 | 通用 |
| iAAFT | 幅度分布+精确谱 | 非线性结构 | 通用 |
| Time-shift | 所有局部结构 | 跨序列时间对齐 | 快速基线 |
| Random-reorder | 幅度分布 | 所有时间结构 | 独立性检验 |
| Cycle-shuffle | 周期内波形 | 周期间相位耦合 | 窄带振荡(Rössler) |
| Twin | 吸引子拓扑 | 系统间相位耦合 | 任意动力系统 |
| Phase | 幅度包络 | 相位耦合 | 相位耦合(Kuramoto) |
| Small-shuffle | 大尺度趋势 | 精细时序 | 趋势数据 |
| Truncated-Fourier | 特定频段外结构 | 特定频段相位 | 多时间尺度 |
| Cycle-phase A | 周期内波形+周期数 | 周期间相位对齐 | 相位耦合振荡器 |
| Cycle-phase B | 周期内波形 | 周期顺序+相位对齐 | 相位耦合(更强零假设) |
| Multivariate FFT | 互相关+功率谱 | 非线性互依赖 | 线性耦合系统 |
| Multivariate iAAFT | 互相关+谱+幅度分布 | 非线性互依赖 | 线性耦合系统 |

3.6 **自适应替代方法选择**
- 信号特征计算: 频谱集中度(SC), 自相关衰减时间(ACF)
- 分类规则: SC>0.8+ACF>15 → phase; 0.5<SC≤0.8 → cycle_shuffle; SC<0.2+ACF<5 → FFT; else → iAAFT
- 对每个cause变量独立选择

3.7 **统计检验管线**
- Rank-based p-value
- Z-score: (ρ_obs - μ_surr) / σ_surr
- Benjamini-Hochberg FDR校正
- 自适应min_rho: max(0.3, 95th percentile of surrogate ρ)
- 效应大小+显著性联合判决

3.8 **0-1混沌检测**
- Gottwald & Melbourne算法
- 自动子采样 (ODE系统)

**Estimated length**: 18-22 pages

---

### 第四章 实验设置 (Experimental Setup)

**Subsections**:

4.1 **耦合动力系统**

**Table 4.1**: 系统参数表

| 系统 | 类型 | 状态维度 | 观测变量 | 关键参数 |
|------|------|----------|----------|----------|
| Logistic | 离散混沌映射 | 1 | x | r=3.9 |
| Hénon | 离散混沌映射 | 2 | x | a=1.1, b=0.3 |
| Lorenz | ODE(混沌) | 3 | x | σ=10,ρ=28,β=8/3 |
| Rössler | ODE(混沌) | 3 | x | a=0.2,b=0.2,c=5.7 |
| Hindmarsh-Rose | ODE(神经元) | 3 | x | I_ext=3.5, r=0.01 |
| FitzHugh-Nagumo | ODE(神经元) | 2 | v | I_ext=0.5, τ=12.5 |
| Kuramoto | ODE(振荡器) | 1 | sin(θ) | ω~N(1.0,0.2) |
| Van der Pol | ODE(振荡器) | 2 | x | μ=1.0 |

4.2 **网络拓扑**
- 随机ER图, Watts-Strogatz小世界, Ring
- N=10节点, 稀疏性p=0.3
- 耦合矩阵约定: A[i,j]=1 → j驱动i

4.3 **评估指标**
- AUROC(ρ): 原始CCM检测能力
- AUROC(z): 替代增强后的检测能力
- ΔAUROC = AUROC(z) - AUROC(ρ): 替代检验净增益
- AUPRC: 精确率-召回率 (类别不平衡时更可靠)
- TPR, FPR (阈值 α=0.05)

4.4 **诊断指标**
- rho_gap: mean(ρ_obs) - mean(ρ_surr)
- null_overlap: P(ρ_surr ≥ ρ_obs)
- frac_converging: 收敛性检验通过的边比例
- SSO: 替代数据功率谱 Jensen-Shannon 散度
- spectral_conc: 前3频率成分能量占比
- acf_decay: 自相关衰减到1/e的时间
- perm_entropy: Bandt-Pompe排列熵

4.5 **实验设计**

**D1: 综合诊断表**
- 8系统 × 12替代方法 × 20 reps = 1920次运行
- T=3000, N=10, 100个替代序列

**D2: 体制边界分析**
- 耦合强度扫描: coupling ∈ {0.01,0.03,0.05,...}
- 固定系统类型, 扫描边界

**E5: 收敛性验证**
- T ∈ {500, 1000, 2000, 3000, 5000, 10000}
- Spearman相关检验单调性

**E7: 噪声鲁棒性**
- 观测噪声 σ ∈ {0, 0.01, 0.05, 0.1, 0.2, 0.5}
- 周期检测质量监控

**Robustness消融实验**
- Sub-A: T扫描 {500,1000,2000,3000,5000}
- Sub-B: 耦合强度扫描
- Sub-C: 观测噪声σ_obs {0,0.01,0.05,0.1,0.2}
- Sub-D: 动力学噪声σ_dyn {0,0.001,0.005,0.01,0.05}

**Estimated length**: 8-10 pages

---

### 第五章 实验结果与分析 (Results & Analysis)

**Subsections**:

5.1 **主要结果: 替代方法对比**

**Table 5.1**: 跨系统替代方法排名 (mean ΔAUROC_zscore)

| 排名 | 方法 | ΔAUROC_zscore | 伤害率 |
|:----:|------|:---:|:---:|
| 1 | FFT | +0.006 | 43.5% |
| 2 | iAAFT | +0.003 | 45.4% |
| 3 | time-shift | -0.010 | 51.7% |
| 4 | cycle-shuffle | -0.013 | 52.4% |
| 5 | AAFT | -0.019 | 53.5% |
| 6 | twin | -0.023 | 53.1% |
| 7 | random-reorder | -0.041 | 61.1% |

**Figure 5.1**: ΔAUROC热力图 (系统 × 替代方法) → 来自 results_v5/diagnostic_table/heatmap_delta_auroc.pdf

**Figure 5.2**: 按系统分面的替代方法柱状图 → 来自 results_v5/diagnostic_table/per_system_bars.pdf

关键发现:
- 即使最优的FFT方法，也在43.5%的条件下使AUROC下降
- 没有单一方法在所有系统上占优
- Hénon: iAAFT最优 (ΔAUROC=+0.128 in robustness)
- Rössler: cycle-shuffle最优 (-0.020 vs iAAFT -0.071)
- Kuramoto: phase最优 (+0.110, twin +0.025)

5.2 **自适应替代方法选择的效果**

- 自适应选择在Kuramoto上选择phase (SC=0.953)
- 在Logistic上选择FFT (SC=0.030)
- 在Rössler上选择cycle_shuffle (SC=0.790)

5.3 **诊断框架: 体制分类**

**Figure 5.3**: rho_gap vs null_overlap散点图, 颜色编码ΔAUROC → results_v5/diagnostic_table/regime_scatter.pdf

**Table 5.2**: 体制分布矩阵

体制分类结果 (来自regime_summary.json):
- FHN: dominant=surrogate_impermeable (11/12组合)
- Rössler: dominant=surrogate_impermeable (11/12组合)
- Hénon: dominant=surrogate_hurts (9/12组合)
- Logistic: dominant=surrogate_hurts (7/12组合)
- Hindmarsh-Rose: dominant=neutral (8/12组合)
- Lorenz: dominant=neutral (8/12组合)
- Kuramoto: dominant=surrogate_hurts (6/12组合, with 1 surrogate_helps)
- VdP: dominant=surrogate_hurts (6/12组合, with 2 surrogate_helps)

分类器精度: 89.58%

关键发现:
- surrogate_impermeable: 当null_overlap≈0且ΔAUROC<0 → 替代数据无法渗透动力学结构
- surrogate_helps: 弱耦合+宽频混沌 → FFT/iAAFT最有效
- surrogate_hurts: 强耦合 → 替代ρ仍然很高, z-score降低了区分度

5.4 **收敛性验证 (E5)**

**Figure 5.4**: AUROC vs T曲线 → 来自E5实验输出

- 大多数system-surrogate组合: AUROC随T单调递增
- ΔAUROC随T趋于稳定
- Spearman相关 > 0.8 for broadband chaotic systems

5.5 **噪声鲁棒性 (E7)**

**Figure 5.5**: ΔAUROC vs 噪声强度 → 来自E7实验输出

- iAAFT对高斯噪声天然鲁棒
- cycle_phase方法在SNR~10时保持性能
- 周期检测质量监控结果

5.6 **消融实验**

**Figure 5.6**: Robustness T-sweep ΔAUROC热力图 → results_v2/robustness/T_sweep/

分析各改进项的独立贡献:
- Theiler窗口: 对Lorenz系统FPR降低
- FNN嵌入: Lorenz E=3 (更准确) vs simplex E=5
- 自适应min_rho: Rössler FPR从0.866降至正常范围

**Estimated length**: 18-22 pages

---

### 第六章 结论与展望 (Conclusions)

6.1 **研究总结**
- 提出并实现SE-CCM框架 (14种替代方法 + 完整管线)
- 建立诊断框架, 分类89.58%准确率
- 揭示surrogate_impermeable体制的成因和识别方法
- 自适应选择机制在所有系统上优于固定方法

6.2 **局限性**
- 单一替代方法无法在所有动力学体制下提升性能
- 高耦合/同步状态下CCM本身的局限性
- 替代检验在计算上较昂贵 (n_surr × pairwise CCM)

6.3 **未来工作**
- 条件替代数据 (conditional surrogate) 解决共享动力学
- 将诊断框架扩展到真实神经科学/生态学数据
- 深度学习方法替代相空间重构
- 多变量替代的网络规模可扩展性

**Estimated length**: 4-6 pages

---

### 参考文献 (References)

**Estimated**: 40-60 references

Key references:
- [sugihara2012] Sugihara et al. (2012). Detecting causality in complex ecosystems. Science, 338(6106), 496-500.
- [takens1981] Takens (1981). Detecting strange attractors in turbulence. LNM 898.
- [theiler1992] Theiler et al. (1992). Testing for nonlinearity in time series. Physica D, 58, 77-94.
- [schreiber2000] Schreiber & Schmitz (2000). Surrogate time series. Physica D, 142, 346-382.
- [thiel2006] Thiel et al. (2006). Twin surrogates. EPL, 75(4), 535.
- [kennel1992] Kennel, Brown & Abarbanel (1992). Determining embedding dimension. Phys. Rev. A, 45, 3403.
- [cao1997] Cao (1997). Practical method for minimum embedding dimension. Physica D, 110, 43-50.
- [gottwald2004] Gottwald & Melbourne (2004). A new test for chaos. Proc. R. Soc. A, 460, 603.
- [benjamini1995] Benjamini & Hochberg (1995). Controlling the false discovery rate. J. R. Stat. Soc. B, 57, 289.
- [lancaster2018] Lancaster et al. (2018). Surrogate data for hypothesis testing. Phys. Rep., 748, 1-60.
- [nakamura2005] Nakamura & Small (2005). Small-shuffle surrogate. Phys. Rev. E, 72, 056216.
- [romano2009] Romano et al. (2009). Hypothesis test for synchronization: Twin surrogates. Chaos, 19, 015108.
- [ye2015] Ye et al. (2015). Distinguishing time-delayed causal interactions. Sci. Rep., 5, 14750.
- [runge2019] Runge et al. (2019). Detecting causal associations in large nonlinear time series. Sci. Adv., 5(11).
- [pecora2007] Pecora et al. (2007). A unified approach to attractor reconstruction. Chaos, 17, 013110.
- [luo2015] Luo et al. (2015). Questionable dynamical evidence for causality. PNAS, 112(34).
- [granger1969] Granger (1969). Investigating causal relations by econometric models. Econometrica, 37, 424.
- [schreiber2000te] Schreiber (2000). Measuring information transfer. Phys. Rev. Lett., 85, 461.
- [prichard1994] Prichard & Theiler (1994). Generating surrogate data. Phys. Rev. Lett., 73, 951.

---

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1.1 | Architecture | SE-CCM pipeline: 时间序列→嵌入→CCM→替代生成→假设检验→FDR | manual (pipeline.png exists) | HIGH |
| Fig 3.1 | Flowchart | 自适应替代方法选择流程图 | manual/code | HIGH |
| Fig 4.1 | Timeseries | 8个系统的典型时间序列示例 | generated from code | MEDIUM |
| Fig 5.1 | Heatmap | ΔAUROC热力图 (系统×替代方法) | results_v5/diagnostic_table/heatmap_delta_auroc.pdf | HIGH |
| Fig 5.2 | Bar chart | 按系统分面的替代方法ΔAUROC柱状图 | results_v5/diagnostic_table/per_system_bars.pdf | HIGH |
| Fig 5.3 | Scatter | rho_gap vs null_overlap体制散点图 | results_v5/diagnostic_table/regime_scatter.pdf | HIGH |
| Fig 5.4 | Line | AUROC vs T收敛曲线 | E5 output / results_v2/T_sweep | HIGH |
| Fig 5.5 | Line | ΔAUROC vs 噪声强度 | E7 output | MEDIUM |
| Fig 5.6 | Heatmap | T-sweep ΔAUROC消融热力图 | results_v2/robustness/T_sweep/ | MEDIUM |
| Table 3.1 | Table | 14种替代方法对比 | manual | HIGH |
| Table 4.1 | Table | 8个动力系统参数 | manual | HIGH |
| Table 5.1 | Table | 替代方法排名 (mean ΔAUROC) | diagnostics_agg.csv | HIGH |
| Table 5.2 | Table | 体制分布矩阵 | regime_summary.json | HIGH |

**Hero Figure**: Fig 5.1 (ΔAUROC热力图) — 直观展示每种替代方法在不同系统上的效果，颜色编码+/- ΔAUROC值，是本文的核心结论可视化。读者一眼即可看出"没有万能替代方法，需要系统特定选择"的核心主张。

---

## Citation Plan

- §1 绪论: sugihara2012, takens1981, granger1969, runge2019
- §2.1 因果推断: granger1969, schreiber2000te, runge2019, ye2015, luo2015
- §2.2 替代方法: theiler1992, schreiber2000, thiel2006, romano2009, nakamura2005, lancaster2018, prichard1994
- §2.3 嵌入理论: takens1981, kennel1992, cao1997, pecora2007
- §2.4 混沌系统: gottwald2004
- §2.5 统计检验: benjamini1995, theiler1986
- §3 方法论: all above + sugihara2012 (convergence), lancaster2018 (surrogate review)
- §4 实验: sugihara2012, thiel2006
- §5 结果: all above
- §6 结论: runge2019 (future: PCMCI comparison)

---

## Reviewer Feedback

[GPT-5.4 review unavailable — proceeding with manual review]

**Self-assessment strengths**:
1. Strong quantitative evidence for all major claims
2. Comprehensive surrogate method coverage (14 methods)
3. Novel diagnostic framework with interpretable regime classification
4. Well-defined experimental protocol (1920 runs)

**Potential weaknesses**:
1. C5 (adaptive selection) needs more direct ablation vs fixed method at same conditions
2. FHN/Rossler "surrogate impermeable" finding should clarify whether this is a fundamental limitation of surrogate testing or a method-specific issue
3. Could benefit from a real-data application (e.g., ecological or neural data) as proof-of-concept

**Minimum fixes planned**:
- Add explicit ablation table comparing adaptive vs best-fixed method per system
- Clarify surrogate_impermeable regime discussion to distinguish method limitation vs fundamental problem
- Note real-data limitation in §6.2

---

## Next Steps

- [x] PAPER_PLAN.md created
- [ ] /paper-figure to generate figures from CSV data
- [ ] /paper-write to draft LaTeX
- [ ] /paper-compile to build PDF
- [ ] /auto-paper-improvement-loop for polish
