# 实验运行指南

## 配置文件一览

| 配置文件 | 用途 | 预计耗时 | 适用场景 |
|---------|------|---------|---------|
| `smoke_test.yaml` | 冒烟测试 | 2-5 分钟 | 代码修改后快速验证 |
| `quick_validation.yaml` | 快速验证 | 15-30 分钟 | 检查各系统 AUROC 是否正常 |
| `method_comparison.yaml` | 方法对比 | 30-60 分钟 | 找每个系统的最佳 surrogate |
| `hr_focused.yaml` | HR 专项 | 10-20 分钟 | 验证 HR 新参数效果 |
| `convergence_test.yaml` | 收敛过滤 | 20-40 分钟 | 测试 convergence_filter |
| `adaptive_ablation.yaml` | 消融实验 | 30-60 分钟 | 量化 adaptive_rho/theiler_w 效果 |
| `full_experiment.yaml` | 完整实验 | 3-8 小时 | 论文级数据收集 |

> 耗时基于 `n_jobs=8` 估算，实际取决于 CPU 核心数。

---

## 运行命令

### 1. 冒烟测试（推荐首先运行）

```bash
python run_experiments.py --experiment robustness --config configs/smoke_test.yaml
```

### 2. 快速验证

```bash
python run_experiments.py --experiment robustness --config configs/quick_validation.yaml
```

### 3. 替代方法对比

```bash
python run_experiments.py --experiment surrogate --config configs/method_comparison.yaml
```

### 4. HR 系统专项

```bash
python run_experiments.py --experiment robustness --config configs/hr_focused.yaml
```

### 5. 收敛过滤测试

```bash
# 开启 convergence_filter（默认）
python run_experiments.py --experiment robustness --config configs/convergence_test.yaml

# 对比：手动将 yaml 中 convergence_filter 改为 false，output_dir 改为 results/convergence_test_off
```

### 6. 消融实验（4 组对比）

```bash
# 组 1：全开（默认）
python run_experiments.py --experiment robustness --config configs/adaptive_ablation.yaml

# 组 2-4：修改 seccm_kwargs 并更改 output_dir 后分别运行
```

### 7. 完整实验（论文数据）

```bash
# 运行所有 6 个实验
python run_experiments.py --experiment all --config configs/full_experiment.yaml --n-jobs 16

# 或只运行鲁棒性消融
python run_experiments.py --experiment robustness --config configs/full_experiment.yaml --n-jobs 16
```

---

## 常用选项

```bash
# 覆盖并行数
python run_experiments.py --experiment robustness --config configs/xxx.yaml --n-jobs 4

# 覆盖输出目录
python run_experiments.py --experiment robustness --config configs/xxx.yaml --output-dir results/my_test

# 可选实验名：bivariate, coupling, noise, topology, surrogate, robustness, all
```

---

## 输出结构

```
results/<experiment_name>/
├── robustness/
│   ├── sub_a_T_sweep/          # AUROC vs T
│   ├── sub_b_coupling_sweep/   # AUROC vs coupling
│   ├── sub_c_obs_noise/        # AUROC vs obs noise
│   ├── sub_d_dyn_noise/        # AUROC vs dyn noise
│   ├── *_line_auroc.pdf        # 折线图
│   ├── *_heatmap_delta.pdf     # 热力图
│   ├── summary_table.csv       # 汇总表
│   └── summary_table.tex       # LaTeX 表
├── surrogate/
│   ├── comparison_heatmap.pdf
│   └── ...
└── ...
```

---

## SECCM 关键参数说明

在 `seccm_kwargs` 中可设置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `adaptive_rho` | `true` | 自适应 min_rho 阈值（基于 surrogate 分布 95th percentile）|
| `theiler_w` | `"auto"` | Theiler 窗口（`"auto"` = 中位 tau，`0` = 关闭）|
| `convergence_filter` | `false` | 收敛性过滤（Kendall-τ 检验）|
| `convergence_threshold` | `0.0` | 收敛分数阈值（τ > threshold 才通过）|
| `E_method` | `"simplex"` | E 选择方法（`"simplex"`, `"fnn"`, `"cao"`）|
