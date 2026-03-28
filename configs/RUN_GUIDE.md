# 实验运行指南

## 1. 前置准备

先安装项目：

```bash
python -m pip install -e .
```

如果你要做完整实验，建议单独指定输出目录，避免覆盖旧结果：

```bash
python run_experiments.py --experiment all \
  --config configs/full_experiment.yaml \
  --n-jobs 16 \
  --output-dir results/full_20260328
```

---

## 2. 当前 CLI 支持的实验名

`run_experiments.py` 当前注册了这些实验名：

- `bivariate`
- `coupling`
- `noise`
- `topology`
- `surrogate`
- `robustness`
- `trivariate`
- `sso_correlation`
- `cycle_phase`
- `diagnostic_table`
- `regime_boundaries`
- `convergence`
- `noise_robustness`
- `node_count`
- `all`

注意：

- `all` 会按上面这 14 个实验顺序全部运行。
- 代码仓库里虽然还有 `edge_density` 和 `subsampling` 模块，但它们目前没有接到 `run_experiments.py` 的 CLI 注册表里，所以 `--experiment all` 现在不会运行它们。

---

## 3. 完整实验怎么跑

### 推荐命令

```bash
python run_experiments.py --experiment all \
  --config configs/full_experiment.yaml \
  --n-jobs 16 \
  --output-dir results/full_20260328
```

这条命令会在：

```text
results/full_20260328/
```

下面生成子目录：

```text
bivariate/
coupling/
noise/
topology/
surrogate/
robustness/
trivariate/
sso_correlation/
cycle_phase/
diagnostic_table/
regime_boundaries/
convergence/
noise_robustness/
node_count/
```

### 当前行为说明

- `full_experiment.yaml` 里已经显式配置了 `bivariate / coupling / noise / topology / node_count / surrogate / robustness / trivariate`。
- 对于 `sso_correlation / cycle_phase / diagnostic_table / regime_boundaries / convergence / noise_robustness`，如果 `full_experiment.yaml` 没写对应配置块，代码会自动回退到各模块内部默认参数。
- `--output-dir` 会覆盖 YAML 里的 `output_dir`。
- `bivariate` 不使用 `n_jobs`，其他实验会使用 `n_jobs`。

---

## 4. 推荐的分步跑法

如果你不想一次跑完整套，建议按下面顺序：

### 第一步：冒烟测试

```bash
python run_experiments.py --experiment robustness --config configs/smoke_test.yaml
```

### 第二步：快速验证

```bash
python run_experiments.py --experiment robustness --config configs/quick_validation.yaml
```

### 第三步：主实验

```bash
python run_experiments.py --experiment all \
  --config configs/full_experiment.yaml \
  --n-jobs 16 \
  --output-dir results/full_main
```

### 第四步：失败后局部补跑

例如只补跑诊断类实验：

```bash
python run_experiments.py --experiment diagnostic_table --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
python run_experiments.py --experiment sso_correlation --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
python run_experiments.py --experiment regime_boundaries --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
python run_experiments.py --experiment convergence --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
python run_experiments.py --experiment noise_robustness --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
python run_experiments.py --experiment cycle_phase --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
```

只补跑核心 benchmark：

```bash
python run_experiments.py --experiment bivariate --config configs/full_experiment.yaml --output-dir results/full_main
python run_experiments.py --experiment coupling --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
python run_experiments.py --experiment noise --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
python run_experiments.py --experiment topology --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
python run_experiments.py --experiment node_count --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
python run_experiments.py --experiment surrogate --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
python run_experiments.py --experiment robustness --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
python run_experiments.py --experiment trivariate --config configs/full_experiment.yaml --n-jobs 16 --output-dir results/full_main
```

---

## 5. 常用实验命令

### 替代方法对比

```bash
python run_experiments.py --experiment surrogate --config configs/method_comparison.yaml
```

### HR 系统专项

```bash
python run_experiments.py --experiment robustness --config configs/hr_focused.yaml
```

### 收敛过滤测试

```bash
python run_experiments.py --experiment robustness --config configs/convergence_test.yaml
```

### 节点数量扫描

```bash
python run_experiments.py --experiment node_count --config configs/default.yaml --n-jobs 8
```

---

## 6. 结果波动与参数建议

目前代码里的 sweep 实验默认已经做了两件事来降低误差条波动：

- 对 `coupling_strength`、`noise`、`network_topology`、`node_count` 等实验，默认固定同一个实验点上的图结构，只重复数据随机性和 surrogate 随机性。
- 图上的误差条默认按 `SEM` 画，不再用 `std`。

如果你想把图结构波动也算进误差条，可以在对应配置块里把：

```yaml
vary_graph_across_reps: true
```

打开。

另外建议：

- `n_surrogates >= 19` 才有可能在 `alpha=0.05` 下拒绝原假设。
- 完整实验更推荐 `n_surrogates=99` 或更高，尤其是看 `z-score / delta AUROC` 时更稳定。
- `--n-jobs` 建议先从物理核心数附近开始，不要盲目开太大。

---

## 7. 常用选项

```bash
# 覆盖并行数
python run_experiments.py --experiment robustness --config configs/xxx.yaml --n-jobs 4

# 覆盖输出目录
python run_experiments.py --experiment robustness --config configs/xxx.yaml --output-dir results/my_test

# 使用默认配置跑全部注册实验
python run_experiments.py --experiment all --config configs/default.yaml
```

---

## 8. 典型输出结构

```text
results/full_20260328/
├── bivariate/
├── coupling/
├── noise/
├── topology/
├── surrogate/
├── robustness/
│   ├── T_sweep/
│   ├── coupling_sweep/
│   ├── obs_noise_sweep/
│   ├── dyn_noise_sweep/
│   ├── ablation_summary.csv
│   └── experiment_parameters.csv
├── trivariate/
├── sso_correlation/
├── cycle_phase/
│   ├── E2_rossler/
│   ├── E3_oscillatory/
│   └── E4_chaotic/
├── diagnostic_table/
├── regime_boundaries/
├── convergence/
├── noise_robustness/
└── node_count/
```

---

## 9. SECCM 常用参数

在各实验配置块的 `seccm_kwargs` 中可设置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `adaptive_rho` | `true` | 自适应 `min_rho` 阈值，基于 surrogate 分布分位数 |
| `adaptive_rho_quantile` | `0.95` | 自适应阈值使用的分位数 |
| `theiler_w` | `"auto"` | Theiler 窗口，`"auto"` 表示用中位 `tau` |
| `convergence_filter` | `false` 或实验自定义 | 是否做收敛性过滤 |
| `convergence_threshold` | `0.0` | 收敛阈值 |
| `E_method` | `"simplex"` | `E` 的选择方法，可用 `simplex/fnn/cao` |
| `iaaft_max_iter` | `200` | iAAFT 最大迭代次数 |
| `use_gpu` | `"auto"` | 自动检测 CuPy/GPU；没有 GPU 时自动退回 CPU |

如果你要复现实验，建议把配置文件、命令行参数和输出目录一起记录下来。
