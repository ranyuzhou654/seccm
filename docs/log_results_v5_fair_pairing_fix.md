# D1 Diagnostic Table Fair-Comparison Fix

## Problem

`results_v5/diagnostic_table` 的 surrogate 排名不是建立在同一批数据 realization 上。

原实现位于 [surrogate_ccm/experiments/exp_diagnostic_table.py](/Users/Zhuanz/Downloads/research/surrogate_v2/surrogate_ccm/experiments/exp_diagnostic_table.py)，使用：

```python
seed = base_seed + hash((sys_name, surr, rep)) % (2**31)
```

这会带来两个直接问题：

1. `surr` 被混进 realization seed，不同 surrogate 方法看到的是不同 adjacency 和不同 time series。
2. `hash(...)` 是 Python 进程相关的，默认不稳定，无法保证跨进程重跑可复现。

结果层面，这会削弱 `diagnostics_agg.csv` 的 surrogate 排名可信度，因为方法间比较不是 paired comparison。

## Root Cause

实验脚本没有区分两类随机性：

- `realization randomness`
  - 网络拓扑生成
  - 动力系统初值 / 数据 realization
- `analysis randomness`
  - surrogate 生成
  - SE-CCM 内部的随机采样

把这两类随机性压成一个 `seed`，又把 `surr` 混进去，就破坏了“同一数据上比较不同 surrogate”的基本前提。

## Changes

修改文件：

- [surrogate_ccm/experiments/exp_diagnostic_table.py](/Users/Zhuanz/Downloads/research/surrogate_v2/surrogate_ccm/experiments/exp_diagnostic_table.py)
- [tests/test_diagnostic_table_pairing.py](/Users/Zhuanz/Downloads/research/surrogate_v2/tests/test_diagnostic_table_pairing.py)

### 1. 引入稳定 seed 派生函数

新增 `_stable_seed(base_seed, *parts)`，使用 `hashlib.blake2b` 派生稳定的 31-bit seed，替代 Python 内建 `hash(...)`。

### 2. 显式拆分两类 seed

新增 `_build_run_args(...)`，对每个 `system × rep`：

- 固定一个 `realization_seed`
- 针对每个 surrogate 单独派生 `analysis_seed`

这样保证：

- 同一个 `system × rep` 下，所有 surrogate 共用同一份数据 realization
- surrogate 自身的随机分析过程仍然独立且可复现

### 3. 调整 worker 的 seed 使用语义

- `generate_network(...)` 使用 `realization_seed`
- `system.generate(...)` 使用 `realization_seed`
- `SECCM(...)` 使用 `analysis_seed`
- SSO 里额外生成 surrogate 时也基于 `analysis_seed`

### 4. 结果表新增更清晰的元信息

`full_diagnostics.csv` 结果行现在包含：

- `rep`
- `seed`
  - 现在表示 `realization_seed`
- `analysis_seed`

这样后处理时可以直接验证：

- 同一 `system + seed` 是否覆盖了全部 surrogate
- 同一 surrogate 是否使用了独立分析 seed

### 5. 增加 paired-run 覆盖率输出

实验完成后会打印：

- `system×seed` 中有多少 realization 拥有全部 surrogate 结果

这让运行时就能暴露不完整 pairing，而不是等结果落盘后再排查。

## Validation

新增回归测试 [tests/test_diagnostic_table_pairing.py](/Users/Zhuanz/Downloads/research/surrogate_v2/tests/test_diagnostic_table_pairing.py)：

- `_stable_seed` 同输入稳定、不同输入可区分
- `_build_run_args` 保证：
  - 同一 `system × rep` 的 `realization_seed` 在 surrogate 间相同
  - 不同 surrogate 的 `analysis_seed` 不同

## Impact

修复后，D1 诊断表实验的 surrogate 排名变成真正的 paired comparison。

这意味着后续重跑生成的新 `full_diagnostics.csv` / `diagnostics_agg.csv` 才能被合理解释为：

- “同一批系统 realization 上，不同 surrogate 方法谁更好”

而不是：

- “不同 surrogate 恰好跑在不同 realization 上时，平均表现如何”

## Follow-up

这次修改只修复了 D1 诊断表实验的公平比较问题，没有重跑 `results_v5`。

如果要恢复 `results_v5/diagnostic_table` 的结论可信度，需要基于修复后的脚本重新运行 D1 实验，并更新：

- `full_diagnostics.csv`
- `diagnostics_agg.csv`
- `regime_classification.csv`
- `regime_summary.json`
- 对应图表输出
