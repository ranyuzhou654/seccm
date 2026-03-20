# Improvement Log: Comments Review Fixes

**Date:** 2026-03-20
**Scope:** 基于 `comments/01-08` 代码审查的系统性修复

## 修复总结

共 8 项修复，按严重程度排序：

### Fix 1: `p < alpha` → `p <= alpha` (HIGH)
- **文件**: `surrogate_ccm/testing/se_ccm.py`
- **问题**: `n_surrogates=19` 时最小可达 p=0.05，`p < 0.05` 永远无法拒绝
- **修复**: 改为 `p <= alpha`，并在 `__init__` 中添加 `n_surrogates` 过低的警告

### Fix 2: twin_surrogate 时间对齐 (HIGH)
- **文件**: `surrogate_ccm/surrogate/twin_surrogate.py`
- **问题**: 嵌入状态 `X[k]` 对应 `x[k+offset]`，但 `_construct_trajectory` 取的是 `x[k]`
- **修复**: 传入 `x_aligned = x[offset:offset+N]`，确保标量轨迹与嵌入状态对齐

### Fix 3: multivariate_iaaft 跨谱保持 (HIGH)
- **文件**: `surrogate_ccm/surrogate/multivariate_surrogate.py`
- **问题**: 每个变量独立保留自己的相位，破坏了跨变量相位关系
- **修复**: 使用参考变量的公共相位 + 原始相位偏差，锁住跨谱结构
- **效果**: 相对跨谱误差从 ~0.12 降至 ~0.045

### Fix 4: Lorenz 耦合归一化 (HIGH)
- **文件**: `surrogate_ccm/generators/lorenz.py`
- **问题**: 其他 6 个系统都按入度归一化耦合，Lorenz 没有
- **修复**: 添加 `k_in_safe` 归一化，与其他系统一致

### Fix 5: ODE t_eval off-by-one (MEDIUM)
- **文件**: 5 个 ODE 生成器 (Lorenz, Rössler, FHN, Kuramoto, HR)
- **问题**: `np.linspace(0, total*dt, total)` 步长为 `total*dt/(total-1)` ≠ `dt`
- **修复**: 改为 `np.arange(total) * dt`

### Fix 6: 网络拓扑方向性不一致 (HIGH)
- **文件**: `surrogate_ccm/generators/network.py`
- **问题**: ER 是有向图，WS/Ring 通过 `.to_directed()` 变成全互惠，混杂了方向性
- **修复**: WS/Ring 改为随机取向（每条无向边随机分配方向），消除互惠性混杂

### Fix 7: iAAFT 收敛判据 (MEDIUM)
- **文件**: `surrogate_ccm/surrogate/iaaft_surrogate.py`
- **问题**: 收敛判据看的是连续迭代差异，不是距目标谱的距离
- **修复**: 改为检查与目标幅度谱的相对误差

### Fix 8: 实验失败处理一致性 (HIGH)
- **文件**: 4 个实验脚本
- **问题**:
  - exp_noise/coupling/topology: 只捕获 `RuntimeError`
  - exp_robustness: 不断换 seed 重试 = survivorship bias
- **修复**:
  - 统一捕获 `Exception`，包裹整个 worker
  - 删除 robustness 的重试循环，使用固定 seed 列表
  - 输出 `success_rate` 指标

## 未修复的问题（评估后认为暂不需要改）

- **可观测量硬编码** (MEDIUM): 各系统只返回一个分量。这是设计选择，CCM 文献中通常也只用单分量
- **自适应选择器阈值硬编码** (MEDIUM): 已有实验验证支撑当前阈值，暂不移到配置
- **AUC 逻辑重复** (MEDIUM): `se_ccm.py:score()` 和 `metrics.py` 有部分重叠，但功能不同
- **aaft_surrogate 未使用的 ranks 变量** (LOW): 不影响正确性
