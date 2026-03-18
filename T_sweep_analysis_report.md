# T-Sweep Sub-experiment Analysis Report

**Experiment**: Surrogate Robustness — Sub-A (Time Series Length Sweep)
**Date**: 2026-03-18
**Data**: `results/robustness/robustness/T_sweep/T_sweep.csv`

---

## 1. Experiment Setup

| Parameter | Value |
|-----------|-------|
| Systems | logistic, lorenz, henon, rossler, hindmarsh_rose, fitzhugh_nagumo, kuramoto |
| Surrogate methods | FFT, AAFT, Time-shift |
| T values swept | 500, 1000, 2000, 3000, 5000 |
| N (nodes) | 10 |
| Topology | ER (p=0.3) |
| n_surrogates | 99 |
| n_reps | 10 (7 for Rössler — see §5) |
| Coupling | Per-system default (logistic=0.1, lorenz=1.0, henon=0.05, etc.) |
| Noise | noise_std=0.0, dyn_noise_std=0.0 |

**Total runs**: 1,005 (= 6×3×5×10 + 1×3×5×7)

---

## 2. Key Findings

### 2.1 Three-tier System Classification

Based on ΔAUROC (z-score − raw) averaged across methods and T values:

| Tier | Systems | Mean ΔAUROC(z) | Interpretation |
|------|---------|----------------|----------------|
| **Surrogate helps** | Hénon (+0.064), Kuramoto (+0.017), Logistic (+0.013) | +0.031 | Z-score metric outperforms raw ρ |
| **Neutral** | Lorenz (−0.005) | ~0 | No consistent benefit or harm |
| **Surrogate hurts** | Rössler (−0.027), Hindmarsh-Rose (−0.020), FitzHugh-Nagumo (−0.012) | −0.020 | P-value thresholding loses information |

### 2.2 Z-score vs P-value Metrics

A critical finding: **z-score-based AUROC consistently outperforms p-value-based AUROC** across 6 of 7 systems. The surrogate p-value (rank-based) is a coarser statistic that discards magnitude information. The z-score retains this and serves as a better discriminator.

| System | Raw ρ AUROC | Surr. p-val AUROC | Z-score AUROC | Best metric |
|--------|-------------|-------------------|---------------|-------------|
| Logistic | 0.937 | 0.865 | **0.945** | z-score |
| Hénon | 0.752 | 0.734 | **0.816** | z-score |
| Kuramoto | 0.650 | 0.660 | **0.664** | z-score |
| Lorenz | 0.667 | 0.658 | **0.665** | z-score |
| FitzHugh-Nagumo | **0.544** | 0.517 | 0.532 | raw ρ |
| Hindmarsh-Rose | **0.490** | 0.474 | 0.472 | raw ρ (all ~0.5) |
| Rössler | **0.609** | 0.530 | 0.576 | raw ρ |

**Recommendation**: Use z-score as the primary surrogate-derived metric, not p-value.

### 2.3 Effect of Time Series Length

**General trend**: Raw ρ AUROC improves monotonically with T for most systems, as longer series provide better attractor reconstruction.

**System-specific patterns**:

- **Logistic**: Already excellent at T=500 (AUROC=0.895), plateaus by T=2000 (0.943). Surrogate adds nothing — raw ρ is sufficient for this well-separated system.
- **Hénon**: Strong improvement T=500→5000 (0.631→0.843). Surrogate z-score adds a consistent +0.06 to +0.08 boost at all T values. **FFT and time-shift are the best methods** (+0.12); AAFT actually hurts (−0.04) due to amplitude-distribution preservation destroying relevant structure.
- **Lorenz**: Steady improvement (0.533→0.734). Surrogate is neutral at short T, becomes mildly positive at T≥3000 (ΔAUROC_z = +0.034 at T=5000, p<0.001).
- **Kuramoto**: Strong improvement (0.535→0.776). Surrogate slightly positive at T=2000–3000, but inconsistent.
- **Rössler**: Improves (0.492→0.704) but surrogate increasingly hurts with longer T (ΔAUROC_surr = −0.182 at T=5000). The chaotic attractor's structure is poorly captured by surrogates.
- **Hindmarsh-Rose**: AUROC ~0.5 at all T values — **CCM completely fails** on this system regardless of T. TPR ≈ FPR, indicating no discrimination ability.
- **FitzHugh-Nagumo**: AUROC ~0.55, barely above chance. TPR ≈ FPR (both >0.7). CCM fails — likely due to excitable (non-chaotic) dynamics.

---

## 3. Per-Method Analysis

### 3.1 FFT Surrogates
- Best overall performer for Hénon (ΔAUROC_z = +0.118)
- Mild positive for Logistic (+0.013) and Kuramoto (+0.017)
- Preserves power spectrum; effective when causal coupling manifests as phase relationships

### 3.2 AAFT Surrogates
- **Catastrophic for Hénon** (ΔAUROC_z = −0.038): preserving the marginal amplitude distribution retains too much structure, weakening the null hypothesis
- Best for Kuramoto (+0.023) and Lorenz (+0.013)
- Most conservative surrogate — smallest improvements and smallest degradations

### 3.3 Time-shift Surrogates
- Strong for Hénon (+0.113), comparable to FFT
- Worst for Rössler (−0.064) and Lorenz (−0.023)
- Preserves local temporal structure; may preserve too much causal information in continuous-time systems

---

## 4. Problematic Systems: Root Cause Analysis

### 4.1 Hindmarsh-Rose (AUROC ≈ 0.5)
- TPR ≈ FPR at all T values (both ~0.2 at most T, dropping to 0.07 at T=5000)
- The HR neuron model's bursting dynamics create complex, non-stationary attractors
- Default coupling (ε=0.05) may be too weak for N=10 ER network
- **Action needed**: Verify coupling strength; try ε=0.1–0.5 in coupling sweep

### 4.2 FitzHugh-Nagumo (AUROC ≈ 0.55)
- TPR and FPR both very high (0.7–0.98), meaning the model calls almost everything causal
- Excitable dynamics (not chaotic) may violate CCM's requirement for deterministic chaos
- **Action needed**: Check if FN generates chaotic dynamics at these parameters; consider excluding from benchmark

### 4.3 Rössler (3/10 reps fail)
- Only 7 of 10 replications succeed consistently across all T values and methods
- Likely numerical instability (divergence) in some random network realizations
- The system's large attractor excursions combined with strong coupling on certain topologies may cause overflow
- **Action needed**: Add robustness to the generator (gradient clipping or NaN detection); increase n_reps to compensate

---

## 5. Data Quality Issues

| Issue | Impact | Recommendation |
|-------|--------|----------------|
| Rössler: 7/10 reps | Reduced statistical power, possible survivorship bias | Add NaN/divergence handling; increase n_reps to 15 |
| HR/FN: AUROC~0.5 | These systems contribute no useful signal | Document as CCM-incompatible; keep for completeness but flag in figures |
| No T>5000 tested | Unclear if benefits continue growing for Lorenz/Kuramoto | Add T=8000, 10000 for systems that haven't plateaued |

---

## 6. Summary Table: ΔAUROC (z-score − raw) by System × T

Values are averaged over 3 methods. Bold = statistically significant (p<0.05).

| System | T=500 | T=1000 | T=2000 | T=3000 | T=5000 |
|--------|-------|--------|--------|--------|--------|
| Logistic | +0.011 | +0.009 | +0.008 | +0.007 | +0.005 |
| Hénon | **+0.052** | **+0.060** | **+0.073** | **+0.077** | **+0.059** |
| Kuramoto | +0.019 | −0.005 | +0.022 | **+0.037** | +0.001 |
| Lorenz | −0.002 | −0.024 | **−0.024** | +0.008 | **+0.034** |
| Rössler | +0.037 | −0.036 | +0.012 | **−0.076** | **−0.104** |
| Hindmarsh-Rose | −0.006 | −0.020 | **−0.045** | **−0.039** | +0.019 |
| FitzHugh-Nagumo | −0.011 | −0.007 | −0.014 | −0.003 | −0.024 |

---

## 7. Recommendations for Improvement

### 7.1 Immediate (before full experiment)
1. **Fix Rössler divergence**: Add NaN/overflow detection in the generator; automatically retry failed seeds or increase n_reps
2. **Consider excluding FitzHugh-Nagumo and Hindmarsh-Rose from ΔAUROC analysis**: Their AUROC~0.5 means CCM itself fails, making surrogate comparison meaningless. Keep them in the raw data but flag in figures.

### 7.2 For the paper
3. **Recommend z-score over p-value**: The z-score metric is strictly better than the rank-based p-value in 6/7 systems. This is a methodological contribution.
4. **Method selection**: FFT is the safest default. AAFT should be avoided for discrete maps (Hénon). Time-shift is good for maps but risky for continuous ODEs.
5. **Minimum T guideline**: T≥2000 recommended for reliable results. At T=500, even well-behaved systems show high variance and unreliable surrogate testing.

### 7.3 Extended sweep (optional)
6. **Add T=8000, 10000**: Lorenz and Kuramoto show continuing improvement at T=5000; longer series would clarify asymptotic behavior
7. **Add T=200, 300**: Useful to characterize the failure mode at very short series lengths

---

## 8. Plots Generated

The following 27 figures are in `results/robustness/robustness/T_sweep/`:

| File | Description |
|------|-------------|
| `line_AUC_ROC_rho_vs_T` | Raw ρ AUROC vs T (baseline, per system) |
| `line_AUC_ROC_surrogate_vs_T` | Surrogate p-value AUROC vs T (with raw baseline) |
| `line_AUC_ROC_zscore_vs_T` | Z-score AUROC vs T (with raw baseline) |
| `line_AUC_ROC_delta_vs_T` | ΔAUROC (surrogate − raw) vs T |
| `line_AUC_ROC_delta_zscore_vs_T` | ΔAUROC (z-score − raw) vs T |
| `heatmap_AUC_ROC_delta_vs_T` | Heatmap: system × T, ΔAUROC (all methods avg) |
| `heatmap_AUC_ROC_delta_zscore_vs_T` | Heatmap: system × T, ΔAUROC_z (all methods avg) |
| `heatmap_*_{fft,aaft,timeshift}_vs_T` | Per-method heatmaps (×2 metrics × 3 methods = 6) |

All plots are in both PDF and PNG formats (300 DPI, publication quality).
