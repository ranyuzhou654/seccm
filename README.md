# SE-CCM: Surrogate-Enhanced Convergent Cross Mapping

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[中文文档 / Chinese Documentation](docs/README_zh.md)**

SE-CCM is a Python framework for statistically rigorous **causal inference in coupled dynamical systems**. It extends the Convergent Cross Mapping (CCM) algorithm ([Sugihara et al., 2012, *Science*](https://doi.org/10.1126/science.1227079)) with surrogate-based hypothesis testing, turning raw cross-map correlations into principled statistical decisions.

---

## Key Idea

CCM detects causal relationships by exploiting **Takens' embedding theorem**: if variable Y causally drives X, then Y's information is encoded in X's reconstructed attractor, and a shadow manifold built from X can cross-predict Y.

**The problem:** raw CCM correlation (ρ) is susceptible to false positives from shared dynamics, synchronization, or finite-sample bias.

**SE-CCM's solution:** generate surrogate time series of the candidate cause under the null hypothesis of no causal influence. Compare the observed ρ against the surrogate distribution to obtain p-values and z-scores, then apply FDR correction for network-wide testing.

<p align="center">
<img src="docs/pipeline.png" alt="SE-CCM Pipeline" width="700">
</p>

## Features

- **8 coupled dynamical systems** — Logistic, Lorenz, Henon, Rossler, Hindmarsh-Rose, FitzHugh-Nagumo, Kuramoto, Van der Pol
- **12 univariate + 2 multivariate surrogate methods + adaptive selection** — FFT, AAFT, iAAFT, time-shift, random reorder, cycle-shuffle, twin, phase, small-shuffle, truncated Fourier, cycle-phase-A, cycle-phase-B; multivariate FFT/iAAFT (cross-correlation preserving); `auto` mode selects per-variable based on spectral concentration
- **Automatic embedding** — data-driven selection of embedding dimension (E) and delay (tau) via mutual information and simplex prediction, FNN, or Cao's method; non-uniform delay embedding for multi-timescale systems
- **Convergence testing** — cross-validated ρ(L) convergence with Kendall-τ scoring
- **Chaos pre-detection** — 0-1 test for chaos (Gottwald & Melbourne) with auto-subsampling for ODE flows
- **Network-scale testing** — pairwise CCM with Benjamini-Hochberg FDR correction, Theiler window, adaptive effect-size thresholds
- **Surrogate diagnostic framework** — SSO (Surrogate Spectral Overlap), AUPRC, rho_gap, null_overlap, convergence_fraction metrics for diagnosing when surrogates help CCM
- **10 experiment modules** — bivariate validation, coupling strength sweep, noise robustness, topology comparison, surrogate method comparison, robustness ablation, diagnostic table, regime boundary sweep, convergence analysis, noise robustness with cycle detection
- **Publication-ready output** — 300 DPI figures (PDF + PNG), LaTeX tables, CSV data

## Installation

```bash
git clone https://github.com/ranyuzhou654/seccm.git
cd seccm
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

**Requirements:** Python >= 3.9, NumPy, SciPy, scikit-learn, NetworkX, matplotlib, seaborn, joblib, h5py, PyYAML, tqdm, statsmodels.

## Quick Start

### Minimal Example

```python
from surrogate_ccm.generators import generate_network, create_system
from surrogate_ccm.testing import SECCM

# Create a 10-node coupled Lorenz network
adj = generate_network("ER", N=10, seed=42, p=0.3)
system = create_system("lorenz", adj, coupling=3.0)
data = system.generate(T=3000, transient=1000, seed=42)  # shape: (3000, 10)

# Run SE-CCM
seccm = SECCM(surrogate_method="iaaft", n_surrogates=99, alpha=0.05, fdr=True)
seccm.fit(data)

# Evaluate against ground truth
metrics = seccm.score(adj)
print(f"AUROC (surrogate): {metrics['AUC_ROC_surrogate']:.3f}")
print(f"AUROC (raw rho):   {metrics['AUC_ROC_rho']:.3f}")
print(f"Delta AUROC:       {metrics['AUC_ROC_delta']:+.3f}")
```

### Run Experiments

```bash
# Run all experiments with default config
python run_experiments.py --experiment all --n-jobs 4

# Run a specific experiment
python run_experiments.py --experiment robustness --n-jobs 16 --output-dir results/robustness

# Smoke test (fast, ~10 seconds)
python run_experiments.py --experiment robustness --config configs/robustness_smoke.yaml

# Run diagnostic framework experiments (D1, D2, E5, E7)
python run_diagnostic_experiments.py --n-jobs 8

# Feasibility mode (small scale, fast)
python run_diagnostic_experiments.py --feasibility --n-jobs 5

# Run specific diagnostic experiment(s)
python run_diagnostic_experiments.py --only D1 D2
```

## Project Structure

```
seccm/
├── run_experiments.py              # CLI entry point (all experiments)
├── run_diagnostic_experiments.py   # CLI for diagnostic framework (D1, D2, E5, E7)
├── configs/
│   ├── default.yaml                # Full experiment configuration
│   └── robustness_smoke.yaml       # Fast smoke test config
├── surrogate_ccm/
│   ├── ccm/                        # CCM core algorithms
│   │   ├── embedding.py            #   delay embedding, tau/E selection (simplex, FNN, Cao), non-uniform embedding
│   │   ├── ccm_core.py             #   cross-map prediction, convergence test, Theiler window
│   │   └── network_ccm.py          #   pairwise CCM for networks
│   ├── surrogate/                  # Surrogate generation methods
│   │   ├── fft_surrogate.py        #   FFT phase randomization
│   │   ├── aaft_surrogate.py       #   amplitude-adjusted FT
│   │   ├── iaaft_surrogate.py      #   iterative AAFT
│   │   ├── timeshift_surrogate.py  #   circular time shift
│   │   ├── random_reorder.py       #   random permutation
│   │   ├── cycle_shuffle_surrogate.py  #   cycle-shuffle for oscillatory systems
│   │   ├── twin_surrogate.py       #   twin surrogate (recurrence-based)
│   │   ├── phase_surrogate.py      #   phase surrogate (Hilbert-based)
│   │   ├── small_shuffle_surrogate.py  #   small-shuffle surrogate
│   │   ├── truncated_fourier_surrogate.py  #   band-selective phase randomization
│   │   ├── cycle_phase_surrogate.py    #   cycle-phase surrogates (A: within-cycle, B: + cycle permutation)
│   │   ├── adaptive.py             #   adaptive method selection by signal profile
│   │   └── multivariate_surrogate.py   #   multivariate FFT/iAAFT (cross-corr preserving)
│   ├── testing/                    # Statistical testing
│   │   ├── hypothesis_test.py      #   p-value, z-score, FDR correction
│   │   └── se_ccm.py              #   SECCM class (full pipeline, AUROC + AUPRC)
│   ├── generators/                 # Coupled dynamical system simulators
│   │   ├── network.py              #   network topology generation (ER, WS, ring)
│   │   ├── logistic.py             #   coupled logistic map
│   │   ├── lorenz.py               #   coupled Lorenz oscillators
│   │   ├── henon.py                #   coupled Henon map
│   │   ├── rossler.py              #   coupled Rossler oscillators
│   │   ├── hindmarsh_rose.py       #   Hindmarsh-Rose neural network
│   │   ├── fitzhugh_nagumo.py      #   FitzHugh-Nagumo neural network
│   │   ├── kuramoto.py             #   Kuramoto phase oscillators
│   │   └── van_der_pol.py          #   coupled van der Pol oscillators
│   ├── evaluation/                 # Detection metrics (TPR, FPR, AUROC, AUPRC, SSO)
│   ├── visualization/              # Plotting utilities
│   ├── experiments/                # Experiment modules
│   │   ├── exp_bivariate.py        #   2-node sanity check
│   │   ├── exp_coupling_strength.py
│   │   ├── exp_noise.py
│   │   ├── exp_network_topology.py
│   │   ├── exp_surrogate_comparison.py
│   │   ├── exp_surrogate_robustness.py
│   │   ├── exp_node_count.py           #   E_node: node-count sweep
│   │   ├── exp_diagnostic_table.py     #   D1: full diagnostic table (8 systems × 12 surrogates)
│   │   ├── exp_regime_boundaries.py    #   D2: regime boundary sweep (coupling × system)
│   │   ├── exp_convergence.py          #   E5: convergence analysis (AUROC vs T)
│   │   └── exp_noise_robustness.py     #   E7: noise robustness + cycle detection quality
│   └── utils/                      # I/O, parallel execution, chaos detection
├── ccm.py                          # Advanced standalone CCM (latent CCM, convergence scoring)
└── system.py                       # Advanced standalone simulator (RK4, process noise, BA/SF graphs)
```

## Experiments

| Experiment | Command | Description |
|---|---|---|
| `bivariate` | `--experiment bivariate` | 2-node validation: verify correct detection of X0 -> X1 |
| `coupling` | `--experiment coupling` | Sweep coupling strength across systems and topologies |
| `noise` | `--experiment noise` | Effect of observation noise on detection accuracy |
| `topology` | `--experiment topology` | Compare ER, WS, ring topologies at varying network sizes |
| `surrogate` | `--experiment surrogate` | Compare 10 surrogate methods across 7 systems |
| `robustness` | `--experiment robustness` | Ablation: sweep T, coupling, obs-noise, dyn-noise |
| `node_count` | `--experiment node_count` | E_node: Sweep node count N and track AUROC/runtime |
| `diagnostic_table` | `--experiment diagnostic_table` | D1: Full diagnostic table (8 systems × 12 surrogates × 20 reps) |
| `regime_boundaries` | `--experiment regime_boundaries` | D2: Regime boundary sweep across coupling strengths |
| `convergence` | `--experiment convergence` | E5: AUROC convergence analysis vs time series length |
| `noise_robustness` | `--experiment noise_robustness` | E7: Noise robustness + cycle detection quality |

### Robustness Experiment Details

The robustness experiment runs 4 sub-experiments, each sweeping one factor while holding others fixed:

| Sub-experiment | Swept parameter | Values |
|---|---|---|
| Sub-A | Time series length T | 500, 1000, 2000, 3000, 5000 |
| Sub-B | Coupling strength eps | Per-system (6 values each) |
| Sub-C | Observation noise sigma_obs | 0.0, 0.01, 0.05, 0.1, 0.2 |
| Sub-D | Dynamical noise sigma_dyn | 0.0, 0.001, 0.005, 0.01, 0.05 |

**Output includes:**
- AUROC line charts with SEM error bars (per system, per method)
- Delta-AUROC heatmaps (system x sweep value)
- Combined 4-panel overview figure
- Ablation summary tables (CSV + LaTeX)
- Experiment parameter documentation table

## Dynamical Systems

| System | Type | State dim | Observed | Key parameters |
|---|---|---|---|---|
| Logistic | Discrete map | 1 | x | r = 3.9 |
| Henon | Discrete map | 2 | x | a = 1.2, b = 0.3 |
| Lorenz | ODE (chaotic) | 3 | x | sigma = 10, rho = 28, beta = 8/3 |
| Rossler | ODE (chaotic) | 3 | x | a = 0.2, b = 0.2, c = 5.7 |
| Hindmarsh-Rose | ODE (neural) | 3 | x (membrane) | I_ext = 3.5, r = 0.01 |
| FitzHugh-Nagumo | ODE (neural) | 2 | v (membrane) | I_ext = 0.5, tau = 12.5 |
| Kuramoto | ODE (oscillator) | 1 | sin(theta) | omega ~ N(1.0, 0.2) |
| Van der Pol | ODE (oscillator) | 2 | x | mu = 1.0 |

All systems support:
- **Observation noise** (`noise_std`): Gaussian noise added to the final output
- **Dynamical noise** (`dyn_noise_std`): Euler-Maruyama SDE integration for ODE systems; additive noise per iteration for maps
- **Configurable coupling** and **network topology** (ER, WS, ring)

## Surrogate Methods

| Method | Preserves | Destroys | Best for | Speed |
|---|---|---|---|---|
| FFT | Power spectrum | Amplitude distribution, nonlinearity | Broadband chaotic | Fast |
| AAFT | Amplitude dist. + approx. spectrum | Nonlinear structure | General purpose | Fast |
| iAAFT | Amplitude dist. + exact spectrum | Nonlinear structure | General purpose | Moderate |
| Time-shift | All local structure | Cross-series temporal alignment | Quick baseline | Very fast |
| Random reorder | Amplitude distribution | All temporal structure | Independence test | Very fast |
| **Cycle-shuffle** | Intra-cycle waveform, amplitude dist. | Inter-cycle phase coupling | **Narrowband oscillatory** (Rossler) | Very fast |
| **Twin** | Attractor topology (recurrence structure) | Inter-system phase coupling | **Any dynamical system** (theoretically optimal) | Moderate |
| **Phase** | Amplitude envelope | Phase coupling | **Phase-coupled oscillatory** (Kuramoto, FHN) | Fast |
| **Small-shuffle** | Large-scale trends | Fine temporal order | Trending data | Very fast |
| **Truncated Fourier** | Structure outside target band | Phase within target band | Multi-timescale analysis | Fast |
| **Multivariate FFT** | Cross-correlations + power spectra | Nonlinear cross-dependencies | Linearly coupled systems | Fast |
| **Cycle-phase A** | Intra-cycle waveform + cycle count | Inter-cycle phase alignment | **Phase-coupled oscillatory** (VdP, Kuramoto) | Fast |
| **Cycle-phase B** | Intra-cycle waveform | Cycle order + phase alignment | **Phase-coupled oscillatory** (stronger null) | Fast |
| **Multivariate iAAFT** | Cross-corr + spectra + amplitude dist. | Nonlinear cross-dependencies | Linearly coupled systems | Moderate |

### Cycle-Shuffle Surrogate

Designed for **narrowband oscillatory systems** where FFT/AAFT/iAAFT preserve the periodic structure that carries causal information, leading to suppressed z-scores and negative ΔAUROC.

**Algorithm:** Detect oscillation cycles via mean-crossing (rising edges) → split signal into complete cycles → randomly permute cycle order → reassemble. Falls back to time-shift if fewer than 3 cycles are detected.

**Reference motivation:** T-sweep experiments show Rossler (top-3 spectral concentration = 0.822), FHN (0.901), and Kuramoto (0.846) have negative ΔAUROC with spectral surrogates, because the surrogate preserves the causal signal embedded in the periodic structure.

### Twin Surrogate

Generates surrogates that preserve the **recurrence structure** (attractor topology) of the original time series — the theoretically most principled surrogate method for testing synchronization and causal coupling.

**Algorithm** ([Thiel et al., 2006, *EPL* 75(4)](https://doi.org/10.1209/epl/i2006-10147-0)):
1. Reconstruct phase space via time-delay embedding
2. Build recurrence neighborhood using KDTree (Chebyshev metric)
3. Identify "twin" states — points with identical recurrence neighborhoods (hash-accelerated)
4. Construct surrogate trajectory by walking the attractor, randomly switching between twin successors

**Performance optimizations:**
- KDTree + sparse neighbor sets instead of dense N×N recurrence matrix (6× faster, 3× less memory)
- Row-hash grouping for twin detection (56× faster than brute-force row comparison)
- Pre-computed twin structure cached across multiple surrogate generations
- Cause-variable caching in SECCM pipeline (9× fewer surrogate generation calls)

### Phase Surrogate

Designed for **phase-coupled oscillatory systems** (Kuramoto, FHN) where cycle-shuffle fails because all cycles are nearly identical (sin(θ) waveform).

**Algorithm:** Hilbert transform → extract instantaneous amplitude A(t) and phase φ(t) → shuffle phase increments Δφ → reconstruct as A(t)·cos(φ_shuffled(t)). Preserves the amplitude envelope while destroying inter-oscillator phase coupling.

### Small-Shuffle Surrogate

A middle ground between time-shift (too conservative) and random-reorder (too aggressive). Perturbs each time index by a small random amount (±δ), preserving large-scale trends while destroying fine temporal alignment.

### Cycle-Phase Surrogate

Designed for **phase-coupled oscillatory systems** (Van der Pol, Kuramoto) where cycle-shuffle fails because cycles are too uniform and spectral surrogates preserve the causal periodic structure.

**Algorithm:** Hilbert transform → extract instantaneous phase → identify cycle boundaries (2π crossings) → **Mode A:** circular-shift time points within each cycle (preserves cycle count and waveform shape) → **Mode B:** additionally permute cycle order (stronger null hypothesis). Falls back to peaks-based cycle detection if Hilbert fails.

### Truncated Fourier Surrogate

Band-selective phase randomization: only randomizes phases within a specified frequency band. Enables testing causality at specific timescales — randomize low frequencies to test slow causality, or high frequencies to test fast coupling.

## Conventions

- **Adjacency matrix:** `A[i, j] = 1` means node j drives node i (row = receiver)
- **CCM direction:** `ccm(x, y)` tests "y causes x" (embed x, predict y)
- **Surrogate target:** the *cause* variable is surrogatized (not the effect)
- **Data shape:** `(T, N)` — rows are time steps, columns are nodes
- **Effect-size gate:** statistical significance requires raw rho >= 0.3

## Configuration

All parameters are controlled via YAML config files. See [`configs/default.yaml`](configs/default.yaml) for the full specification. Key sections:

```yaml
time_series:
  T: 3000
  transient: 1000

surrogate:
  methods: [fft, aaft, iaaft, timeshift, random_reorder, cycle_shuffle, twin]
  n_surrogates: 100

surrogate_robustness:
  systems: [logistic, lorenz, henon, rossler, hindmarsh_rose, fitzhugh_nagumo, kuramoto]
  methods: [fft, aaft, iaaft, timeshift, random_reorder, cycle_shuffle, twin, phase, small_shuffle, truncated_fourier]
  n_surrogates: 99
  N: 10
  n_reps: 5
  # ... sweep values per sub-experiment
```

## API Reference

```python
# ── Generators ──────────────────────────────────────────
from surrogate_ccm.generators import generate_network, create_system

adj = generate_network("ER", N=10, seed=0, p=0.3)    # -> (10, 10) binary matrix
system = create_system("lorenz", adj, coupling=3.0)
data = system.generate(T=3000, transient=1000, seed=0,
                       noise_std=0.0, dyn_noise_std=0.0)  # -> (3000, 10)

# ── CCM ─────────────────────────────────────────────────
from surrogate_ccm.ccm import delay_embed, select_parameters, ccm, compute_pairwise_ccm
from surrogate_ccm.ccm import convergence_score, select_E_fnn, select_E_cao
from surrogate_ccm.ccm import delay_embed_nonuniform, select_delays_nonuniform

E, tau = select_parameters(data[:, 0])                 # auto-select (E, tau)
E, tau = select_parameters(data[:, 0], E_method="fnn") # FNN method
E, tau = select_parameters(data[:, 0], E_method="cao") # Cao's method
rho = ccm(data[:, 0], data[:, 1], E=3, tau=2)         # test "node 1 causes node 0"
rho = ccm(data[:, 0], data[:, 1], E=3, tau=2, theiler_w=5)  # with Theiler window
score, rho = convergence_score(data[:, 0], data[:, 1], E=3, tau=2,
                               cross_validate=True)    # convergence test

# Non-uniform embedding (for multi-timescale systems like Hindmarsh-Rose)
delays = select_delays_nonuniform(data[:, 0], E_max=5, tau_max=100)  # e.g. [0, 5, 6, 17, 18]
M = delay_embed_nonuniform(data[:, 0], delays)         # custom delay embedding

# ── Surrogates ──────────────────────────────────────────
from surrogate_ccm.surrogate import generate_surrogate, select_surrogate_method
from surrogate_ccm.surrogate import generate_multivariate_surrogate

surrogates = generate_surrogate(data[:, 0], method="aaft", n_surrogates=99)           # -> (99, T)
surrogates = generate_surrogate(data[:, 0], method="cycle_shuffle", n_surrogates=99) # oscillatory
surrogates = generate_surrogate(data[:, 0], method="twin", n_surrogates=99)          # recurrence-based
surrogates = generate_surrogate(data[:, 0], method="cycle_phase_A", n_surrogates=99) # cycle-phase (within-cycle)
surrogates = generate_surrogate(data[:, 0], method="cycle_phase_B", n_surrogates=99) # cycle-phase (+ permute order)
method, profile = select_surrogate_method(data[:, 0])                                # auto-select method

# Multivariate surrogates (preserves cross-correlations between variables)
mv_surr = generate_multivariate_surrogate(data, method="multivariate_fft", n_surrogates=99)  # list of (T,N)

# ── Chaos Detection ────────────────────────────────────
from surrogate_ccm.utils import test_01_chaos, is_chaotic

K, K_values = test_01_chaos(data[:, 0])                # K ≈ 1 → chaotic, K ≈ 0 → regular
chaotic, K = is_chaotic(data[:, 0])                    # quick boolean check

# ── Diagnostic Metrics ─────────────────────────────────
from surrogate_ccm.evaluation.metrics import compute_sso

sso = compute_sso(data[:, 0], surrogates)  # Surrogate Spectral Overlap (JSD of PSDs)

# ── Full Pipeline ───────────────────────────────────────
from surrogate_ccm.testing import SECCM

seccm = SECCM(surrogate_method="auto", n_surrogates=99, alpha=0.05, fdr=True)
# Defaults: theiler_w="auto" (median tau), adaptive_rho=True
seccm.fit(data)
metrics = seccm.score(adj)     # -> dict with AUROC, TPR, FPR, etc.

# Convergence filtering is enabled by default (convergence_filter=True)
# To disable it:
seccm = SECCM(convergence_filter=False)
seccm.fit(data)

# Access internals
seccm.ccm_matrix_              # raw CCM rho matrix (N, N)
seccm.pvalue_matrix_           # p-value matrix (N, N)
seccm.zscore_matrix_           # z-score matrix (N, N)
seccm.detected_                # binary detection matrix (N, N)
seccm.surrogate_methods_used_  # per-variable method (auto mode)
seccm.min_rho_matrix_          # per-pair adaptive thresholds
seccm.theiler_w_used_          # resolved Theiler window value
seccm.convergence_matrix_      # convergence scores (if convergence_filter=True)
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{seccm2026,
  title  = {SE-CCM: Surrogate-Enhanced Convergent Cross Mapping},
  year   = {2026},
  url    = {https://github.com/ranyuzhou654/seccm}
}
```

## License

This project is licensed under the MIT License.
