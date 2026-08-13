# Hidden-Liquidity Markov-Modulated Marked Hawkes Model

Numerical implementation of a two-state hidden-liquidity Markov-modulated marked Hawkes order-flow model with:

- **Finite-dimensional filter** for posterior regime-belief propagation from observed buy/sell order flow.
- **Reduced dynamic-programming solver** for the impulse-control optimal liquidation problem under incomplete information.
- **Exercise/continuation region visualisation** as a function of belief and candidate intensities.
- **Optimal order-size policy maps** with nonlinear market-impact sensitivity analysis.

## Project Structure

```
repo/
├── configs/
│   └── filter_default.yaml     # All model parameters (T, zeta, grid sizes, etc.)
├── data/                        # (empty; all data are synthetic)
├── exp/
│   └── run_all.py               # Master runner: executes Experiments 1–4 in sequence
├── run_experiments_2_4.py       # Combined runner sharing DP solution across Exps 2–4
├── src/
│   ├── model/
│   │   ├── ctmc.py              # Two-state CTMC: generator, holding times, stationary
│   │   ├── hawkes_marked.py     # Regime-modulated marked Hawkes: intensities + Ogata thinning
│   │   ├── filter.py            # Finite-dimensional filter: ODE propagation + Bayes update
│   │   └── impact.py            # Order-book model: Q(v), TC(v), price process S/D/P
│   ├── control/
│   │   └── intervention.py      # Reduced DP solver: g_N recursion, argmax ξ*, exercise map
│   ├── experiments/
│   │   ├── filter_paths.py      # Exp 1: simulate and filter regime paths
│   │   ├── dp_convergence.py    # Exp 2: convergence of W_N in N
│   │   ├── exercise_regions.py  # Exp 3: exercise/continuation region geometry
│   │   └── order_size_maps.py   # Exp 4: optimal order size vs inventory/belief/impact
│   └── utils/
│       └── plotting.py          # Shared figure helpers
├── tests/                       # 148 pytest tests across all modules
├── results/                     # Generated figures, CSVs, JSON summaries, RESULTS.md
└── README.md
```

## Installation

```bash
pip install numpy scipy matplotlib pandas pyyaml pytest
```

All required libraries (numpy, scipy, matplotlib, pandas, pyyaml, scipy, scikit-learn) are available in the standard Python environment used by this project.

## Configuration

All parameters are stored in `configs/filter_default.yaml`. Researcher-chosen values (T, ζ, d_fixed) are documented with comments. Parameters taken directly from the paper are labelled accordingly.

| Parameter | Value | Source |
|---|---|---|
| T | 20.0 | Researcher-chosen |
| ζ (zeta) | 2.0 | Researcher-chosen |
| Ψ | [[-0.2, 0.2],[0.2, -0.2]] | Paper |
| β | 0.5 | Paper |
| ν | 0.8 | Paper |
| ρ | 0.1 | Paper |
| c₀ | 0.1 | Paper |
| η | 0.05 | Paper |
| λ∞ | 1.0 | Paper |
| c | 1.0 | Paper |
| e | 3.0 | Paper |

## Running Experiments

### Experiment 1 — Filter Paths (standalone)
```bash
python -m src.experiments.filter_paths
```

### Experiments 2–4 — DP Solver + Regions + Order Size (combined runner)
```bash
python run_experiments_2_4.py
```

### All experiments via master runner
```bash
python exp/run_all.py
```

All outputs are written to `results/`.

## Running Tests

```bash
pytest tests/ -v
```

Expected: **148 passed**.

## Key Results

| Experiment | Result |
|---|---|
| **Exp 1: Filtering** | Posterior tracks hidden regime. MAP accuracy up to 89.1% (kp=1, km=5, prior=0.1). Simplex validity and symmetry verified. |
| **Exp 2: Convergence** | δ_N decays from 8.33 → 8.2×10⁻¹² over N=20 iterations. Practical convergence (δ<10⁻⁴) at iteration 5. Monotone confirmed. |
| **Exp 3: Exercise Regions** | Exercise region covers x ∈ (0, ~2.22]. Exact numerical symmetry (error=0.0) under regime swap. |
| **Exp 4: Order Size** | c=0.7 produces 43% smaller mean optimal orders than c=1.0. Impact sensitivity claim confirmed. |

See `results/RESULTS.md` for full quantitative summaries.

## Model Overview

### Hidden Regime

The latent liquidity regime I_t is a 2-state CTMC with generator Ψ = [[-0.2, 0.2], [0.2, -0.2]], simulated via exact exponential holding times.

### Marked Hawkes Process

Candidate intensities for each regime i ∈ {1,2} evolve as:

```
dλᵢ⁺ = -β(λᵢ⁺ - λ∞ⁱ)dt + φ_s(v/m₁)dN⁺ + φ_c(v/m₁)dN⁻
```

with excitation φ(v) = η · (ζ^η / Γ(1+η)) · v^η and η=0.05. Events are generated via Ogata thinning adapted to the latent-regime active intensity.

### Finite-Dimensional Filter

Between jumps, the posterior propagates by the generator-induced ODE. At each observed jump of side k with mark v, the Bayes update is:

```
π_t(i) ∝ π_{t-}(i) · λ_{t-}^{i,k} · ν_i(v)
```

where ν_i(v) = ζ · exp(-ζv) is the mark density.

### Impact Model

```
f(x) = c · x^{-1+e}          (order-book depth)
Q(v) = (e·v/c)^{1/e}          (price impact of trade size v)
TC(v) = c/(e+1) · Q(v)^{e+1}  (transaction cost)
```

with c=1.0, e=3.0 (default).

### Reduced DP Solver

The value function is represented as V(t,y) = x·s + g(t,x,d,κ⁺,κ⁻,μ). The reduced recursion iterates:

```
g_{N+1} = max(g_N, M̃g_N)
```

where M̃ is the intervention operator and g₀ is the forced terminal liquidation value.

## Library Usage

- **scipy** — ODE propagation (solve_ivp), Gauss-Laguerre quadrature (fixed_quad), interpolation (RegularGridInterpolator), Gamma function — *Context7 confirmed*
- **hmmlearn** — referenced for HMM pattern guidance; actual filter is custom due to marked point-process observation model — *Context7 confirmed*
- **numpy** — array operations throughout
- **matplotlib** — all visualizations

Custom implementations (verified no library equivalent via Context7):
- Regime-modulated marked Hawkes simulation with Ogata thinning
- Finite-dimensional marked point-process filter with Bayes updates
- Reduced impulse-control DP recursion with first-jump operator
