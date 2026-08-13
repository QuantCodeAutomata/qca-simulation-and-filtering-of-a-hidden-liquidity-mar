# Repository Knowledge — Hidden-Liquidity Markov-Modulated Marked Hawkes

## Project Layout
```
src/model/          Core model (ctmc, hawkes_marked, filter, impact)
src/control/        DP / control solver (to be implemented)
src/experiments/    Experiment runners (to be implemented)
src/utils/          Shared utilities (to be implemented)
configs/            YAML configuration files
tests/              Unit / integration tests
```

## Key Parameters (from configs/filter_default.yaml)
| Parameter | Value | Notes |
|-----------|-------|-------|
| n_states | 2 | Hidden CTMC states |
| generator Ψ | [[-0.2,0.2],[0.2,-0.2]] | Symmetric → stationary = [0.5, 0.5] |
| zeta | 2.0 | Mark Exp rate; mean mark = 0.5 |
| eta | 0.05 | Excitation exponent (near-flat φ(v)≈1) |
| beta | 0.5 | Mean-reversion rate (WARNING: supercritical, see below) |
| lambda_infty | 1.0 each | Baseline intensities |
| nu | 0.8 | Permanent impact fraction |
| rho | 0.1 | Transient-deviation decay rate |
| c | 1.0, e=3.0 | Order-book depth/shape |

## Critical Stability Insight
Because φ(v) = (ζ^η / Γ(1+η)) · v^η and v ~ Exp(ζ), the mark distribution
cancels the normalisation constant exactly:
    E[φ(v)] = 1  (regardless of η, ζ)

Each jump adds φ(v) to **both** λ^{i,+} and λ^{i,-} for **all** regimes.
The effective branching ratio for the active regime is:
    ρ = 2 · E[φ(v)] / β = 2/β

Subcriticality requires **β > 2**.  The config value β=0.5 is supercritical
(ρ=4).  `simulate_marked_hawkes` has a `max_events=100_000` safety guard to
prevent infinite loops in supercritical runs.

## Model Equations Summary
- **CTMC**: exponential holding times; `simulate_ctmc` returns (times, states)
  with `times[0]=0.0` and `states[k]` active during [times[k], times[k+1]).

- **Intensity dynamics** (between jumps, no-jump evolution):
    λ^{i,k}(t+dt) = λ∞_i + (κ_k[i] − λ∞_i)·exp(−β·dt)

- **Excitation** (at each accepted jump, mark v, ANY side):
    κ_plus[i]  += φ(v)  for ALL i
    κ_minus[i] += φ(v)  for ALL i

- **Filter ODE** (between jumps):
    dσ_i/dt = (σ @ Ψ)_i − σ_i·(λ^{i,+}_t + λ^{i,-}_t)

- **Bayes update** (at jump with side k, mark v):
    π(i) ∝ π(i) · λ^{i,k} · ν_i(v),   ν_i(v) = ζ·exp(−ζv)

- **Price paths** (permanent S, transient D, mid P=S+D):
    dS = ν·Q(v)·dn,   dD = −ρ·D dt + (1−ν)·Q(v)·dn
    Q(v) = (e·v/c)^{1/e}

## Implementation Notes
- `filter_between_jumps`: uses `scipy.integrate.solve_ivp` (RK45) with
  `max_step = dt/n_steps`.  Fast (≈0.001s/call for n_states=2).
- `simulate_price_path`: tracks `t_last` (time of last event, not last grid
  point) so D decays correctly without being applied twice.
- All intensities are clipped to ≥ 0 via `np.maximum`.
- Posteriors are renormalised after every Bayes update and ODE step.
- Avoid bash `<< 'EOF'` heredoc for long Python scripts — terminal echo causes
  timeouts.  Write to a `.py` file and execute instead.
