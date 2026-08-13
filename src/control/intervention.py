"""Reduced dynamic-programming solver for the optimal liquidation problem.

State representation: g(t, x, kp1, km1, mu).  Transient deviation d enters via a
deterministic scalar path d(t) = d_fixed * exp(-rho * t) (researcher-chosen initial
level d_fixed; paper omits exact value and requires it to be documented).  The d
update from own trades, d -> d - (1-nu)*Q(xi), is small (nu=0.8) and neglected for
numerical tractability; this is noted as an approximation in the results.

Key formulas (paper notation):
  Q(v)     = (e*v/c)^{1/e}
  TC(xi)   = c/(e+1) * Q(xi)^{e+1}
  phi(v)   = (zeta^eta / Gamma(1+eta)) * v^eta
  S(t,u)   = exp(-[(kp1+km1)-2*lam_inf]*(1-exp(-beta*dt))/beta - 2*lam_inf*dt)
  mu_u     = 0.5 + (mu-0.5)*exp(-2*alpha*dt)  [no-jump belief under symmetric Psi]
  d(u)     = d_fixed * exp(-rho * u)           [deterministic transient-deviation path]

Symmetry restriction: kp^2 = km^1, km^2 = kp^1 (grid tracks only regime-1 values).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gamma as gamma_fn
from typing import Dict, Optional, Tuple
import yaml


# ---------------------------------------------------------------------------
# Elementary impact/excitation functions
# ---------------------------------------------------------------------------

def _Q(xi: float, c: float, e: float) -> float:
    """Price impact Q(xi) = (e*xi/c)^{1/e}."""
    return float((e * max(xi, 0.0) / c) ** (1.0 / e))


def _TC(xi: float, c: float, e: float) -> float:
    """Transaction cost TC(xi) = c/(e+1) * Q(xi)^{e+1}."""
    Qxi = _Q(xi, c, e)
    return float(c / (e + 1.0) * Qxi ** (e + 1.0))


def _phi(v: float, eta: float, zeta: float) -> float:
    """Excitation magnitude phi(v) = eta * (zeta^eta / Gamma(1+eta)) * v^eta.

    The leading eta factor ensures subcriticality (E[phi]=eta, branching ratio
    2*eta/beta = 2*0.05/0.5 = 0.2).  Consistent with hawkes_marked.excitation_phi.
    """
    if v <= 0.0:
        return 0.0
    return float(eta * (zeta ** eta / gamma_fn(1.0 + eta)) * v ** eta)


# ---------------------------------------------------------------------------
# Analytical no-jump evolution
# ---------------------------------------------------------------------------

def no_jump_intensity(kappa: float, lam_inf: float, beta: float, dt: float) -> float:
    """Deterministically evolved intensity after dt with no jumps."""
    return float(max(lam_inf + (kappa - lam_inf) * np.exp(-beta * dt), 0.0))


def no_jump_belief(mu: float, alpha: float, dt: float) -> float:
    """No-jump belief propagation under symmetric generator with off-diag rate alpha.

    mu_u = 0.5 + (mu - 0.5) * exp(-2*alpha*dt)
    """
    return float(np.clip(0.5 + (mu - 0.5) * np.exp(-2.0 * alpha * dt), 0.0, 1.0))


def survival_prob(kp1: float, km1: float, lam_inf: float,
                  beta: float, dt: float) -> float:
    """No-jump survival probability (closed-form via symmetry).

    Both regimes have total intensity kp1+km1, so:
    S(t,t+dt) = exp(-integral_0^{dt} [kp1(s)+km1(s)] ds)
    where kp1(s)+km1(s) = (kp1+km1-2*lam_inf)*exp(-beta*s) + 2*lam_inf.
    """
    total0 = kp1 + km1
    total_inf = 2.0 * lam_inf
    if beta > 1e-10:
        integral = (total0 - total_inf) * (1.0 - np.exp(-beta * dt)) / beta + total_inf * dt
    else:
        integral = total0 * dt
    return float(np.exp(-max(integral, 0.0)))


def no_jump_evolution(
    kp1: float, km1: float, lam_inf: float, beta: float, dt: float
) -> Tuple[float, float]:
    """Evolve (kp1, km1) by dt with no jumps."""
    kp1_new = no_jump_intensity(kp1, lam_inf, beta, dt)
    km1_new = no_jump_intensity(km1, lam_inf, beta, dt)
    return kp1_new, km1_new


# ---------------------------------------------------------------------------
# Intervention operator
# ---------------------------------------------------------------------------

def reduced_intervention_objective(
    xi: float,
    x: float,
    d: float,
    kp1: float,
    km1: float,
    mu: float,
    t: float,
    g_interp: RegularGridInterpolator,
    nu: float,
    c: float,
    e: float,
    c0: float,
    kappa_max: float,
    x_max: float,
) -> float:
    """Reduced intervention objective for selling xi shares.

    obj(xi) = d*xi - TC(xi) - c0 - nu*Q(xi)*(x-xi)
              + g_interp(t, x-xi, d-(1-nu)*Q(xi), kp1, km1, mu)

    For xi=0: returns 0.0.

    Parameters
    ----------
    xi:
        Sell size (in [0, x]).
    x, d, kp1, km1, mu:
        Current state.
    t:
        Current time.
    g_interp:
        Interpolator for g_{N-1} over grid (t, x, kp1, km1, mu).
    nu, c, e, c0:
        Model parameters.
    kappa_max, x_max:
        Grid bounds for clipping.

    Returns
    -------
    Objective value (scalar).
    """
    if xi <= 0.0:
        return 0.0
    xi = min(xi, x)
    Qxi = _Q(xi, c, e)
    tc = _TC(xi, c, e)

    # Revenue: d*xi - transaction_cost(xi) - fixed_cost - permanent_impact_on_remainder
    rev = d * xi - tc - c0 - nu * Qxi * (x - xi)

    # Post-trade state
    x_new = x - xi
    d_new = d - (1.0 - nu) * Qxi  # transient deviation after sale

    # Clip to grid bounds for interpolation
    x_new_c = float(np.clip(x_new, 0.0, x_max))
    kp1_c = float(np.clip(kp1, 0.0, kappa_max))
    km1_c = float(np.clip(km1, 0.0, kappa_max))
    mu_c = float(np.clip(mu, 0.0, 1.0))

    # g_interp is over (t, x, kp1, km1, mu) — d not in grid (d=0 assumed)
    cont = float(g_interp([[t, x_new_c, kp1_c, km1_c, mu_c]])[0])
    return rev + cont


def argmax_intervention(
    x: float,
    d: float,
    kp1: float,
    km1: float,
    mu: float,
    t: float,
    g_interp: RegularGridInterpolator,
    nu: float,
    c: float,
    e: float,
    c0: float,
    kappa_max: float,
    x_max: float,
    n_xi: int = 40,
) -> Tuple[float, float]:
    """Find xi* and M_tilde g value via grid search.

    Parameters
    ----------
    x:
        Current inventory (>= 0).
    Returns
    -------
    (xi_star, M_tilde_value): optimal sell size and intervention value.
    """
    if x <= 0.0:
        return 0.0, 0.0

    # Include xi=0 exactly; scale grid with x
    xi_grid = np.linspace(0.0, x, n_xi + 1)
    best_obj = 0.0  # xi=0 gives 0
    xi_star = 0.0

    for xi in xi_grid[1:]:  # skip xi=0 (already initialised to 0)
        obj = reduced_intervention_objective(
            xi, x, d, kp1, km1, mu, t, g_interp,
            nu, c, e, c0, kappa_max, x_max
        )
        if obj > best_obj:
            best_obj = obj
            xi_star = xi

    return xi_star, best_obj


# ---------------------------------------------------------------------------
# Grid initialisation (g_0)
# ---------------------------------------------------------------------------

def build_g0(
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    kappa_grid: np.ndarray,
    mu_grid: np.ndarray,
    T: float,
    rho: float,
    c: float,
    e: float,
    c0: float,
    d_fixed: float = 1.5,
) -> np.ndarray:
    """Build g_0: no-intervention reduced value function.

    Under no interventions the entire position x is liquidated at terminal
    time T.  Transient deviation at T is d(T) = d_fixed * exp(-rho*T).
    For earlier times t the forced-liquidation value uses the same d_T
    (since d decays independently of interventions in g_0).

    g_0(t, x) = d_T * x - TC(x) - c0   for x > 0
    g_0(t, 0) = 0

    d_fixed (researcher-chosen, documented in config) is the initial
    transient deviation at t=0.  The paper does not specify this value;
    1.5 is the default chosen to produce non-trivial exercise regions.

    Shape: (n_t, n_x, n_kp, n_km, n_mu).
    """
    n_t, n_x, n_kp, n_km, n_mu = (
        len(t_grid), len(x_grid), len(kappa_grid), len(kappa_grid), len(mu_grid)
    )
    g0 = np.zeros((n_t, n_x, n_kp, n_km, n_mu), dtype=float)

    d_T = d_fixed * np.exp(-rho * T)  # transient deviation at terminal time

    for xi, x in enumerate(x_grid):
        if x <= 0.0:
            g0[:, xi, :, :, :] = 0.0
        else:
            tc = _TC(x, c, e)
            g0[:, xi, :, :, :] = d_T * x - tc - c0

    return g0


# ---------------------------------------------------------------------------
# One DP iteration
# ---------------------------------------------------------------------------

def _gl_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Laguerre nodes and weights for int_0^inf f(x) exp(-x) dx."""
    return np.polynomial.laguerre.laggauss(n)


def _legendre_nodes_weights(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights for int_a^b f(x) dx."""
    xi_ref, wi_ref = np.polynomial.legendre.leggauss(n)
    # Transform from [-1,1] to [a,b]
    nodes = 0.5 * (b - a) * xi_ref + 0.5 * (b + a)
    weights = 0.5 * (b - a) * wi_ref
    return nodes, weights


def dp_iteration(
    g_prev: np.ndarray,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    kappa_grid: np.ndarray,
    mu_grid: np.ndarray,
    lam_inf: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
    c: float,
    e: float,
    c0: float,
    eta: float,
    zeta: float,
    d_fixed: float = 1.5,
    n_xi: int = 30,
    n_quad_time: int = 4,
    n_quad_mark: int = 6,
) -> Tuple[np.ndarray, float]:
    """One DP iteration via vectorised numpy operations.

    Vectorises over all (x, kp1, km1, mu) state combinations simultaneously
    for each (ti, ui) decision-time pair, calling the RegularGridInterpolator
    in large batches to avoid Python-loop overhead.

    Parameters
    ----------
    g_prev:
        Previous iterate g_{N-1}, shape (n_t, n_x, n_kp, n_km, n_mu).
    t_grid, x_grid, kappa_grid, mu_grid:
        Grid arrays.
    lam_inf:
        Baseline intensity (same for both regimes by symmetry).
    alpha:
        Off-diagonal transition rate in the symmetric generator.
    beta, nu, rho, c, e, c0, eta, zeta:
        Model parameters.
    n_xi:
        Action grid resolution for the sell-size argmax.
    n_quad_time:
        Gauss-Legendre nodes for the first-jump time integral.
    n_quad_mark:
        Gauss-Laguerre nodes for the mark integral.

    Returns
    -------
    g_new:
        New iterate g_N, shape matching g_prev.
    delta_N:
        max|g_N - g_prev| over the grid.
    """
    T = float(t_grid[-1])
    n_t, n_x, n_kp, n_km, n_mu = g_prev.shape
    kappa_max = float(kappa_grid[-1])
    x_max = float(x_grid[-1])

    g_interp = RegularGridInterpolator(
        (t_grid, x_grid, kappa_grid, kappa_grid, mu_grid),
        g_prev,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    # Gauss-Laguerre for mark integral
    gl_nodes, gl_weights = _gl_nodes_weights(n_quad_mark)
    phi_marks = np.array([_phi(v / zeta, eta, zeta) for v in gl_nodes])

    # Precompute d-independent parts of action-grid revenues for every x level.
    # Full revenue at intervention time u: d(u)*xi - TC(xi) - c0 - nu*Q(xi)*(x-xi)
    # where d(u) = d_fixed * exp(-rho*u) is added per (ti,ui) pair.
    xi_actions_list = []
    rev_base_list = []  # -TC(xi) - c0 - nu*Q(xi)*(x - xi)  [d-free part]
    Qxi_list = []
    for xi_idx in range(n_x):
        x_now = float(x_grid[xi_idx])
        if x_now <= 0.0:
            xi_actions_list.append(np.array([0.0]))
            rev_base_list.append(np.array([0.0]))
            Qxi_list.append(np.array([0.0]))
        else:
            xia = np.linspace(0.0, x_now, n_xi + 1)[1:]  # (n_xi,)
            Qxi = np.array([_Q(float(xi), c, e) for xi in xia])
            tc = np.array([_TC(float(xi), c, e) for xi in xia])
            rev_base = -tc - c0 - nu * Qxi * (x_now - xia)
            xi_actions_list.append(xia)
            rev_base_list.append(rev_base)
            Qxi_list.append(Qxi)

    # Initialise best_val for all (ti, x, kp, km, mu)
    best_val = np.full((n_t, n_x, n_kp, n_km, n_mu), -np.inf, dtype=float)

    # Build flat indices for the (kp, km, mu) triplet — reused across loops
    kp_idx_all, km_idx_all, mu_idx_all = np.meshgrid(
        np.arange(n_kp), np.arange(n_km), np.arange(n_mu), indexing="ij"
    )
    kp_flat = kp_idx_all.ravel()  # (n_kp*n_km*n_mu,)
    km_flat = km_idx_all.ravel()
    mu_flat = mu_idx_all.ravel()
    n_state = n_kp * n_km * n_mu

    for ti in range(n_t):
        t_now = float(t_grid[ti])

        for ui in range(ti, n_t):
            u = float(t_grid[ui])
            dt_u = u - t_now

            # ---- No-jump evolved intensities and belief ----
            exp_beta = np.exp(-beta * dt_u)
            kp1_u = np.maximum(lam_inf + (kappa_grid - lam_inf) * exp_beta, 0.0)  # (n_kp,)
            km1_u = np.maximum(lam_inf + (kappa_grid - lam_inf) * exp_beta, 0.0)  # (n_km,)
            mu_u = 0.5 + (mu_grid - 0.5) * np.exp(-2.0 * alpha * dt_u)       # (n_mu,)
            mu_u = np.clip(mu_u, 0.0, 1.0)
            # Transient deviation at intervention time u (deterministic decay from t=0)
            d_at_u = d_fixed * np.exp(-rho * u)

            # Survival probability: both regimes share same total intensity
            # S(dt) = exp(-integral_0^dt [kp+km](s) ds)
            total0 = kappa_grid[:, np.newaxis] + kappa_grid[np.newaxis, :]  # (n_kp, n_km)
            total_inf = 2.0 * lam_inf
            if beta > 1e-10:
                integral_u = (total0 - total_inf) * (1.0 - exp_beta) / beta + total_inf * dt_u
            else:
                integral_u = total0 * dt_u
            surv_u = np.exp(-np.maximum(integral_u, 0.0))  # (n_kp, n_km)

            # ---- M_tilde computation (batched over all states) ----
            # m_tilde_all[x_idx, kp_idx, km_idx, mu_idx]
            m_tilde_all = np.zeros((n_x, n_kp, n_km, n_mu), dtype=float)

            for xi_idx in range(n_x):
                x_now = float(x_grid[xi_idx])
                if x_now <= 0.0:
                    continue
                xia = xi_actions_list[xi_idx]               # (n_xi,)
                rev = rev_base_list[xi_idx] + d_at_u * xia  # (n_xi,) full revenue
                n_xi_a = len(xia)
                x_new = np.clip(x_now - xia, 0.0, x_max)  # (n_xi,)

                # Build batched query: (n_xi * n_state, 5)
                # xi_a: (n_xi,), (kp, km, mu): (n_state,) -> outer product
                xi_rep = np.repeat(np.arange(n_xi_a), n_state)     # (n_xi*n_state,)
                kp_rep = np.tile(kp_flat, n_xi_a)
                km_rep = np.tile(km_flat, n_xi_a)
                mu_rep = np.tile(mu_flat, n_xi_a)

                query = np.stack([
                    np.full(len(xi_rep), u),
                    x_new[xi_rep],
                    np.clip(kp1_u[kp_rep], 0.0, kappa_max),
                    np.clip(km1_u[km_rep], 0.0, kappa_max),
                    np.clip(mu_u[mu_rep], 0.0, 1.0),
                ], axis=1)  # (n_xi*n_state, 5)

                cont = g_interp(query).reshape(n_xi_a, n_state)  # (n_xi, n_state)
                # obj[xi_a, state] = rev[xi_a] + cont[xi_a, state]
                obj = rev[:, np.newaxis] + cont  # (n_xi, n_state)
                # m_tilde = max over xi_a, compare against xi=0 (value=0)
                m_tilde_flat = np.maximum(np.max(obj, axis=0), 0.0)  # (n_state,)
                m_tilde_all[xi_idx] = m_tilde_flat.reshape(n_kp, n_km, n_mu)

            # term_no_jump[x, kp, km, mu] = surv_u[kp, km] * m_tilde_all[x, kp, km, mu]
            term_nj = surv_u[np.newaxis, :, :, np.newaxis] * m_tilde_all  # (n_x, n_kp, n_km, n_mu)

            # ---- First-jump integral ----
            fj_integral = np.zeros((n_x, n_kp, n_km, n_mu), dtype=float)

            if dt_u > 1e-8:
                gl_t_nodes, gl_t_weights = _legendre_nodes_weights(n_quad_time, t_now, u)

                for q_i, u_prime in enumerate(gl_t_nodes):
                    dt_p = u_prime - t_now
                    exp_beta_p = np.exp(-beta * dt_p)
                    kp1_up = np.maximum(lam_inf + (kappa_grid - lam_inf) * exp_beta_p, 0.0)
                    km1_up = np.maximum(lam_inf + (kappa_grid - lam_inf) * exp_beta_p, 0.0)

                    if beta > 1e-10:
                        integral_p = (total0 - total_inf) * (1.0 - exp_beta_p) / beta + total_inf * dt_p
                    else:
                        integral_p = total0 * dt_p
                    surv_up = np.exp(-np.maximum(integral_p, 0.0))  # (n_kp, n_km)

                    # Post-jump beliefs: (n_kp, n_km, n_mu)
                    mu3 = mu_grid[np.newaxis, np.newaxis, :]
                    kp3 = kp1_up[:, np.newaxis, np.newaxis]
                    km3 = km1_up[np.newaxis, :, np.newaxis]
                    denom_plus  = mu3 * kp3 + (1.0 - mu3) * km3
                    denom_minus = mu3 * km3 + (1.0 - mu3) * kp3
                    safe_p = np.where(denom_plus  > 1e-15, denom_plus,  1.0)
                    safe_m = np.where(denom_minus > 1e-15, denom_minus, 1.0)
                    mu_plus  = np.clip(mu3 * kp3 / safe_p, 0.0, 1.0)  # (n_kp, n_km, n_mu)
                    mu_minus = np.clip(mu3 * km3 / safe_m, 0.0, 1.0)

                    # Mark integral: accumulate over Gauss-Laguerre nodes
                    mark_int_p = np.zeros((n_x, n_kp, n_km, n_mu), dtype=float)
                    mark_int_m = np.zeros((n_x, n_kp, n_km, n_mu), dtype=float)

                    for mk_i, phi_k in enumerate(phi_marks):
                        kp1_j = np.clip(kp1_up + phi_k, 0.0, kappa_max)  # (n_kp,)
                        km1_j = np.clip(km1_up + phi_k, 0.0, kappa_max)  # (n_km,)
                        t_q = float(np.clip(u_prime, t_grid[0], T))

                        # Batch query for ALL (x, kp, km, mu) simultaneously
                        # Side +: mu_plus[kp, km, mu]
                        # Side -: mu_minus[kp, km, mu]

                        # Build (n_x * n_state, 5) query
                        x_rep = np.repeat(np.arange(n_x), n_state)
                        kp_rep2 = np.tile(kp_flat, n_x)
                        km_rep2 = np.tile(km_flat, n_x)
                        mu_rep2 = np.tile(mu_flat, n_x)

                        x_col = np.clip(x_grid[x_rep], 0.0, x_max)
                        kp_j_col = np.clip(kp1_j[kp_rep2], 0.0, kappa_max)
                        km_j_col = np.clip(km1_j[km_rep2], 0.0, kappa_max)

                        mu_p_col = np.clip(
                            mu_plus.ravel()[kp_rep2 * n_km * n_mu + km_rep2 * n_mu + mu_rep2],
                            0.0, 1.0
                        )
                        mu_m_col = np.clip(
                            mu_minus.ravel()[kp_rep2 * n_km * n_mu + km_rep2 * n_mu + mu_rep2],
                            0.0, 1.0
                        )

                        t_col_q = np.full(len(x_rep), t_q)

                        gp = g_interp(np.stack([t_col_q, x_col, kp_j_col, km_j_col, mu_p_col], axis=1))
                        gm = g_interp(np.stack([t_col_q, x_col, kp_j_col, km_j_col, mu_m_col], axis=1))

                        mark_int_p += gl_weights[mk_i] * gp.reshape(n_x, n_kp, n_km, n_mu)
                        mark_int_m += gl_weights[mk_i] * gm.reshape(n_x, n_kp, n_km, n_mu)

                    # surv_up[kp, km], denom[kp, km, mu] -> fj contribution (n_x, n_kp, n_km, n_mu)
                    s3 = surv_up[np.newaxis, :, :, np.newaxis]
                    dp3 = denom_plus[np.newaxis, :, :, :]
                    dm3 = denom_minus[np.newaxis, :, :, :]
                    fj_integral += gl_t_weights[q_i] * s3 * (dp3 * mark_int_p + dm3 * mark_int_m)

            total_val = term_nj + fj_integral  # (n_x, n_kp, n_km, n_mu)
            best_val[ti] = np.maximum(best_val[ti], total_val)

    # Monotone enforcement
    g_new = np.maximum(best_val, g_prev)
    delta_N = float(np.max(np.abs(g_new - g_prev)))
    return g_new, delta_N


# ---------------------------------------------------------------------------
# Full DP solver
# ---------------------------------------------------------------------------

def run_dp_solver(
    config_path: str = "/workspace/project/configs/filter_default.yaml",
    n_t: int = 8,
    n_x: int = 10,
    n_kp: int = 8,
    n_km: int = 8,
    n_mu: int = 9,
) -> Dict:
    """Run the complete DP solver for N_iter iterations.

    Uses a small grid for computational feasibility.  Grid sizes can be
    overridden via the function arguments.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    n_t, n_x, n_kp, n_km, n_mu:
        Grid sizes (override config defaults for speed).

    Returns
    -------
    dict with keys:
        'g_iterations': list of g_N arrays (len = N_iter+1, index 0 = g_0).
        'delta_N':      array of max differences, length N_iter.
        't_grid', 'x_grid', 'kappa_grid', 'mu_grid': grid arrays.
        'config':       loaded config dict.
        'g_interp_final': RegularGridInterpolator for the final g_N.
    """
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    model = cfg["model"]
    T = float(model["T"])
    zeta = float(model["zeta"])
    beta = float(model["beta"])
    nu = float(model["nu"])
    rho = float(model["rho"])
    c0 = float(model["c0"])
    eta = float(model["eta"])
    lam_inf = float(model["lambda_infty_1"])
    c = float(model["c"])
    e = float(model["e"])
    psi_mat = np.array(model["generator"], dtype=float)
    alpha = float(psi_mat[0, 1])  # off-diagonal rate

    N_iter = int(cfg["dp_solver"]["N_iter"])
    kappa_max = float(cfg["dp_solver"]["kappa_max"])
    x_max = float(cfg["dp_solver"]["x_max"])
    d_fixed = float(cfg["dp_solver"].get("d_fixed", 1.5))

    # Build grids
    t_grid = np.linspace(0.0, T, n_t)
    x_grid = np.linspace(0.0, x_max, n_x)
    kappa_grid = np.linspace(0.0, kappa_max, n_kp)
    mu_grid = np.linspace(0.0, 1.0, n_mu)

    print(f"Grid: n_t={n_t}, n_x={n_x}, n_kp={n_kp}, n_km={n_km}, n_mu={n_mu}")
    print(f"Total states: {n_t * n_x * n_kp * n_km * n_mu:,}")
    print(f"Running {N_iter} DP iterations...")

    # Initialise g_0 (d_fixed is the initial transient deviation at t=0)
    g = build_g0(t_grid, x_grid, kappa_grid, mu_grid, T, rho, c, e, c0, d_fixed)
    g_iterations = [g.copy()]
    delta_N_list = []

    for n_iter in range(1, N_iter + 1):
        g_new, delta = dp_iteration(
            g, t_grid, x_grid, kappa_grid, mu_grid,
            lam_inf, alpha, beta, nu, rho, c, e, c0, eta, zeta, d_fixed,
        )
        g = g_new
        g_iterations.append(g.copy())
        delta_N_list.append(delta)
        print(f"  Iter {n_iter:2d}/{N_iter}: delta_N = {delta:.6f}")

    delta_N = np.array(delta_N_list)

    g_interp_final = RegularGridInterpolator(
        (t_grid, x_grid, kappa_grid, kappa_grid, mu_grid),
        g,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    return {
        "g_iterations": g_iterations,
        "delta_N": delta_N,
        "t_grid": t_grid,
        "x_grid": x_grid,
        "kappa_grid": kappa_grid,
        "mu_grid": mu_grid,
        "g_interp_final": g_interp_final,
        "config": cfg,
        "d_fixed": d_fixed,
        "rho": rho,
    }


# ---------------------------------------------------------------------------
# Convenience: compute exercise region and optimal order size on a 2D slice
# ---------------------------------------------------------------------------

def compute_exercise_map(
    g_solution: Dict,
    mu_values: np.ndarray,
    x_values: np.ndarray,
    kp1: float,
    km1: float,
    t: float,
    tol: float = 1e-5,
    n_xi: int = 40,
) -> np.ndarray:
    """Classify each (mu, x) as exercise or continuation at fixed (kp1, km1, t).

    Parameters
    ----------
    g_solution:
        Output of run_dp_solver.
    mu_values:
        Belief grid for the plot (1D).
    x_values:
        Inventory grid for the plot (1D).
    kp1, km1:
        Fixed candidate intensities.
    t:
        Fixed time.
    tol:
        Exercise if M_tilde_g >= g - tol.
    n_xi:
        Action grid resolution.

    Returns
    -------
    exercise_map: bool array of shape (len(mu_values), len(x_values)).
    """
    g_interp = g_solution["g_interp_final"]
    kappa_max = float(g_solution["kappa_grid"][-1])
    x_max = float(g_solution["x_grid"][-1])
    cfg = g_solution["config"]
    model = cfg["model"]
    nu = float(model["nu"])
    c = float(model["c"])
    e = float(model["e"])
    c0 = float(model["c0"])

    M, N = len(mu_values), len(x_values)
    exercise_map = np.zeros((M, N), dtype=bool)

    rho = float(g_solution.get("rho", 0.1))
    d_fixed = float(g_solution.get("d_fixed", 1.5))
    d_at_t = d_fixed * np.exp(-rho * t)  # deterministic d at evaluation time t

    for mi, mu in enumerate(mu_values):
        mu_c = float(np.clip(mu, 0.0, 1.0))
        for xi_i, x in enumerate(x_values):
            # Nothing to sell when inventory is zero.
            if x <= 0.0:
                exercise_map[mi, xi_i] = False
                continue

            x_c = float(np.clip(x, 0.0, x_max))
            kp1_c = float(np.clip(kp1, 0.0, kappa_max))
            km1_c = float(np.clip(km1, 0.0, kappa_max))

            g_val = float(g_interp([[t, x_c, kp1_c, km1_c, mu_c]])[0])

            _, m_val = argmax_intervention(
                x_c, d_at_t, kp1_c, km1_c, mu_c, t, g_interp,
                nu, c, e, c0, kappa_max, x_max, n_xi
            )
            exercise_map[mi, xi_i] = (m_val >= g_val - tol)

    return exercise_map


def compute_order_size_map(
    g_solution: Dict,
    mu_values: np.ndarray,
    x_values: np.ndarray,
    kp1: float,
    km1: float,
    t: float,
    c_override: Optional[float] = None,
    n_xi: int = 50,
) -> np.ndarray:
    """Compute optimal order size xi*(mu, x) at fixed (kp1, km1, t).

    Parameters
    ----------
    g_solution:
        Output of run_dp_solver.
    mu_values, x_values:
        2D plot grids.
    kp1, km1:
        Fixed candidate intensities.
    t:
        Fixed time.
    c_override:
        If given, use this value for c instead of the config value.
    n_xi:
        Action grid resolution.

    Returns
    -------
    xi_star: float array of shape (len(mu_values), len(x_values)).
    """
    g_interp = g_solution["g_interp_final"]
    kappa_max = float(g_solution["kappa_grid"][-1])
    x_max = float(g_solution["x_grid"][-1])
    cfg = g_solution["config"]
    model = cfg["model"]
    nu = float(model["nu"])
    c = c_override if c_override is not None else float(model["c"])
    e = float(model["e"])
    c0 = float(model["c0"])

    M, N = len(mu_values), len(x_values)
    xi_star = np.zeros((M, N), dtype=float)

    rho = float(g_solution.get("rho", 0.1))
    d_fixed = float(g_solution.get("d_fixed", 1.5))
    d_at_t = d_fixed * np.exp(-rho * t)  # deterministic d at evaluation time t

    for mi, mu in enumerate(mu_values):
        mu_c = float(np.clip(mu, 0.0, 1.0))
        for xi_i, x in enumerate(x_values):
            x_c = float(np.clip(x, 0.0, x_max))
            kp1_c = float(np.clip(kp1, 0.0, kappa_max))
            km1_c = float(np.clip(km1, 0.0, kappa_max))

            xi_opt, _ = argmax_intervention(
                x_c, d_at_t, kp1_c, km1_c, mu_c, t, g_interp,
                nu, c, e, c0, kappa_max, x_max, n_xi
            )
            xi_star[mi, xi_i] = float(np.clip(xi_opt, 0.0, x_c))

    return xi_star
