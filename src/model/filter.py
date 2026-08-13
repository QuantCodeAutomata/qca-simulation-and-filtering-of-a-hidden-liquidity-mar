"""Finite-dimensional filter for the hidden Markov-modulated marked Hawkes process.

Between jumps the unnormalised filter sigma evolves as:
    d sigma_i/dt = (sigma @ Psi)_i - sigma_i * (lambda^{i,+}_t + lambda^{i,-}_t)

where the candidate intensities decay deterministically:
    lambda^{i,k}_t = lambda_infty_i + (kappa_k_i - lambda_infty_i) * exp(-beta * t_rel)

At a jump (side k, mark v) a Bayes update is applied:
    pi(i) <- pi(i) * lambda^{i,k} * nu_i(v)  then normalised.

After each jump the candidate intensities gain phi(v) (see hawkes_marked.py).
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple

from .hawkes_marked import excitation_phi, mark_density

_EPS = 1e-15


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def filter_bayes_update(
    pi: np.ndarray,
    lambda_k: np.ndarray,
    nu_v: np.ndarray,
    eps: float = _EPS,
) -> np.ndarray:
    """Apply a Bayes update at a jump event.

    pi_new(i) = pi(i) * lambda_k(i) * nu_v(i) / Z
    where Z is the normalising constant.

    Parameters
    ----------
    pi:
        Current posterior, shape (n_states,).
    lambda_k:
        Candidate intensity for the observed side at jump time,
        shape (n_states,).
    nu_v:
        Mark densities nu_i(v) evaluated at the observed mark v,
        shape (n_states,).
    eps:
        Floor added to Z to avoid division by zero.

    Returns
    -------
    pi_new:
        Updated and normalised posterior, shape (n_states,).
    """
    weights = pi * lambda_k * nu_v
    total = float(weights.sum())
    if total < eps:
        return pi.copy()
    return weights / total


def filter_between_jumps(
    pi: np.ndarray,
    generator: np.ndarray,
    kappa_plus: np.ndarray,
    kappa_minus: np.ndarray,
    lambda_infty: np.ndarray,
    beta: float,
    dt: float,
    n_steps: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evolve the posterior and candidate intensities over [t, t+dt] with no jumps.

    The unnormalised filter sigma satisfies:
        d sigma_i/dt = (sigma @ Psi)_i - sigma_i * (lambda^{i,+}_t + lambda^{i,-}_t)

    with lambda^{i,k}_t = lambda_infty_i + (kappa_k_i - lambda_infty_i)*exp(-beta*t_rel).

    The ODE is integrated with ``scipy.integrate.solve_ivp`` (RK45).

    Parameters
    ----------
    pi:
        Current posterior (normalised), shape (n_states,).
    generator:
        CTMC rate matrix Psi, shape (n_states, n_states).
    kappa_plus:
        Candidate buy intensities at t, shape (n_states,).
    kappa_minus:
        Candidate sell intensities at t, shape (n_states,).
    lambda_infty:
        Baseline intensities, shape (n_states,).
    beta:
        Mean-reversion speed.
    dt:
        Time increment (must be >= 0).
    n_steps:
        Controls the maximum ODE step size: max_step = dt / n_steps.

    Returns
    -------
    pi_new:
        Updated normalised posterior at t+dt, shape (n_states,).
    kappa_plus_new:
        Evolved buy intensities at t+dt, shape (n_states,).
    kappa_minus_new:
        Evolved sell intensities at t+dt, shape (n_states,).
    """
    if dt <= 0.0:
        kp = np.maximum(kappa_plus.copy(), 0.0)
        km = np.maximum(kappa_minus.copy(), 0.0)
        return pi.copy(), kp, km

    kappa_plus = np.asarray(kappa_plus, dtype=float)
    kappa_minus = np.asarray(kappa_minus, dtype=float)
    lambda_infty = np.asarray(lambda_infty, dtype=float)

    # Pre-compute amplitude of exponential decay for each side.
    amp_plus = kappa_plus - lambda_infty    # (kappa_plus[i] - lambda_infty[i])
    amp_minus = kappa_minus - lambda_infty  # (kappa_minus[i] - lambda_infty[i])

    def ode_rhs(t_rel: float, sigma: np.ndarray) -> np.ndarray:
        decay = np.exp(-beta * t_rel)
        kp_t = np.maximum(lambda_infty + amp_plus * decay, 0.0)
        km_t = np.maximum(lambda_infty + amp_minus * decay, 0.0)
        total_lambda = kp_t + km_t
        # d sigma / dt = sigma @ Psi  -  sigma * total_lambda
        return sigma @ generator - sigma * total_lambda

    sol = solve_ivp(
        ode_rhs,
        t_span=(0.0, dt),
        y0=pi.copy(),
        method="RK45",
        max_step=dt / max(n_steps, 1),
        dense_output=False,
        rtol=1e-6,
        atol=1e-9,
    )

    sigma_new = np.maximum(sol.y[:, -1], 0.0)
    total = float(sigma_new.sum())
    pi_new = sigma_new / total if total > _EPS else pi.copy()

    # Closed-form intensity evolution (same formula used in MarkovHawkesState).
    decay_dt = np.exp(-beta * dt)
    kappa_plus_new = np.maximum(lambda_infty + amp_plus * decay_dt, 0.0)
    kappa_minus_new = np.maximum(lambda_infty + amp_minus * decay_dt, 0.0)

    return pi_new, kappa_plus_new, kappa_minus_new


# ---------------------------------------------------------------------------
# Full filter pass
# ---------------------------------------------------------------------------

def run_filter(
    event_times: np.ndarray,
    event_sides: np.ndarray,
    event_marks: np.ndarray,
    T: float,
    generator: np.ndarray,
    init_kappa_plus: np.ndarray,
    init_kappa_minus: np.ndarray,
    lambda_infty: np.ndarray,
    beta: float,
    eta: float,
    zeta: float,
    pi0: np.ndarray,
    eps: float = _EPS,
) -> Dict:
    """Run the complete filter over [0, T].

    Processes each event in chronological order:
      1. Evolve posterior and intensities from previous event time to current
         event time (no-jump ODE).
      2. Bayes update using the observed side and mark.
      3. Add phi(v) to all candidate intensities (jump excitation).

    After the last event the filter is evolved to T.

    Parameters
    ----------
    event_times:
        Jump times, shape (n_events,); need not be sorted.
    event_sides:
        +1 / -1, shape (n_events,).
    event_marks:
        Mark values v > 0, shape (n_events,).
    T:
        Horizon.
    generator:
        CTMC rate matrix Psi, shape (n_states, n_states).
    init_kappa_plus:
        Initial candidate buy intensities at t=0, shape (n_states,).
    init_kappa_minus:
        Initial candidate sell intensities at t=0, shape (n_states,).
    lambda_infty:
        Baseline intensities, shape (n_states,).
    beta:
        Mean-reversion speed.
    eta:
        Excitation exponent for phi(v).
    zeta:
        Exponential mark rate.
    pi0:
        Prior (initial posterior) at t=0, shape (n_states,).
    eps:
        Numerical floor for normalisation.

    Returns
    -------
    dict with keys:
        'filter_times': 1-D array including 0, all event times, and T.
        'posterior':    2-D array (n_times, n_states), pi_t at each time.
        'kappa_plus':   2-D array (n_times, n_states), lambda^{i,+}.
        'kappa_minus':  2-D array (n_times, n_states), lambda^{i,-}.
    """
    # Ensure normalised prior.
    pi = np.asarray(pi0, dtype=float).copy()
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()

    kp = np.asarray(init_kappa_plus, dtype=float).copy()
    km = np.asarray(init_kappa_minus, dtype=float).copy()
    lam_inf = np.asarray(lambda_infty, dtype=float)

    # Sort events chronologically.
    if len(event_times) > 0:
        order = np.argsort(event_times)
        ev_times = np.asarray(event_times, dtype=float)[order]
        ev_sides = np.asarray(event_sides, dtype=int)[order]
        ev_marks = np.asarray(event_marks, dtype=float)[order]
    else:
        ev_times = np.empty(0, dtype=float)
        ev_sides = np.empty(0, dtype=int)
        ev_marks = np.empty(0, dtype=float)

    filter_times_list = [0.0]
    posterior_list = [pi.copy()]
    kappa_plus_list = [kp.copy()]
    kappa_minus_list = [km.copy()]

    t_prev = 0.0

    for idx in range(len(ev_times)):
        t_n = float(ev_times[idx])
        side = int(ev_sides[idx])
        v = float(ev_marks[idx])

        # ---- 1. No-jump evolution from t_prev to t_n ----
        dt = t_n - t_prev
        if dt > 0.0:
            pi, kp, km = filter_between_jumps(
                pi, generator, kp, km, lam_inf, beta, dt
            )

        # ---- 2. Bayes update ----
        lambda_k = kp if side > 0 else km
        nu_v = mark_density(np.array([v]), zeta)   # shape (1,)
        pi = filter_bayes_update(pi, lambda_k, nu_v[0] * np.ones_like(pi), eps)

        # ---- 3. Jump excitation ----
        phi_v = excitation_phi(np.array([v]), eta, zeta)[0]
        kp = kp + phi_v
        km = km + phi_v

        filter_times_list.append(t_n)
        posterior_list.append(pi.copy())
        kappa_plus_list.append(kp.copy())
        kappa_minus_list.append(km.copy())
        t_prev = t_n

    # ---- Final no-jump evolution to T ----
    dt_final = T - t_prev
    if dt_final > 0.0:
        pi, kp, km = filter_between_jumps(
            pi, generator, kp, km, lam_inf, beta, dt_final
        )

    filter_times_list.append(T)
    posterior_list.append(pi.copy())
    kappa_plus_list.append(kp.copy())
    kappa_minus_list.append(km.copy())

    return {
        "filter_times": np.array(filter_times_list, dtype=float),
        "posterior": np.array(posterior_list, dtype=float),
        "kappa_plus": np.array(kappa_plus_list, dtype=float),
        "kappa_minus": np.array(kappa_minus_list, dtype=float),
    }
