"""Order-book impact and price-path model.

Order-book depth density: f(x) = c * x^{-1+e}
Price impact function:    Q(v) = (e*v/c)^{1/e}
Transaction cost:         TC(xi) = c/(e+1) * Q(xi)^{e+1}
Transaction revenue:      C(p, xi) = p*xi - TC(xi) - c0

Price dynamics (P = S + D):
    dS_t = nu * Q(v) * dn_t          (permanent impact, pure jump)
    dD_t = -rho * D_t dt + (1-nu)*Q(v)*dn_t   (transient deviation)

where dn = +1 for buy, -1 for sell.

Default paper parameters: c=1.0, e=3.0, nu=0.8, rho=0.1, c0=0.1.
"""

import numpy as np
from typing import Tuple


def price_impact(v: np.ndarray, c: float = 1.0, e: float = 3.0) -> np.ndarray:
    """Compute price impact Q(v) = (e*v/c)^{1/e}.

    Parameters
    ----------
    v:
        Trade size (volume), shape (m,). Must be >= 0.
    c:
        Order-book depth parameter.
    e:
        Order-book shape exponent (e > 0).

    Returns
    -------
    Q:
        Price impact, same shape as v.
    """
    v = np.asarray(v, dtype=float)
    return (e * v / c) ** (1.0 / e)


def transaction_cost(xi: float, c: float = 1.0, e: float = 3.0) -> float:
    """Compute the transaction cost for trading volume xi.

    TC(xi) = integral_0^{Q(xi)} y * f(y) dy = c/(e+1) * Q(xi)^{e+1}

    where f(y) = c * y^{-1+e} is the order-book density.

    Parameters
    ----------
    xi:
        Trade volume (>= 0).
    c:
        Order-book depth parameter.
    e:
        Order-book shape exponent (e > 0).

    Returns
    -------
    TC:
        Transaction cost (scalar, >= 0).
    """
    Q = float(price_impact(np.array([xi]), c, e)[0])
    return (c / (e + 1.0)) * Q ** (e + 1.0)


def transaction_revenue(
    p: float,
    xi: float,
    c: float = 1.0,
    e: float = 3.0,
    c0: float = 0.1,
) -> float:
    """Compute net transaction revenue C(p, xi) = p*xi - TC(xi) - c0.

    Parameters
    ----------
    p:
        Current mid-price.
    xi:
        Trade volume (>= 0).
    c:
        Order-book depth parameter.
    e:
        Order-book shape exponent.
    c0:
        Fixed trading cost.

    Returns
    -------
    C:
        Net revenue (can be negative).
    """
    return p * xi - transaction_cost(xi, c, e) - c0


def simulate_price_path(
    event_times: np.ndarray,
    event_sides: np.ndarray,
    event_marks: np.ndarray,
    T: float,
    nu: float = 0.8,
    rho: float = 0.1,
    c: float = 1.0,
    e: float = 3.0,
    n_grid: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the price paths (S_t, D_t, P_t) on a uniform time grid.

    Between events:
        S_t is constant (no drift).
        D_t decays: D(t) = D(t_prev) * exp(-rho * (t - t_prev)).

    At each event (time t_n, side s_n in {+1,-1}, mark v_n):
        S += nu       * Q(v_n) * s_n
        D += (1 - nu) * Q(v_n) * s_n

    Initial conditions: S(0) = D(0) = 0.

    Parameters
    ----------
    event_times:
        Jump times, shape (n_events,).
    event_sides:
        +1 (buy) or -1 (sell), shape (n_events,).
    event_marks:
        Mark values v > 0, shape (n_events,).
    T:
        Simulation horizon.
    nu:
        Permanent impact fraction in [0, 1].
    rho:
        Transient-deviation decay rate (> 0).
    c:
        Order-book depth parameter.
    e:
        Order-book shape exponent.
    n_grid:
        Number of uniform grid points in [0, T].

    Returns
    -------
    t_grid:
        Time grid, shape (n_grid,).
    S_path:
        Permanent price component, shape (n_grid,).
    D_path:
        Transient deviation, shape (n_grid,).
    P_path:
        Total price P = S + D, shape (n_grid,).
    """
    t_grid = np.linspace(0.0, T, n_grid)

    # Sort events chronologically.
    if len(event_times) > 0:
        order = np.argsort(event_times)
        ev_times = np.asarray(event_times, dtype=float)[order]
        ev_sides = np.asarray(event_sides, dtype=float)[order]
        ev_marks = np.asarray(event_marks, dtype=float)[order]
    else:
        ev_times = np.empty(0, dtype=float)
        ev_sides = np.empty(0, dtype=float)
        ev_marks = np.empty(0, dtype=float)

    S_path = np.zeros(n_grid)
    D_path = np.zeros(n_grid)
    P_path = np.zeros(n_grid)

    # `S` and `D` track the state at time `t_last`.
    # `D` is the post-jump value; continuous decay is applied on demand.
    S = 0.0
    D = 0.0
    t_last = 0.0   # time at which S and D were last updated by a jump
    ev_idx = 0     # index of the next unprocessed event

    for grid_idx, t in enumerate(t_grid):
        # Apply all events with times <= t.
        while ev_idx < len(ev_times) and ev_times[ev_idx] <= t:
            t_ev = float(ev_times[ev_idx])
            # Continuous decay of D from t_last to t_ev.
            D *= np.exp(-rho * (t_ev - t_last))
            t_last = t_ev
            # Jump contribution.
            v = float(ev_marks[ev_idx])
            s = float(ev_sides[ev_idx])
            Q_val = float(price_impact(np.array([v]), c, e)[0])
            S += nu * Q_val * s
            D += (1.0 - nu) * Q_val * s
            ev_idx += 1

        # Decay D from t_last to current grid point (read-only; no mutation).
        D_now = D * np.exp(-rho * (t - t_last))

        S_path[grid_idx] = S
        D_path[grid_idx] = D_now
        P_path[grid_idx] = S + D_now

    return t_grid, S_path, D_path, P_path
