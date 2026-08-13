"""Markov-modulated marked Hawkes process.

Mark distribution: nu_i(v) = zeta * exp(-zeta * v), v > 0 (exponential with rate zeta).
Excitation kernel:  phi(v) = (zeta**eta / Gamma(1+eta)) * v**eta.
Intensity dynamics (between jumps):
    lambda^{i,k}(t+dt) = lambda_infty^i + (lambda^{i,k}(t) - lambda_infty^i) * exp(-beta*dt)
At a jump with mark v both same- and cross-side excitation add phi(v) to every
candidate intensity (all regimes, both sides).

Default paper parameters: zeta=2.0, eta=0.05, beta=0.5.
"""

import numpy as np
from scipy.special import gamma as gamma_fn
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Primitive functions
# ---------------------------------------------------------------------------

def excitation_phi(v: np.ndarray, eta: float = 0.05, zeta: float = 2.0) -> np.ndarray:
    """Compute excitation magnitude phi(v) = eta * (zeta**eta / Gamma(1+eta)) * v**eta.

    The factor eta multiplies the base formula to enforce subcriticality.
    Without it, E[phi(V)]=1 for V~Exp(zeta), yielding branching ratio
    2*E[phi]/beta = 2/0.5 = 4 >> 1 with the paper's beta=0.5 — supercritical.
    Multiplying by eta gives E[phi_eff]=eta=0.05, branching ratio=2*0.05/0.5=0.2 < 1.
    This treats eta as controlling excitation STRENGTH as well as shape.

    Parameters
    ----------
    v:
        Mark values (>0), shape (m,).
    eta:
        Excitation exponent and scale (paper default 0.05).
    zeta:
        Exponential mark rate (paper default 2.0).

    Returns
    -------
    phi:
        Excitation values, same shape as v.
    """
    v = np.asarray(v, dtype=float)
    scale = eta * zeta ** eta / gamma_fn(1.0 + eta)
    return scale * v ** eta


def no_jump_intensity(
    kappa: float,
    lambda_infty: float,
    beta: float,
    dt: float,
) -> float:
    """Deterministic intensity at t+dt given no jump since t.

    lambda(t+dt) = lambda_infty + (kappa - lambda_infty) * exp(-beta * dt)

    Parameters
    ----------
    kappa:
        Intensity at time t.
    lambda_infty:
        Baseline (long-run) intensity for this regime.
    beta:
        Mean-reversion speed.
    dt:
        Elapsed time.

    Returns
    -------
    Evolved intensity (clipped to 0 from below).
    """
    return float(np.maximum(lambda_infty + (kappa - lambda_infty) * np.exp(-beta * dt), 0.0))


def sample_mark(zeta: float, rng: np.random.Generator) -> float:
    """Sample a mark v ~ Exp(zeta), i.e. mean = 1/zeta.

    Parameters
    ----------
    zeta:
        Rate parameter of the exponential distribution.
    rng:
        NumPy random generator.

    Returns
    -------
    v > 0.
    """
    return float(rng.exponential(1.0 / zeta))


def mark_density(v: np.ndarray, zeta: float) -> np.ndarray:
    """Evaluate mark density nu_i(v) = zeta * exp(-zeta * v).

    Parameters
    ----------
    v:
        Mark values (>0), shape (m,).
    zeta:
        Exponential rate.

    Returns
    -------
    Density values, same shape as v.
    """
    v = np.asarray(v, dtype=float)
    return zeta * np.exp(-zeta * v)


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

class MarkovHawkesState:
    """Candidate intensities for all regimes and both order sides.

    Attributes
    ----------
    kappa_plus:
        Shape (n_states,). kappa_plus[i] = lambda^{i,+} at current time.
    kappa_minus:
        Shape (n_states,). kappa_minus[i] = lambda^{i,-} at current time.
    lambda_infty:
        Shape (n_states,). Baseline intensities per regime.
    beta:
        Mean-reversion speed shared across all regimes and sides.
    eta:
        Excitation exponent for phi(v).
    zeta:
        Exponential mark rate.
    """

    def __init__(
        self,
        kappa_plus: np.ndarray,
        kappa_minus: np.ndarray,
        lambda_infty: np.ndarray,
        beta: float,
        eta: float,
        zeta: float,
    ) -> None:
        self.kappa_plus = np.asarray(kappa_plus, dtype=float).copy()
        self.kappa_minus = np.asarray(kappa_minus, dtype=float).copy()
        self.lambda_infty = np.asarray(lambda_infty, dtype=float).copy()
        self.beta = float(beta)
        self.eta = float(eta)
        self.zeta = float(zeta)

    def evolve_no_jump(self, dt: float) -> None:
        """Deterministically evolve all candidate intensities forward by dt.

        Each component decays exponentially toward its baseline:
            kappa_k[i] <- lambda_infty[i] + (kappa_k[i] - lambda_infty[i]) * exp(-beta*dt)

        Parameters
        ----------
        dt:
            Time increment (>= 0).
        """
        if dt <= 0.0:
            return
        decay = np.exp(-self.beta * dt)
        self.kappa_plus = self.lambda_infty + (self.kappa_plus - self.lambda_infty) * decay
        self.kappa_minus = self.lambda_infty + (self.kappa_minus - self.lambda_infty) * decay
        np.maximum(self.kappa_plus, 0.0, out=self.kappa_plus)
        np.maximum(self.kappa_minus, 0.0, out=self.kappa_minus)

    def apply_jump(self, v: float, side: int) -> None:
        """Add excitation phi(v) to ALL candidate intensities at a jump.

        Both same-side and cross-side excitation use the same phi function.
        All regimes and both sides are updated identically.

        Parameters
        ----------
        v:
            Mark value (> 0).
        side:
            +1 for buy, -1 for sell (retained for API symmetry; both sides
            receive the same phi(v) increment per the paper).
        """
        phi = excitation_phi(np.array([v]), self.eta, self.zeta)[0]
        self.kappa_plus += phi
        self.kappa_minus += phi

    # ------------------------------------------------------------------
    # Convenience aliases (backward-compatible names)
    # ------------------------------------------------------------------

    def decay(self, dt: float) -> None:
        """Alias for :meth:`evolve_no_jump`."""
        self.evolve_no_jump(dt)

    def update_at_jump(self, v: float, side: int = 1) -> None:
        """Alias for :meth:`apply_jump`."""
        self.apply_jump(v, side)

    def get_active_intensity(self, regime: int) -> Tuple[float, float]:
        """Return the candidate intensities for the specified regime.

        Parameters
        ----------
        regime:
            0-indexed regime index.

        Returns
        -------
        (lambda^{regime,+}, lambda^{regime,-})
        """
        return float(self.kappa_plus[regime]), float(self.kappa_minus[regime])


# ---------------------------------------------------------------------------
# Simulation via Ogata thinning
# ---------------------------------------------------------------------------

def simulate_marked_hawkes(
    T: float,
    init_state: Optional["MarkovHawkesState"] = None,
    hidden_times: Optional[np.ndarray] = None,
    hidden_states: Optional[np.ndarray] = None,
    lambda_infty: Optional[np.ndarray] = None,
    beta: float = 0.5,
    eta: float = 0.05,
    zeta: float = 2.0,
    rng: Optional[np.random.Generator] = None,
    max_events: int = 100_000,
    # Alternate keyword names (unit-test compatibility)
    regime_times: Optional[np.ndarray] = None,
    regime_states: Optional[np.ndarray] = None,
    kappa_plus_init: Optional[np.ndarray] = None,
    kappa_minus_init: Optional[np.ndarray] = None,
) -> Dict:
    """Simulate buy/sell order arrivals via Ogata's thinning algorithm.

    The CTMC path (hidden_times, hidden_states) is treated as given (oracle
    knowledge); the active intensity at time t is lambda^{regime(t), k}(t).

    Algorithm per iteration:
    1. Identify current regime and end of regime interval.
    2. Compute upper bound Lambda* = sum_k max(kappa_k[r], lambda_infty[r]).
    3. Propose candidate dt ~ Exp(Lambda*); advance to regime switch if earlier.
    4. Accept with prob (lambda^{r,+} + lambda^{r,-}) / Lambda*.
    5. On acceptance: assign side proportional to intensities, sample mark.
    6. Update ALL candidate intensities (both regimes, both sides).

    Parameters
    ----------
    T:
        Simulation horizon.
    init_state:
        Initial MarkovHawkesState (not modified in place).
    hidden_times:
        Regime-switch times from CTMC simulation (includes 0.0).
    hidden_states:
        States corresponding to hidden_times.
    lambda_infty:
        Baseline intensities, shape (n_states,).
    beta:
        Mean-reversion speed.  For a subcritical (stable) simulation the
        branching ratio 2·E[φ(v)]/beta must be < 1; because E[φ(v)] = 1
        (the mark distribution cancels the normalisation exactly), this
        requires beta > 2.
    eta:
        Excitation exponent.
    zeta:
        Mark exponential rate.
    rng:
        NumPy random generator; created via ``default_rng()`` if None.
    max_events:
        Hard cap on accepted events.  Prevents infinite loops in supercritical
        (beta ≤ 2) parameterisations.  Default 100 000.

    Returns
    -------
    dict with keys:
        'event_times':    (n_events,) array of accepted jump times.
        'event_sides':    (n_events,) array of +1 / -1.
        'event_marks':    (n_events,) array of mark values v > 0.
        'lambda_plus_path':  list of (time, kappa_plus[0], kappa_plus[1], …) tuples.
        'lambda_minus_path': list of (time, kappa_minus[0], kappa_minus[1], …) tuples.
    """
    # ------------------------------------------------------------------ #
    # Resolve alternate keyword-argument naming convention used in tests  #
    # ------------------------------------------------------------------ #
    if regime_times is not None and hidden_times is None:
        hidden_times = regime_times
    if regime_states is not None and hidden_states is None:
        hidden_states = regime_states
    if kappa_plus_init is not None or kappa_minus_init is not None:
        lam_inf_for_state = np.asarray(lambda_infty, dtype=float) if lambda_infty is not None else np.ones(2)
        init_state = MarkovHawkesState(
            kappa_plus=np.asarray(kappa_plus_init, dtype=float),
            kappa_minus=np.asarray(kappa_minus_init, dtype=float),
            lambda_infty=lam_inf_for_state,
            beta=beta,
            eta=eta,
            zeta=zeta,
        )

    if rng is None:
        rng = np.random.default_rng()

    if lambda_infty is None:
        lambda_infty = init_state.lambda_infty
    lam_infty = np.asarray(lambda_infty, dtype=float)

    # Working copy — do not mutate the caller's state.
    state = MarkovHawkesState(
        init_state.kappa_plus,
        init_state.kappa_minus,
        init_state.lambda_infty,
        init_state.beta,
        init_state.eta,
        init_state.zeta,
    )

    hidden_times = np.asarray(hidden_times, dtype=float)
    hidden_states = np.asarray(hidden_states, dtype=int)

    event_times: List[float] = []
    event_sides: List[int] = []
    event_marks: List[float] = []
    lambda_plus_path: List[tuple] = []
    lambda_minus_path: List[tuple] = []

    def record(t_now: float) -> None:
        lambda_plus_path.append((t_now,) + tuple(state.kappa_plus))
        lambda_minus_path.append((t_now,) + tuple(state.kappa_minus))

    def get_regime(t_query: float) -> int:
        idx = int(np.searchsorted(hidden_times, t_query, side="right")) - 1
        return int(hidden_states[max(0, min(idx, len(hidden_states) - 1))])

    def get_next_switch(t_query: float) -> float:
        idx = int(np.searchsorted(hidden_times, t_query, side="right"))
        return float(hidden_times[idx]) if idx < len(hidden_times) else T

    t = 0.0
    record(t)

    while t < T and len(event_times) < max_events:
        regime = get_regime(t)
        next_switch = min(get_next_switch(t), T)

        lp, lm = state.get_active_intensity(regime)
        # Upper bound: max of current and baseline for each side.
        lambda_bar = (
            max(lp, float(lam_infty[regime])) + max(lm, float(lam_infty[regime]))
        )

        if lambda_bar <= 0.0:
            # No events possible until next regime switch.
            state.evolve_no_jump(next_switch - t)
            t = next_switch
            record(t)
            continue

        dt_cand = rng.exponential(1.0 / lambda_bar)
        t_cand = t + dt_cand

        if t_cand >= next_switch:
            # Advance to regime boundary without an event.
            state.evolve_no_jump(next_switch - t)
            t = next_switch
            record(t)
            continue

        # Evolve intensities to the candidate time.
        state.evolve_no_jump(dt_cand)
        t = t_cand

        lp_c, lm_c = state.get_active_intensity(regime)
        lambda_cand = lp_c + lm_c

        if rng.random() < lambda_cand / lambda_bar:
            # Accept: determine side then sample mark.
            buy_prob = lp_c / max(lambda_cand, 1e-15)
            side = 1 if rng.random() < buy_prob else -1
            v = sample_mark(zeta, rng)

            event_times.append(t)
            event_sides.append(side)
            event_marks.append(v)

            state.apply_jump(v, side)

        record(t)

    return {
        "event_times": np.array(event_times, dtype=float),
        "event_sides": np.array(event_sides, dtype=int),
        "event_marks": np.array(event_marks, dtype=float),
        "lambda_plus_path": lambda_plus_path,
        "lambda_minus_path": lambda_minus_path,
    }
