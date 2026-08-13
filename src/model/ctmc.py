"""Continuous-time Markov chain (CTMC) utilities.

Default paper parameters: n_states=2, generator Psi=[[-0.2, 0.2], [0.2, -0.2]].
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple, Optional


def simulate_ctmc(
    generator: np.ndarray,
    T: float,
    init_state: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a CTMC path over [0, T] using exact exponential holding times.

    Parameters
    ----------
    generator:
        Rate matrix Psi of shape (n_states, n_states).  Rows sum to 0;
        off-diagonal entries are non-negative.
    T:
        Finite horizon.
    init_state:
        Initial hidden state (0-indexed).
    rng:
        NumPy random generator; created via ``default_rng()`` if None.

    Returns
    -------
    times:
        Jump times including 0.0 as first element.  The last entry is the
        time of the last jump, which may be strictly less than T.
    states:
        State held during [times[k], times[k+1]).  states[k] gives the
        regime active immediately after times[k].
    """
    if rng is None:
        rng = np.random.default_rng()

    n_states = generator.shape[0]
    times: list[float] = [0.0]
    states: list[int] = [init_state]

    t = 0.0
    state = init_state

    while t < T:
        hold_rate = -generator[state, state]
        if hold_rate <= 0.0:
            # Absorbing state — no further transitions.
            break

        dt = rng.exponential(1.0 / hold_rate)
        t_next = t + dt
        if t_next >= T:
            break

        # Choose next state proportional to off-diagonal rates.
        row = generator[state].copy()
        row[state] = 0.0
        probs = row / hold_rate  # normalised transition probabilities
        next_state = int(rng.choice(n_states, p=probs))

        t = t_next
        state = next_state
        times.append(t)
        states.append(state)

    return np.array(times, dtype=float), np.array(states, dtype=int)


def ctmc_stationary(generator: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution pi satisfying pi @ Psi = 0, sum(pi) = 1.

    Parameters
    ----------
    generator:
        Rate matrix Psi of shape (n_states, n_states).

    Returns
    -------
    pi:
        Stationary distribution of shape (n_states,).
    """
    n = generator.shape[0]
    # System: Psi^T @ pi = 0, 1^T @ pi = 1.
    # Stack into an over-determined least-squares problem for robustness.
    A = np.vstack([generator.T, np.ones((1, n))])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()
    return pi


def ctmc_propagator(generator: np.ndarray, dt: float) -> np.ndarray:
    """Compute the matrix exponential exp(Psi * dt).

    Entry (i, j) is the probability of being in state j at time t+dt
    given state i at time t.

    Parameters
    ----------
    generator:
        Rate matrix Psi of shape (n_states, n_states).
    dt:
        Time increment (>= 0).

    Returns
    -------
    P:
        Transition matrix of shape (n_states, n_states).
    """
    return expm(generator * dt)
