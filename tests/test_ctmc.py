"""Tests for src/model/ctmc.py — CTMC simulation and stationary distribution."""

import numpy as np
import pytest
from src.model.ctmc import simulate_ctmc, ctmc_stationary


# ---------------------------------------------------------------------------
# Paper-default generator
# ---------------------------------------------------------------------------
PSI = np.array([[-0.2, 0.2], [0.2, -0.2]], dtype=float)


class TestSimulateCTMC:
    """Tests for simulate_ctmc."""

    def test_returns_times_and_states(self):
        times, states = simulate_ctmc(PSI, T=20.0, init_state=0, rng=np.random.default_rng(0))
        assert isinstance(times, np.ndarray)
        assert isinstance(states, np.ndarray)
        assert len(times) == len(states)

    def test_first_time_is_zero(self):
        times, _ = simulate_ctmc(PSI, T=10.0, init_state=0, rng=np.random.default_rng(1))
        assert times[0] == pytest.approx(0.0)

    def test_first_state_matches_init(self):
        for init in [0, 1]:
            _, states = simulate_ctmc(PSI, T=10.0, init_state=init, rng=np.random.default_rng(2))
            assert states[0] == init

    def test_all_jump_times_within_horizon(self):
        T = 20.0
        times, _ = simulate_ctmc(PSI, T=T, init_state=0, rng=np.random.default_rng(3))
        # All recorded jump times must be strictly before T
        assert np.all(times < T + 1e-12)

    def test_times_strictly_increasing(self):
        times, _ = simulate_ctmc(PSI, T=50.0, init_state=0, rng=np.random.default_rng(4))
        if len(times) > 1:
            assert np.all(np.diff(times) > 0)

    def test_states_valid_range(self):
        _, states = simulate_ctmc(PSI, T=20.0, init_state=0, rng=np.random.default_rng(5))
        assert np.all(states >= 0)
        assert np.all(states < PSI.shape[0])

    def test_at_least_one_switch_over_long_horizon(self):
        """Expect at least one regime switch over a long horizon with high prob."""
        times, states = simulate_ctmc(PSI, T=200.0, init_state=0, rng=np.random.default_rng(6))
        assert len(times) >= 2, "Expected at least one switch over T=200"

    def test_stationary_frequencies_approx_correct(self):
        """Empirical regime fractions should approach stationary distribution."""
        T = 50000.0
        rng = np.random.default_rng(7)
        times, states = simulate_ctmc(PSI, T=T, init_state=0, rng=rng)
        # Compute time in each state
        all_times = np.append(times, T)
        durations = np.diff(all_times)
        time_in = np.zeros(2)
        for i, s in enumerate(states):
            time_in[s] += durations[i]
        frac = time_in / time_in.sum()
        # Symmetric generator => stationary = [0.5, 0.5]
        assert frac[0] == pytest.approx(0.5, abs=0.03)
        assert frac[1] == pytest.approx(0.5, abs=0.03)

    def test_absorbing_state_no_transitions(self):
        """If off-diagonal rate is 0, state never changes."""
        gen_absorbing = np.array([[0.0, 0.0], [0.0, 0.0]])
        _, states = simulate_ctmc(gen_absorbing, T=100.0, init_state=0,
                                  rng=np.random.default_rng(8))
        assert np.all(states == 0)

    def test_short_horizon_may_have_no_jump(self):
        """For a very short horizon, may see zero jumps (only the initial state)."""
        times, states = simulate_ctmc(PSI, T=1e-6, init_state=1,
                                      rng=np.random.default_rng(9))
        assert len(times) >= 1
        assert states[0] == 1

    def test_single_state_generator(self):
        """Single-state chain: no transitions, constant state 0."""
        gen_one = np.array([[0.0]])
        times, states = simulate_ctmc(gen_one, T=10.0, init_state=0,
                                      rng=np.random.default_rng(10))
        assert np.all(states == 0)


class TestCtmcStationary:
    """Tests for ctmc_stationary."""

    def test_symmetric_generator_gives_uniform(self):
        pi = ctmc_stationary(PSI)
        assert pi == pytest.approx([0.5, 0.5], abs=1e-10)

    def test_stationary_sums_to_one(self):
        pi = ctmc_stationary(PSI)
        assert pi.sum() == pytest.approx(1.0, abs=1e-10)

    def test_stationary_satisfies_balance(self):
        """pi @ Psi ≈ 0."""
        pi = ctmc_stationary(PSI)
        residual = pi @ PSI
        assert np.allclose(residual, 0.0, atol=1e-10)

    def test_stationary_nonnegative(self):
        pi = ctmc_stationary(PSI)
        assert np.all(pi >= 0)

    def test_asymmetric_generator(self):
        """Verify against known solution for a non-symmetric chain."""
        # Psi = [[-1, 1], [3, -3]] => stationary = [3/4, 1/4]
        gen = np.array([[-1.0, 1.0], [3.0, -3.0]])
        pi = ctmc_stationary(gen)
        assert pi == pytest.approx([0.75, 0.25], abs=1e-8)
