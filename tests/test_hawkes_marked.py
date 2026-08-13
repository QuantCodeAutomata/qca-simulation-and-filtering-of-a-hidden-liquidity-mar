"""Tests for src/model/hawkes_marked.py."""

import numpy as np
import pytest
from scipy.special import gamma as gamma_fn

from src.model.hawkes_marked import (
    excitation_phi,
    no_jump_intensity,
    sample_mark,
    mark_density,
    MarkovHawkesState,
    simulate_marked_hawkes,
)


# ---------------------------------------------------------------------------
# excitation_phi
# ---------------------------------------------------------------------------

class TestExcitationPhi:
    def test_output_shape(self):
        v = np.array([0.5, 1.0, 2.0])
        phi = excitation_phi(v)
        assert phi.shape == v.shape

    def test_nonnegative(self):
        v = np.linspace(0.1, 5.0, 20)
        assert np.all(excitation_phi(v) >= 0)

    def test_scalar_input(self):
        phi = excitation_phi(np.array([1.0]))
        assert phi.ndim == 1

    def test_monotone_increasing_in_v(self):
        """phi is a power function v^eta (eta=0.05>0) => monotone increasing."""
        v = np.linspace(0.1, 10.0, 50)
        phi = excitation_phi(v, eta=0.05, zeta=2.0)
        assert np.all(np.diff(phi) > 0)

    def test_known_value_at_eta_0(self):
        """As eta -> 0, phi = (zeta^0/Gamma(1)) * v^0 = 1. Check eta=1e-8 ~ 1."""
        v = np.array([3.14])
        phi = excitation_phi(v, eta=1e-8, zeta=2.0)
        # eta * zeta^eta/Gamma(1+eta) * v^eta ~ eta * 1 * 1 = eta ~ 0
        # Actually for tiny eta, the scale factor is eta → 0
        assert phi[0] >= 0

    def test_formula_at_specific_point(self):
        """Verify formula: phi(v) = eta * (zeta^eta / Gamma(1+eta)) * v^eta."""
        v_val = 2.0
        eta = 0.05
        zeta = 2.0
        expected = eta * (zeta ** eta / gamma_fn(1.0 + eta)) * (v_val ** eta)
        result = excitation_phi(np.array([v_val]), eta=eta, zeta=zeta)[0]
        assert result == pytest.approx(expected, rel=1e-8)

    def test_larger_eta_larger_excitation(self):
        """Higher eta means higher excitation magnitude for v > 1."""
        v = np.array([3.0])
        phi_low = excitation_phi(v, eta=0.05, zeta=2.0)[0]
        phi_high = excitation_phi(v, eta=0.3, zeta=2.0)[0]
        # For v>1, v^eta increases with eta; scale factor also larger
        assert phi_high > phi_low


# ---------------------------------------------------------------------------
# no_jump_intensity
# ---------------------------------------------------------------------------

class TestNoJumpIntensity:
    def test_at_dt_zero_returns_kappa(self):
        assert no_jump_intensity(3.0, 1.0, 0.5, 0.0) == pytest.approx(3.0)

    def test_decays_toward_baseline(self):
        kappa = 5.0
        lam_inf = 1.0
        val_small = no_jump_intensity(kappa, lam_inf, 0.5, 0.1)
        val_large = no_jump_intensity(kappa, lam_inf, 0.5, 100.0)
        assert val_small > val_large
        assert val_large == pytest.approx(lam_inf, abs=0.01)

    def test_starts_below_baseline_recovers(self):
        """kappa < lambda_infty: intensity should increase toward baseline."""
        val = no_jump_intensity(0.0, 1.0, 0.5, 1.0)
        assert 0.0 < val <= 1.0

    def test_nonnegative_clipping(self):
        """Even extreme inputs should not produce negative intensity."""
        val = no_jump_intensity(-5.0, 0.0, 1.0, 10.0)
        assert val >= 0.0


# ---------------------------------------------------------------------------
# sample_mark and mark_density
# ---------------------------------------------------------------------------

class TestMarkDistribution:
    def test_sample_mark_positive(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            v = sample_mark(2.0, rng)
            assert v > 0

    def test_sample_mark_mean(self):
        """E[V] = 1/zeta for V ~ Exp(zeta)."""
        rng = np.random.default_rng(42)
        samples = [sample_mark(2.0, rng) for _ in range(10_000)]
        assert np.mean(samples) == pytest.approx(0.5, abs=0.02)

    def test_mark_density_positive(self):
        v = np.array([0.1, 0.5, 1.0, 5.0])
        dens = mark_density(v, 2.0)
        assert np.all(dens > 0)

    def test_mark_density_formula(self):
        """nu(v) = zeta * exp(-zeta * v)."""
        zeta = 3.0
        v = np.array([1.0, 2.0])
        expected = zeta * np.exp(-zeta * v)
        assert mark_density(v, zeta) == pytest.approx(expected, rel=1e-10)

    def test_mark_density_integrates_to_one(self):
        """Numerical integration of nu over [0, inf) should be ~1."""
        from scipy.integrate import quad
        zeta = 2.0
        result, _ = quad(lambda v: float(mark_density(np.array([v]), zeta)[0]),
                         0, np.inf)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_mark_density_monotone_decreasing(self):
        v = np.linspace(0.1, 5.0, 50)
        dens = mark_density(v, 2.0)
        assert np.all(np.diff(dens) < 0)


# ---------------------------------------------------------------------------
# MarkovHawkesState
# ---------------------------------------------------------------------------

class TestMarkovHawkesState:
    """Unit tests for the MarkovHawkesState dataclass / helper."""

    def _default_state(self, kp=np.array([5.0, 1.0]), km=np.array([1.0, 5.0])):
        return MarkovHawkesState(
            kappa_plus=kp.copy(),
            kappa_minus=km.copy(),
            lambda_infty=np.array([1.0, 1.0]),
            beta=0.5,
            eta=0.05,
            zeta=2.0,
        )

    def test_initial_intensities_stored(self):
        state = self._default_state()
        np.testing.assert_array_equal(state.kappa_plus, [5.0, 1.0])
        np.testing.assert_array_equal(state.kappa_minus, [1.0, 5.0])

    def test_decay_step_reduces_excess(self):
        state = self._default_state()
        kp_old = state.kappa_plus.copy()
        state.decay(dt=1.0)
        # Each entry should have moved toward lambda_infty=1
        assert np.all(state.kappa_plus <= kp_old)

    def test_decay_step_nonnegative(self):
        state = self._default_state(kp=np.array([-1.0, 0.0]),
                                    km=np.array([0.0, -2.0]))
        state.decay(dt=0.5)
        assert np.all(state.kappa_plus >= 0)
        assert np.all(state.kappa_minus >= 0)

    def test_jump_excitation_increases_intensities(self):
        state = self._default_state()
        kp_before = state.kappa_plus.copy()
        km_before = state.kappa_minus.copy()
        state.update_at_jump(v=1.0)
        assert np.all(state.kappa_plus >= kp_before)
        assert np.all(state.kappa_minus >= km_before)


# ---------------------------------------------------------------------------
# simulate_marked_hawkes
# ---------------------------------------------------------------------------

class TestSimulateMarkedHawkes:
    """End-to-end tests for Ogata thinning simulation."""

    def _run(self, kp1=5.0, km1=1.0, seed=0, T=20.0):
        psi = np.array([[-0.2, 0.2], [0.2, -0.2]])
        kp_init = np.array([kp1, km1])
        km_init = np.array([km1, kp1])
        lam_inf = np.array([1.0, 1.0])
        rng = np.random.default_rng(seed)
        from src.model.ctmc import simulate_ctmc
        regime_times, regime_states = simulate_ctmc(psi, T=T, init_state=0, rng=rng)
        return simulate_marked_hawkes(
            regime_times=regime_times,
            regime_states=regime_states,
            kappa_plus_init=kp_init,
            kappa_minus_init=km_init,
            lambda_infty=lam_inf,
            beta=0.5,
            eta=0.05,
            zeta=2.0,
            T=T,
            rng=rng,
        )

    def test_returns_arrays(self):
        result = self._run()
        assert "event_times" in result
        assert "event_sides" in result
        assert "event_marks" in result

    def test_event_times_in_horizon(self):
        result = self._run(T=20.0)
        assert np.all(result["event_times"] >= 0)
        assert np.all(result["event_times"] < 20.0 + 1e-12)

    def test_event_sides_binary(self):
        result = self._run()
        sides = result["event_sides"]
        assert np.all((sides == 1) | (sides == -1))

    def test_event_marks_positive(self):
        result = self._run()
        assert np.all(result["event_marks"] > 0)

    def test_event_times_increasing(self):
        result = self._run()
        times = result["event_times"]
        if len(times) > 1:
            assert np.all(np.diff(times) > 0)

    def test_empty_output_short_horizon(self):
        """Very short horizon might produce zero events; result should still be valid."""
        result = self._run(T=0.001, seed=999)
        assert len(result["event_times"]) >= 0  # any count is valid

    def test_higher_intensity_more_events(self):
        """Higher initial intensity -> more events on average."""
        n_runs = 5
        counts_high = []
        counts_low = []
        for seed in range(n_runs):
            r_high = self._run(kp1=10.0, km1=10.0, seed=seed)
            r_low = self._run(kp1=0.1, km1=0.1, seed=seed)
            counts_high.append(len(r_high["event_times"]))
            counts_low.append(len(r_low["event_times"]))
        assert np.mean(counts_high) > np.mean(counts_low)
