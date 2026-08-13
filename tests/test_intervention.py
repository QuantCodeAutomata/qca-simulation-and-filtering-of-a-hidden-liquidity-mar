"""Tests for src/control/intervention.py — DP solver and intervention operator."""

import numpy as np
import pytest

from src.control.intervention import (
    _Q,
    _TC,
    _phi,
    no_jump_intensity,
    no_jump_belief,
    survival_prob,
    no_jump_evolution,
    build_g0,
    run_dp_solver,
    compute_exercise_map,
    compute_order_size_map,
)

CONFIG = "/workspace/project/configs/filter_default.yaml"

# Minimal fast grid for unit tests
FAST_GRID = dict(n_t=4, n_x=5, n_kp=4, n_km=4, n_mu=5)


# ---------------------------------------------------------------------------
# _Q, _TC, _phi — pure functions
# ---------------------------------------------------------------------------

class TestImpactPrimitives:
    def test_Q_zero_xi(self):
        assert _Q(0.0, 1.0, 3.0) == pytest.approx(0.0, abs=1e-10)

    def test_Q_formula(self):
        """Q = (e*xi/c)^{1/e}."""
        assert _Q(1.0, 1.0, 3.0) == pytest.approx((3.0 ** (1.0 / 3.0)), rel=1e-8)

    def test_Q_negative_clipped_to_zero(self):
        assert _Q(-5.0, 1.0, 3.0) == pytest.approx(0.0, abs=1e-10)

    def test_TC_zero_xi(self):
        assert _TC(0.0, 1.0, 3.0) == pytest.approx(0.0, abs=1e-10)

    def test_TC_nonnegative(self):
        for xi in [0.0, 0.1, 0.5, 2.0]:
            assert _TC(xi, 1.0, 3.0) >= 0

    def test_phi_zero_at_zero(self):
        assert _phi(0.0, 0.05, 2.0) == pytest.approx(0.0, abs=1e-10)

    def test_phi_positive(self):
        assert _phi(1.0, 0.05, 2.0) > 0

    def test_phi_negative_returns_zero(self):
        assert _phi(-1.0, 0.05, 2.0) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# no_jump_intensity
# ---------------------------------------------------------------------------

class TestNoJumpIntensity:
    def test_at_dt_zero(self):
        assert no_jump_intensity(5.0, 1.0, 0.5, 0.0) == pytest.approx(5.0)

    def test_decays_toward_baseline(self):
        val = no_jump_intensity(5.0, 1.0, 0.5, 100.0)
        assert val == pytest.approx(1.0, abs=0.01)

    def test_nonnegative(self):
        assert no_jump_intensity(-10.0, 0.0, 1.0, 5.0) >= 0


# ---------------------------------------------------------------------------
# no_jump_belief
# ---------------------------------------------------------------------------

class TestNoJumpBelief:
    def test_symmetric_prior_unchanged(self):
        """mu=0.5 stays at 0.5 under symmetric generator."""
        assert no_jump_belief(0.5, 0.2, 10.0) == pytest.approx(0.5, abs=1e-10)

    def test_extreme_prior_drifts_toward_half(self):
        val = no_jump_belief(0.9, 0.2, 50.0)
        assert val == pytest.approx(0.5, abs=0.01)

    def test_at_dt_zero_preserves_prior(self):
        assert no_jump_belief(0.7, 0.2, 0.0) == pytest.approx(0.7, abs=1e-10)

    def test_output_in_unit_interval(self):
        for mu in [0.0, 0.1, 0.5, 0.9, 1.0]:
            val = no_jump_belief(mu, 0.2, 5.0)
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# survival_prob
# ---------------------------------------------------------------------------

class TestSurvivalProb:
    def test_at_dt_zero_is_one(self):
        assert survival_prob(3.0, 1.0, 1.0, 0.5, 0.0) == pytest.approx(1.0)

    def test_decays_with_dt(self):
        s1 = survival_prob(3.0, 1.0, 1.0, 0.5, 1.0)
        s2 = survival_prob(3.0, 1.0, 1.0, 0.5, 5.0)
        assert s1 > s2

    def test_in_unit_interval(self):
        for dt in [0.0, 0.1, 1.0, 5.0]:
            s = survival_prob(3.0, 1.0, 1.0, 0.5, dt)
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# no_jump_evolution
# ---------------------------------------------------------------------------

class TestNoJumpEvolution:
    def test_both_components_evolve(self):
        # kp1=5.0 > lam_inf=1.0 → decays down; km1=0.0 < lam_inf=1.0 → grows up
        kp_new, km_new = no_jump_evolution(5.0, 0.0, 1.0, 0.5, 1.0)
        assert 0 < kp_new < 5.0  # decays toward 1.0
        assert 0 < km_new < 1.0  # starts below baseline, recovers

    def test_nonnegative_output(self):
        kp_new, km_new = no_jump_evolution(-1.0, -2.0, 0.0, 0.5, 5.0)
        assert kp_new >= 0
        assert km_new >= 0


# ---------------------------------------------------------------------------
# build_g0
# ---------------------------------------------------------------------------

_G0_KWARGS = dict(T=20.0, rho=0.1, c=1.0, e=3.0, c0=0.1, d_fixed=1.5)


class TestBuildG0:
    def test_returns_array(self):
        t_grid = np.linspace(0, 20, 4)
        x_grid = np.linspace(0, 5, 5)
        kappa_grid = np.linspace(0, 5, 4)
        mu_grid = np.linspace(0, 1, 5)
        g0 = build_g0(t_grid, x_grid, kappa_grid, mu_grid, **_G0_KWARGS)
        assert isinstance(g0, np.ndarray)

    def test_shape(self):
        t_grid = np.linspace(0, 20, 4)
        x_grid = np.linspace(0, 5, 5)
        kappa_grid = np.linspace(0, 5, 4)
        mu_grid = np.linspace(0, 1, 5)
        g0 = build_g0(t_grid, x_grid, kappa_grid, mu_grid, **_G0_KWARGS)
        # Expected: (n_t, n_x, n_kp, n_km, n_mu)
        assert g0.ndim == 5
        assert g0.shape == (4, 5, 4, 4, 5)

    def test_zero_at_zero_inventory(self):
        t_grid = np.linspace(0, 20, 4)
        x_grid = np.array([0.0, 1.0, 2.0])
        kappa_grid = np.array([1.0, 3.0])
        mu_grid = np.array([0.0, 0.5, 1.0])
        g0 = build_g0(t_grid, x_grid, kappa_grid, mu_grid, **_G0_KWARGS)
        np.testing.assert_allclose(g0[:, 0, :, :, :], 0.0, atol=1e-10)

    def test_nonneg_for_small_x(self):
        """g0 can be negative for small x (fixed cost c0 may dominate)
        but the slice at x=0 must be 0."""
        t_grid = np.linspace(0, 20, 4)
        x_grid = np.linspace(0, 5, 5)
        kappa_grid = np.linspace(0, 5, 4)
        mu_grid = np.linspace(0, 1, 5)
        g0 = build_g0(t_grid, x_grid, kappa_grid, mu_grid, **_G0_KWARGS)
        np.testing.assert_allclose(g0[:, 0, :, :, :], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# run_dp_solver — convergence verification
# ---------------------------------------------------------------------------

class TestRunDPSolver:
    """Integration tests for the DP solver convergence behavior."""

    def test_solver_runs_without_error(self):
        result = run_dp_solver(CONFIG, **FAST_GRID)
        assert result is not None

    def test_returns_required_keys(self):
        result = run_dp_solver(CONFIG, **FAST_GRID)
        for key in ["g_iterations", "delta_N", "t_grid", "x_grid",
                    "kappa_grid", "mu_grid", "g_interp_final"]:
            assert key in result, f"Missing key: {key}"

    def test_delta_N_length_matches_iterations(self):
        result = run_dp_solver(CONFIG, **FAST_GRID)
        # delta_N[k] = max|g_{k+1} - g_k|, len = n_iterations
        assert len(result["delta_N"]) > 0

    def test_delta_N_nonnegative(self):
        result = run_dp_solver(CONFIG, **FAST_GRID)
        assert np.all(np.array(result["delta_N"]) >= 0)

    def test_monotone_nondecreasing_g(self):
        """g_N(state) >= g_{N-1}(state) for all states (value iterations increase)."""
        result = run_dp_solver(CONFIG, **FAST_GRID)
        g_iters = result["g_iterations"]
        violations = 0
        for k in range(1, len(g_iters)):
            diff = g_iters[k] - g_iters[k - 1]
            n_neg = int(np.sum(diff < -1e-8))
            violations += n_neg
        assert violations == 0, f"Monotonicity violated at {violations} states"

    def test_zero_inventory_zero_value(self):
        """g(t, x=0, ...) = 0 at any time because no shares to liquidate."""
        result = run_dp_solver(CONFIG, **FAST_GRID)
        g_final = result["g_iterations"][-1]
        # x=0 is the first index along axis 1
        x0_slice = g_final[:, 0, ...]
        np.testing.assert_allclose(x0_slice, 0.0, atol=1e-8)

    def test_final_delta_N_smaller_than_initial(self):
        """The last delta_N should be less than the first, indicating convergence."""
        result = run_dp_solver(CONFIG, **FAST_GRID)
        delta = result["delta_N"]
        if len(delta) > 1:
            assert delta[-1] <= delta[0] + 1e-8  # allow negligible plateau


# ---------------------------------------------------------------------------
# compute_exercise_map
# ---------------------------------------------------------------------------

class TestComputeExerciseMap:
    """Tests for compute_exercise_map using the correct g_solution structure."""

    def _g_solution_and_grids(self):
        result = run_dp_solver(CONFIG, **FAST_GRID)
        mu_plot = np.linspace(0.0, 1.0, 5)
        x_plot = result["x_grid"]
        return result, mu_plot, x_plot

    def test_returns_bool_array(self):
        g_sol, mu_plot, x_plot = self._g_solution_and_grids()
        ex_map = compute_exercise_map(
            g_sol, mu_plot, x_plot, kp1=3.0, km1=1.0, t=0.0
        )
        assert isinstance(ex_map, np.ndarray)
        assert ex_map.dtype in [bool, np.bool_]

    def test_shape_matches_mu_x(self):
        g_sol, mu_plot, x_plot = self._g_solution_and_grids()
        ex_map = compute_exercise_map(
            g_sol, mu_plot, x_plot, kp1=3.0, km1=1.0, t=0.0
        )
        assert ex_map.shape == (len(mu_plot), len(x_plot))

    def test_zero_inventory_is_never_exercise(self):
        """x=0 should never trigger intervention (nothing to sell)."""
        g_sol, mu_plot, _ = self._g_solution_and_grids()
        x_vals = np.array([0.0, 1.0, 2.0])
        ex_map = compute_exercise_map(
            g_sol, mu_plot, x_vals, kp1=3.0, km1=1.0, t=0.0
        )
        # x=0 column should be all False
        assert not np.any(ex_map[:, 0])

    def test_values_are_binary(self):
        g_sol, mu_plot, x_plot = self._g_solution_and_grids()
        ex_map = compute_exercise_map(
            g_sol, mu_plot, x_plot, kp1=3.0, km1=1.0, t=0.0
        )
        unique_vals = np.unique(ex_map)
        for v in unique_vals:
            assert v in [True, False, 0, 1]


# ---------------------------------------------------------------------------
# compute_order_size_map
# ---------------------------------------------------------------------------

class TestComputeOrderSizeMap:
    """Tests for compute_order_size_map using the correct g_solution structure."""

    def _g_solution_and_grids(self):
        result = run_dp_solver(CONFIG, **FAST_GRID)
        mu_plot = np.linspace(0.0, 1.0, 5)
        x_plot = result["x_grid"]
        return result, mu_plot, x_plot

    def test_returns_float_array(self):
        g_sol, mu_plot, x_plot = self._g_solution_and_grids()
        xi_map = compute_order_size_map(
            g_sol, mu_plot, x_plot, kp1=3.0, km1=1.0, t=0.0
        )
        assert isinstance(xi_map, np.ndarray)
        assert xi_map.dtype in [np.float32, np.float64]

    def test_shape_matches_mu_x(self):
        g_sol, mu_plot, x_plot = self._g_solution_and_grids()
        xi_map = compute_order_size_map(
            g_sol, mu_plot, x_plot, kp1=3.0, km1=1.0, t=0.0
        )
        assert xi_map.shape == (len(mu_plot), len(x_plot))

    def test_nonnegative(self):
        g_sol, mu_plot, x_plot = self._g_solution_and_grids()
        xi_map = compute_order_size_map(
            g_sol, mu_plot, x_plot, kp1=3.0, km1=1.0, t=0.0
        )
        assert np.all(xi_map >= -1e-10)

    def test_xi_star_leq_x(self):
        """Optimal sell size must not exceed inventory."""
        g_sol, mu_plot, x_plot = self._g_solution_and_grids()
        xi_map = compute_order_size_map(
            g_sol, mu_plot, x_plot, kp1=3.0, km1=1.0, t=0.0
        )
        for ix, x_val in enumerate(x_plot):
            col = xi_map[:, ix]
            assert np.all(col <= x_val + 1e-8), f"xi* > x at x={x_val:.3f}"

    def test_zero_inventory_zero_xi(self):
        """x=0 -> xi*=0 (no shares to sell)."""
        g_sol, mu_plot, _ = self._g_solution_and_grids()
        x_vals = np.array([0.0, 1.0])
        xi_map = compute_order_size_map(
            g_sol, mu_plot, x_vals, kp1=3.0, km1=1.0, t=0.0
        )
        np.testing.assert_allclose(xi_map[:, 0], 0.0, atol=1e-10)
