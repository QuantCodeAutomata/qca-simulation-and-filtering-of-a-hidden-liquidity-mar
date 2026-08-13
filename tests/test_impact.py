"""Tests for src/model/impact.py — order-book impact and price path."""

import numpy as np
import pytest

from src.model.impact import (
    price_impact,
    transaction_cost,
    transaction_revenue,
    simulate_price_path,
)

# Paper defaults
C = 1.0
E = 3.0
NU = 0.8
RHO = 0.1
C0 = 0.1


class TestPriceImpact:
    def test_zero_trade_zero_impact(self):
        assert price_impact(np.array([0.0]), C, E)[0] == pytest.approx(0.0)

    def test_output_nonnegative(self):
        v = np.linspace(0.0, 10.0, 50)
        assert np.all(price_impact(v, C, E) >= 0)

    def test_monotone_increasing_in_v(self):
        v = np.linspace(0.01, 10.0, 100)
        Q = price_impact(v, C, E)
        assert np.all(np.diff(Q) > 0)

    def test_formula_at_known_point(self):
        """Q(v) = (e * v / c)^{1/e}. With c=1, e=3, v=1: Q = 3^{1/3}."""
        expected = (3.0 * 1.0 / 1.0) ** (1.0 / 3.0)
        result = float(price_impact(np.array([1.0]), C, E)[0])
        assert result == pytest.approx(expected, rel=1e-8)

    def test_shape_preserved(self):
        v = np.ones((3, 4))
        Q = price_impact(v, C, E)
        assert Q.shape == (3, 4)

    def test_smaller_c_larger_impact(self):
        """Smaller c -> larger Q(v) since Q = (e*v/c)^{1/e}."""
        v = np.array([1.0])
        Q_large_c = float(price_impact(v, c=2.0, e=E)[0])
        Q_small_c = float(price_impact(v, c=0.5, e=E)[0])
        assert Q_small_c > Q_large_c


class TestTransactionCost:
    def test_zero_volume_zero_cost(self):
        assert transaction_cost(0.0, C, E) == pytest.approx(0.0, abs=1e-10)

    def test_nonnegative(self):
        for xi in [0.1, 0.5, 1.0, 5.0]:
            assert transaction_cost(xi, C, E) >= 0

    def test_monotone_increasing(self):
        prev = 0.0
        for xi in [0.1, 0.2, 0.5, 1.0, 2.0]:
            tc = transaction_cost(xi, C, E)
            assert tc >= prev
            prev = tc

    def test_formula_at_xi_one(self):
        """TC(xi) = c/(e+1) * Q(xi)^{e+1}. With c=1, e=3, xi=1:
           Q(1) = 3^{1/3}, Q^4 = 3^{4/3}, TC = 3^{4/3}/4.
        """
        Q_1 = (E * 1.0 / C) ** (1.0 / E)
        expected = (C / (E + 1.0)) * Q_1 ** (E + 1.0)
        assert transaction_cost(1.0, C, E) == pytest.approx(expected, rel=1e-8)

    def test_sensitivity_to_c(self):
        """Smaller c -> larger Q -> larger TC (convex cost scaling)."""
        tc_large_c = transaction_cost(1.0, c=2.0, e=E)
        tc_small_c = transaction_cost(1.0, c=0.5, e=E)
        assert tc_small_c > tc_large_c


class TestTransactionRevenue:
    def test_zero_volume_minus_fixed_cost(self):
        """C(p, 0) = p*0 - TC(0) - c0 = -c0."""
        rev = transaction_revenue(10.0, 0.0, C, E, C0)
        assert rev == pytest.approx(-C0, abs=1e-10)

    def test_revenue_increases_with_price(self):
        """Higher price p -> higher revenue for same xi > 0."""
        xi = 1.0
        rev_low = transaction_revenue(1.0, xi, C, E, C0)
        rev_high = transaction_revenue(5.0, xi, C, E, C0)
        assert rev_high > rev_low

    def test_revenue_may_be_negative_small_trade(self):
        """Very small trade vs. fixed cost: revenue can be negative."""
        rev = transaction_revenue(0.0, 1e-6, C, E, C0)
        assert rev < 0  # fixed cost c0 dominates

    def test_revenue_formula(self):
        """C(p, xi) = p*xi - TC(xi) - c0."""
        p, xi = 5.0, 0.5
        expected = p * xi - transaction_cost(xi, C, E) - C0
        assert transaction_revenue(p, xi, C, E, C0) == pytest.approx(expected, rel=1e-8)


class TestSimulatePricePath:
    def _run(self, n_events=10, seed=0, T=10.0):
        rng = np.random.default_rng(seed)
        times = np.sort(rng.uniform(0, T, n_events))
        sides = rng.choice([-1, 1], size=n_events).astype(float)
        marks = rng.exponential(0.5, n_events)
        return simulate_price_path(times, sides, marks, T=T, nu=NU, rho=RHO, c=C, e=E)

    def test_returns_four_arrays(self):
        result = self._run()
        assert len(result) == 4

    def test_t_grid_shape(self):
        t_grid, S, D, P = self._run()
        assert len(t_grid) == len(S) == len(D) == len(P)

    def test_t_grid_starts_at_zero(self):
        t_grid, _, _, _ = self._run()
        assert t_grid[0] == pytest.approx(0.0)

    def test_t_grid_ends_at_T(self):
        T = 15.0
        t_grid, _, _, _ = self._run(T=T)
        assert t_grid[-1] == pytest.approx(T)

    def test_P_equals_S_plus_D(self):
        _, S, D, P = self._run()
        np.testing.assert_allclose(P, S + D, atol=1e-12)

    def test_initial_S_D_are_zero(self):
        _, S, D, _ = self._run()
        assert S[0] == pytest.approx(0.0, abs=1e-12)
        assert D[0] == pytest.approx(0.0, abs=1e-12)

    def test_zero_events_flat_price(self):
        """With no events, price should remain 0 throughout."""
        t_grid, S, D, P = simulate_price_path(
            np.empty(0), np.empty(0), np.empty(0), T=5.0,
            nu=NU, rho=RHO, c=C, e=E,
        )
        np.testing.assert_allclose(S, 0.0, atol=1e-15)
        np.testing.assert_allclose(D, 0.0, atol=1e-15)

    def test_permanent_component_does_not_decay(self):
        """S is permanent and should not decay between events."""
        times = np.array([1.0])
        sides = np.array([1.0])     # buy
        marks = np.array([1.0])
        t_grid, S, D, P = simulate_price_path(
            times, sides, marks, T=10.0, nu=NU, rho=RHO, c=C, e=E, n_grid=100,
        )
        # After the buy event, S should be constant to the end
        jump_idx = np.searchsorted(t_grid, 1.0, side="right")
        assert S[jump_idx] == pytest.approx(S[-1], abs=1e-10)

    def test_transient_component_decays(self):
        """D should decay exponentially after a single event."""
        times = np.array([0.5])
        sides = np.array([1.0])
        marks = np.array([1.0])
        t_grid, _, D, _ = simulate_price_path(
            times, sides, marks, T=20.0, nu=NU, rho=RHO, c=C, e=E, n_grid=200,
        )
        # D should decrease after the jump
        jump_idx = np.searchsorted(t_grid, 0.5, side="right")
        if jump_idx < len(D) - 1:
            assert D[-1] < D[jump_idx]

    def test_nu_zero_no_permanent_impact(self):
        """With nu=0, all impact is transient; S should remain zero."""
        times = np.array([1.0, 2.0])
        sides = np.array([1.0, -1.0])
        marks = np.array([0.5, 0.5])
        _, S, _, _ = simulate_price_path(
            times, sides, marks, T=5.0, nu=0.0, rho=RHO, c=C, e=E,
        )
        np.testing.assert_allclose(S, 0.0, atol=1e-12)

    def test_nu_one_no_transient_impact(self):
        """With nu=1, all impact is permanent; D should remain zero."""
        times = np.array([1.0])
        sides = np.array([1.0])
        marks = np.array([1.0])
        _, _, D, _ = simulate_price_path(
            times, sides, marks, T=5.0, nu=1.0, rho=RHO, c=C, e=E,
        )
        np.testing.assert_allclose(D, 0.0, atol=1e-12)
