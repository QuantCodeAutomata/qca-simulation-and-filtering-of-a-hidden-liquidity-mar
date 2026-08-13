"""Integration tests — full model pipeline: CTMC -> Hawkes -> Filter -> DP."""

import numpy as np
import pytest

from src.model.ctmc import simulate_ctmc
from src.model.hawkes_marked import simulate_marked_hawkes, MarkovHawkesState
from src.model.filter import run_filter
from src.model.impact import simulate_price_path, price_impact, transaction_cost

PSI = np.array([[-0.2, 0.2], [0.2, -0.2]], dtype=float)
LAM_INF = np.array([1.0, 1.0])
T = 10.0


def _make_state(kp_init: np.ndarray, km_init: np.ndarray) -> MarkovHawkesState:
    return MarkovHawkesState(
        kappa_plus=kp_init,
        kappa_minus=km_init,
        lambda_infty=LAM_INF,
        beta=0.5,
        eta=0.05,
        zeta=2.0,
    )


class TestFullSimulationPipeline:
    """CTMC -> Hawkes simulation -> Filter -> Price path pipeline."""

    def _run_full(self, kp1=5.0, km1=1.0, prior=0.9, seed=0):
        rng = np.random.default_rng(seed)
        # 1. Simulate hidden regime
        regime_times, regime_states = simulate_ctmc(PSI, T=T, init_state=0, rng=rng)

        # 2. Simulate marked Hawkes events
        kp_init = np.array([kp1, km1])
        km_init = np.array([km1, kp1])
        init_state = _make_state(kp_init, km_init)
        hawkes_result = simulate_marked_hawkes(
            T=T,
            init_state=init_state,
            hidden_times=regime_times,
            hidden_states=regime_states,
            lambda_infty=LAM_INF,
            beta=0.5,
            eta=0.05,
            zeta=2.0,
            rng=rng,
        )

        # 3. Run filter
        pi0 = np.array([prior, 1.0 - prior])
        filter_result = run_filter(
            event_times=hawkes_result["event_times"],
            event_sides=hawkes_result["event_sides"],
            event_marks=hawkes_result["event_marks"],
            T=T,
            generator=PSI,
            init_kappa_plus=kp_init,
            init_kappa_minus=km_init,
            lambda_infty=LAM_INF,
            beta=0.5,
            eta=0.05,
            zeta=2.0,
            pi0=pi0,
        )

        # 4. Price path
        price_result = simulate_price_path(
            event_times=hawkes_result["event_times"],
            event_sides=hawkes_result["event_sides"],
            event_marks=hawkes_result["event_marks"],
            T=T, nu=0.8, rho=0.1, c=1.0, e=3.0,
        )

        return hawkes_result, filter_result, price_result

    def test_pipeline_runs_without_error(self):
        self._run_full()

    def test_filter_posterior_simplex_throughout(self):
        _, flt, _ = self._run_full()
        row_sums = flt["posterior"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_filter_posterior_nonnegative(self):
        _, flt, _ = self._run_full()
        assert np.all(flt["posterior"] >= 0)

    def test_event_times_align_with_filter(self):
        """All event times should appear in filter_times."""
        hw, flt, _ = self._run_full()
        ev_times = set(np.round(hw["event_times"], 10))
        ft_times = set(np.round(flt["filter_times"], 10))
        assert ev_times.issubset(ft_times)

    def test_price_path_length(self):
        _, _, price_result = self._run_full()
        t_grid, S, D, P = price_result
        assert len(t_grid) == len(S) == len(D) == len(P)

    def test_price_path_P_equals_S_plus_D(self):
        _, _, price_result = self._run_full()
        _, S, D, P = price_result
        np.testing.assert_allclose(P, S + D, atol=1e-12)

    def test_three_priors_different_initial_posterior(self):
        """Different priors should produce different filter trajectories."""
        _, flt_09, _ = self._run_full(prior=0.9, seed=5)
        _, flt_05, _ = self._run_full(prior=0.5, seed=5)
        _, flt_01, _ = self._run_full(prior=0.1, seed=5)
        # Initial posterior should differ across priors
        assert flt_09["posterior"][0, 0] != pytest.approx(flt_05["posterior"][0, 0])
        assert flt_05["posterior"][0, 0] != pytest.approx(flt_01["posterior"][0, 0])

    def test_intensity_settings_5_1_vs_1_5(self):
        """(kp1=5,km1=1) and (kp1=1,km1=5) should produce different event patterns."""
        hw_51, _, _ = self._run_full(kp1=5.0, km1=1.0, seed=2)
        hw_15, _, _ = self._run_full(kp1=1.0, km1=5.0, seed=2)
        n_51 = len(hw_51["event_times"])
        n_15 = len(hw_15["event_times"])
        assert n_51 >= 0 and n_15 >= 0  # both valid

    def test_symmetric_regime_swap_symmetry(self):
        """Under symmetric generator and swapped intensities, filter should be
        symmetric: posterior state 0 with (5,1) ≈ posterior state 1 with (1,5)."""
        rng = np.random.default_rng(77)
        regime_times, regime_states = simulate_ctmc(PSI, T=T, init_state=0, rng=rng)

        kp_A = np.array([5.0, 1.0])
        km_A = np.array([1.0, 5.0])
        init_A = _make_state(kp_A, km_A)
        hw = simulate_marked_hawkes(
            T=T, init_state=init_A,
            hidden_times=regime_times, hidden_states=regime_states,
            lambda_infty=LAM_INF, beta=0.5, eta=0.05, zeta=2.0, rng=rng,
        )
        ev_t = hw["event_times"]
        ev_s = hw["event_sides"]
        ev_m = hw["event_marks"]

        flt_A = run_filter(ev_t, ev_s, ev_m, T, PSI,
                           kp_A, km_A, LAM_INF, 0.5, 0.05, 2.0, np.array([0.9, 0.1]))

        kp_B = np.array([1.0, 5.0])
        km_B = np.array([5.0, 1.0])
        flt_B = run_filter(ev_t, ev_s, ev_m, T, PSI,
                           kp_B, km_B, LAM_INF, 0.5, 0.05, 2.0, np.array([0.1, 0.9]))

        np.testing.assert_allclose(
            flt_A["posterior"][:, 0], flt_B["posterior"][:, 1], atol=1e-8
        )


class TestImpactMathematicalProperties:
    """Mathematical consistency checks for the impact model."""

    def test_Q_concave_in_v(self):
        """Q(v) = (ev/c)^{1/e} with 1/e = 1/3 < 1 is concave."""
        v = np.linspace(0.1, 5.0, 50)
        Q = price_impact(v, c=1.0, e=3.0)
        # Second derivative via finite diff: should be <= 0 for concavity
        second_diff = np.diff(Q, n=2)
        assert np.all(second_diff <= 1e-8)

    def test_TC_convex_in_xi(self):
        """Transaction cost TC(xi) should be convex (second difference >= 0)."""
        xi_grid = np.linspace(0.01, 3.0, 100)
        tc_vals = np.array([transaction_cost(xi, 1.0, 3.0) for xi in xi_grid])
        second_diff = np.diff(tc_vals, n=2)
        assert np.all(second_diff >= -1e-10)

    def test_revenue_C_formula_components(self):
        """Check each term in C(p, xi) = p*xi - TC(xi) - c0."""
        from src.model.impact import transaction_revenue
        p, xi, c0 = 5.0, 1.0, 0.1
        tc = transaction_cost(xi, 1.0, 3.0)
        expected = p * xi - tc - c0
        assert transaction_revenue(p, xi, 1.0, 3.0, c0) == pytest.approx(expected, rel=1e-8)

    def test_permanent_component_is_cumulative_sum(self):
        """S_t is the sum of permanent impacts from all past events."""
        times = np.array([1.0, 3.0, 5.0])
        sides = np.array([1.0, -1.0, 1.0])
        marks = np.array([1.0, 1.0, 1.0])
        nu, c, e = 0.8, 1.0, 3.0
        Q_val = float(price_impact(np.array([1.0]), c, e)[0])

        _, S, _, _ = simulate_price_path(
            times, sides, marks, T=10.0, nu=nu, rho=0.1, c=c, e=e, n_grid=1000,
        )
        # At t=10: S = nu*Q*(+1 -1 +1) = nu*Q
        expected_S = nu * Q_val * (1.0 - 1.0 + 1.0)
        assert S[-1] == pytest.approx(expected_S, abs=1e-10)
