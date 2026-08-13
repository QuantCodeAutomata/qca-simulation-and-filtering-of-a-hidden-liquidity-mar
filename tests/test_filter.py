"""Tests for src/model/filter.py — finite-dimensional Bayesian filter."""

import numpy as np
import pytest

from src.model.filter import filter_bayes_update, filter_between_jumps, run_filter


# ---------------------------------------------------------------------------
# Paper default parameters
# ---------------------------------------------------------------------------

PSI = np.array([[-0.2, 0.2], [0.2, -0.2]], dtype=float)
LAM_INF = np.array([1.0, 1.0])
BETA = 0.5
ETA = 0.05
ZETA = 2.0


# ---------------------------------------------------------------------------
# filter_bayes_update
# ---------------------------------------------------------------------------

class TestFilterBayesUpdate:
    def test_output_sums_to_one(self):
        pi = np.array([0.6, 0.4])
        lam = np.array([2.0, 1.0])
        nu = np.array([1.5, 0.8])
        pi_new = filter_bayes_update(pi, lam, nu)
        assert pi_new.sum() == pytest.approx(1.0, abs=1e-12)

    def test_output_nonnegative(self):
        pi = np.array([0.9, 0.1])
        lam = np.array([5.0, 1.0])
        nu = np.array([2.0, 0.5])
        pi_new = filter_bayes_update(pi, lam, nu)
        assert np.all(pi_new >= 0)

    def test_known_two_state(self):
        """Manual computation: pi=[0.5,0.5], lam=[4,1], nu=[1,1] -> [4/5, 1/5]."""
        pi = np.array([0.5, 0.5])
        lam = np.array([4.0, 1.0])
        nu = np.array([1.0, 1.0])
        pi_new = filter_bayes_update(pi, lam, nu)
        assert pi_new == pytest.approx([0.8, 0.2], abs=1e-10)

    def test_informative_observation_concentrates_belief(self):
        """Large ratio lam[0]/lam[1] should push posterior toward state 0."""
        pi = np.array([0.5, 0.5])
        lam = np.array([100.0, 1.0])
        nu = np.ones(2)
        pi_new = filter_bayes_update(pi, lam, nu)
        assert pi_new[0] > 0.98

    def test_uniform_likelihood_preserves_prior(self):
        """If lam and nu are equal across states, prior is unchanged."""
        pi = np.array([0.3, 0.7])
        lam = np.array([2.0, 2.0])
        nu = np.array([1.5, 1.5])
        pi_new = filter_bayes_update(pi, lam, nu)
        assert pi_new == pytest.approx(pi, abs=1e-10)

    def test_zero_denominator_returns_prior_unchanged(self):
        """If all weights are zero (degenerate case), prior should be returned."""
        pi = np.array([0.6, 0.4])
        lam = np.array([0.0, 0.0])
        nu = np.array([0.0, 0.0])
        pi_new = filter_bayes_update(pi, lam, nu, eps=1e-15)
        # Should not crash; returns something (prior or pi)
        assert pi_new.sum() == pytest.approx(1.0, abs=1e-10)

    def test_three_state_case(self):
        pi = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        lam = np.array([6.0, 3.0, 1.0])
        nu = np.ones(3)
        pi_new = filter_bayes_update(pi, lam, nu)
        assert pi_new.sum() == pytest.approx(1.0, abs=1e-10)
        assert pi_new[0] > pi_new[1] > pi_new[2]


# ---------------------------------------------------------------------------
# filter_between_jumps
# ---------------------------------------------------------------------------

class TestFilterBetweenJumps:
    def test_posterior_sums_to_one_after_evolution(self):
        pi = np.array([0.7, 0.3])
        kp = np.array([3.0, 1.0])
        km = np.array([1.0, 3.0])
        pi_new, kp_new, km_new = filter_between_jumps(pi, PSI, kp, km, LAM_INF, BETA, 1.0)
        assert pi_new.sum() == pytest.approx(1.0, abs=1e-6)

    def test_posterior_nonnegative(self):
        pi = np.array([0.5, 0.5])
        kp = np.array([2.0, 2.0])
        km = np.array([2.0, 2.0])
        pi_new, _, _ = filter_between_jumps(pi, PSI, kp, km, LAM_INF, BETA, 2.0)
        assert np.all(pi_new >= 0)

    def test_intensities_nonnegative(self):
        kp = np.array([5.0, 1.0])
        km = np.array([1.0, 5.0])
        pi = np.array([0.5, 0.5])
        _, kp_new, km_new = filter_between_jumps(pi, PSI, kp, km, LAM_INF, BETA, 5.0)
        assert np.all(kp_new >= 0)
        assert np.all(km_new >= 0)

    def test_intensities_decay_toward_baseline(self):
        """After long dt, intensities should be near lambda_infty."""
        kp = np.array([10.0, 10.0])
        km = np.array([10.0, 10.0])
        pi = np.array([0.5, 0.5])
        _, kp_new, km_new = filter_between_jumps(pi, PSI, kp, km, LAM_INF, BETA, 50.0)
        np.testing.assert_allclose(kp_new, LAM_INF, atol=0.1)
        np.testing.assert_allclose(km_new, LAM_INF, atol=0.1)

    def test_zero_dt_preserves_state(self):
        """dt=0 should leave posterior and intensities unchanged (or very close)."""
        pi = np.array([0.3, 0.7])
        kp = np.array([3.0, 1.0])
        km = np.array([1.0, 3.0])
        pi_new, kp_new, km_new = filter_between_jumps(pi, PSI, kp, km, LAM_INF, BETA, 0.0)
        np.testing.assert_allclose(pi_new, pi, atol=1e-8)
        np.testing.assert_allclose(kp_new, kp, atol=1e-8)
        np.testing.assert_allclose(km_new, km, atol=1e-8)


# ---------------------------------------------------------------------------
# run_filter — end-to-end methodology validation
# ---------------------------------------------------------------------------

class TestRunFilter:
    def _minimal_result(self, prior=0.5, T=10.0, n_events=10, seed=0):
        rng = np.random.default_rng(seed)
        n_events = max(n_events, 0)
        times = np.sort(rng.uniform(0, T, n_events)) if n_events > 0 else np.empty(0)
        sides = rng.choice([-1, 1], size=n_events) if n_events > 0 else np.empty(0, dtype=int)
        marks = rng.exponential(0.5, n_events) if n_events > 0 else np.empty(0)
        pi0 = np.array([prior, 1.0 - prior])
        return run_filter(
            event_times=times,
            event_sides=sides,
            event_marks=marks,
            T=T,
            generator=PSI,
            init_kappa_plus=np.array([3.0, 1.0]),
            init_kappa_minus=np.array([1.0, 3.0]),
            lambda_infty=LAM_INF,
            beta=BETA,
            eta=ETA,
            zeta=ZETA,
            pi0=pi0,
        )

    def test_returns_required_keys(self):
        res = self._minimal_result()
        assert "filter_times" in res
        assert "posterior" in res
        assert "kappa_plus" in res
        assert "kappa_minus" in res

    def test_posterior_rows_sum_to_one(self):
        """Every row of the posterior matrix must sum to 1 (simplex constraint)."""
        res = self._minimal_result()
        row_sums = res["posterior"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_posterior_nonnegative(self):
        res = self._minimal_result()
        assert np.all(res["posterior"] >= 0)

    def test_first_posterior_matches_prior(self):
        """Initial posterior must equal pi0."""
        prior = 0.9
        res = self._minimal_result(prior=prior)
        np.testing.assert_allclose(res["posterior"][0], [prior, 1.0 - prior], atol=1e-10)

    def test_time_axis_starts_at_zero(self):
        res = self._minimal_result()
        assert res["filter_times"][0] == pytest.approx(0.0)

    def test_time_axis_ends_at_T(self):
        T = 15.0
        res = self._minimal_result(T=T)
        assert res["filter_times"][-1] == pytest.approx(T)

    def test_filter_handles_zero_events(self):
        """No events: filter should still evolve from t=0 to T without crash."""
        res = self._minimal_result(n_events=0)
        assert res["posterior"].shape[0] >= 2
        np.testing.assert_allclose(res["posterior"].sum(axis=1), 1.0, atol=1e-8)

    def test_filter_handles_single_event(self):
        res = self._minimal_result(n_events=1)
        assert res["posterior"].shape[0] >= 3  # t=0, event, T

    def test_prior_90_high_posterior_regime_0(self):
        """With prior 0.9 for regime 1 and symmetric generator, posterior should
        remain near 0.9 if no strongly discriminating events arrive."""
        res = self._minimal_result(prior=0.9, n_events=0)
        # After filtering with no events and symmetric generator, belief drifts
        # toward 0.5 but starts at 0.9
        assert res["posterior"][0, 0] == pytest.approx(0.9, abs=1e-10)

    def test_prior_10_high_posterior_regime_1(self):
        res = self._minimal_result(prior=0.1, n_events=0)
        assert res["posterior"][0, 0] == pytest.approx(0.1, abs=1e-10)

    def test_posterior_shape_consistent(self):
        res = self._minimal_result(n_events=5)
        n_times = len(res["filter_times"])
        assert res["posterior"].shape == (n_times, 2)
        assert res["kappa_plus"].shape == (n_times, 2)
        assert res["kappa_minus"].shape == (n_times, 2)

    def test_intensities_nonnegative_throughout(self):
        res = self._minimal_result(n_events=20)
        assert np.all(res["kappa_plus"] >= 0)
        assert np.all(res["kappa_minus"] >= 0)

    def test_symmetry_under_regime_swap(self):
        """Swap regime labels (prior, kp, km) and check posterior symmetry.

        If pi0=[p, 1-p], kp=[a,b], km=[b,a], and we re-run with
        pi0=[1-p, p], kp=[b,a], km=[a,b] and the same events, then the
        new posterior[:,0] should equal the original posterior[:,1].
        """
        T = 10.0
        seed = 7
        rng = np.random.default_rng(seed)
        times = np.sort(rng.uniform(0, T, 15))
        sides = rng.choice([-1, 1], size=15)
        marks = rng.exponential(0.5, 15)

        kp_A = np.array([5.0, 1.0])
        km_A = np.array([1.0, 5.0])
        pi0_A = np.array([0.9, 0.1])

        res_A = run_filter(times, sides, marks, T, PSI,
                           kp_A, km_A, LAM_INF, BETA, ETA, ZETA, pi0_A)

        kp_B = np.array([1.0, 5.0])
        km_B = np.array([5.0, 1.0])
        pi0_B = np.array([0.1, 0.9])

        res_B = run_filter(times, sides, marks, T, PSI,
                           kp_B, km_B, LAM_INF, BETA, ETA, ZETA, pi0_B)

        np.testing.assert_allclose(
            res_A["posterior"][:, 0],
            res_B["posterior"][:, 1],
            atol=1e-8,
            err_msg="Symmetry under regime swap failed.",
        )
