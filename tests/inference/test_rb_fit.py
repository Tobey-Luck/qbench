"""
Tests for qbench.inference.rb_fit

Covers:
- rb_decay model evaluation
- log_prior boundary conditions
- log_likelihood correctness
- RBFit posterior summaries and credible intervals
- fit_rb end-to-end on synthetic data
- Error rate recovery within credible interval
"""

import numpy as np
import pytest

from qbench.inference.rb_fit import (
    RBFit,
    fit_rb,
    log_likelihood,
    log_prior,
    rb_decay,
)
from qbench.protocols.rb import RBResults


# ---------------------------------------------------------------------------
# rb_decay model
# ---------------------------------------------------------------------------

class TestRBDecay:
    def test_zero_length(self):
        # A * p^0 + B = A + B
        assert rb_decay(np.array([0.0]), 0.5, 0.99, 0.25) == pytest.approx(0.75)

    def test_large_length_approaches_B(self):
        # As m -> inf, A * p^m -> 0, so result -> B
        result = rb_decay(np.array([10000.0]), 0.5, 0.99, 0.25)
        assert result == pytest.approx(0.25, abs=1e-3)

    def test_noiseless_p_equals_one(self):
        # p=1 means no decay
        result = rb_decay(np.array([1.0, 10.0, 100.0]), 0.5, 1.0, 0.25)
        assert np.allclose(result, 0.75)

    def test_shape_preserved(self):
        m = np.array([1, 2, 4, 8, 16])
        result = rb_decay(m, 0.4, 0.95, 0.25)
        assert result.shape == (5,)

    def test_monotone_decay(self):
        m = np.arange(1, 20, dtype=float)
        result = rb_decay(m, 0.5, 0.95, 0.25)
        assert np.all(np.diff(result) < 0)


# ---------------------------------------------------------------------------
# log_prior
# ---------------------------------------------------------------------------

class TestLogPrior:
    def test_valid_params(self):
        assert np.isfinite(log_prior(np.array([0.5, 0.99, 0.25])))

    def test_p_out_of_bounds_low(self):
        assert log_prior(np.array([0.5, 0.0, 0.25])) == -np.inf

    def test_p_out_of_bounds_high(self):
        assert log_prior(np.array([0.5, 1.0, 0.25])) == -np.inf

    def test_A_zero(self):
        assert log_prior(np.array([0.0, 0.99, 0.25])) == -np.inf

    def test_B_zero(self):
        assert log_prior(np.array([0.5, 0.99, 0.0])) == -np.inf

    def test_A_plus_B_exceeds_one(self):
        assert log_prior(np.array([0.7, 0.99, 0.5])) == -np.inf

    def test_boundary_A_plus_B_equals_one(self):
        # A + B = 1.0 is physically valid (perfect SPAM), should be accepted
        assert np.isfinite(log_prior(np.array([0.75, 0.99, 0.25])))

    def test_returns_zero_for_valid(self):
        # Uniform prior => log(1) = 0
        assert log_prior(np.array([0.4, 0.95, 0.2])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# log_likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    def _make_data(self, A, p, B, lengths, n_shots, seed=0):
        rng = np.random.default_rng(seed)
        mu = rb_decay(lengths, A, p, B)
        counts = rng.binomial(n_shots, mu)
        return counts

    def test_finite_for_valid_params(self):
        lengths = np.array([1.0, 2.0, 4.0, 8.0])
        counts = np.array([95, 90, 80, 65])
        ll = log_likelihood(np.array([0.5, 0.99, 0.25]), lengths, counts, 100)
        assert np.isfinite(ll)

    def test_true_params_higher_likelihood(self):
        # True params should give higher likelihood than a wrong guess
        A_true, p_true, B_true = 0.45, 0.97, 0.25
        lengths = np.arange(1, 30, dtype=float)
        counts = self._make_data(A_true, p_true, B_true, lengths, n_shots=500)

        ll_true = log_likelihood(
            np.array([A_true, p_true, B_true]), lengths, counts, 500
        )
        ll_wrong = log_likelihood(
            np.array([0.3, 0.80, 0.15]), lengths, counts, 500
        )
        assert ll_true > ll_wrong

    def test_more_data_increases_log_likelihood_magnitude(self):
        lengths = np.array([1.0, 4.0, 16.0])
        theta = np.array([0.4, 0.95, 0.25])
        counts_small = np.array([40, 35, 28])
        counts_large = counts_small * 10

        ll_small = log_likelihood(theta, lengths, counts_small, 100)
        ll_large = log_likelihood(theta, lengths, counts_large, 1000)
        assert abs(ll_large) > abs(ll_small)


# ---------------------------------------------------------------------------
# RBFit container
# ---------------------------------------------------------------------------

class TestRBFit:
    def _make_fit(self, n_samples=1000) -> RBFit:
        rng = np.random.default_rng(0)
        # Simulate posterior samples around known values
        A_samples = rng.normal(0.45, 0.02, n_samples)
        p_samples = rng.normal(0.97, 0.005, n_samples)
        B_samples = rng.normal(0.25, 0.01, n_samples)
        samples = np.column_stack([A_samples, p_samples, B_samples])
        return RBFit(
            samples=samples,
            n_walkers=32,
            n_steps=500,
            acceptance_fraction=0.35,
        )

    def test_samples_shape(self):
        fit = self._make_fit(500)
        assert fit.samples.shape == (500, 3)

    def test_A_samples(self):
        fit = self._make_fit()
        assert fit.A_samples.shape == (1000,)

    def test_p_samples(self):
        fit = self._make_fit()
        assert fit.p_samples.shape == (1000,)

    def test_error_rate_samples(self):
        fit = self._make_fit()
        r = fit.error_rate_samples
        assert np.all(r >= 0)
        assert np.all(r <= 0.5)

    def test_error_rate_formula(self):
        fit = self._make_fit()
        expected = (1.0 - fit.p_samples) / 2.0
        assert np.allclose(fit.error_rate_samples, expected)

    def test_posterior_mean(self):
        fit = self._make_fit()
        mean_p = fit.posterior_mean("p")
        assert mean_p == pytest.approx(0.97, abs=0.02)

    def test_posterior_std(self):
        fit = self._make_fit()
        std_p = fit.posterior_std("p")
        assert std_p == pytest.approx(0.005, abs=0.002)

    def test_credible_interval_ordering(self):
        fit = self._make_fit()
        lo, hi = fit.credible_interval("error_rate", level=0.95)
        assert lo < hi

    def test_credible_interval_coverage(self):
        fit = self._make_fit()
        lo, hi = fit.credible_interval("p", level=0.95)
        samples = fit.p_samples
        coverage = np.mean((samples >= lo) & (samples <= hi))
        assert coverage == pytest.approx(0.95, abs=0.01)

    def test_invalid_param_raises(self):
        fit = self._make_fit()
        with pytest.raises(ValueError, match="Unknown parameter"):
            fit.posterior_mean("gamma")

    def test_summary_is_string(self):
        fit = self._make_fit()
        s = fit.summary()
        assert isinstance(s, str)
        assert "error_rate" in s
        assert "acceptance" in s.lower()


# ---------------------------------------------------------------------------
# fit_rb end-to-end
# ---------------------------------------------------------------------------

class TestFitRB:
    def _make_synthetic_results(
        self,
        A: float = 0.45,
        p: float = 0.97,
        B: float = 0.25,
        sequence_lengths=None,
        n_shots: int = 300,
        n_sequences: int = 20,
        seed: int = 7,
    ) -> RBResults:
        """Construct synthetic RBResults from known ground truth."""
        if sequence_lengths is None:
            sequence_lengths = [1, 2, 4, 8, 16, 32, 64]
        rng = np.random.default_rng(seed)
        survival_probs = {}
        for m in sequence_lengths:
            mu = rb_decay(np.array([float(m)]), A, p, B)[0]
            probs = rng.binomial(n_shots, mu, size=n_sequences) / n_shots
            survival_probs[m] = probs
        return RBResults(
            sequence_lengths=sequence_lengths,
            survival_probs=survival_probs,
            n_shots=n_shots,
            n_sequences=n_sequences,
        )

    def test_returns_rbfit(self):
        results = self._make_synthetic_results()
        fit = fit_rb(results, n_walkers=16, n_steps=200, n_burn=100, seed=0)
        assert isinstance(fit, RBFit)

    def test_samples_shape(self):
        results = self._make_synthetic_results()
        fit = fit_rb(results, n_walkers=16, n_steps=200, n_burn=100, seed=0)
        assert fit.samples.shape == (16 * 200, 3)

    def test_acceptance_fraction_healthy(self):
        results = self._make_synthetic_results()
        fit = fit_rb(results, n_walkers=16, n_steps=300, n_burn=100, seed=0)
        assert 0.1 < fit.acceptance_fraction < 0.8

    def test_error_rate_recovers_true_value(self):
        """
        The posterior mean error rate should be close to the true value,
        and the true value should lie within the 95% credible interval.
        """
        true_p = 0.97
        true_r = (1.0 - true_p) / 2.0  # = 0.015

        results = self._make_synthetic_results(p=true_p, n_shots=500, n_sequences=30)
        fit = fit_rb(results, n_walkers=32, n_steps=500, n_burn=200, seed=42)

        mean_r = fit.posterior_mean("error_rate")
        lo, hi = fit.credible_interval("error_rate", level=0.95)

        # True value should be within the credible interval
        assert lo <= true_r <= hi, (
            f"True r={true_r:.4f} not in 95% CI [{lo:.4f}, {hi:.4f}]"
        )
        # Posterior mean should be within 50% of the true value
        assert abs(mean_r - true_r) < true_r * 0.5, (
            f"Posterior mean r={mean_r:.4f} too far from true r={true_r:.4f}"
        )

    def test_p_samples_in_valid_range(self):
        results = self._make_synthetic_results()
        fit = fit_rb(results, n_walkers=16, n_steps=200, n_burn=100, seed=0)
        assert np.all(fit.p_samples > 0)
        assert np.all(fit.p_samples < 1)

    def test_summary_runs(self):
        results = self._make_synthetic_results()
        fit = fit_rb(results, n_walkers=16, n_steps=200, n_burn=100, seed=0)
        s = fit.summary()
        assert "error_rate" in s