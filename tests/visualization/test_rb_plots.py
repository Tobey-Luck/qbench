"""
Tests for qbench.visualization.rb_plots

Visualization tests focus on API correctness, output types, and that plots
render without errors rather than pixel-level comparisons. We use a non-
interactive matplotlib backend to avoid display requirements.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, must be set before pyplot import
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from qbench.inference.rb_fit import RBFit
from qbench.protocols.rb import RBResults
from qbench.visualization import plot_decay_curve, plot_posterior


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_results(seed: int = 0) -> RBResults:
    """Synthetic RBResults with a realistic decay."""
    rng = np.random.default_rng(seed)
    lengths = [1, 2, 4, 8, 16, 32, 64]
    A, p, B = 0.45, 0.97, 0.25
    survival_probs = {}
    for m in lengths:
        mu = A * p**m + B
        probs = rng.binomial(300, mu, size=20) / 300.0
        survival_probs[m] = probs
    return RBResults(
        sequence_lengths=lengths,
        survival_probs=survival_probs,
        n_shots=300,
        n_sequences=20,
    )


def _make_fit(n_samples: int = 500, seed: int = 0) -> RBFit:
    """Synthetic RBFit with plausible posterior samples."""
    rng = np.random.default_rng(seed)
    A_s = rng.normal(0.45, 0.02, n_samples)
    p_s = rng.normal(0.97, 0.005, n_samples)
    B_s = rng.normal(0.25, 0.01, n_samples)
    # Clip to valid range
    A_s = np.clip(A_s, 0.01, 0.9)
    p_s = np.clip(p_s, 0.01, 0.999)
    B_s = np.clip(B_s, 0.01, 0.9)
    samples = np.column_stack([A_s, p_s, B_s])
    return RBFit(
        samples=samples,
        n_walkers=32,
        n_steps=n_samples // 32,
        acceptance_fraction=0.35,
    )


# ---------------------------------------------------------------------------
# plot_decay_curve
# ---------------------------------------------------------------------------

class TestPlotDecayCurve:
    def setup_method(self):
        plt.close("all")

    def test_returns_figure(self):
        results = _make_results()
        fig = plot_decay_curve(results)
        assert isinstance(fig, Figure)

    def test_returns_figure_with_fit(self):
        results = _make_results()
        fit = _make_fit()
        fig = plot_decay_curve(results, fit=fit)
        assert isinstance(fig, Figure)

    def test_accepts_existing_axes(self):
        results = _make_results()
        fig_in, ax = plt.subplots()
        fig_out = plot_decay_curve(results, ax=ax)
        assert fig_out is fig_in

    def test_axes_has_xlabel(self):
        results = _make_results()
        fig = plot_decay_curve(results)
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""

    def test_axes_has_ylabel(self):
        results = _make_results()
        fig = plot_decay_curve(results)
        ax = fig.axes[0]
        assert ax.get_ylabel() != ""

    def test_axes_has_title(self):
        results = _make_results()
        fig = plot_decay_curve(results)
        ax = fig.axes[0]
        assert ax.get_title() != ""

    def test_ylim_reasonable(self):
        results = _make_results()
        fig = plot_decay_curve(results)
        ax = fig.axes[0]
        ylo, yhi = ax.get_ylim()
        assert ylo >= -0.1
        assert yhi <= 1.2

    def test_legend_present(self):
        results = _make_results()
        fig = plot_decay_curve(results)
        ax = fig.axes[0]
        assert ax.get_legend() is not None

    def test_credible_band_with_fit(self):
        results = _make_results()
        fit = _make_fit()
        fig = plot_decay_curve(results, fit=fit, credible_level=0.90)
        ax = fig.axes[0]
        # Should have more artists (fill_between adds a PolyCollection)
        collections = ax.collections
        assert len(collections) > 0

    def test_show_individual_false(self):
        results = _make_results()
        fig_with = plot_decay_curve(results, show_individual=True)
        fig_without = plot_decay_curve(results, show_individual=False)
        # More scatter points expected when show_individual=True
        n_with = sum(len(c.get_offsets()) for c in fig_with.axes[0].collections)
        n_without = sum(len(c.get_offsets()) for c in fig_without.axes[0].collections)
        assert n_with >= n_without

    def test_custom_color(self):
        results = _make_results()
        # Should not raise with a custom colour
        fig = plot_decay_curve(results, color="#FF5733")
        assert isinstance(fig, Figure)

    def test_no_fit_no_annotation(self):
        results = _make_results()
        fig = plot_decay_curve(results, fit=None)
        ax = fig.axes[0]
        # No annotation text about error rate without fit
        texts = [t.get_text() for t in ax.texts]
        assert not any("r =" in t for t in texts)

    def test_fit_adds_annotation(self):
        results = _make_results()
        fit = _make_fit()
        fig = plot_decay_curve(results, fit=fit)
        ax = fig.axes[0]
        # Annotation with error rate should be present
        texts = [t.get_text() for t in ax.texts]
        assert any("r" in t for t in texts)


# ---------------------------------------------------------------------------
# plot_posterior
# ---------------------------------------------------------------------------

class TestPlotPosterior:
    def setup_method(self):
        plt.close("all")

    def test_returns_figure(self):
        fit = _make_fit()
        fig = plot_posterior(fit)
        assert isinstance(fig, Figure)

    def test_default_params_four_panels(self):
        fit = _make_fit()
        fig = plot_posterior(fit)
        # Default 4 params -> 4x4 grid, lower triangle = 4+3+2+1 = 10 axes
        assert len(fig.axes) == 10

    def test_two_params(self):
        fit = _make_fit()
        fig = plot_posterior(fit, params=["p", "error_rate"])
        # 2 params -> 2+1 = 3 axes (lower triangle of 2x2)
        assert len(fig.axes) == 3

    def test_custom_color(self):
        fit = _make_fit()
        fig = plot_posterior(fit, color="#E11D48")
        assert isinstance(fig, Figure)

    def test_custom_n_bins(self):
        fit = _make_fit()
        fig = plot_posterior(fit, params=["p"], n_bins=20)
        assert isinstance(fig, Figure)

    def test_diagonal_axes_have_no_yticks(self):
        fit = _make_fit()
        fig = plot_posterior(fit, params=["A", "p"])
        # Diagonal axes (index 0 and 2 in lower triangle)
        for ax in fig.axes:
            # Check if it's a diagonal by seeing if it's a histogram (no ylabel)
            if ax.get_ylabel() == "":
                assert ax.get_yticks().size == 0 or list(ax.get_yticks()) == []

    def test_has_suptitle(self):
        fit = _make_fit()
        fig = plot_posterior(fit)
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() != ""

    def test_invalid_param_raises(self):
        fit = _make_fit()
        with pytest.raises((ValueError, KeyError)):
            plot_posterior(fit, params=["p", "not_a_param"])

    def test_single_param(self):
        fit = _make_fit()
        fig = plot_posterior(fit, params=["error_rate"])
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestVisualizationIntegration:
    def test_results_to_decay_plot(self):
        """End-to-end: RBResults -> plot_decay_curve."""
        from qbench.backends import SimulatedBackend
        from qbench.protocols import RandomizedBenchmarking

        backend = SimulatedBackend(depolarizing_rate=2e-3, seed=0)
        rb = RandomizedBenchmarking(backend=backend, seed=0)
        results = rb.run(
            sequence_lengths=[1, 4, 16],
            n_shots=100,
            n_sequences=5,
        )
        fig = plot_decay_curve(results)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_fit_to_posterior_plot(self):
        """End-to-end: RBFit -> plot_posterior."""
        fit = _make_fit(n_samples=200)
        fig = plot_posterior(fit, params=["p", "error_rate"])
        assert isinstance(fig, Figure)
        plt.close(fig)