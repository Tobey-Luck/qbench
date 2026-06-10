"""
qbench.visualization.rb_plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting utilities for randomized benchmarking results and posteriors.

Two main functions:

plot_decay_curve(results, fit)
    Scatter plot of survival probabilities vs sequence length, with the
    posterior median decay curve and a 95% credible band.

plot_posterior(fit)
    Corner plot of the joint posterior over (A, p, B) and the derived
    error rate r = (1-p)/2, with marginal histograms on the diagonal.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

from qbench.protocols.rb import RBResults
from qbench.inference.rb_fit import RBFit, rb_decay


# ---------------------------------------------------------------------------
# Decay curve plot
# ---------------------------------------------------------------------------

def plot_decay_curve(
    results: RBResults,
    fit: RBFit | None = None,
    ax: plt.Axes | None = None,
    n_curve_samples: int = 200,
    credible_level: float = 0.95,
    color: str = "#2563EB",
    show_individual: bool = True,
) -> Figure:
    """
    Plot the RB decay curve with optional Bayesian uncertainty band.

    Parameters
    ----------
    results : RBResults
        Raw experiment data from RandomizedBenchmarking.run().
    fit : RBFit, optional
        Posterior from results.fit(). If provided, draws the posterior
        median curve and a credible band.
    ax : matplotlib Axes, optional
        Axes to draw on. If None, a new figure is created.
    n_curve_samples : int
        Number of posterior samples used to compute the credible band.
    credible_level : float
        Width of the credible band (default 0.95).
    color : str
        Base colour for the plot elements.
    show_individual : bool
        If True, plot individual sequence survival probabilities as faint
        scatter points in addition to the per-length means.

    Returns
    -------
    matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.get_figure()

    lengths = np.array(results.sequence_lengths)
    means = np.array([np.mean(results.survival_probs[m]) for m in lengths])
    stds = np.array([np.std(results.survival_probs[m]) for m in lengths])

    # Individual sequence points
    if show_individual:
        for m in lengths:
            probs = results.survival_probs[m]
            ax.scatter(
                np.full_like(probs, m), probs,
                color=color, alpha=0.25, s=12, zorder=2,
            )

    # Mean ± 1 std error bars
    ax.errorbar(
        lengths, means, yerr=stds / np.sqrt(results.n_sequences),
        fmt="o", color=color, markersize=6, linewidth=1.5,
        capsize=4, zorder=3, label="Mean ± SE",
    )

    # Posterior curve and band
    if fit is not None:
        m_fine = np.linspace(0, max(lengths) * 1.05, 300)
        alpha = (1.0 - credible_level) / 2.0

        # Draw n_curve_samples posterior curves
        rng = np.random.default_rng(0)
        idx = rng.choice(len(fit.samples), size=min(n_curve_samples, len(fit.samples)), replace=False)
        curves = np.array([
            rb_decay(m_fine, *fit.samples[i])
            for i in idx
        ])

        lo = np.percentile(curves, 100 * alpha, axis=0)
        hi = np.percentile(curves, 100 * (1 - alpha), axis=0)
        median = np.median(curves, axis=0)

        ax.fill_between(
            m_fine, lo, hi,
            color=color, alpha=0.15, zorder=1,
            label=f"{int(credible_level * 100)}% credible band",
        )
        ax.plot(
            m_fine, median,
            color=color, linewidth=2.0, zorder=4,
            label="Posterior median",
        )

        # Annotate error rate
        mean_r = fit.posterior_mean("error_rate")
        lo_r, hi_r = fit.credible_interval("error_rate", level=credible_level)
        ax.annotate(
            f"$r = {mean_r:.4f}$ [{lo_r:.4f}, {hi_r:.4f}]",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    ax.set_xlabel("Sequence length $m$", fontsize=11)
    ax.set_ylabel("Survival probability $P(|0\\rangle)$", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=-0.5)
    ax.legend(fontsize=9, loc="upper right" if fit is None else "lower left")
    ax.set_title("Randomized Benchmarking Decay Curve", fontsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Corner / posterior plot
# ---------------------------------------------------------------------------

def plot_posterior(
    fit: RBFit,
    params: list[str] | None = None,
    color: str = "#7C3AED",
    n_bins: int = 40,
) -> Figure:
    """
    Corner plot of the joint posterior over RB model parameters.

    Diagonal panels show marginal histograms with the posterior mean and
    95% credible interval marked. Off-diagonal panels show 2D scatter
    density plots of parameter pairs.

    Parameters
    ----------
    fit : RBFit
        Posterior from fit_rb().
    params : list of str, optional
        Which parameters to include. Defaults to ['A', 'p', 'B', 'error_rate'].
    color : str
        Base colour for all plot elements.
    n_bins : int
        Number of histogram bins for marginal distributions.

    Returns
    -------
    matplotlib Figure
    """
    if params is None:
        params = ["A", "p", "B", "error_rate"]

    param_labels = {
        "A": "$A$",
        "p": "$p$",
        "B": "$B$",
        "error_rate": r"$r = \frac{1-p}{2}$",
    }

    n = len(params)
    samples_dict = {param: fit._get_samples(param) for param in params}

    fig = plt.figure(figsize=(3.2 * n, 3.0 * n))
    gs = gridspec.GridSpec(n, n, figure=fig, hspace=0.08, wspace=0.08)

    axes = [[None] * n for _ in range(n)]

    for row in range(n):
        for col in range(n):
            if col > row:
                continue  # upper triangle: leave blank

            ax = fig.add_subplot(gs[row, col])
            axes[row][col] = ax

            if row == col:
                # Diagonal: marginal histogram
                s = samples_dict[params[row]]
                ax.hist(s, bins=n_bins, color=color, alpha=0.75, density=True)

                mean = np.mean(s)
                lo, hi = np.percentile(s, [2.5, 97.5])
                ax.axvline(mean, color="black", linewidth=1.5, linestyle="-")
                ax.axvline(lo, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
                ax.axvline(hi, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

                ax.set_yticks([])
                if row == n - 1:
                    ax.set_xlabel(param_labels[params[col]], fontsize=10)
                else:
                    ax.set_xticklabels([])

            else:
                # Off-diagonal: 2D scatter density
                x = samples_dict[params[col]]
                y = samples_dict[params[row]]
                ax.scatter(x, y, s=1.5, color=color, alpha=0.15, rasterized=True)

                # Contours at 68% and 95%
                try:
                    _add_contours(ax, x, y, color=color)
                except Exception:
                    pass  # contours are cosmetic; don't fail if they error

                if row == n - 1:
                    ax.set_xlabel(param_labels[params[col]], fontsize=10)
                else:
                    ax.set_xticklabels([])

                if col == 0:
                    ax.set_ylabel(param_labels[params[row]], fontsize=10)
                else:
                    ax.set_yticklabels([])

            # Shared x-axis limits per column
            s_col = samples_dict[params[col]]
            lo_x, hi_x = np.percentile(s_col, [0.5, 99.5])
            ax.set_xlim(lo_x, hi_x)
            if row != col:
                s_row = samples_dict[params[row]]
                lo_y, hi_y = np.percentile(s_row, [0.5, 99.5])
                ax.set_ylim(lo_y, hi_y)

    fig.suptitle("RB Posterior Distribution", fontsize=13, y=1.01)
    return fig


def _add_contours(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    levels: tuple[float, float] = (0.68, 0.95),
) -> None:
    """Add smoothed 2D density contours at the given probability levels."""
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(np.vstack([x, y]))
    x_min, x_max = np.percentile(x, [0.5, 99.5])
    y_min, y_max = np.percentile(y, [0.5, 99.5])
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 60),
        np.linspace(y_min, y_max, 60),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Convert levels to density thresholds
    z_sorted = np.sort(zz.ravel())[::-1]
    z_cumsum = np.cumsum(z_sorted)
    z_cumsum /= z_cumsum[-1]
    thresholds = [
        z_sorted[np.searchsorted(z_cumsum, lvl)] for lvl in levels
    ]

    ax.contour(
        xx, yy, zz,
        levels=sorted(thresholds),
        colors=[color],
        alpha=0.6,
        linewidths=1.0,
    )