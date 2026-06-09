"""
qbench.inference.rb_fit
~~~~~~~~~~~~~~~~~~~~~~~

Bayesian inference for the standard RB decay model:

    E[P(0) | m] = A * p^m + B

where:
    p  = (1 - 4r/3)  is the depolarizing parameter
    r  = (1 - p) / 2 is the average gate error rate (EPC)
    A  captures state preparation and measurement (SPAM) asymmetry
    B  is the offset (ideally 0.5 for fully depolarized state)

Rather than a least-squares point estimate, we compute the full posterior
P(A, p, B | data) via MCMC using emcee. This gives calibrated credible
intervals on all parameters, and in particular on the error rate r.

Prior
-----
    p  ~ Uniform(0, 1)
    A  ~ Uniform(0, 1)
    B  ~ Uniform(0, 1)
    with the constraint A + B <= 1 (survival probability bounded by 1)

Likelihood
----------
For each (m, k) observation (sequence length m, sequence index k):
    n_k  ~ Binomial(n_shots, A * p^m + B)

where n_k = round(p_k * n_shots) is the observed survival count.

References
----------
Magesan et al., PRL 106, 180504 (2011)
Foreman-Mackey et al., PASP 125, 306 (2013) — emcee
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

try:
    import emcee
    _EMCEE_AVAILABLE = True
except ImportError:
    _EMCEE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Decay model
# ---------------------------------------------------------------------------

def rb_decay(m: np.ndarray, A: float, p: float, B: float) -> np.ndarray:
    """
    Evaluate the RB decay model E[P(0) | m] = A * p^m + B.

    Parameters
    ----------
    m : array of sequence lengths
    A, p, B : model parameters
    """
    return A * np.power(p, m) + B


# ---------------------------------------------------------------------------
# Log-likelihood and log-prior
# ---------------------------------------------------------------------------

def log_prior(theta: np.ndarray) -> float:
    """
    Log prior over (A, p, B).

    Uniform on the physically valid region:
        0 < p < 1
        0 < A
        0 < B
        A + B <= 1
        A + B >= 0  (implied)
    """
    A, p, B = theta
    if not (0.0 < p < 1.0):
        return -np.inf
    if not (0.0 < A):
        return -np.inf
    if not (0.0 < B):
        return -np.inf
    if A + B > 1.0:
        return -np.inf
    return 0.0  # log(1) = 0 for uniform prior


def log_likelihood(
    theta: np.ndarray,
    lengths: np.ndarray,
    counts: np.ndarray,
    n_shots: int,
) -> float:
    """
    Log-likelihood under the binomial observation model.

    Parameters
    ----------
    theta : (A, p, B)
    lengths : 1D array of sequence lengths, one per observation
    counts : 1D array of survival counts (integers), one per observation
    n_shots : number of shots per circuit
    """
    A, p, B = theta
    mu = rb_decay(lengths, A, p, B)

    # Clip to avoid log(0)
    mu = np.clip(mu, 1e-10, 1.0 - 1e-10)

    # Binomial log-likelihood: sum_i [ k_i log(mu_i) + (n-k_i) log(1-mu_i) ]
    log_p = counts * np.log(mu) + (n_shots - counts) * np.log(1.0 - mu)
    return float(np.sum(log_p))


def log_posterior(
    theta: np.ndarray,
    lengths: np.ndarray,
    counts: np.ndarray,
    n_shots: int,
) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, lengths, counts, n_shots)


# ---------------------------------------------------------------------------
# RBFit result container
# ---------------------------------------------------------------------------

@dataclass
class RBFit:
    """
    Posterior samples and summaries from a Bayesian RB fit.

    Attributes
    ----------
    samples : ndarray, shape (n_samples, 3)
        Posterior samples over (A, p, B) after burn-in.
    n_walkers : int
        Number of MCMC walkers used.
    n_steps : int
        Number of steps per walker (after burn-in).
    acceptance_fraction : float
        Mean acceptance fraction across walkers (healthy range: 0.2-0.5).
    """

    samples: np.ndarray       # (n_samples, 3): columns are A, p, B
    n_walkers: int
    n_steps: int
    acceptance_fraction: float

    # ------------------------------------------------------------------
    # Derived parameter posteriors
    # ------------------------------------------------------------------

    @property
    def A_samples(self) -> np.ndarray:
        return self.samples[:, 0]

    @property
    def p_samples(self) -> np.ndarray:
        return self.samples[:, 1]

    @property
    def B_samples(self) -> np.ndarray:
        return self.samples[:, 2]

    @property
    def error_rate_samples(self) -> np.ndarray:
        """
        Posterior samples of the average gate error rate (EPC):
            r = (1 - p) / 2
        """
        return (1.0 - self.p_samples) / 2.0

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def credible_interval(
        self,
        param: str = "error_rate",
        level: float = 0.95,
    ) -> tuple[float, float]:
        """
        Return the equal-tailed credible interval for a parameter.

        Parameters
        ----------
        param : one of 'A', 'p', 'B', 'error_rate'
        level : credible level (default 0.95)
        """
        samples = self._get_samples(param)
        alpha = (1.0 - level) / 2.0
        lo = float(np.percentile(samples, 100 * alpha))
        hi = float(np.percentile(samples, 100 * (1 - alpha)))
        return lo, hi

    def posterior_mean(self, param: str = "error_rate") -> float:
        """Return the posterior mean of a parameter."""
        return float(np.mean(self._get_samples(param)))

    def posterior_std(self, param: str = "error_rate") -> float:
        """Return the posterior standard deviation of a parameter."""
        return float(np.std(self._get_samples(param)))

    def summary(self) -> str:
        """Return a formatted summary of the posterior."""
        params = ["A", "p", "B", "error_rate"]
        lines = [
            "RB Fit Summary (Bayesian, MCMC)",
            f"  Walkers: {self.n_walkers}  Steps: {self.n_steps}  "
            f"Acceptance: {self.acceptance_fraction:.3f}",
            "",
            f"  {'Parameter':>12}  {'Mean':>10}  {'Std':>10}  "
            f"{'95% CI lo':>10}  {'95% CI hi':>10}",
            "  " + "-" * 58,
        ]
        for param in params:
            mean = self.posterior_mean(param)
            std = self.posterior_std(param)
            lo, hi = self.credible_interval(param)
            lines.append(
                f"  {param:>12}  {mean:>10.6f}  {std:>10.6f}  "
                f"{lo:>10.6f}  {hi:>10.6f}"
            )
        return "\n".join(lines)

    def _get_samples(self, param: str) -> np.ndarray:
        mapping = {
            "A": self.A_samples,
            "p": self.p_samples,
            "B": self.B_samples,
            "error_rate": self.error_rate_samples,
        }
        if param not in mapping:
            raise ValueError(
                f"Unknown parameter '{param}'. "
                f"Choose from: {list(mapping.keys())}"
            )
        return mapping[param]


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_rb(
    results,
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burn: int = 500,
    seed: int | None = None,
) -> RBFit:
    """
    Fit the RB decay model to RBResults using MCMC.

    Parameters
    ----------
    results : RBResults
        Output from RandomizedBenchmarking.run().
    n_walkers : int
        Number of emcee ensemble walkers. Must be even and >= 6.
    n_steps : int
        Number of MCMC steps per walker (after burn-in is discarded).
    n_burn : int
        Number of burn-in steps to discard.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    RBFit
        Posterior samples and summary statistics.
    """
    if not _EMCEE_AVAILABLE:
        raise ImportError(
            "emcee is required for Bayesian RB fitting. "
            "Install it with: pip install emcee"
        )

    # Flatten all observations into (length, count) pairs
    lengths_list = []
    counts_list = []
    for m, probs in results.survival_probs.items():
        for p_k in probs:
            lengths_list.append(m)
            counts_list.append(int(round(p_k * results.n_shots)))

    lengths = np.array(lengths_list, dtype=float)
    counts = np.array(counts_list, dtype=int)
    n_shots = results.n_shots

    # Initial guess via least-squares on the mean survival probabilities
    means = results.mean_survival()
    m_arr = np.array(sorted(means.keys()), dtype=float)
    p_arr = np.array([means[m] for m in sorted(means.keys())])
    A0, p0, B0 = _lsq_init(m_arr, p_arr)

    # Initialise walkers in a small ball around the LS estimate
    rng = np.random.default_rng(seed)
    ndim = 3
    p_init = np.array([A0, p0, B0])
    p_init = np.clip(p_init, 0.05, 0.95)

    # Perturb initial positions ensuring they stay in the prior
    walkers = []
    attempts = 0
    while len(walkers) < n_walkers:
        candidate = p_init + rng.normal(0, 0.02, size=ndim)
        if np.isfinite(log_prior(candidate)):
            walkers.append(candidate)
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("Could not initialise walkers in valid prior region.")
    pos = np.array(walkers)

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        log_posterior,
        args=(lengths, counts, n_shots),
    )

    # Burn-in
    state = sampler.run_mcmc(pos, n_burn, progress=False)
    sampler.reset()

    # Production run
    sampler.run_mcmc(state, n_steps, progress=False)

    flat_samples = sampler.get_chain(flat=True)  # (n_walkers * n_steps, 3)
    acceptance = float(np.mean(sampler.acceptance_fraction))

    return RBFit(
        samples=flat_samples,
        n_walkers=n_walkers,
        n_steps=n_steps,
        acceptance_fraction=acceptance,
    )


def _lsq_init(m: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    """
    Rough least-squares initialisation for (A, p0, B).
    Fits p(m) = A * decay^m + B by trying a grid of decay values.
    """
    best_residual = np.inf
    best = (0.5, 0.99, 0.25)

    for decay in np.linspace(0.8, 0.9999, 50):
        X = np.column_stack([np.power(decay, m), np.ones_like(m)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, p, rcond=None)
            A_hat, B_hat = coeffs
            residual = np.sum((A_hat * np.power(decay, m) + B_hat - p) ** 2)
            if residual < best_residual and A_hat > 0 and B_hat > 0 and A_hat + B_hat <= 1:
                best_residual = residual
                best = (float(A_hat), float(decay), float(B_hat))
        except np.linalg.LinAlgError:
            continue

    return best