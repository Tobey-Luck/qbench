"""
qbench.inference.irb_fit
~~~~~~~~~~~~~~~~~~~~~~~~

Bayesian inference for interleaved randomized benchmarking.

Model
-----
We fit two decay curves jointly:

    E[P_rb(0)  | m] = A_rb  * p_rb^m  + B
    E[P_irb(0) | m] = A_irb * p_irb^m + B

where B is shared between both experiments (same fully-depolarised offset),
and the gate error rate is derived as:

    r_gate = (d - 1) / d * (1 - p_irb / p_rb)

with d = 2 for a single qubit.

The posterior over (A_rb, A_irb, p_rb, p_irb, B) is sampled via MCMC.
The derived posterior over r_gate is computed from the samples.

Prior
-----
    p_rb, p_irb ~ Uniform(0, 1)
    A_rb, A_irb ~ Uniform(0, 1)
    B           ~ Uniform(0, 1)
    A_rb + B <= 1,  A_irb + B <= 1
    p_irb <= p_rb  (IRB decay is at least as fast as RB)

References
----------
Magesan et al., PRL 109, 080505 (2012)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import emcee
    _EMCEE_AVAILABLE = True
except ImportError:
    _EMCEE_AVAILABLE = False

from qbench.inference.rb_fit import rb_decay, _lsq_init


# ---------------------------------------------------------------------------
# Prior and likelihood
# ---------------------------------------------------------------------------

def log_prior_irb(theta: np.ndarray) -> float:
    """
    Log prior over (A_rb, A_irb, p_rb, p_irb, B).

    Enforces:
    - All parameters positive
    - p_rb, p_irb in (0, 1)
    - A_rb + B <= 1,  A_irb + B <= 1
    - p_irb <= p_rb (IRB decays at least as fast)
    """
    A_rb, A_irb, p_rb, p_irb, B = theta

    if not (0.0 < p_rb < 1.0):
        return -np.inf
    if not (0.0 < p_irb <= p_rb):
        return -np.inf
    if not (0.0 < A_rb):
        return -np.inf
    if not (0.0 < A_irb):
        return -np.inf
    if not (0.0 < B):
        return -np.inf
    if A_rb + B > 1.0:
        return -np.inf
    if A_irb + B > 1.0:
        return -np.inf

    return 0.0


def log_likelihood_irb(
    theta: np.ndarray,
    rb_lengths: np.ndarray,
    rb_counts: np.ndarray,
    irb_lengths: np.ndarray,
    irb_counts: np.ndarray,
    n_shots: int,
) -> float:
    """
    Joint binomial log-likelihood for RB and IRB data.

    Parameters
    ----------
    theta : (A_rb, A_irb, p_rb, p_irb, B)
    rb_lengths, rb_counts : RB observations
    irb_lengths, irb_counts : IRB observations
    n_shots : shots per circuit
    """
    A_rb, A_irb, p_rb, p_irb, B = theta

    mu_rb = np.clip(rb_decay(rb_lengths, A_rb, p_rb, B), 1e-10, 1 - 1e-10)
    mu_irb = np.clip(rb_decay(irb_lengths, A_irb, p_irb, B), 1e-10, 1 - 1e-10)

    ll_rb = float(np.sum(
        rb_counts * np.log(mu_rb) + (n_shots - rb_counts) * np.log(1.0 - mu_rb)
    ))
    ll_irb = float(np.sum(
        irb_counts * np.log(mu_irb) + (n_shots - irb_counts) * np.log(1.0 - mu_irb)
    ))

    return ll_rb + ll_irb


def log_posterior_irb(theta, rb_lengths, rb_counts, irb_lengths, irb_counts, n_shots):
    lp = log_prior_irb(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_irb(
        theta, rb_lengths, rb_counts, irb_lengths, irb_counts, n_shots
    )


# ---------------------------------------------------------------------------
# IRBFit result container
# ---------------------------------------------------------------------------

@dataclass
class IRBFit:
    """
    Posterior samples and summaries from a Bayesian IRB fit.

    Attributes
    ----------
    samples : ndarray, shape (n_samples, 5)
        Posterior samples over (A_rb, A_irb, p_rb, p_irb, B).
    target_gate : str
        Name of the interleaved gate.
    n_walkers, n_steps, acceptance_fraction : MCMC diagnostics.
    """

    samples: np.ndarray      # (n_samples, 5)
    target_gate: str
    n_walkers: int
    n_steps: int
    acceptance_fraction: float

    # Column indices
    _COL = {"A_rb": 0, "A_irb": 1, "p_rb": 2, "p_irb": 3, "B": 4}

    @property
    def p_rb_samples(self) -> np.ndarray:
        return self.samples[:, 2]

    @property
    def p_irb_samples(self) -> np.ndarray:
        return self.samples[:, 3]

    @property
    def gate_error_rate_samples(self) -> np.ndarray:
        """
        Posterior samples of the gate error rate:
            r_gate = (d-1)/d * (1 - p_irb / p_rb),  d=2
        """
        return 0.5 * (1.0 - self.p_irb_samples / self.p_rb_samples)

    @property
    def rb_error_rate_samples(self) -> np.ndarray:
        """Average Clifford error rate from the reference RB."""
        return (1.0 - self.p_rb_samples) / 2.0

    def _get_samples(self, param: str) -> np.ndarray:
        mapping = {
            "A_rb":           self.samples[:, 0],
            "A_irb":          self.samples[:, 1],
            "p_rb":           self.p_rb_samples,
            "p_irb":          self.p_irb_samples,
            "B":              self.samples[:, 4],
            "gate_error_rate": self.gate_error_rate_samples,
            "rb_error_rate":  self.rb_error_rate_samples,
        }
        if param not in mapping:
            raise ValueError(
                f"Unknown parameter '{param}'. "
                f"Choose from: {list(mapping.keys())}"
            )
        return mapping[param]

    def posterior_mean(self, param: str = "gate_error_rate") -> float:
        return float(np.mean(self._get_samples(param)))

    def posterior_std(self, param: str = "gate_error_rate") -> float:
        return float(np.std(self._get_samples(param)))

    def credible_interval(
        self,
        param: str = "gate_error_rate",
        level: float = 0.95,
    ) -> tuple[float, float]:
        samples = self._get_samples(param)
        alpha = (1.0 - level) / 2.0
        return (
            float(np.percentile(samples, 100 * alpha)),
            float(np.percentile(samples, 100 * (1 - alpha))),
        )

    def summary(self) -> str:
        params = ["p_rb", "p_irb", "B", "rb_error_rate", "gate_error_rate"]
        lines = [
            f"IRB Fit Summary  (gate: {self.target_gate})",
            f"  Walkers: {self.n_walkers}  Steps: {self.n_steps}  "
            f"Acceptance: {self.acceptance_fraction:.3f}",
            "",
            f"  {'Parameter':>16}  {'Mean':>10}  {'Std':>10}  "
            f"{'95% CI lo':>10}  {'95% CI hi':>10}",
            "  " + "-" * 64,
        ]
        for param in params:
            mean = self.posterior_mean(param)
            std = self.posterior_std(param)
            lo, hi = self.credible_interval(param)
            lines.append(
                f"  {param:>16}  {mean:>10.6f}  {std:>10.6f}  "
                f"{lo:>10.6f}  {hi:>10.6f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_irb(
    results,
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burn: int = 500,
    seed: int | None = None,
) -> IRBFit:
    """
    Jointly fit the RB and IRB decay curves using MCMC.

    Parameters
    ----------
    results : IRBResults
        Output from InterleavedRandomizedBenchmarking.run().
    n_walkers, n_steps, n_burn : MCMC parameters.
    seed : random seed.

    Returns
    -------
    IRBFit
    """
    if not _EMCEE_AVAILABLE:
        raise ImportError("emcee is required. Install with: pip install emcee")

    def _flatten(rb_results):
        lengths, counts = [], []
        for m, probs in rb_results.survival_probs.items():
            for p_k in probs:
                lengths.append(m)
                counts.append(int(round(p_k * rb_results.n_shots)))
        return np.array(lengths, dtype=float), np.array(counts, dtype=int)

    rb_lengths, rb_counts = _flatten(results.rb_results)
    irb_lengths, irb_counts = _flatten(results.irb_results)
    n_shots = results.rb_results.n_shots

    # Initial guess from least-squares on each experiment separately
    rb_means = results.rb_results.mean_survival()
    irb_means = results.irb_results.mean_survival()
    m_arr = np.array(sorted(rb_means.keys()), dtype=float)

    A_rb0, p_rb0, B0 = _lsq_init(m_arr, np.array([rb_means[m] for m in sorted(rb_means)]))
    A_irb0, p_irb0, _ = _lsq_init(m_arr, np.array([irb_means[m] for m in sorted(irb_means)]))

    # Ensure p_irb0 <= p_rb0
    p_irb0 = min(p_irb0, p_rb0 * 0.99)

    p_init = np.array([
        np.clip(A_rb0, 0.05, 0.9),
        np.clip(A_irb0, 0.05, 0.9),
        np.clip(p_rb0, 0.05, 0.999),
        np.clip(p_irb0, 0.05, 0.999),
        np.clip(B0, 0.05, 0.4),
    ])

    rng = np.random.default_rng(seed)
    ndim = 5

    walkers = []
    attempts = 0
    while len(walkers) < n_walkers:
        candidate = p_init + rng.normal(0, 0.015, size=ndim)
        if np.isfinite(log_prior_irb(candidate)):
            walkers.append(candidate)
        attempts += 1
        if attempts > 20000:
            raise RuntimeError("Could not initialise walkers in valid prior region.")
    pos = np.array(walkers)

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        log_posterior_irb,
        args=(rb_lengths, rb_counts, irb_lengths, irb_counts, n_shots),
    )

    state = sampler.run_mcmc(pos, n_burn, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, n_steps, progress=False)

    flat_samples = sampler.get_chain(flat=True)
    acceptance = float(np.mean(sampler.acceptance_fraction))

    return IRBFit(
        samples=flat_samples,
        target_gate=results.target_gate,
        n_walkers=n_walkers,
        n_steps=n_steps,
        acceptance_fraction=acceptance,
    )