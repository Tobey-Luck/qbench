# qbench

A Python library for quantum gate characterization and benchmarking with rigorous Bayesian inference.

## Overview

`qbench` implements standard quantum benchmarking protocols -- randomized benchmarking (RB), interleaved randomized benchmarking (IRB), and gate set tomography (GST) -- with a focus on statistically rigorous parameter estimation. Rather than reporting point estimates, `qbench` provides full posterior distributions over gate error parameters via MCMC sampling, giving calibrated uncertainty quantification on characterization results.

The library is designed around a hardware-agnostic backend interface: all protocols run against an abstract `Backend`, with a high-fidelity `SimulatedBackend` included. The abstraction is clean enough to support real hardware backends (e.g. via Qiskit or Cirq) as future extensions.

## Features

- Randomized benchmarking (standard and interleaved)
- Gate set tomography (coming soon)
- Cross-entropy benchmarking (coming soon)
- Bayesian posterior estimation via MCMC (emcee)
- Credible intervals and model comparison, not just point estimates
- Hardware-agnostic backend interface
- Comprehensive test suite

## Installation

```bash
git clone https://github.com/Tobey-Luck/qbench.git
cd qbench
pip install -e ".[dev]"
```

## Quickstart

```python
import qbench
from qbench.backends import SimulatedBackend
from qbench.protocols import RandomizedBenchmarking

# Configure a noisy simulated backend
backend = SimulatedBackend(n_qubits=1, depolarizing_rate=1e-3)

# Run randomized benchmarking
rb = RandomizedBenchmarking(backend=backend, n_qubits=1)
results = rb.run(sequence_lengths=[1, 2, 4, 8, 16, 32, 64], n_shots=200)

# Bayesian inference on decay parameter
fit = results.fit()
print(fit.summary())          # posterior mean, std, credible intervals
fit.plot_posterior()          # marginal posterior over error rate
fit.plot_decay_curve()        # survival probability with uncertainty band
```

## Project Structure

```
qbench/
  backends/       # Backend interface and simulated implementation
  protocols/      # Benchmarking protocol implementations
  inference/      # Bayesian inference, MCMC, model fitting
  visualization/  # Plotting utilities
tests/
examples/         # Jupyter notebooks
docs/
```

## Design Philosophy

Most benchmarking tools report a single decay parameter and a standard error from a least-squares fit. This is statistically inadequate when sequence counts are low, when the likelihood is non-Gaussian, or when comparing competing error models. `qbench` treats inference as a first-class concern: every protocol produces a posterior, and summaries (mean, credible intervals, model Bayes factors) are derived from that posterior.

## Dependencies

- numpy
- scipy
- emcee
- qutip
- matplotlib
- pytest (dev)

## License

MIT
