"""
qbench.protocols.rb
~~~~~~~~~~~~~~~~~~~

Standard single-qubit randomized benchmarking (RB).

Protocol
--------
For each sequence length m in `sequence_lengths`:
  1. Sample `n_sequences` random Clifford sequences of length m.
  2. Append a recovery gate so the ideal output is |0>.
  3. Execute each sequence on the backend for `n_shots` shots.
  4. Record the survival probability P(|0>) for each sequence.

The result is an RBResults object containing, for each sequence length, the
array of survival probabilities across all sequences. The RBResults object
exposes a `fit()` method (day 3) for Bayesian inference on the decay model.

References
----------
Magesan et al., PRL 106, 180504 (2011)
Knill et al., PRA 77, 012307 (2008)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from qbench.backends.base import Backend, Circuit
from .clifford import CLIFFORD_DECOMPOSITIONS, random_clifford_sequence


@dataclass
class RBResults:
    """
    Raw results from a randomized benchmarking experiment.

    Attributes
    ----------
    sequence_lengths:
        The sequence lengths m that were run.
    survival_probs:
        Dict mapping each sequence length to a 1D array of survival
        probabilities, one per random sequence. Shape: (n_sequences,).
    n_shots:
        Number of measurement shots per sequence.
    n_sequences:
        Number of random sequences per sequence length.
    backend_name:
        String identifier for the backend used.
    """

    sequence_lengths: list[int]
    survival_probs: dict[int, np.ndarray]
    n_shots: int
    n_sequences: int
    backend_name: str = "unknown"

    def mean_survival(self) -> dict[int, float]:
        """Return the mean survival probability at each sequence length."""
        return {m: float(np.mean(probs)) for m, probs in self.survival_probs.items()}

    def std_survival(self) -> dict[int, float]:
        """Return the std of survival probability at each sequence length."""
        return {m: float(np.std(probs)) for m, probs in self.survival_probs.items()}

    def summary(self) -> str:
        """Return a formatted summary table."""
        lines = [
            f"RB Results  ({self.n_sequences} sequences x {self.n_shots} shots each)",
            f"{'Length':>8}  {'Mean P(0)':>10}  {'Std P(0)':>10}",
            "-" * 34,
        ]
        means = self.mean_survival()
        stds = self.std_survival()
        for m in self.sequence_lengths:
            lines.append(f"{m:>8}  {means[m]:>10.4f}  {stds[m]:>10.4f}")
        return "\n".join(lines)

    def fit(self):
        """
        Run Bayesian inference on the RB decay curve.

        Returns an RBFit object with posterior samples over (A, p, B) in
        the model:  E[P(0) | m] = A * p^m + B

        This method is implemented in qbench.inference and will be available
        in day 3.
        """
        from qbench.inference.rb_fit import fit_rb
        return fit_rb(self)


class RandomizedBenchmarking:
    """
    Standard single-qubit randomized benchmarking protocol.

    Parameters
    ----------
    backend:
        Backend to run circuits on. Must have n_qubits == 1.
    n_qubits:
        Number of qubits. Currently only 1 is supported.
    seed:
        Optional random seed for reproducible sequence generation.
    """

    def __init__(
        self,
        backend: Backend,
        n_qubits: int = 1,
        seed: int | None = None,
    ) -> None:
        if n_qubits != 1:
            raise NotImplementedError("RB currently supports only n_qubits=1.")
        if backend.n_qubits != n_qubits:
            raise ValueError(
                f"Backend has {backend.n_qubits} qubits but protocol "
                f"requires {n_qubits}."
            )
        self._backend = backend
        self._n_qubits = n_qubits
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        sequence_lengths: Sequence[int],
        n_shots: int = 200,
        n_sequences: int = 30,
    ) -> RBResults:
        """
        Run the RB experiment.

        Parameters
        ----------
        sequence_lengths:
            List of Clifford sequence lengths m to benchmark at.
            Typical values: [1, 2, 4, 8, 16, 32, 64, 128].
        n_shots:
            Number of measurement shots per circuit.
        n_sequences:
            Number of independently sampled random sequences per length.
            More sequences give better statistics. 30-50 is typical.

        Returns
        -------
        RBResults
        """
        lengths = sorted(set(sequence_lengths))
        survival_probs: dict[int, np.ndarray] = {}

        for m in lengths:
            probs = np.zeros(n_sequences)
            for k in range(n_sequences):
                circuit = self._make_rb_circuit(m)
                result = self._backend.run(circuit, n_shots)
                probs[k] = result.survival_probability("0")
            survival_probs[m] = probs

        return RBResults(
            sequence_lengths=lengths,
            survival_probs=survival_probs,
            n_shots=n_shots,
            n_sequences=n_sequences,
            backend_name=type(self._backend).__name__,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_rb_circuit(self, length: int) -> Circuit:
        """
        Build a Circuit for a single RB sequence of the given length.

        The circuit consists of `length` random Clifford gates followed by
        a recovery gate, each decomposed into the backend's native gate set.
        """
        clifford_indices = random_clifford_sequence(length, self._rng)

        gates: list[tuple[str, tuple[int, ...]]] = []
        for cliff_idx in clifford_indices:
            decomp = CLIFFORD_DECOMPOSITIONS[cliff_idx]
            for gate_name in reversed(decomp):
                gates.append((gate_name, (0,)))

        return Circuit.from_gate_list(self._n_qubits, gates)
