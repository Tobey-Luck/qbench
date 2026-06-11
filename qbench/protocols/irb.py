"""
qbench.protocols.irb
~~~~~~~~~~~~~~~~~~~~

Interleaved randomized benchmarking (IRB) for single-qubit gates.

Protocol
--------
IRB estimates the error rate of a specific target gate G by running two
experiments:

1. Standard RB (already implemented in qbench.protocols.rb)
2. Interleaved RB: the same random Clifford sequences, but with G inserted
   after every random Clifford gate.

The two decay parameters p_rb and p_irb are then compared to isolate the
gate-specific error rate:

    r_gate = (d - 1) / d * (1 - p_irb / p_rb)

where d = 2^n_qubits (d = 2 for a single qubit).

The gate G must be an element of the Clifford group. Its index in
CLIFFORD_GROUP and its decomposition into native gates are looked up
automatically from the gate name.

References
----------
Magesan et al., PRL 109, 080505 (2012) — original IRB paper
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from qbench.backends.base import Backend, Circuit
from qbench.protocols.clifford import (
    CLIFFORD_DECOMPOSITIONS,
    CLIFFORD_GROUP,
    _matrices_equal_up_to_phase,
    _NAMED_GATES,
    random_clifford_sequence,
    compose_cliffords,
    clifford_inverse,
)
from qbench.protocols.rb import RBResults


# ---------------------------------------------------------------------------
# Gate lookup: map a named gate to its Clifford index
# ---------------------------------------------------------------------------

def _gate_name_to_clifford_index(gate_name: str) -> int:
    """
    Find the index of a named gate in CLIFFORD_GROUP.

    Parameters
    ----------
    gate_name : str
        A gate name from the backend's gate set (e.g. "X", "H", "S").

    Returns
    -------
    int
        Index into CLIFFORD_GROUP.

    Raises
    ------
    ValueError
        If gate_name is not in the named gate set or not a Clifford.
    """
    if gate_name not in _NAMED_GATES:
        raise ValueError(
            f"Gate '{gate_name}' is not a recognised named gate. "
            f"Available: {sorted(_NAMED_GATES.keys())}"
        )
    U = _NAMED_GATES[gate_name]
    for k, C in enumerate(CLIFFORD_GROUP):
        if _matrices_equal_up_to_phase(U, C):
            return k
    raise ValueError(
        f"Gate '{gate_name}' is not an element of the single-qubit Clifford group."
    )


# ---------------------------------------------------------------------------
# IRB results container
# ---------------------------------------------------------------------------

@dataclass
class IRBResults:
    """
    Raw results from an interleaved randomized benchmarking experiment.

    Contains both the reference RB results and the interleaved RB results
    so that they can be jointly fit.

    Attributes
    ----------
    rb_results : RBResults
        Standard RB decay data (reference experiment).
    irb_results : RBResults
        Interleaved RB decay data (target gate interleaved).
    target_gate : str
        Name of the gate that was interleaved.
    """

    rb_results: RBResults
    irb_results: RBResults
    target_gate: str

    def summary(self) -> str:
        rb_means = self.rb_results.mean_survival()
        irb_means = self.irb_results.mean_survival()
        lengths = self.rb_results.sequence_lengths

        lines = [
            f"IRB Results  (target gate: {self.target_gate})",
            f"  {self.rb_results.n_sequences} sequences x "
            f"{self.rb_results.n_shots} shots each",
            "",
            f"  {'Length':>8}  {'RB P(0)':>10}  {'IRB P(0)':>10}",
            "  " + "-" * 34,
        ]
        for m in lengths:
            lines.append(
                f"  {m:>8}  {rb_means[m]:>10.4f}  {irb_means[m]:>10.4f}"
            )
        return "\n".join(lines)

    def fit(self):
        """
        Run Bayesian inference jointly on the RB and IRB decay curves.

        Returns an IRBFit with posteriors over p_rb, p_irb, and the
        derived gate error rate r_gate.
        """
        from qbench.inference.irb_fit import fit_irb
        return fit_irb(self)


# ---------------------------------------------------------------------------
# IRB protocol
# ---------------------------------------------------------------------------

class InterleavedRandomizedBenchmarking:
    """
    Interleaved randomized benchmarking for a single target gate.

    Runs both a reference RB experiment and an interleaved RB experiment
    using the same random sequences (up to re-seeding), returning an
    IRBResults object for joint inference.

    Parameters
    ----------
    backend : Backend
        Backend to execute circuits on.
    target_gate : str
        Name of the gate to characterise. Must be in the backend's gate set
        and must be a single-qubit Clifford gate.
    n_qubits : int
        Number of qubits. Currently only 1 is supported.
    seed : int or None
        Random seed for reproducible sequence generation.
    """

    def __init__(
        self,
        backend: Backend,
        target_gate: str,
        n_qubits: int = 1,
        seed: int | None = None,
    ) -> None:
        if n_qubits != 1:
            raise NotImplementedError("IRB currently supports only n_qubits=1.")
        if backend.n_qubits != n_qubits:
            raise ValueError(
                f"Backend has {backend.n_qubits} qubits but protocol "
                f"requires {n_qubits}."
            )
        if target_gate not in backend.gate_set:
            raise ValueError(
                f"Gate '{target_gate}' is not in the backend's gate set: "
                f"{backend.gate_set}"
            )

        self._backend = backend
        self._target_gate = target_gate
        self._target_clifford_idx = _gate_name_to_clifford_index(target_gate)
        self._n_qubits = n_qubits
        self._rng = np.random.default_rng(seed)

    @property
    def target_gate(self) -> str:
        return self._target_gate

    def run(
        self,
        sequence_lengths: Sequence[int],
        n_shots: int = 200,
        n_sequences: int = 30,
    ) -> IRBResults:
        """
        Run the IRB experiment.

        For each sequence length, generates n_sequences random Clifford
        sequences and runs both the standard and interleaved versions.

        Parameters
        ----------
        sequence_lengths : sequence of int
            Clifford sequence lengths to benchmark at.
        n_shots : int
            Measurement shots per circuit.
        n_sequences : int
            Number of random sequences per length.

        Returns
        -------
        IRBResults
        """
        lengths = sorted(set(sequence_lengths))

        rb_probs: dict[int, np.ndarray] = {}
        irb_probs: dict[int, np.ndarray] = {}

        for m in lengths:
            rb_p = np.zeros(n_sequences)
            irb_p = np.zeros(n_sequences)

            for k in range(n_sequences):
                # Generate a single random sequence, use it for both RB and IRB
                clifford_indices = self._rng.integers(0, 24, size=m).tolist()

                # Standard RB circuit
                rb_circuit = self._make_rb_circuit(clifford_indices)
                rb_result = self._backend.run(rb_circuit, n_shots)
                rb_p[k] = rb_result.survival_probability("0")

                # Interleaved RB circuit (same random Cliffords + target gate)
                irb_circuit = self._make_irb_circuit(clifford_indices)
                irb_result = self._backend.run(irb_circuit, n_shots)
                irb_p[k] = irb_result.survival_probability("0")

            rb_probs[m] = rb_p
            irb_probs[m] = irb_p

        rb_results = RBResults(
            sequence_lengths=lengths,
            survival_probs=rb_probs,
            n_shots=n_shots,
            n_sequences=n_sequences,
            backend_name=type(self._backend).__name__,
        )
        irb_results = RBResults(
            sequence_lengths=lengths,
            survival_probs=irb_probs,
            n_shots=n_shots,
            n_sequences=n_sequences,
            backend_name=type(self._backend).__name__,
        )

        return IRBResults(
            rb_results=rb_results,
            irb_results=irb_results,
            target_gate=self._target_gate,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_rb_circuit(self, clifford_indices: list[int]) -> Circuit:
        """
        Build a standard RB circuit from a list of Clifford indices.
        Appends the recovery gate.
        """
        net = 0  # identity
        for idx in clifford_indices:
            net = compose_cliffords(idx, net)
        recovery = clifford_inverse(net)

        all_indices = clifford_indices + [recovery]
        gates: list[tuple[str, tuple[int, ...]]] = []
        for cliff_idx in all_indices:
            for gate_name in reversed(CLIFFORD_DECOMPOSITIONS[cliff_idx]):
                gates.append((gate_name, (0,)))

        return Circuit.from_gate_list(self._n_qubits, gates)

    def _make_irb_circuit(self, clifford_indices: list[int]) -> Circuit:
        """
        Build an interleaved RB circuit.

        Structure: [C_1, G, C_2, G, ..., C_m, G, recovery]

        The recovery gate is computed so that the full sequence is the
        identity, accounting for the interleaved G gates.
        """
        G_idx = self._target_clifford_idx

        # Net unitary: product of (C_i * G) for i=1..m, then recovery
        # net = G * C_m * G * C_{m-1} * ... * G * C_1
        net = 0  # identity
        for idx in clifford_indices:
            net = compose_cliffords(idx, net)   # apply C_i
            net = compose_cliffords(G_idx, net)  # apply G

        recovery = clifford_inverse(net)

        # Build gate list: C_1, G, C_2, G, ..., C_m, G, recovery
        gates: list[tuple[str, tuple[int, ...]]] = []
        for cliff_idx in clifford_indices:
            for gate_name in reversed(CLIFFORD_DECOMPOSITIONS[cliff_idx]):
                gates.append((gate_name, (0,)))
            # Interleaved target gate
            gates.append((self._target_gate, (0,)))

        # Recovery gate
        for gate_name in reversed(CLIFFORD_DECOMPOSITIONS[recovery]):
            gates.append((gate_name, (0,)))

        return Circuit.from_gate_list(self._n_qubits, gates)