"""
qbench.backends.simulated
~~~~~~~~~~~~~~~~~~~~~~~~~

High-fidelity simulated backend using density matrix evolution.

The SimulatedBackend applies depolarizing noise after each gate, giving a
realistic model of incoherent gate errors. This is the primary backend used
for testing and demonstration.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import Backend, Circuit, MeasurementResult

try:
    import qutip as qt
    _QUTIP_AVAILABLE = True
except ImportError:
    _QUTIP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Gate definitions as unitary matrices (NumPy)
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

SINGLE_QUBIT_GATES: dict[str, np.ndarray] = {
    "I": _I,
    "X": _X,
    "Y": _Y,
    "Z": _Z,
    "H": _H,
    "S": _S,
    "Sdg": _Sdg,
    "T": _T,
}

_CLIFFORD_1Q: frozenset[str] = frozenset({"I", "X", "Y", "Z", "H", "S", "Sdg"})


class SimulatedBackend(Backend):
    """
    Density matrix simulator with depolarizing noise.

    Each gate application is followed by a depolarizing channel with
    rate `depolarizing_rate`. Measurement is in the computational basis.

    Parameters
    ----------
    n_qubits:
        Number of qubits. Currently only n_qubits=1 is fully implemented;
        multi-qubit support is scaffolded for future extension.
    depolarizing_rate:
        Per-gate depolarizing error probability p. After each gate U,
        the state rho is mapped to:
            (1 - p) * U rho U† + (p/3) * (X rho X + Y rho Y + Z rho Z)
        For p=0 the backend is noiseless.
    seed:
        Optional random seed for reproducible measurement sampling.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        depolarizing_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if n_qubits != 1:
            raise NotImplementedError(
                "SimulatedBackend currently supports only n_qubits=1. "
                "Multi-qubit support is planned."
            )
        if not (0.0 <= depolarizing_rate <= 1.0):
            raise ValueError(
                f"depolarizing_rate must be in [0, 1], got {depolarizing_rate}"
            )

        self._n_qubits = n_qubits
        self._depolarizing_rate = depolarizing_rate
        self._rng = np.random.default_rng(seed)

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def depolarizing_rate(self) -> float:
        return self._depolarizing_rate

    @property
    def gate_set(self) -> frozenset[str]:
        return _CLIFFORD_1Q

    def run(self, circuit: Circuit, n_shots: int) -> MeasurementResult:
        """
        Simulate the circuit and return sampled measurement counts.

        The qubit is initialised in |0><0|. Gates are applied sequentially
        with depolarizing noise after each. The final state is measured in
        the computational basis.
        """
        if circuit.n_qubits != self._n_qubits:
            raise ValueError(
                f"Circuit has {circuit.n_qubits} qubits but backend has "
                f"{self._n_qubits}."
            )

        rho = self._initial_state()

        for gate_name, qubit_indices in circuit.gates:
            if gate_name not in SINGLE_QUBIT_GATES:
                raise ValueError(
                    f"Gate '{gate_name}' is not in this backend's gate set."
                )
            U = SINGLE_QUBIT_GATES[gate_name]
            rho = self._apply_gate(rho, U)
            if self._depolarizing_rate > 0.0:
                rho = self._depolarize(rho, self._depolarizing_rate)

        counts = self._sample(rho, n_shots)
        return MeasurementResult(counts=counts, n_shots=n_shots, circuit=circuit)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initial_state(self) -> np.ndarray:
        """Return the |0><0| density matrix."""
        rho = np.zeros((2, 2), dtype=complex)
        rho[0, 0] = 1.0
        return rho

    def _apply_gate(self, rho: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Apply unitary U: rho -> U rho U†."""
        return U @ rho @ U.conj().T

    def _depolarize(self, rho: np.ndarray, p: float) -> np.ndarray:
        """
        Apply the single-qubit depolarizing channel with rate p.

            E(rho) = (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)

        This is equivalent to replacing rho with the mixture
            (1 - 4p/3) rho + (4p/3)(I/2)
        which is the standard form used in RB decay analysis.
        """
        pauli_sum = (
            _X @ rho @ _X
            + _Y @ rho @ _Y
            + _Z @ rho @ _Z
        )
        return (1 - p) * rho + (p / 3) * pauli_sum

    def _sample(self, rho: np.ndarray, n_shots: int) -> dict[str, int]:
        """
        Sample n_shots measurements in the computational basis.

        Returns a counts dictionary, e.g. {"0": 193, "1": 7}.
        """
        p0 = float(np.real(rho[0, 0]))
        p0 = np.clip(p0, 0.0, 1.0)

        n_zero = int(self._rng.binomial(n_shots, p0))
        n_one = n_shots - n_zero

        counts: dict[str, int] = {}
        if n_zero > 0:
            counts["0"] = n_zero
        if n_one > 0:
            counts["1"] = n_one
        return counts
