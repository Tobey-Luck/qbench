"""
qbench.backends
~~~~~~~~~~~~~~~

Backend interface and implementations.

A Backend is responsible for executing quantum circuits and returning
measurement outcomes. All protocols in qbench operate against the Backend
abstraction, making them independent of any specific simulator or hardware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Circuit:
    """
    Minimal circuit representation.

    A circuit is an ordered sequence of named gates applied to a set of
    qubits. This is intentionally lightweight -- qbench does not implement
    a full circuit IR. For hardware backends, this would be translated to
    the target framework's native circuit type.

    Attributes
    ----------
    n_qubits:
        Number of qubits in the circuit.
    gates:
        Ordered list of (gate_name, qubit_indices) pairs.
        Gate names are strings corresponding to keys in the backend's
        gate set (e.g. "X", "Y", "H", "CNOT").
    """

    n_qubits: int
    gates: tuple[tuple[str, tuple[int, ...]], ...] = field(default_factory=tuple)

    @classmethod
    def from_gate_list(
        cls,
        n_qubits: int,
        gates: Sequence[tuple[str, Sequence[int]]],
    ) -> "Circuit":
        """Construct a Circuit from a list of (gate_name, qubits) pairs."""
        return cls(
            n_qubits=n_qubits,
            gates=tuple((name, tuple(qubits)) for name, qubits in gates),
        )


@dataclass
class MeasurementResult:
    """
    Raw measurement outcomes from a single circuit execution.

    Attributes
    ----------
    counts:
        Dictionary mapping bitstring outcomes (e.g. "00", "01") to
        integer counts. Bitstrings are ordered qubit-0 first.
    n_shots:
        Total number of shots. Equal to sum(counts.values()).
    circuit:
        The circuit that produced these results.
    """

    counts: dict[str, int]
    n_shots: int
    circuit: Circuit

    def survival_probability(self, target_state: str = "0" * 1) -> float:
        """
        Return the fraction of shots that produced `target_state`.

        Parameters
        ----------
        target_state:
            Bitstring to treat as the 'survival' outcome.
            Defaults to the all-zeros state.
        """
        return self.counts.get(target_state, 0) / self.n_shots


class Backend(ABC):
    """
    Abstract base class for all qbench backends.

    A Backend exposes two things:
      1. A gate set: the named single- and two-qubit gates it supports.
      2. A run() method: execute a Circuit for n_shots and return counts.

    Subclasses must implement `gate_set` and `run`. They may optionally
    override `run_batch` for more efficient batched execution.
    """

    @property
    @abstractmethod
    def n_qubits(self) -> int:
        """Number of qubits supported by this backend."""
        ...

    @property
    @abstractmethod
    def gate_set(self) -> frozenset[str]:
        """
        The set of gate names this backend supports.

        Gate names are strings (e.g. "X", "Y", "H", "S", "CNOT").
        Protocols will only construct circuits using gates in this set.
        """
        ...

    @abstractmethod
    def run(self, circuit: Circuit, n_shots: int) -> MeasurementResult:
        """
        Execute a circuit and return measurement outcomes.

        Parameters
        ----------
        circuit:
            The circuit to execute. All gate names must be in self.gate_set.
        n_shots:
            Number of times to execute the circuit.

        Returns
        -------
        MeasurementResult
            Raw counts over all n_shots executions.
        """
        ...

    def run_batch(
        self,
        circuits: Sequence[Circuit],
        n_shots: int,
    ) -> list[MeasurementResult]:
        """
        Execute a batch of circuits.

        Default implementation calls run() sequentially. Subclasses may
        override this for parallelism or hardware-native batching.

        Parameters
        ----------
        circuits:
            Circuits to execute.
        n_shots:
            Number of shots per circuit.
        """
        return [self.run(c, n_shots) for c in circuits]
