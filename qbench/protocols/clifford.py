"""
qbench.protocols.clifford
~~~~~~~~~~~~~~~~~~~~~~~~~

Single-qubit Clifford group: elements, composition, and random sequence
generation for randomized benchmarking.

The single-qubit Clifford group has 24 elements. Each is represented as a
2x2 unitary matrix and identified by an integer index in [0, 24).
"""

from __future__ import annotations

import numpy as np

_I   = np.eye(2, dtype=complex)
_X   = np.array([[0, 1], [1, 0]], dtype=complex)
_Y   = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z   = np.array([[1, 0], [0, -1]], dtype=complex)
_H   = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S   = np.array([[1, 0], [0, 1j]], dtype=complex)
_Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)

_NAMED_GATES: dict[str, np.ndarray] = {
    "I": _I, "X": _X, "Y": _Y, "Z": _Z,
    "H": _H, "S": _S, "Sdg": _Sdg,
}


def _matrices_equal_up_to_phase(A: np.ndarray, B: np.ndarray, atol: float = 1e-9) -> bool:
    """Return True if A == e^{i*phi} * B for some scalar phi."""
    # Find first non-tiny element of B to extract phase
    for i in range(2):
        for j in range(2):
            if abs(B[i, j]) > atol:
                phase = A[i, j] / B[i, j]
                if abs(abs(phase) - 1.0) > atol:
                    return False
                return np.allclose(A, phase * B, atol=atol)
    return False  # B is zero matrix


def _build_clifford_group() -> list[np.ndarray]:
    """
    Enumerate all 24 single-qubit Clifford matrices via closure under H and S.
    Elements are stored as-is (not phase-normalized) to keep them unitary.
    """
    group: list[np.ndarray] = []
    queue: list[np.ndarray] = [_I.copy()]

    while queue:
        U = queue.pop(0)
        # Check if U is already in the group up to global phase
        already_present = any(_matrices_equal_up_to_phase(U, V) for V in group)
        if already_present:
            continue
        group.append(U)
        for G in [_H, _S]:
            queue.append(G @ U)
            queue.append(U @ G)

    # Trim to exactly 24 (the BFS may produce duplicates in the queue)
    assert len(group) == 24, f"Expected 24 Cliffords, got {len(group)}"
    return group


CLIFFORD_GROUP: list[np.ndarray] = _build_clifford_group()


def _find_clifford_index(U: np.ndarray) -> int:
    """Find the index of U in CLIFFORD_GROUP (up to global phase)."""
    for k, C in enumerate(CLIFFORD_GROUP):
        if _matrices_equal_up_to_phase(U, C):
            return k
    raise RuntimeError("Matrix not found in Clifford group.")


# Precompute inverse table
_CLIFFORD_INVERSE: list[int] = [
    _find_clifford_index(C.conj().T) for C in CLIFFORD_GROUP
]


def clifford_inverse(index: int) -> int:
    """Return the index of the inverse of Clifford element `index`."""
    return _CLIFFORD_INVERSE[index]


def compose_cliffords(i: int, j: int) -> int:
    """Return index of C_i @ C_j."""
    return _find_clifford_index(CLIFFORD_GROUP[i] @ CLIFFORD_GROUP[j])


def random_clifford_sequence(length: int, rng: np.random.Generator) -> list[int]:
    """
    Generate a random Clifford sequence with a recovery gate appended.

    Returns length + 1 indices. The recovery gate ensures the full sequence
    acts as the identity on |0>.
    """
    indices = rng.integers(0, 24, size=length).tolist()
    net = 0  # identity
    for idx in indices:
        net = compose_cliffords(idx, net)
    recovery = clifford_inverse(net)
    return indices + [recovery]


def _build_decompositions() -> list[list[str]]:
    """
    BFS: for each Clifford find shortest decomposition into named gates.
    """
    decompositions: dict[int, list[str]] = {}
    identity_idx = _find_clifford_index(_I)
    decompositions[identity_idx] = []
    queue = [(identity_idx, [])]

    while queue and len(decompositions) < 24:
        current_idx, current_seq = queue.pop(0)
        for gate_name, gate_mat in _NAMED_GATES.items():
            new_mat = gate_mat @ CLIFFORD_GROUP[current_idx]
            new_idx = _find_clifford_index(new_mat)
            if new_idx not in decompositions:
                new_seq = [gate_name] + current_seq
                decompositions[new_idx] = new_seq
                queue.append((new_idx, new_seq))

    return [decompositions[i] for i in range(24)]


CLIFFORD_DECOMPOSITIONS: list[list[str]] = _build_decompositions()
