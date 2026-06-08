"""
Tests for qbench.protocols: Clifford group and randomized benchmarking.

Covers:
- Clifford group closure, inverse, and composition
- Sequence generation and recovery gate correctness
- RB experiment loop and RBResults structure
- Noiseless backend survival probability = 1.0
- Noisy backend survival probability < 1.0 and decays with length
"""

import numpy as np
import pytest

from qbench.backends import SimulatedBackend
from qbench.protocols import RandomizedBenchmarking, RBResults
from qbench.protocols.clifford import (
    CLIFFORD_GROUP,
    CLIFFORD_DECOMPOSITIONS,
    clifford_inverse,
    compose_cliffords,
    random_clifford_sequence,
)


# ---------------------------------------------------------------------------
# Clifford group tests
# ---------------------------------------------------------------------------

class TestCliffordGroup:
    def test_group_size(self):
        assert len(CLIFFORD_GROUP) == 24

    def test_all_unitary(self):
        for i, C in enumerate(CLIFFORD_GROUP):
            product = C @ C.conj().T
            assert np.allclose(product, np.eye(2), atol=1e-10), \
                f"Clifford {i} is not unitary"

    def test_closure(self):
        # Composition of any two Cliffords should be in the group
        for i in range(24):
            for j in range(5):  # sample a few to keep test fast
                result = compose_cliffords(i, j)
                assert 0 <= result < 24

    def test_inverse_is_inverse(self):
        I = np.eye(2, dtype=complex)
        for i in range(24):
            inv_idx = clifford_inverse(i)
            product = CLIFFORD_GROUP[i] @ CLIFFORD_GROUP[inv_idx]
            # Should be identity up to global phase
            ratio = product / product[0, 0]
            assert np.allclose(ratio, I, atol=1e-10), \
                f"Clifford {i} * Clifford {inv_idx} != I"

    def test_decompositions_length(self):
        assert len(CLIFFORD_DECOMPOSITIONS) == 24

    def test_decompositions_implement_cliffords(self):
        gate_matrices = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
            "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            "S": np.array([[1, 0], [0, 1j]], dtype=complex),
            "Sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
        }

        def phase_normalize(U):
            flat = U.ravel()
            idx = np.argmax(np.abs(flat))
            phase = flat[idx] / abs(flat[idx])
            return U / phase

        for i, decomp in enumerate(CLIFFORD_DECOMPOSITIONS):
            U = np.eye(2, dtype=complex)
            for gate_name in reversed(decomp):
                U = gate_matrices[gate_name] @ U
            U_norm = phase_normalize(U)
            C_norm = phase_normalize(CLIFFORD_GROUP[i])
            assert np.allclose(U_norm, C_norm, atol=1e-10), \
                f"Decomposition of Clifford {i} does not match"


class TestCliffordSequence:
    def test_sequence_length(self):
        rng = np.random.default_rng(0)
        for m in [1, 5, 10, 20]:
            seq = random_clifford_sequence(m, rng)
            assert len(seq) == m + 1

    def test_sequence_is_identity(self):
        """The full sequence (including recovery) should implement identity."""
        rng = np.random.default_rng(42)

        def phase_normalize(U):
            flat = U.ravel()
            idx = np.argmax(np.abs(flat))
            phase = flat[idx] / abs(flat[idx])
            return U / phase

        for _ in range(20):
            m = np.random.randint(1, 15)
            seq = random_clifford_sequence(m, rng)
            U = np.eye(2, dtype=complex)
            for idx in seq:
                U = CLIFFORD_GROUP[idx] @ U
            U_norm = phase_normalize(U)
            I_norm = phase_normalize(np.eye(2, dtype=complex))
            assert np.allclose(U_norm, I_norm, atol=1e-8), \
                f"Sequence of length {m} is not identity"

    def test_all_indices_valid(self):
        rng = np.random.default_rng(1)
        seq = random_clifford_sequence(10, rng)
        for idx in seq:
            assert 0 <= idx < 24


# ---------------------------------------------------------------------------
# RandomizedBenchmarking tests
# ---------------------------------------------------------------------------

class TestRandomizedBenchmarkingInit:
    def test_valid_construction(self):
        backend = SimulatedBackend(n_qubits=1)
        rb = RandomizedBenchmarking(backend=backend, n_qubits=1, seed=0)
        assert rb is not None

    def test_multi_qubit_raises(self):
        backend = SimulatedBackend(n_qubits=1)
        with pytest.raises(NotImplementedError):
            RandomizedBenchmarking(backend=backend, n_qubits=2)


class TestRBNoiseless:
    """Noiseless backend: survival probability should be ~1.0."""

    def setup_method(self):
        self.backend = SimulatedBackend(depolarizing_rate=0.0, seed=0)
        self.rb = RandomizedBenchmarking(backend=self.backend, seed=42)

    def test_noiseless_survival_is_one(self):
        results = self.rb.run(
            sequence_lengths=[1, 2, 4, 8],
            n_shots=200,
            n_sequences=10,
        )
        for m, probs in results.survival_probs.items():
            mean_p = np.mean(probs)
            assert mean_p == pytest.approx(1.0, abs=0.05), \
                f"Noiseless RB at length {m}: mean P(0) = {mean_p:.3f}"

    def test_results_structure(self):
        results = self.rb.run(
            sequence_lengths=[1, 4, 16],
            n_shots=100,
            n_sequences=5,
        )
        assert results.sequence_lengths == [1, 4, 16]
        for m in [1, 4, 16]:
            assert m in results.survival_probs
            assert len(results.survival_probs[m]) == 5

    def test_results_n_shots(self):
        results = self.rb.run([2], n_shots=300, n_sequences=3)
        assert results.n_shots == 300

    def test_results_n_sequences(self):
        results = self.rb.run([2], n_shots=100, n_sequences=7)
        assert results.n_sequences == 7

    def test_sequence_lengths_sorted(self):
        results = self.rb.run([16, 1, 4], n_shots=50, n_sequences=3)
        assert results.sequence_lengths == [1, 4, 16]


class TestRBNoisy:
    """Noisy backend: survival should degrade with sequence length."""

    def test_decay_with_length(self):
        backend = SimulatedBackend(depolarizing_rate=5e-3, seed=0)
        rb = RandomizedBenchmarking(backend=backend, seed=0)
        results = rb.run(
            sequence_lengths=[1, 64],
            n_shots=500,
            n_sequences=20,
        )
        mean_short = np.mean(results.survival_probs[1])
        mean_long = np.mean(results.survival_probs[64])
        assert mean_short > mean_long, \
            f"Expected decay: short={mean_short:.3f}, long={mean_long:.3f}"

    def test_survival_in_bounds(self):
        backend = SimulatedBackend(depolarizing_rate=1e-2, seed=1)
        rb = RandomizedBenchmarking(backend=backend, seed=1)
        results = rb.run([1, 8, 32], n_shots=200, n_sequences=10)
        for m, probs in results.survival_probs.items():
            assert np.all(probs >= 0.0), f"Negative prob at length {m}"
            assert np.all(probs <= 1.0), f"Prob > 1 at length {m}"


# ---------------------------------------------------------------------------
# RBResults tests
# ---------------------------------------------------------------------------

class TestRBResults:
    def _make_results(self) -> RBResults:
        probs = {
            1: np.array([0.98, 0.99, 0.97]),
            4: np.array([0.90, 0.92, 0.91]),
            16: np.array([0.75, 0.78, 0.76]),
        }
        return RBResults(
            sequence_lengths=[1, 4, 16],
            survival_probs=probs,
            n_shots=200,
            n_sequences=3,
        )

    def test_mean_survival(self):
        r = self._make_results()
        means = r.mean_survival()
        assert means[1] == pytest.approx(np.mean([0.98, 0.99, 0.97]))
        assert means[16] == pytest.approx(np.mean([0.75, 0.78, 0.76]))

    def test_std_survival(self):
        r = self._make_results()
        stds = r.std_survival()
        assert stds[4] == pytest.approx(np.std([0.90, 0.92, 0.91]))

    def test_summary_is_string(self):
        r = self._make_results()
        s = r.summary()
        assert isinstance(s, str)
        assert "16" in s
