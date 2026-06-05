"""
Tests for qbench.backends.

Covers:
- Circuit construction
- MeasurementResult survival probability
- SimulatedBackend noiseless and noisy execution
- Depolarizing channel limits
"""

import numpy as np
import pytest

from qbench.backends import Circuit, MeasurementResult, SimulatedBackend
from qbench.backends.base import Backend


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------

class TestCircuit:
    def test_from_gate_list(self):
        c = Circuit.from_gate_list(1, [("X", [0]), ("H", [0])])
        assert c.n_qubits == 1
        assert c.gates == (("X", (0,)), ("H", (0,)))

    def test_empty_circuit(self):
        c = Circuit.from_gate_list(1, [])
        assert c.gates == ()

    def test_frozen(self):
        c = Circuit.from_gate_list(1, [("X", [0])])
        with pytest.raises((AttributeError, TypeError)):
            c.n_qubits = 2  # type: ignore


# ---------------------------------------------------------------------------
# MeasurementResult
# ---------------------------------------------------------------------------

class TestMeasurementResult:
    def _make_result(self, counts):
        c = Circuit.from_gate_list(1, [])
        n_shots = sum(counts.values())
        return MeasurementResult(counts=counts, n_shots=n_shots, circuit=c)

    def test_survival_all_zero(self):
        r = self._make_result({"0": 100, "1": 0})
        assert r.survival_probability("0") == pytest.approx(1.0)

    def test_survival_all_one(self):
        r = self._make_result({"1": 100})
        assert r.survival_probability("0") == pytest.approx(0.0)

    def test_survival_mixed(self):
        r = self._make_result({"0": 75, "1": 25})
        assert r.survival_probability("0") == pytest.approx(0.75)

    def test_survival_missing_key(self):
        r = self._make_result({"1": 100})
        assert r.survival_probability("0") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SimulatedBackend
# ---------------------------------------------------------------------------

class TestSimulatedBackendInit:
    def test_valid_construction(self):
        b = SimulatedBackend(n_qubits=1, depolarizing_rate=1e-3)
        assert b.n_qubits == 1
        assert b.depolarizing_rate == pytest.approx(1e-3)

    def test_multi_qubit_raises(self):
        with pytest.raises(NotImplementedError):
            SimulatedBackend(n_qubits=2)

    def test_invalid_rate_high(self):
        with pytest.raises(ValueError):
            SimulatedBackend(depolarizing_rate=1.5)

    def test_invalid_rate_low(self):
        with pytest.raises(ValueError):
            SimulatedBackend(depolarizing_rate=-0.1)

    def test_gate_set(self):
        b = SimulatedBackend()
        assert "X" in b.gate_set
        assert "H" in b.gate_set
        assert "S" in b.gate_set


class TestSimulatedBackendNoiseless:
    """Noiseless backend should give deterministic results."""

    def setup_method(self):
        self.backend = SimulatedBackend(depolarizing_rate=0.0, seed=42)

    def _run(self, gates, n_shots=1000):
        c = Circuit.from_gate_list(1, gates)
        return self.backend.run(c, n_shots)

    def test_identity_stays_zero(self):
        result = self._run([("I", [0])])
        assert result.survival_probability("0") == pytest.approx(1.0)

    def test_x_gate_flips_to_one(self):
        result = self._run([("X", [0])])
        assert result.survival_probability("0") == pytest.approx(0.0)
        assert result.survival_probability("1") == pytest.approx(1.0)

    def test_x_x_returns_to_zero(self):
        result = self._run([("X", [0]), ("X", [0])])
        assert result.survival_probability("0") == pytest.approx(1.0)

    def test_h_gate_gives_half(self):
        # H|0> = |+>, measuring in Z basis gives 50/50
        result = self._run([("H", [0])], n_shots=10000)
        p0 = result.survival_probability("0")
        assert abs(p0 - 0.5) < 0.02  # statistical tolerance

    def test_z_gate_leaves_zero(self):
        # Z|0> = |0> (phase doesn't affect measurement in Z basis)
        result = self._run([("Z", [0])])
        assert result.survival_probability("0") == pytest.approx(1.0)

    def test_shot_count(self):
        result = self._run([("I", [0])], n_shots=500)
        assert result.n_shots == 500
        assert sum(result.counts.values()) == 500

    def test_unknown_gate_raises(self):
        c = Circuit.from_gate_list(1, [("UNKNOWN", [0])])
        with pytest.raises(ValueError, match="gate set"):
            self.backend.run(c, 100)

    def test_qubit_mismatch_raises(self):
        c = Circuit(n_qubits=2, gates=())
        with pytest.raises(ValueError):
            self.backend.run(c, 100)


class TestSimulatedBackendNoisy:
    """Noisy backend should degrade survival probability."""

    def test_high_noise_degrades_identity(self):
        # With very high depolarizing noise, repeated identity gates
        # should drive the state toward the maximally mixed state (p0 -> 0.5)
        backend = SimulatedBackend(depolarizing_rate=0.5, seed=0)
        gates = [("I", [0])] * 50
        c = Circuit.from_gate_list(1, gates)
        result = backend.run(c, n_shots=5000)
        p0 = result.survival_probability("0")
        # Should be well below 1.0 and heading toward 0.5
        assert p0 < 0.8

    def test_zero_noise_is_deterministic(self):
        backend = SimulatedBackend(depolarizing_rate=0.0, seed=0)
        c = Circuit.from_gate_list(1, [("X", [0])])
        result = backend.run(c, 200)
        assert result.survival_probability("1") == pytest.approx(1.0)

    def test_seed_reproducibility(self):
        c = Circuit.from_gate_list(1, [("H", [0])])
        b1 = SimulatedBackend(depolarizing_rate=1e-3, seed=7)
        b2 = SimulatedBackend(depolarizing_rate=1e-3, seed=7)
        r1 = b1.run(c, 500)
        r2 = b2.run(c, 500)
        assert r1.counts == r2.counts


class TestBackendIsAbstract:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Backend()  # type: ignore
