"""
Tests for qbench.protocols.irb and qbench.inference.irb_fit

Covers:
- Gate-to-Clifford index lookup
- IRB circuit construction (noiseless recovery)
- IRBResults structure and summary
- InterleavedRandomizedBenchmarking init and run
- Noiseless IRB survival == 1.0
- Noisy IRB decays faster than RB
- IRBFit posterior structure and derived quantities
- fit_irb end-to-end gate error rate recovery
"""

import numpy as np
import pytest

from qbench.backends import SimulatedBackend
from qbench.inference.irb_fit import IRBFit, fit_irb, log_prior_irb, log_likelihood_irb
from qbench.protocols.irb import (
    InterleavedRandomizedBenchmarking,
    IRBResults,
    _gate_name_to_clifford_index,
)
from qbench.protocols.rb import RBResults


# ---------------------------------------------------------------------------
# Gate lookup
# ---------------------------------------------------------------------------

class TestGateNameToCliffordIndex:
    def test_identity(self):
        idx = _gate_name_to_clifford_index("I")
        assert 0 <= idx < 24

    def test_x_gate(self):
        idx = _gate_name_to_clifford_index("X")
        assert 0 <= idx < 24

    def test_h_gate(self):
        idx = _gate_name_to_clifford_index("H")
        assert 0 <= idx < 24

    def test_all_named_gates_are_cliffords(self):
        for gate in ["I", "X", "Y", "Z", "H", "S", "Sdg"]:
            idx = _gate_name_to_clifford_index(gate)
            assert 0 <= idx < 24

    def test_unknown_gate_raises(self):
        with pytest.raises(ValueError, match="not a recognised"):
            _gate_name_to_clifford_index("CNOT")

    def test_different_gates_have_different_indices(self):
        idx_x = _gate_name_to_clifford_index("X")
        idx_h = _gate_name_to_clifford_index("H")
        assert idx_x != idx_h


# ---------------------------------------------------------------------------
# InterleavedRandomizedBenchmarking init
# ---------------------------------------------------------------------------

class TestIRBInit:
    def test_valid_construction(self):
        backend = SimulatedBackend(n_qubits=1)
        irb = InterleavedRandomizedBenchmarking(backend=backend, target_gate="X")
        assert irb.target_gate == "X"

    def test_multi_qubit_raises(self):
        backend = SimulatedBackend(n_qubits=1)
        with pytest.raises(NotImplementedError):
            InterleavedRandomizedBenchmarking(backend=backend, target_gate="X", n_qubits=2)

    def test_gate_not_in_gate_set_raises(self):
        backend = SimulatedBackend(n_qubits=1)
        with pytest.raises(ValueError, match="gate set"):
            InterleavedRandomizedBenchmarking(backend=backend, target_gate="T")

    def test_all_clifford_gates_accepted(self):
        backend = SimulatedBackend(n_qubits=1)
        for gate in ["X", "Y", "Z", "H", "S", "Sdg"]:
            irb = InterleavedRandomizedBenchmarking(backend=backend, target_gate=gate)
            assert irb.target_gate == gate


# ---------------------------------------------------------------------------
# Noiseless IRB: survival should be ~1.0
# ---------------------------------------------------------------------------

class TestIRBNoiseless:
    def setup_method(self):
        self.backend = SimulatedBackend(depolarizing_rate=0.0, seed=0)

    def _run(self, gate: str, lengths=None, n_sequences=10):
        if lengths is None:
            lengths = [1, 2, 4, 8]
        irb = InterleavedRandomizedBenchmarking(
            backend=self.backend, target_gate=gate, seed=42
        )
        return irb.run(lengths, n_shots=200, n_sequences=n_sequences)

    def test_rb_survival_is_one(self):
        results = self._run("X")
        for m, probs in results.rb_results.survival_probs.items():
            assert np.mean(probs) == pytest.approx(1.0, abs=0.05), \
                f"RB at m={m}: mean={np.mean(probs):.3f}"

    def test_irb_survival_is_one_x_gate(self):
        results = self._run("X")
        for m, probs in results.irb_results.survival_probs.items():
            assert np.mean(probs) == pytest.approx(1.0, abs=0.05), \
                f"IRB(X) at m={m}: mean={np.mean(probs):.3f}"

    def test_irb_survival_is_one_h_gate(self):
        results = self._run("H")
        for m, probs in results.irb_results.survival_probs.items():
            assert np.mean(probs) == pytest.approx(1.0, abs=0.05), \
                f"IRB(H) at m={m}: mean={np.mean(probs):.3f}"

    def test_irb_survival_is_one_s_gate(self):
        results = self._run("S")
        for m, probs in results.irb_results.survival_probs.items():
            assert np.mean(probs) == pytest.approx(1.0, abs=0.05), \
                f"IRB(S) at m={m}: mean={np.mean(probs):.3f}"

    def test_results_structure(self):
        results = self._run("X", lengths=[1, 4, 16])
        assert results.target_gate == "X"
        assert results.rb_results.sequence_lengths == [1, 4, 16]
        assert results.irb_results.sequence_lengths == [1, 4, 16]

    def test_n_sequences(self):
        results = self._run("X", n_sequences=7)
        for m in results.rb_results.sequence_lengths:
            assert len(results.rb_results.survival_probs[m]) == 7
            assert len(results.irb_results.survival_probs[m]) == 7


# ---------------------------------------------------------------------------
# Noisy IRB: IRB should decay faster than RB
# ---------------------------------------------------------------------------

class TestIRBNoisy:
    def test_irb_decays_faster_than_rb(self):
        """With noise, the interleaved experiment should decay faster."""
        backend = SimulatedBackend(depolarizing_rate=5e-3, seed=0)
        irb = InterleavedRandomizedBenchmarking(
            backend=backend, target_gate="X", seed=0
        )
        results = irb.run(
            sequence_lengths=[1, 32],
            n_shots=500,
            n_sequences=30,
        )
        rb_decay = (
            np.mean(results.rb_results.survival_probs[1])
            - np.mean(results.rb_results.survival_probs[32])
        )
        irb_decay = (
            np.mean(results.irb_results.survival_probs[1])
            - np.mean(results.irb_results.survival_probs[32])
        )
        assert irb_decay >= rb_decay, (
            f"Expected IRB to decay at least as fast as RB: "
            f"rb_decay={rb_decay:.3f}, irb_decay={irb_decay:.3f}"
        )

    def test_survival_probabilities_in_bounds(self):
        backend = SimulatedBackend(depolarizing_rate=1e-2, seed=1)
        irb = InterleavedRandomizedBenchmarking(backend=backend, target_gate="H", seed=1)
        results = irb.run([1, 8, 32], n_shots=200, n_sequences=10)
        for probs in results.irb_results.survival_probs.values():
            assert np.all(probs >= 0.0)
            assert np.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# IRBResults
# ---------------------------------------------------------------------------

class TestIRBResults:
    def _make_results(self) -> IRBResults:
        def make_rb(lengths, means):
            rng = np.random.default_rng(0)
            return RBResults(
                sequence_lengths=lengths,
                survival_probs={m: rng.uniform(mu - 0.02, mu + 0.02, 5)
                                for m, mu in zip(lengths, means)},
                n_shots=200,
                n_sequences=5,
            )
        lengths = [1, 4, 16]
        rb = make_rb(lengths, [0.95, 0.85, 0.70])
        irb = make_rb(lengths, [0.93, 0.80, 0.60])
        return IRBResults(rb_results=rb, irb_results=irb, target_gate="X")

    def test_summary_is_string(self):
        r = self._make_results()
        s = r.summary()
        assert isinstance(s, str)
        assert "X" in s

    def test_summary_contains_lengths(self):
        r = self._make_results()
        s = r.summary()
        assert "16" in s

    def test_target_gate_stored(self):
        r = self._make_results()
        assert r.target_gate == "X"


# ---------------------------------------------------------------------------
# log_prior_irb
# ---------------------------------------------------------------------------

class TestLogPriorIRB:
    def _valid(self):
        return np.array([0.45, 0.43, 0.97, 0.94, 0.25])

    def test_valid_params(self):
        assert np.isfinite(log_prior_irb(self._valid()))

    def test_p_rb_out_of_bounds(self):
        theta = self._valid().copy(); theta[2] = 1.1
        assert log_prior_irb(theta) == -np.inf

    def test_p_irb_greater_than_p_rb(self):
        theta = self._valid().copy(); theta[3] = theta[2] + 0.01
        assert log_prior_irb(theta) == -np.inf

    def test_negative_A_rb(self):
        theta = self._valid().copy(); theta[0] = -0.1
        assert log_prior_irb(theta) == -np.inf

    def test_A_rb_plus_B_exceeds_one(self):
        theta = self._valid().copy(); theta[0] = 0.8; theta[4] = 0.3
        assert log_prior_irb(theta) == -np.inf

    def test_A_irb_plus_B_exceeds_one(self):
        theta = self._valid().copy(); theta[1] = 0.8; theta[4] = 0.3
        assert log_prior_irb(theta) == -np.inf


# ---------------------------------------------------------------------------
# IRBFit container
# ---------------------------------------------------------------------------

class TestIRBFit:
    def _make_fit(self, n_samples=500) -> IRBFit:
        rng = np.random.default_rng(0)
        samples = np.column_stack([
            rng.normal(0.45, 0.02, n_samples),   # A_rb
            rng.normal(0.43, 0.02, n_samples),   # A_irb
            rng.normal(0.970, 0.003, n_samples), # p_rb
            rng.normal(0.955, 0.004, n_samples), # p_irb
            rng.normal(0.250, 0.010, n_samples), # B
        ])
        return IRBFit(
            samples=samples,
            target_gate="X",
            n_walkers=32,
            n_steps=200,
            acceptance_fraction=0.35,
        )

    def test_samples_shape(self):
        fit = self._make_fit()
        assert fit.samples.shape == (500, 5)

    def test_p_rb_samples(self):
        fit = self._make_fit()
        assert fit.p_rb_samples.shape == (500,)

    def test_p_irb_samples(self):
        fit = self._make_fit()
        assert fit.p_irb_samples.shape == (500,)

    def test_gate_error_rate_formula(self):
        fit = self._make_fit()
        expected = 0.5 * (1.0 - fit.p_irb_samples / fit.p_rb_samples)
        assert np.allclose(fit.gate_error_rate_samples, expected)

    def test_gate_error_rate_positive(self):
        fit = self._make_fit()
        assert np.all(fit.gate_error_rate_samples >= 0)

    def test_posterior_mean_gate_error(self):
        fit = self._make_fit()
        mean = fit.posterior_mean("gate_error_rate")
        assert 0.0 < mean < 0.1

    def test_credible_interval_ordering(self):
        fit = self._make_fit()
        lo, hi = fit.credible_interval("gate_error_rate")
        assert lo < hi

    def test_invalid_param_raises(self):
        fit = self._make_fit()
        with pytest.raises(ValueError, match="Unknown parameter"):
            fit.posterior_mean("nonsense")

    def test_summary_contains_gate_name(self):
        fit = self._make_fit()
        s = fit.summary()
        assert "X" in s
        assert "gate_error_rate" in s


# ---------------------------------------------------------------------------
# fit_irb end-to-end
# ---------------------------------------------------------------------------

class TestFitIRB:
    def _make_synthetic_irb_results(
        self,
        p_rb: float = 0.97,
        gate_error: float = 0.008,
        seed: int = 7,
    ) -> "IRBResults":
        from qbench.protocols.irb import IRBResults

        # p_irb from the IRB formula: r_gate = 0.5*(1 - p_irb/p_rb)
        # => p_irb = p_rb * (1 - 2*r_gate)
        p_irb = p_rb * (1.0 - 2.0 * gate_error)
        A, B = 0.45, 0.25

        rng = np.random.default_rng(seed)
        lengths = [1, 2, 4, 8, 16, 32, 64]
        n_shots, n_seq = 300, 20

        def make_rb_results(p_val):
            probs = {}
            for m in lengths:
                mu = A * p_val**m + B
                probs[m] = rng.binomial(n_shots, mu, size=n_seq) / n_shots
            return RBResults(
                sequence_lengths=lengths,
                survival_probs=probs,
                n_shots=n_shots,
                n_sequences=n_seq,
            )

        return IRBResults(
            rb_results=make_rb_results(p_rb),
            irb_results=make_rb_results(p_irb),
            target_gate="X",
        )

    def test_returns_irb_fit(self):
        results = self._make_synthetic_irb_results()
        fit = fit_irb(results, n_walkers=16, n_steps=200, n_burn=100, seed=0)
        assert isinstance(fit, IRBFit)

    def test_samples_shape(self):
        results = self._make_synthetic_irb_results()
        fit = fit_irb(results, n_walkers=16, n_steps=200, n_burn=100, seed=0)
        assert fit.samples.shape == (16 * 200, 5)

    def test_acceptance_fraction_healthy(self):
        results = self._make_synthetic_irb_results()
        fit = fit_irb(results, n_walkers=16, n_steps=300, n_burn=100, seed=0)
        assert 0.1 < fit.acceptance_fraction < 0.8

    def test_gate_error_rate_recovery(self):
        """True gate error rate should lie within the 95% credible interval."""
        true_gate_error = 0.008
        results = self._make_synthetic_irb_results(gate_error=true_gate_error, seed=42)
        fit = fit_irb(results, n_walkers=32, n_steps=500, n_burn=200, seed=42)

        lo, hi = fit.credible_interval("gate_error_rate", level=0.95)
        mean = fit.posterior_mean("gate_error_rate")

        assert lo - 1e-4 <= true_gate_error <= hi, (
            f"True r_gate={true_gate_error:.4f} not in 95% CI [{lo:.4f}, {hi:.4f}]"
        )
        assert abs(mean - true_gate_error) < true_gate_error * 0.8, (
            f"Posterior mean {mean:.4f} too far from true {true_gate_error:.4f}"
        )

    def test_target_gate_stored(self):
        results = self._make_synthetic_irb_results()
        fit = fit_irb(results, n_walkers=16, n_steps=200, n_burn=100, seed=0)
        assert fit.target_gate == "X"

    def test_summary_runs(self):
        results = self._make_synthetic_irb_results()
        fit = fit_irb(results, n_walkers=16, n_steps=200, n_burn=100, seed=0)
        s = fit.summary()
        assert "gate_error_rate" in s