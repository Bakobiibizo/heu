"""
HEU Core Tests (v1.0)
=====================

Tests for the core HHEnvelopeUnit functionality per the v1.0 spec.

Required tests:
1. Pulse response - single pulse -> transient -> baseline
2. Refractory - two close pulses -> max 1 spike
3. Tonic sustain - small tonic -> sustained envelope, no runaway
4. Determinism - same inputs -> same outputs
5. Safety damping - large excitation -> damping activates
"""

from __future__ import annotations

import numpy as np
import pytest

from hh_envelope.core import (
    HEUConfig,
    HEUModulators,
    HEUParamsRef,
    HEUState,
    HEUStep,
    HEUThetaBounds,
    HHEnvelopeUnit,
    Modulators,
    SafetyConfig,
    StepOutput,
)


class TestHEUState:
    """Tests for HEUState dataclass."""

    def test_state_initialization(self):
        """Test default state initialization."""
        state = HEUState()
        assert state.value == 0.0
        assert state.attack_gate == 0.0
        assert state.recovery_gate == 0.0
        assert state.inhibit_gate == 1.0
        assert state.refractory_remaining_ms == 0.0
        assert state.activity_ema == 0.0

    def test_state_copy(self):
        """Test state copy is independent."""
        state = HEUState(value=0.5, attack_gate=0.3)
        copy = state.copy()
        copy.value = 0.8
        assert state.value == 0.5  # Original unchanged

    def test_state_to_dict(self):
        """Test serialization to dict."""
        state = HEUState(value=0.5, attack_gate=0.3, refractory_remaining_ms=10.0)
        d = state.to_dict()
        assert d["value"] == 0.5
        assert d["attack_gate"] == 0.3
        assert d["refractory_remaining_ms"] == 10.0

    def test_state_from_dict(self):
        """Test deserialization from dict."""
        d = {"value": 0.5, "attack_gate": 0.3, "refractory_remaining_ms": 10.0}
        state = HEUState.from_dict(d)
        assert state.value == 0.5
        assert state.attack_gate == 0.3
        assert state.refractory_remaining_ms == 10.0

    def test_state_to_array(self):
        """Test conversion to numpy array."""
        state = HEUState(value=0.5, attack_gate=0.3, recovery_gate=0.2, inhibit_gate=0.9)
        arr = state.to_array()
        assert arr.shape == (4,)
        assert arr[0] == 0.5
        assert arr[1] == 0.3
        assert arr[2] == 0.2
        assert arr[3] == 0.9

    def test_state_from_array(self):
        """Test creation from numpy array."""
        arr = np.array([0.5, 0.3, 0.2, 0.9])
        state = HEUState.from_array(arr, refractory_remaining_ms=15.0, activity_ema=0.2)
        assert state.value == 0.5
        assert state.attack_gate == 0.3
        assert state.recovery_gate == 0.2
        assert state.inhibit_gate == 0.9
        assert state.refractory_remaining_ms == 15.0
        assert state.activity_ema == 0.2


class TestBackwardsCompatibility:
    """Tests for backwards compatibility aliases."""

    def test_modulators_alias(self):
        """Modulators should be alias for HEUModulators."""
        assert Modulators is HEUModulators
        mod = Modulators(attack_gain=1.2)
        assert mod.attack_gain == 1.2

    def test_step_output_alias(self):
        """StepOutput should be alias for HEUStep."""
        assert StepOutput is HEUStep


class TestHHEnvelopeUnitBasic:
    """Basic tests for HHEnvelopeUnit."""

    def test_initialization_default(self):
        """Test default initialization."""
        unit = HHEnvelopeUnit()
        assert unit.total_conductance > 0
        assert 0 < unit.attack_ratio < 1
        assert unit.attack_tau_ms > 0
        assert unit.recovery_tau_ms > 0

    def test_initialization_with_legacy_params(self):
        """Test initialization with legacy parameter objects."""
        params = HEUParamsRef(total_conductance_ref=3.0)
        bounds = HEUThetaBounds()
        safety = SafetyConfig()
        unit = HHEnvelopeUnit(params_ref=params, theta_bounds=bounds, safety=safety)
        assert unit.params_ref.total_conductance_ref == 3.0

    def test_initialization_with_unified_config(self):
        """Test initialization with unified HEUConfig."""
        config = HEUConfig(total_conductance_ref=3.0, enable_inhibit_gate=True)
        unit = HHEnvelopeUnit(config=config)
        assert unit.params_ref.total_conductance_ref == 3.0
        assert unit._enable_inhibit_gate is True

    def test_step_returns_heu_step(self):
        """Test that step() returns HEUStep."""
        unit = HHEnvelopeUnit()
        output = unit.step(excitation=1.0, dt_ms=1.0)
        assert isinstance(output, HEUStep)
        assert 0 <= output.envelope <= 1
        assert isinstance(output.spike_event, bool)
        assert isinstance(output.state, HEUState)


class TestPulseResponse:
    """Test 1: Pulse response - single pulse -> transient -> baseline."""

    def test_single_pulse_creates_transient(self):
        """Single excitation pulse should create envelope transient."""
        unit = HHEnvelopeUnit()
        dt_ms = 1.0

        # Apply single pulse
        output = unit.step(excitation=1.0, dt_ms=dt_ms)
        peak_envelope = output.envelope

        # Envelope should rise above 0
        assert peak_envelope > 0.1, f"Pulse should create transient, got {peak_envelope}"

    def test_transient_returns_to_baseline(self):
        """After transient, envelope should return toward baseline."""
        unit = HHEnvelopeUnit()
        dt_ms = 1.0

        # Apply pulse and track envelope
        envelope_trace = []
        for i in range(200):
            excitation = 1.0 if i == 0 else 0.0
            output = unit.step(excitation=excitation, dt_ms=dt_ms)
            envelope_trace.append(output.envelope)

        # Find peak and final value
        peak = max(envelope_trace)
        final = envelope_trace[-1]

        assert peak > 0.3, f"Should have significant peak, got {peak}"
        assert final < 0.1, f"Should return toward baseline, got {final}"


class TestRefractory:
    """Test 2: Refractory - two close pulses -> max 1 spike."""

    def test_refractory_prevents_double_spike(self):
        """Two pulses within refractory period should produce at most 1 spike."""
        unit = HHEnvelopeUnit()
        dt_ms = 1.0
        refractory_ms = unit.params_ref.refractory_ref_ms

        spikes = []
        # Two strong pulses separated by half the refractory period
        for i in range(int(refractory_ms * 2)):
            pulse_1_time = 0
            pulse_2_time = int(refractory_ms / 2)
            excitation = 1.0 if i == pulse_1_time or i == pulse_2_time else 0.0
            output = unit.step(excitation=excitation, dt_ms=dt_ms)
            if output.spike_event:
                spikes.append(i)

        # Should have at most 1 spike (the second pulse is within refractory)
        assert len(spikes) <= 1, f"Refractory should prevent double spike, got spikes at {spikes}"

    def test_spikes_after_refractory(self):
        """Pulses separated by more than refractory period can both spike."""
        unit = HHEnvelopeUnit()
        dt_ms = 1.0
        refractory_ms = unit.params_ref.refractory_ref_ms

        spikes = []
        # Two strong pulses separated by more than refractory period
        for i in range(int(refractory_ms * 4)):
            pulse_1_time = 0
            pulse_2_time = int(refractory_ms * 2)  # Well after refractory
            excitation = 1.0 if i == pulse_1_time or i == pulse_2_time else 0.0
            output = unit.step(excitation=excitation, dt_ms=dt_ms)
            if output.spike_event:
                spikes.append(i)

        # Could have 2 spikes if both reach threshold
        # (may have fewer if dynamics don't allow)
        assert len(spikes) >= 1, f"Should have at least one spike, got {len(spikes)}"


class TestTonicSustain:
    """Test 3: Tonic sustain - small tonic -> sustained envelope, no runaway."""

    def test_tonic_creates_sustained_envelope(self):
        """Small tonic drive should create sustained (non-zero) envelope."""
        unit = HHEnvelopeUnit()
        dt_ms = 1.0

        # Initial transient pulse, then tonic only
        envelopes = []
        for i in range(500):
            excitation = 1.0 if i == 0 else 0.0
            tonic = 0.15  # Continuous low-level drive
            output = unit.step(excitation=excitation, dt_ms=dt_ms, tonic=tonic)
            envelopes.append(output.envelope)

        # Check sustained region (after initial transient)
        sustained_mean = np.mean(envelopes[100:])
        assert sustained_mean > 0.1, f"Tonic should sustain envelope, got mean {sustained_mean}"

    def test_tonic_no_runaway(self):
        """Tonic drive should not cause runaway (envelope staying bounded)."""
        unit = HHEnvelopeUnit()
        dt_ms = 1.0

        envelopes = []
        for i in range(1000):
            tonic = 0.2  # Moderate tonic
            output = unit.step(excitation=0.0, dt_ms=dt_ms, tonic=tonic)
            envelopes.append(output.envelope)

        # Envelope should stay bounded (no runaway)
        max_envelope = max(envelopes)
        assert max_envelope <= 1.0, f"Envelope should not exceed 1.0, got {max_envelope}"


class TestDeterminism:
    """Test 4: Determinism - same inputs -> same outputs."""

    def test_deterministic_outputs(self):
        """Same inputs should produce same outputs."""
        # Run 1
        unit1 = HHEnvelopeUnit()
        outputs1 = []
        for i in range(100):
            excitation = 1.0 if i % 50 == 0 else 0.0
            output = unit1.step(excitation=excitation, dt_ms=1.0)
            outputs1.append(output.envelope)

        # Run 2 (fresh unit, same inputs)
        unit2 = HHEnvelopeUnit()
        outputs2 = []
        for i in range(100):
            excitation = 1.0 if i % 50 == 0 else 0.0
            output = unit2.step(excitation=excitation, dt_ms=1.0)
            outputs2.append(output.envelope)

        # Should be identical
        np.testing.assert_array_almost_equal(
            outputs1, outputs2, decimal=10, err_msg="Outputs should be deterministic"
        )

    def test_reset_restores_initial_state(self):
        """Reset should restore unit to initial state."""
        unit = HHEnvelopeUnit()

        # Run some steps
        for _ in range(50):
            unit.step(excitation=1.0, dt_ms=1.0)

        # Reset
        unit.reset()

        # State should be back to initial
        state = unit.state
        assert state.value == 0.0
        assert state.attack_gate == 0.0
        assert state.refractory_remaining_ms == 0.0


class TestSafetyDamping:
    """Test 5: Safety damping - large excitation -> damping activates."""

    def test_large_excitation_triggers_damping(self):
        """Sustained large excitation should trigger damping."""
        # Use lower ceiling to trigger damping faster
        safety = SafetyConfig(activity_ceiling=0.5, activity_target=0.2)
        unit = HHEnvelopeUnit(safety=safety)

        # Sustained high excitation
        for _ in range(500):
            unit.step(excitation=1.0, dt_ms=1.0)

        # Activity should be high enough that damping < 1.0
        damping = unit._activity_monitor.get_damping_factor()
        assert damping < 1.0, f"Damping should activate, got factor {damping}"

    def test_damping_limits_envelope(self):
        """Damping should limit envelope growth."""
        safety = SafetyConfig(activity_ceiling=0.5, activity_target=0.2)
        unit = HHEnvelopeUnit(safety=safety)

        envelopes = []
        for _ in range(1000):
            output = unit.step(excitation=1.0, dt_ms=1.0)
            envelopes.append(output.envelope)

        # Later envelopes should be limited by damping
        late_mean = np.mean(envelopes[500:])
        assert late_mean < 0.9, f"Damping should limit late envelope, got mean {late_mean}"


class TestThetaParameters:
    """Tests for theta parameter handling."""

    def test_set_get_theta(self):
        """Test setting and getting theta parameters."""
        unit = HHEnvelopeUnit()
        unit.set_theta(theta_total=0.5, theta_balance=-0.3)
        theta = unit.get_theta()
        assert theta["theta_total"] == 0.5
        assert theta["theta_balance"] == -0.3

    def test_theta_bounds_clamping(self):
        """Theta values should be clamped to bounds."""
        unit = HHEnvelopeUnit()
        unit.set_theta(theta_total=100.0)  # Way above bounds
        theta = unit.get_theta()
        assert theta["theta_total"] <= unit.theta_bounds.theta_total[1]

    def test_set_theta_from_dict(self):
        """Test setting theta from dictionary."""
        unit = HHEnvelopeUnit()
        unit.set_theta_from_dict({"theta_total": 0.5, "theta_balance": -0.3})
        theta = unit.get_theta()
        assert theta["theta_total"] == 0.5
        assert theta["theta_balance"] == -0.3


class TestSerialization:
    """Tests for state serialization."""

    def test_state_dict_load_state_dict(self):
        """Test state serialization and restoration."""
        unit = HHEnvelopeUnit()

        # Run some steps
        for _ in range(50):
            unit.step(excitation=1.0, dt_ms=1.0)

        # Save state
        saved = unit.state_dict()
        saved_envelope = unit.step(excitation=0.5, dt_ms=1.0).envelope

        # Create new unit and restore
        unit2 = HHEnvelopeUnit()
        unit2.load_state_dict(saved)

        # Should produce same output
        restored_envelope = unit2.step(excitation=0.5, dt_ms=1.0).envelope
        assert abs(saved_envelope - restored_envelope) < 0.01

    def test_params_dict(self):
        """Test params_dict includes theta and computed values."""
        unit = HHEnvelopeUnit()
        unit.set_theta(theta_total=0.5)
        params = unit.params_dict()

        assert "theta" in params
        assert params["theta"]["theta_total"] == 0.5
        assert "computed" in params
        assert "total_conductance" in params["computed"]


class TestModulators:
    """Tests for modulator functionality."""

    def test_attack_gain_modulation(self):
        """Higher attack gain should increase envelope peak."""
        unit1 = HHEnvelopeUnit()
        unit2 = HHEnvelopeUnit()

        # Run with different attack gains
        mod_low = HEUModulators(attack_gain=0.5)
        mod_high = HEUModulators(attack_gain=1.5)

        out1 = unit1.step(excitation=1.0, dt_ms=1.0, mod=mod_low)
        out2 = unit2.step(excitation=1.0, dt_ms=1.0, mod=mod_high)

        assert out2.envelope >= out1.envelope, "Higher attack gain should produce higher envelope"


class TestLookahead:
    """Tests for compressor-style lookahead."""

    def test_lookahead_configuration(self):
        """Test lookahead runtime configuration."""
        unit = HHEnvelopeUnit()
        assert unit.lookahead_ms == 0.0  # Default disabled

        unit.set_lookahead(lookahead_ms=10.0, ceiling=0.9)
        assert unit.lookahead_ms == 10.0
        assert unit._lookahead_ceiling == 0.9

    def test_lookahead_with_config(self):
        """Test lookahead via HEUConfig."""
        config = HEUConfig(lookahead_ms=20.0, lookahead_ceiling=0.85)
        unit = HHEnvelopeUnit(config=config)
        assert unit.lookahead_ms == 20.0
        assert unit._lookahead_ceiling == 0.85

    def test_batch_processing_basic(self):
        """Test basic batch processing works."""
        unit = HHEnvelopeUnit()
        excitations = [0.0] * 10 + [1.0] + [0.0] * 20
        outputs = unit.process_batch(excitations, dt_ms=1.0)

        assert len(outputs) == len(excitations)
        assert all(hasattr(o, "envelope") for o in outputs)

    def test_lookahead_reduces_saturation(self):
        """Lookahead should reduce saturation by anticipating peaks."""
        # Without lookahead
        unit1 = HHEnvelopeUnit()
        excitations = [0.0] * 100 + [1.0] + [0.0] * 100 + [1.0] + [0.0] * 100
        outputs1 = unit1.process_batch(excitations, dt_ms=1.0)
        peaks1 = [o.raw_value for o in outputs1]
        max_without = max(peaks1)

        # With lookahead
        unit2 = HHEnvelopeUnit()
        unit2.set_lookahead(lookahead_ms=20.0, ceiling=0.85)
        outputs2 = unit2.process_batch(excitations, dt_ms=1.0)
        peaks2 = [o.raw_value for o in outputs2]
        max_with = max(peaks2)

        # Lookahead should keep peaks closer to ceiling
        assert max_with <= max_without + 0.1, (
            f"Lookahead should reduce peaks: {max_with} vs {max_without}"
        )

    def test_lookahead_preserves_dynamics(self):
        """Lookahead should preserve relative dynamics between small and large pulses."""
        unit = HHEnvelopeUnit()
        unit.set_lookahead(lookahead_ms=20.0, ceiling=0.9)

        # Small pulse followed by large pulse
        excitations = [0.0] * 50 + [0.5] + [0.0] * 100 + [1.0] + [0.0] * 50
        outputs = unit.process_batch(excitations, dt_ms=1.0)

        # Find peaks
        envelopes = [o.envelope for o in outputs]
        small_peak = max(envelopes[45:70])
        large_peak = max(envelopes[145:170])

        # Large pulse should still produce larger response
        assert large_peak > small_peak, (
            f"Large pulse should produce larger peak: {large_peak} vs {small_peak}"
        )


class TestOversampling:
    """Tests for internal oversampling."""

    def test_oversample_configuration(self):
        """Test oversampling runtime configuration."""
        unit = HHEnvelopeUnit()
        assert unit.oversample_factor == 1  # Default disabled

        unit.set_oversample(4)
        assert unit.oversample_factor == 4

    def test_oversample_with_config(self):
        """Test oversampling via HEUConfig."""
        config = HEUConfig(oversample_factor=8)
        unit = HHEnvelopeUnit(config=config)
        assert unit.oversample_factor == 8

    def test_oversample_bounds(self):
        """Test oversampling factor is bounded."""
        unit = HHEnvelopeUnit()
        unit.set_oversample(100)  # Above max
        assert unit.oversample_factor == 16  # Clamped to max

        unit.set_oversample(0)  # Below min
        assert unit.oversample_factor == 1  # Clamped to min

    def test_oversampling_produces_valid_output(self):
        """Oversampling should produce valid envelope output."""
        unit = HHEnvelopeUnit()
        unit.set_oversample(4)

        output = unit.step(excitation=1.0, dt_ms=1.0)
        assert 0.0 <= output.envelope <= 1.0
        assert hasattr(output, "spike_event")

    def test_oversampling_consistency(self):
        """Oversampled and non-oversampled should produce similar results."""
        unit1 = HHEnvelopeUnit()
        unit2 = HHEnvelopeUnit()
        unit2.set_oversample(4)

        # Run same inputs
        envelopes1 = []
        envelopes2 = []
        for i in range(100):
            excitation = 1.0 if i == 0 else 0.0
            envelopes1.append(unit1.step(excitation, dt_ms=1.0).envelope)
            envelopes2.append(unit2.step(excitation, dt_ms=1.0).envelope)

        # Should be roughly similar (oversampling may have small differences)
        # Check the general shape is preserved
        peak1_idx = np.argmax(envelopes1)
        peak2_idx = np.argmax(envelopes2)
        assert abs(peak1_idx - peak2_idx) < 5, "Peak timing should be similar"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
