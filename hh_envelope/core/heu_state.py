"""
HEU State Dataclasses (v1.0)
============================

State containers for the HH-like Envelope Unit.

Components:
-----------
1. HEUState: Core state variables
   - value: signal value (analog of membrane voltage)
   - attack_gate: fast gate (Na-like activation)
   - recovery_gate: slow gate (K-like activation)
   - inhibit_gate: optional inactivation gate
   - refractory_remaining_ms: countdown timer for refractory period
   - activity_ema: exponential moving average of activity (for safety)

2. HEUModulators: External modulation from higher layers
   - attack_gain, recovery_gain, excitability, refractory_scale

3. HEUStep: Output from a single step
   - envelope, spike_event, state, raw_value

Usage:
------
    state = HEUState()  # Initialize at rest
    state_dict = state.to_dict()  # Serialize
    new_state = HEUState.from_dict(state_dict)  # Deserialize

    mods = HEUModulators(attack_gain=1.2)  # Boost attack
    output = unit.step(excitation, dt_sec, mods)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HEUState:
    """State of an HH-like Envelope Unit.

    Analogs to Hodgkin-Huxley model:
    - value: membrane voltage equivalent
    - attack_gate: Na activation (m) - fast positive feedback
    - recovery_gate: K activation (n) - slow negative feedback
    - inhibit_gate: Na inactivation (h) - transient cutoff
    - refractory_remaining_ms: countdown timer (replaces timestamp tracking)
    - activity_ema: activity tracking for homeostatic safety
    """

    value: float = 0.0
    attack_gate: float = 0.0
    recovery_gate: float = 0.0
    inhibit_gate: float = 1.0
    refractory_remaining_ms: float = 0.0
    activity_ema: float = 0.0

    def copy(self) -> HEUState:
        """Create a deep copy of this state."""
        return HEUState(
            value=self.value,
            attack_gate=self.attack_gate,
            recovery_gate=self.recovery_gate,
            inhibit_gate=self.inhibit_gate,
            refractory_remaining_ms=self.refractory_remaining_ms,
            activity_ema=self.activity_ema,
        )

    def to_array(self) -> np.ndarray:
        """Convert continuous state variables to array for integration.

        Note: refractory_remaining_ms and activity_ema are NOT included
        (discrete/tracking, not integrated).
        Order: [value, attack_gate, recovery_gate, inhibit_gate]
        """
        return np.array(
            [self.value, self.attack_gate, self.recovery_gate, self.inhibit_gate],
            dtype=np.float64,
        )

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        refractory_remaining_ms: float = 0.0,
        activity_ema: float = 0.0,
    ) -> HEUState:
        """Create state from integration array.

        Args:
            arr: Array of [value, attack_gate, recovery_gate, inhibit_gate]
            refractory_remaining_ms: Countdown timer (passed through)
            activity_ema: Activity tracking (passed through)
        """
        return cls(
            value=float(arr[0]),
            attack_gate=float(arr[1]),
            recovery_gate=float(arr[2]),
            inhibit_gate=float(arr[3]),
            refractory_remaining_ms=refractory_remaining_ms,
            activity_ema=activity_ema,
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize state to dictionary.

        Returns:
            Dictionary with all state fields
        """
        return {
            "value": self.value,
            "attack_gate": self.attack_gate,
            "recovery_gate": self.recovery_gate,
            "inhibit_gate": self.inhibit_gate,
            "refractory_remaining_ms": self.refractory_remaining_ms,
            "activity_ema": self.activity_ema,
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> HEUState:
        """Deserialize state from dictionary.

        Args:
            d: Dictionary with state fields

        Returns:
            HEUState instance
        """
        return cls(
            value=d.get("value", 0.0),
            attack_gate=d.get("attack_gate", 0.0),
            recovery_gate=d.get("recovery_gate", 0.0),
            inhibit_gate=d.get("inhibit_gate", 1.0),
            refractory_remaining_ms=d.get("refractory_remaining_ms", 0.0),
            activity_ema=d.get("activity_ema", 0.0),
        )


@dataclass
class HEUModulators:
    """External modulation signals from higher layers or context.

    These scale the effective parameters without changing theta values.
    Used by hierarchy for top-down modulation.
    """

    attack_gain: float = 1.0
    recovery_gain: float = 1.0
    excitability: float = 1.0
    refractory_scale: float = 1.0
    threshold_offset: float = 0.0


# Backward compatibility alias
Modulators = HEUModulators


@dataclass
class HEUStep:
    """Output from a single HEU step.

    Attributes:
        envelope: Normalized output (clipped to 0..1)
        spike_event: True if a spike was triggered this step
        note_off_event: True if envelope fell below threshold (for MIDI note-off)
        state: Updated internal state
        raw_value: Unbounded internal value (for debugging/analysis)
    """

    envelope: float
    spike_event: bool
    note_off_event: bool
    state: HEUState
    raw_value: float


# Backward compatibility alias
StepOutput = HEUStep
