"""
Voice Class (v1.0)
==================

Wrapper that owns an HEU and provides voice-level interface.

A Voice owns:
- one HHEnvelopeUnit (or could own a stack for multiple articulators)
- optional local memory (pitch hint, voice ID, spike history)

A Voice receives:
- excitation (voice-specific onset energy)
- tonic (continuous low-level drive for sustained envelopes) [NEW in v1.0]
- shared_context (harmonic field, pulse phase, density pressure)
- coupling_input (from other voices)

It produces:
- envelope control signal
- spike events

Usage:
------
    from hh_envelope.polyphony import Voice, SharedContext
    from hh_envelope.core import HHEnvelopeUnit

    unit = HHEnvelopeUnit()
    voice = Voice(unit, pitch_hint=60, voice_id=0)  # Middle C

    context = SharedContext(pulse_phase=0.0, harmonic_bias=0.7)
    output = voice.step(onset_signal=1.0, tonic=0.15, shared_context=context, dt_ms=1.0)

    print(f"Envelope: {output.envelope}, Spike: {output.spike_event}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hh_envelope.core import HEUModulators, HHEnvelopeUnit

if TYPE_CHECKING:
    from hh_envelope.coupling import CouplingContribution, VoiceExport


@dataclass
class SharedContext:
    """Shared context available to all voices.

    This represents global musical context that affects voice behavior.
    """

    pulse_phase: float = 0.0
    """Current phase within the beat (0..1)."""

    harmonic_bias: float = 0.5
    """Harmonic context bias (0..1, higher = favor consonance)."""

    density_pressure: float = 0.5
    """Density pressure from other voices (0..1, higher = reduce excitability)."""

    global_modulator: float = 0.0
    """Global modulation signal (reward, novelty, error) for learning."""


@dataclass
class VoiceOutput:
    """Output from a voice step."""

    envelope: float
    """Current envelope value (0..1)."""

    spike_event: bool
    """True if a spike occurred this step."""

    note_on_event: bool
    """True if a note-on event occurred (spike with low prior activity)."""

    note_off_event: bool
    """True if a note-off event occurred (envelope below threshold)."""

    voice_id: int
    """Identifier of this voice."""


class Voice:
    """A voice in the polyphonic system.

    Wraps an HHEnvelopeUnit with additional context for
    pitch, coupling, and shared musical context.
    """

    def __init__(
        self,
        unit: HHEnvelopeUnit | None = None,
        pitch_hint: int | None = None,
        voice_id: int = 0,
    ):
        """Initialize a voice.

        Args:
            unit: HHEnvelopeUnit to use (creates default if None)
            pitch_hint: MIDI note number or pitch class (for coupling)
            voice_id: Unique identifier for this voice
        """
        self._unit = unit if unit is not None else HHEnvelopeUnit()
        self._pitch_hint = pitch_hint
        self._voice_id = voice_id

        # State tracking
        self._last_envelope: float = 0.0
        self._last_spike_event: bool = False
        self._spike_history: list[float] = []
        self._current_time_ms: float = 0.0

    @property
    def pitch_hint(self) -> int | None:
        """MIDI note or pitch class hint."""
        return self._pitch_hint

    @pitch_hint.setter
    def pitch_hint(self, value: int | None) -> None:
        self._pitch_hint = value

    @property
    def voice_id(self) -> int:
        """Voice identifier."""
        return self._voice_id

    @property
    def last_envelope(self) -> float:
        """Last computed envelope value."""
        return self._last_envelope

    @property
    def spike_history(self) -> list[float]:
        """List of spike times in ms."""
        return self._spike_history.copy()

    @property
    def unit(self) -> HHEnvelopeUnit:
        """The underlying HHEnvelopeUnit."""
        return self._unit

    def step(
        self,
        onset_signal: float,
        shared_context: SharedContext | None = None,
        coupling_input: float = 0.0,
        dt_ms: float = 1.0,
        tonic: float = 0.0,
        coupling_contributions: "CouplingContribution | None" = None,
    ) -> VoiceOutput:
        """Advance the voice by one timestep.

        Args:
            onset_signal: Voice-specific excitation (onset energy)
            shared_context: Shared musical context
            coupling_input: Legacy coupling from other voices (scalar)
            dt_ms: Timestep in milliseconds
            tonic: Continuous low-level drive for sustained envelopes
            coupling_contributions: New-style coupling from CouplingEngine.
                If provided, overrides the legacy coupling_input path.

        Returns:
            VoiceOutput with envelope, spike_event, voice_id
        """
        if shared_context is None:
            shared_context = SharedContext()

        # Build modulators from shared context
        # Density pressure reduces excitability (competition between voices)
        excitability = 1.0 - 0.5 * shared_context.density_pressure
        # Harmonic bias can boost attack for consonant contexts
        attack_gain = 0.8 + 0.4 * shared_context.harmonic_bias

        threshold_offset = 0.0

        if coupling_contributions is not None:
            # New coupling path: apply CouplingContribution fields
            onset_signal += coupling_contributions.excitation_add
            attack_gain *= coupling_contributions.attack_gain_mul
            threshold_offset = coupling_contributions.threshold_add
        else:
            # Legacy coupling path: scalar modulates attack_gain
            coupling_modulation = 1.0 + coupling_input
            coupling_modulation = max(0.5, min(1.5, coupling_modulation))
            attack_gain *= coupling_modulation

        modulators = HEUModulators(
            attack_gain=attack_gain,
            recovery_gain=1.0,
            excitability=excitability,
            refractory_scale=1.0,
            threshold_offset=threshold_offset,
        )

        # Step the unit (pass tonic for sustained envelopes)
        step_output = self._unit.step(onset_signal, dt_ms, modulators, tonic=tonic)

        # Update internal state
        self._last_envelope = step_output.envelope
        self._last_spike_event = step_output.spike_event
        self._current_time_ms += dt_ms

        # Track spikes
        if step_output.spike_event:
            self._spike_history.append(self._current_time_ms)

        # Compute note_on_event: spike with low prior activity (envelope below threshold)
        note_on_event = step_output.spike_event and self._last_envelope < 0.3

        return VoiceOutput(
            envelope=step_output.envelope,
            spike_event=step_output.spike_event,
            note_on_event=note_on_event,
            note_off_event=step_output.note_off_event,
            voice_id=self._voice_id,
        )

    def get_export(self) -> "VoiceExport":
        """Get current voice state as a VoiceExport for the CouplingEngine.

        Returns:
            VoiceExport with current value, activity_ema, and spike state.
        """
        from hh_envelope.coupling import VoiceExport as _VoiceExport

        # Use envelope property instead of accessing private _state
        return _VoiceExport(
            voice_id=self._voice_id,
            pitch_hint=self._pitch_hint,
            value=self._last_envelope,
            activity_ema=self._unit.activity_ema,
            spike_event=self._last_spike_event,
        )

    def reset(self) -> None:
        """Reset the voice to initial state."""
        self._unit.reset()
        self._last_envelope = 0.0
        self._spike_history.clear()
        self._current_time_ms = 0.0

    def get_spike_times_array(self) -> np.ndarray:
        """Get spike times as numpy array."""

        return np.array(self._spike_history)
