"""Voice Allocator (Optional for v0.1)
======================================

Voice allocation for polyphonic systems (future expansion).

For v0.1 experiments, hardcode 1-2 voices. This module provides
dynamic voice allocation for future versions.

Allocation strategies (implement later):
    - Oldest-voice stealing
    - Lowest-amplitude stealing
    - Priority-based (some pitches more important)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from hh_envelope.core import HEUParamsRef, HEUThetaBounds, HHEnvelopeUnit, SafetyConfig

from .voice import Voice

if TYPE_CHECKING:
    pass


class AllocationStrategy(Enum):
    """Strategy for voice stealing when all voices are in use."""

    OLDEST = "oldest"  # Steal the oldest active voice
    LOWEST_AMPLITUDE = "lowest_amplitude"  # Steal the quietest voice
    SAME_PITCH = "same_pitch"  # Prefer stealing a voice with same pitch class


@dataclass
class VoiceState:
    """Internal state for tracking a voice in the allocator."""

    voice: Voice
    is_active: bool = False
    note_on_time_ms: float = 0.0
    pitch: int | None = None


class VoiceAllocator:
    """Manages allocation and deallocation of voices in a polyphonic system.

    Pre-allocates a pool of voices and handles voice stealing when
    all voices are in use.
    """

    def __init__(
        self,
        max_voices: int = 8,
        params_ref: HEUParamsRef | None = None,
        theta_bounds: HEUThetaBounds | None = None,
        safety: SafetyConfig | None = None,
        strategy: AllocationStrategy = AllocationStrategy.OLDEST,
    ):
        """Initialize the voice allocator.

        Args:
            max_voices: Maximum number of simultaneous voices
            params_ref: HEU parameter reference (shared by all voices)
            theta_bounds: Theta bounds (shared by all voices)
            safety: Safety config (shared by all voices)
            strategy: Voice stealing strategy
        """
        self._max_voices = max_voices
        self._params_ref = params_ref or HEUParamsRef()
        self._theta_bounds = theta_bounds or HEUThetaBounds()
        self._safety = safety or SafetyConfig()
        self._strategy = strategy

        # Pre-allocate voice pool
        self._voice_states: list[VoiceState] = []
        for i in range(max_voices):
            unit = HHEnvelopeUnit(self._params_ref, self._theta_bounds, self._safety)
            voice = Voice(unit=unit, voice_id=i)
            self._voice_states.append(VoiceState(voice=voice))

        self._current_time_ms: float = 0.0

    @property
    def max_voices(self) -> int:
        """Maximum number of voices."""
        return self._max_voices

    @property
    def active_count(self) -> int:
        """Number of currently active voices."""
        return sum(1 for vs in self._voice_states if vs.is_active)

    def allocate(
        self,
        pitch: int,
        onset_energy: float = 1.0,
    ) -> Voice | None:
        """Allocate a voice for a new note.

        Args:
            pitch: MIDI note number
            onset_energy: Initial onset energy (for prioritization)

        Returns:
            Allocated voice, or None if allocation failed
        """
        # First, try to find a free voice
        for vs in self._voice_states:
            if not vs.is_active:
                vs.is_active = True
                vs.note_on_time_ms = self._current_time_ms
                vs.pitch = pitch
                vs.voice.pitch_hint = pitch
                vs.voice.reset()
                return vs.voice

        # All voices are active - need to steal one
        stolen = self._select_voice_to_steal(pitch)
        if stolen is not None:
            stolen.is_active = True
            stolen.note_on_time_ms = self._current_time_ms
            stolen.pitch = pitch
            stolen.voice.pitch_hint = pitch
            stolen.voice.reset()
            return stolen.voice

        return None

    def release(self, voice_id: int) -> None:
        """Release a voice (mark as available for reuse).

        Args:
            voice_id: ID of the voice to release
        """
        for vs in self._voice_states:
            if vs.voice.voice_id == voice_id:
                vs.is_active = False
                vs.pitch = None
                break

    def release_if_silent(self, threshold: float = 0.01) -> list[int]:
        """Release voices that have decayed below threshold.

        Args:
            threshold: Envelope level below which to release

        Returns:
            List of released voice IDs
        """
        released = []
        for vs in self._voice_states:
            if vs.is_active and vs.voice.last_envelope < threshold:
                vs.is_active = False
                vs.pitch = None
                released.append(vs.voice.voice_id)
        return released

    def get_active_voices(self) -> list[Voice]:
        """Get all currently active voices.

        Returns:
            List of active voices
        """
        return [vs.voice for vs in self._voice_states if vs.is_active]

    def get_all_voices(self) -> list[Voice]:
        """Get all voices (active and inactive).

        Returns:
            List of all voices
        """
        return [vs.voice for vs in self._voice_states]

    def advance_time(self, dt_ms: float) -> None:
        """Advance the allocator's internal clock.

        Args:
            dt_ms: Time to advance in milliseconds
        """
        self._current_time_ms += dt_ms

    def reset(self) -> None:
        """Reset all voices and allocator state."""
        for vs in self._voice_states:
            vs.is_active = False
            vs.note_on_time_ms = 0.0
            vs.pitch = None
            vs.voice.reset()
        self._current_time_ms = 0.0

    def _select_voice_to_steal(self, new_pitch: int) -> VoiceState | None:
        """Select a voice to steal based on the configured strategy.

        Args:
            new_pitch: Pitch of the note requesting allocation

        Returns:
            VoiceState to steal, or None
        """
        active_states = [vs for vs in self._voice_states if vs.is_active]
        if not active_states:
            return None

        if self._strategy == AllocationStrategy.OLDEST:
            # Steal the oldest (earliest note_on_time)
            return min(active_states, key=lambda vs: vs.note_on_time_ms)

        elif self._strategy == AllocationStrategy.LOWEST_AMPLITUDE:
            # Steal the quietest
            return min(active_states, key=lambda vs: vs.voice.last_envelope)

        elif self._strategy == AllocationStrategy.SAME_PITCH:
            # Prefer stealing a voice with the same pitch class
            new_pitch_class = new_pitch % 12
            same_pitch = [
                vs
                for vs in active_states
                if vs.pitch is not None and vs.pitch % 12 == new_pitch_class
            ]
            if same_pitch:
                return min(same_pitch, key=lambda vs: vs.note_on_time_ms)
            # Fall back to oldest
            return min(active_states, key=lambda vs: vs.note_on_time_ms)

        return None
