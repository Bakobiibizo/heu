"""
Articulated Note: Multi-voice articulation management.

This module provides the ArticulatedNote class for managing multi-voice
articulations like grace notes, trills, and mordents without modifying
the underlying HEU internals.

Architecture:
    ArticulatedNote  →  spawns Voice instances with different configs
            ↓
         Voice       →  owns HEU + lifetime + tonic config (unchanged)
            ↓
     HHEnvelopeUnit  →  pure dynamics, no articulation logic (unchanged)

Key Design Decisions:
    - Combination mode: max(grace, main) for perceptual maximum
    - Grace voice: no tonic, short lifetime (~40ms active)
    - Main voice: tonic drive (0.12 level), longer lifetime
    - Each voice can have independent theta parameters for optimization

Usage:
    from hh_envelope.polyphony.articulation import (
        ArticulatedNote,
        create_grace_note_config,
    )

    config = create_grace_note_config(main_duration_ms=360.0)
    note = ArticulatedNote(config)

    # Step through time
    for _ in range(int(config.total_duration_ms)):
        output = note.step(dt_ms=1.0)
        combined_envelope = output.combined_envelope

    # For optimization, set theta per voice
    note.set_voice_theta("grace", {"theta_total": -0.5, ...})
    note.set_voice_theta("main", {"theta_total": -1.0, ...})
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from hh_envelope.core import HEUParamsRef, HEUThetaBounds, HHEnvelopeUnit, SafetyConfig
from hh_envelope.polyphony.voice import SharedContext, Voice


class CombinationMode(Enum):
    """How to combine multiple voice envelopes."""

    MAX = "max"  # Perceptual maximum (default)
    SUM = "sum"  # Additive (may exceed 1.0)
    MEAN = "mean"  # Average
    CROSSFADE = "crossfade"  # Smooth crossfade between voices


@dataclass
class TonicConfig:
    """Configuration for tonic (sustained) excitation drive."""

    level: float = 0.12
    """Tonic excitation level (0..1)."""

    start_ms: float = 30.0
    """When tonic drive begins after voice onset."""

    end_before_ms: float = 80.0
    """How long before voice end to stop tonic (for release)."""


@dataclass
class VoiceRole:
    """Configuration for one voice within an articulation.

    Each VoiceRole defines timing, amplitude, and optional tonic drive
    for a single voice in a multi-voice articulation.
    """

    name: str
    """Role name (e.g., 'grace', 'main', 'upper', 'lower')."""

    onset_time_ms: float
    """When this voice's onset pulse occurs."""

    pulse_amplitude: float = 1.0
    """Amplitude of the onset pulse (0..1+)."""

    active_duration_ms: float | None = None
    """How long this voice is active (None = until articulation ends)."""

    tonic_config: TonicConfig | None = None
    """Optional tonic drive configuration for sustained notes."""


@dataclass
class ArticulationConfig:
    """Full configuration for a multi-voice articulation pattern."""

    name: str
    """Articulation name (e.g., 'grace_note', 'trill', 'mordent')."""

    voice_roles: list[VoiceRole]
    """Configuration for each voice in the articulation."""

    total_duration_ms: float
    """Total articulation duration."""

    combination_mode: CombinationMode = CombinationMode.MAX
    """How to combine voice envelopes."""


@dataclass
class ArticulatedNoteOutput:
    """Output from ArticulatedNote.step()."""

    combined_envelope: float
    """Combined envelope value from all voices."""

    voice_envelopes: dict[str, float]
    """Individual envelope values per voice role."""

    any_spike: bool
    """True if any voice spiked this step."""

    current_time_ms: float
    """Current simulation time."""


class ArticulatedNote:
    """Manages multi-voice articulations.

    Creates and coordinates multiple Voice instances, each with potentially
    different theta parameters, to produce complex articulation patterns
    like grace notes, trills, and mordents.
    """

    def __init__(
        self,
        config: ArticulationConfig,
        params_ref: HEUParamsRef | None = None,
        theta_bounds: HEUThetaBounds | None = None,
        safety: SafetyConfig | None = None,
    ):
        """Initialize an articulated note.

        Args:
            config: Articulation configuration
            params_ref: HEU reference parameters (shared across voices)
            theta_bounds: Theta bounds (shared across voices)
            safety: Safety config (shared across voices)
        """
        self._config = config
        self._params_ref = params_ref or HEUParamsRef()
        self._theta_bounds = theta_bounds or HEUThetaBounds()
        self._safety = safety or SafetyConfig()

        # Create voices for each role
        self._voices: dict[str, Voice] = {}
        self._voice_configs: dict[str, VoiceRole] = {}

        for i, role in enumerate(config.voice_roles):
            unit = HHEnvelopeUnit(
                params_ref=self._params_ref,
                theta_bounds=self._theta_bounds,
                safety=self._safety,
            )
            voice = Voice(unit=unit, voice_id=i)
            self._voices[role.name] = voice
            self._voice_configs[role.name] = role

        # Timing state
        self._current_time_ms: float = 0.0

        # Shared context (neutral by default)
        self._shared_context = SharedContext()

    @property
    def config(self) -> ArticulationConfig:
        """Get the articulation configuration."""
        return self._config

    @property
    def voice_names(self) -> list[str]:
        """Get names of all voice roles."""
        return list(self._voices.keys())

    @property
    def current_time_ms(self) -> float:
        """Current simulation time in ms."""
        return self._current_time_ms

    def set_voice_theta(self, role_name: str, theta_dict: dict[str, float]) -> None:
        """Set theta parameters for a specific voice.

        Args:
            role_name: Voice role name (e.g., 'grace', 'main')
            theta_dict: Dictionary of theta parameters

        Raises:
            KeyError: If role_name not found
        """
        if role_name not in self._voices:
            raise KeyError(f"Voice role '{role_name}' not found. Available: {self.voice_names}")
        self._voices[role_name].unit.set_theta_from_dict(theta_dict)

    def get_voice_theta(self, role_name: str) -> dict[str, float]:
        """Get theta parameters for a specific voice.

        Args:
            role_name: Voice role name

        Returns:
            Dictionary of current theta parameters
        """
        if role_name not in self._voices:
            raise KeyError(f"Voice role '{role_name}' not found. Available: {self.voice_names}")
        return self._voices[role_name].unit.get_theta()

    def step(self, dt_ms: float = 1.0) -> ArticulatedNoteOutput:
        """Advance the articulation by one timestep.

        Computes excitation and tonic for each voice based on its role configuration,
        steps each voice, and combines the envelopes.

        Args:
            dt_ms: Timestep in milliseconds

        Returns:
            ArticulatedNoteOutput with combined and per-voice envelopes
        """
        voice_envelopes: dict[str, float] = {}
        any_spike = False

        for role_name, voice in self._voices.items():
            role = self._voice_configs[role_name]

            # Compute excitation and tonic for this voice at current time
            excitation, tonic = self._compute_excitation_and_tonic(role)

            # Check if voice is still active
            if role.active_duration_ms is not None:
                voice_end_time = role.onset_time_ms + role.active_duration_ms
                if self._current_time_ms > voice_end_time:
                    # Voice inactive - still step with zero excitation to let it decay
                    excitation = 0.0
                    tonic = 0.0

            # Step the voice (passing tonic for sustained envelopes)
            output = voice.step(
                onset_signal=excitation,
                shared_context=self._shared_context,
                coupling_input=0.0,  # No inter-voice coupling in articulations
                dt_ms=dt_ms,
                tonic=tonic,
            )

            voice_envelopes[role_name] = output.envelope
            if output.spike_event:
                any_spike = True

        # Combine envelopes
        combined = self._combine_envelopes(voice_envelopes)

        # Advance time
        self._current_time_ms += dt_ms

        return ArticulatedNoteOutput(
            combined_envelope=combined,
            voice_envelopes=voice_envelopes,
            any_spike=any_spike,
            current_time_ms=self._current_time_ms,
        )

    def _compute_excitation_and_tonic(self, role: VoiceRole) -> tuple[float, float]:
        """Compute excitation and tonic for a voice role at current time.

        Separates pulse excitation from tonic drive per v1.0 HEU interface.

        Args:
            role: Voice role configuration

        Returns:
            Tuple of (excitation, tonic) for this timestep
        """
        excitation = 0.0
        tonic_level = 0.0
        time_since_onset = self._current_time_ms - role.onset_time_ms

        # Onset pulse (at onset time, within 1ms window)
        if 0.0 <= time_since_onset < 1.0:
            excitation = role.pulse_amplitude

        # Tonic drive (if configured and within active window)
        if role.tonic_config is not None and time_since_onset >= 0:
            tonic_cfg = role.tonic_config

            # Determine voice end time
            if role.active_duration_ms is not None:
                voice_end_ms = role.active_duration_ms
            else:
                voice_end_ms = self._config.total_duration_ms - role.onset_time_ms

            tonic_start = tonic_cfg.start_ms
            tonic_end = voice_end_ms - tonic_cfg.end_before_ms

            if tonic_start <= time_since_onset < tonic_end:
                tonic_level = tonic_cfg.level

        return excitation, tonic_level

    def _compute_excitation(self, role: VoiceRole) -> float:
        """Compute excitation for a voice role at current time.

        DEPRECATED: Use _compute_excitation_and_tonic() instead.

        Args:
            role: Voice role configuration

        Returns:
            Excitation value for this timestep (tonic added to excitation for legacy compat)
        """
        excitation, tonic = self._compute_excitation_and_tonic(role)
        # Legacy behavior: tonic was added to excitation
        return max(excitation, tonic)

    def _combine_envelopes(self, voice_envelopes: dict[str, float]) -> float:
        """Combine voice envelopes according to combination mode.

        Args:
            voice_envelopes: Envelope values per voice

        Returns:
            Combined envelope value
        """
        values = list(voice_envelopes.values())
        if not values:
            return 0.0

        mode = self._config.combination_mode
        if mode == CombinationMode.MAX:
            return max(values)
        elif mode == CombinationMode.SUM:
            return sum(values)
        elif mode == CombinationMode.MEAN:
            return sum(values) / len(values)
        elif mode == CombinationMode.CROSSFADE:
            return self._crossfade_combine(voice_envelopes)
        else:
            return max(values)

    def _crossfade_combine(self, voice_envelopes: dict[str, float]) -> float:
        """Combine voices with smooth crossfade based on timing.

        For grace note: grace voice fades out as main voice fades in.

        Args:
            voice_envelopes: Envelope values per voice

        Returns:
            Crossfaded envelope value
        """
        # For two voices (grace + main), do a smooth crossfade
        if len(voice_envelopes) == 2 and "grace" in voice_envelopes and "main" in voice_envelopes:
            grace_env = voice_envelopes["grace"]
            main_env = voice_envelopes["main"]

            # Find the main voice onset time
            main_role = self._voice_configs.get("main")
            if main_role is None:
                return max(grace_env, main_env)

            main_onset = main_role.onset_time_ms
            crossfade_duration = 20.0  # ms to transition

            # Compute crossfade factor (0 = all grace, 1 = all main)
            if self._current_time_ms < main_onset:
                factor = 0.0
            elif self._current_time_ms > main_onset + crossfade_duration:
                factor = 1.0
            else:
                factor = (self._current_time_ms - main_onset) / crossfade_duration

            # Smooth crossfade with sigmoid-like curve
            # Use cosine interpolation for smoother transition
            import math

            smooth_factor = (1.0 - math.cos(factor * math.pi)) / 2.0

            return grace_env * (1.0 - smooth_factor) + main_env * smooth_factor
        else:
            # Fallback to max for other configurations
            return max(voice_envelopes.values())

    def reset(self) -> None:
        """Reset all voices and timing to initial state."""
        for voice in self._voices.values():
            voice.reset()
        self._current_time_ms = 0.0

    def run_full(self, dt_ms: float = 1.0) -> np.ndarray:
        """Run the full articulation and return combined envelope.

        Convenience method for testing and optimization.

        Args:
            dt_ms: Timestep in milliseconds

        Returns:
            Array of combined envelope values
        """
        self.reset()
        num_samples = int(self._config.total_duration_ms / dt_ms)
        envelope = np.zeros(num_samples)

        for i in range(num_samples):
            output = self.step(dt_ms)
            envelope[i] = output.combined_envelope

        return envelope


# --- Factory functions for common articulation patterns ---


def create_grace_note_config(
    main_duration_ms: float = 360.0,
    grace_duration_ms: float = 30.0,
    gap_duration_ms: float = 10.0,
    grace_amplitude: float = 0.9,
    main_amplitude: float = 1.0,
    main_tonic_level: float = 0.18,
    grace_sustain_level: float = 0.08,
) -> ArticulationConfig:
    """Create a grace note articulation configuration.

    Two voices: a short grace note followed by the main note.
    Grace voice has a small sustain tonic to maintain the release floor.
    Main voice has tonic drive for sustained envelope.

    Args:
        main_duration_ms: Duration of main note (after grace + gap)
        grace_duration_ms: Duration of grace note region
        gap_duration_ms: Gap between grace decay and main onset
        grace_amplitude: Grace note onset amplitude
        main_amplitude: Main note onset amplitude
        main_tonic_level: Tonic drive level for main note sustain
        grace_sustain_level: Tonic to maintain floor during gap (0.08 = ~0.15 output)

    Returns:
        ArticulationConfig for a grace note
    """
    total_duration = grace_duration_ms + gap_duration_ms + main_duration_ms
    main_onset = grace_duration_ms + gap_duration_ms

    # Grace voice: quick transient with small sustain tonic to maintain floor
    # The tonic starts after the attack peak and maintains the release threshold
    grace_role = VoiceRole(
        name="grace",
        onset_time_ms=0.0,
        pulse_amplitude=grace_amplitude,
        active_duration_ms=grace_duration_ms + gap_duration_ms,  # Active through gap
        tonic_config=TonicConfig(
            level=grace_sustain_level,  # Small tonic to maintain ~0.15 floor
            start_ms=10.0,  # Start after attack peak
            end_before_ms=0.0,  # Continue through gap until handoff
        ),
    )

    main_role = VoiceRole(
        name="main",
        onset_time_ms=main_onset,
        pulse_amplitude=main_amplitude,
        active_duration_ms=None,  # Active until end
        tonic_config=TonicConfig(
            level=main_tonic_level,
            start_ms=30.0,  # Start tonic after attack settles
            end_before_ms=80.0,  # Stop before release
        ),
    )

    return ArticulationConfig(
        name="grace_note",
        voice_roles=[grace_role, main_role],
        total_duration_ms=total_duration,
        combination_mode=CombinationMode.MAX,  # Perceptual maximum
    )


def create_trill_config(
    duration_ms: float = 500.0,
    rate_hz: float = 8.0,
    upper_amplitude: float = 0.9,
    lower_amplitude: float = 0.9,
    tonic_level: float = 0.08,
) -> ArticulationConfig:
    """Create a trill articulation configuration.

    Two alternating voices (upper and lower notes) at specified rate.

    Args:
        duration_ms: Total trill duration
        rate_hz: Trill rate in Hz (alternations per second)
        upper_amplitude: Upper note amplitude
        lower_amplitude: Lower note amplitude
        tonic_level: Tonic level for both voices

    Returns:
        ArticulationConfig for a trill
    """
    period_ms = 1000.0 / rate_hz
    half_period = period_ms / 2

    # Upper voice starts at t=0
    upper_role = VoiceRole(
        name="upper",
        onset_time_ms=0.0,
        pulse_amplitude=upper_amplitude,
        active_duration_ms=None,
        tonic_config=TonicConfig(
            level=tonic_level,
            start_ms=half_period,  # Tonic during sustain
            end_before_ms=50.0,
        ),
    )

    # Lower voice starts at half period
    lower_role = VoiceRole(
        name="lower",
        onset_time_ms=half_period,
        pulse_amplitude=lower_amplitude,
        active_duration_ms=None,
        tonic_config=TonicConfig(
            level=tonic_level,
            start_ms=half_period,
            end_before_ms=50.0,
        ),
    )

    return ArticulationConfig(
        name="trill",
        voice_roles=[upper_role, lower_role],
        total_duration_ms=duration_ms,
        combination_mode=CombinationMode.MAX,
    )


def create_mordent_config(
    main_duration_ms: float = 300.0,
    ornament_duration_ms: float = 15.0,
    main_amplitude: float = 1.0,
    ornament_amplitude: float = 0.6,
    tonic_level: float = 0.10,
) -> ArticulationConfig:
    """Create a mordent articulation configuration.

    Three voices: main note, quick upper auxiliary, back to main.
    The mordent is: main → upper → main (very quick ornament).

    Args:
        main_duration_ms: Duration of final main note portion
        ornament_duration_ms: Duration of each ornament note
        main_amplitude: Main note amplitude
        ornament_amplitude: Ornament note amplitude
        tonic_level: Tonic level for main note sustain

    Returns:
        ArticulationConfig for a mordent
    """
    total_duration = ornament_duration_ms * 2 + main_duration_ms

    # First main hit
    first_main = VoiceRole(
        name="first_main",
        onset_time_ms=0.0,
        pulse_amplitude=main_amplitude,
        active_duration_ms=ornament_duration_ms,
        tonic_config=None,
    )

    # Upper auxiliary
    upper_aux = VoiceRole(
        name="upper_aux",
        onset_time_ms=ornament_duration_ms,
        pulse_amplitude=ornament_amplitude,
        active_duration_ms=ornament_duration_ms,
        tonic_config=None,
    )

    # Final main note with sustain
    final_main = VoiceRole(
        name="final_main",
        onset_time_ms=ornament_duration_ms * 2,
        pulse_amplitude=main_amplitude,
        active_duration_ms=None,
        tonic_config=TonicConfig(
            level=tonic_level,
            start_ms=20.0,
            end_before_ms=60.0,
        ),
    )

    return ArticulationConfig(
        name="mordent",
        voice_roles=[first_main, upper_aux, final_main],
        total_duration_ms=total_duration,
        combination_mode=CombinationMode.MAX,
    )
