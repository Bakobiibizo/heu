"""CouplingEngine: computes inter-voice coupling contributions.

Reads voice exports (value, activity_ema, spike events), maintains
spike impulse traces, and produces per-voice CouplingContribution
objects that voices apply to their HEU parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .intervals import get_interval_weight
from .spec import CouplingConfig


@dataclass
class VoiceExport:
    """Snapshot of a voice's observable state for coupling computation.

    Attributes:
        voice_id: Unique voice identifier.
        pitch_hint: MIDI note number (for interval weight lookup).
        value: Current envelope value (instantaneous).
        activity_ema: Smoothed activity level.
        spike_event: Whether a spike occurred this timestep.
    """

    voice_id: int
    pitch_hint: int | None = None
    value: float = 0.0
    activity_ema: float = 0.0
    spike_event: bool = False


@dataclass
class CouplingContribution:
    """Coupling effects to apply to a voice's HEU.

    Depending on CouplingConfig.site, one of these fields will be
    non-default. The voice applies all of them.

    Attributes:
        excitation_add: Additive coupling to excitation signal.
        attack_gain_mul: Multiplicative coupling to attack_gain.
        threshold_add: Additive coupling to spike threshold.
    """

    excitation_add: float = 0.0
    attack_gain_mul: float = 1.0
    threshold_add: float = 0.0


class CouplingEngine:
    """Computes coupling contributions between voices.

    Maintains spike impulse traces (exponentially decaying) and
    computes per-voice contributions based on interval weights
    and the configured signal/site/strength.

    Usage:
        engine = CouplingEngine(num_voices=2, config=CouplingConfig())
        # Each timestep:
        contributions = engine.step(voice_exports, dt_ms=1.0)
        # Apply contributions[i] to voice i
    """

    def __init__(self, num_voices: int, config: CouplingConfig):
        """Initialize the coupling engine.

        Args:
            num_voices: Number of voices to track.
            config: Coupling configuration.
        """
        self._num_voices = num_voices
        self._config = config
        self._impulse_traces: list[float] = [0.0] * num_voices

    @property
    def config(self) -> CouplingConfig:
        """Current coupling configuration."""
        return self._config

    @property
    def impulse_traces(self) -> list[float]:
        """Current spike impulse trace values (read-only copy)."""
        return self._impulse_traces.copy()

    def reset(self) -> None:
        """Reset all impulse traces to zero."""
        self._impulse_traces = [0.0] * self._num_voices

    def step(self, voice_exports: list[VoiceExport], dt_ms: float) -> list[CouplingContribution]:
        """Compute coupling contributions for all voices.

        Args:
            voice_exports: Current state exports from each voice.
            dt_ms: Timestep in milliseconds.

        Returns:
            List of CouplingContribution, one per voice (same order as exports).
        """
        num = len(voice_exports)
        config = self._config

        # 1. Update impulse traces: decay then add spike
        decay_factor = math.exp(-dt_ms / config.impulse_kernel_tau_ms)
        for idx in range(num):
            self._impulse_traces[idx] *= decay_factor
            if voice_exports[idx].spike_event:
                self._impulse_traces[idx] += 1.0

        # 2. For each voice, compute coupling from all other voices
        contributions: list[CouplingContribution] = []
        for target_idx in range(num):
            coupling_raw = 0.0
            target_export = voice_exports[target_idx]

            for source_idx in range(num):
                if source_idx == target_idx:
                    continue

                source_export = voice_exports[source_idx]

                # Interval weight
                weight = get_interval_weight(target_export.pitch_hint, source_export.pitch_hint)
                weight *= config.interval_weight_scale

                # Signal selection
                if config.signal == "value":
                    signal_value = source_export.value
                elif config.signal == "activity_ema":
                    signal_value = source_export.activity_ema
                elif config.signal == "spike_impulse":
                    signal_value = self._impulse_traces[source_idx]
                else:
                    signal_value = 0.0

                coupling_raw += weight * signal_value * config.strength

            # 3. Clamp to modulation_clip
            coupling_raw = max(-config.modulation_clip, min(config.modulation_clip, coupling_raw))

            # 4. Map to site
            contrib = CouplingContribution()
            if config.site == "excitation_add":
                contrib = CouplingContribution(excitation_add=coupling_raw)
            elif config.site == "attack_gain_mul":
                gain_mul = 1.0 + coupling_raw
                gain_mul = max(
                    config.min_attack_gain_mul, min(config.max_attack_gain_mul, gain_mul)
                )
                contrib = CouplingContribution(attack_gain_mul=gain_mul)
            elif config.site == "threshold_add":
                contrib = CouplingContribution(threshold_add=coupling_raw)

            contributions.append(contrib)

        return contributions
