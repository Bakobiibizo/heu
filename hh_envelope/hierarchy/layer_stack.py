"""Multi-Scale Layer Stack
=========================

Manages three hierarchical layers with different time scales:
- Note-level (fast): Transient articulation, short envelopes
- Phrase-level (medium): Parameter modulation to note-level
- Form-level (slow): Long-term tension, density control

Higher layers modulate PARAMETERS of lower layers (attack_gain,
excitability, refractory_scale), not direct signal injection.

Usage:
    from hh_envelope.hierarchy import LayerStack
    from hh_envelope.core import LayerConfig

    stack = LayerStack(
        note_config=LayerConfig(time_scale_factor=1.0),
        phrase_config=LayerConfig(time_scale_factor=20.0),
        form_config=LayerConfig(time_scale_factor=200.0),
    )

    for excitation in excitation_signal:
        output = stack.step(excitation, dt_ms=1.0)
        print(f"Note: {output.note_envelope:.2f}, Phrase: {output.phrase_envelope:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass

from hh_envelope.core import (
    HEUModulators,
    HEUParamsRef,
    HEUThetaBounds,
    HHEnvelopeUnit,
    LayerConfig,
    SafetyConfig,
)


@dataclass
class LayerStackOutput:
    """Output from a LayerStack step."""

    note_envelope: float
    """Note-level envelope (fast, drives articulation)."""

    note_spike: bool
    """Note-level spike event."""

    note_on_event: bool
    """Note-level note-on event."""

    note_off_event: bool
    """Note-level note-off event."""

    phrase_envelope: float
    """Phrase-level envelope (medium, modulates note-level)."""

    phrase_spike: bool
    """Phrase-level spike event."""

    phrase_on_event: bool
    """Phrase-level note-on event."""

    phrase_off_event: bool
    """Phrase-level note-off event."""

    form_envelope: float
    """Form-level envelope (slow, modulates phrase-level)."""

    form_spike: bool
    """Form-level spike event."""

    form_on_event: bool
    """Form-level note-on event."""

    form_off_event: bool
    """Form-level note-off event."""


class LayerStack:
    """Multi-scale hierarchy of HEUs with different time scales.

    Three layers:
    - Note-level (fast): Handles transient articulation, short envelopes
    - Phrase-level (medium): Parameter modulation to note-level
    - Form-level (slow): Long-term tension, density control

    Critical rule: Higher layers modulate PARAMETERS, not direct signal injection.
    """

    def __init__(
        self,
        note_config: LayerConfig | None = None,
        phrase_config: LayerConfig | None = None,
        form_config: LayerConfig | None = None,
        base_params: HEUParamsRef | None = None,
        theta_bounds: HEUThetaBounds | None = None,
        safety: SafetyConfig | None = None,
    ):
        """Initialize the layer stack.

        Args:
            note_config: Configuration for note-level layer (fast)
            phrase_config: Configuration for phrase-level layer (medium)
            form_config: Configuration for form-level layer (slow)
            base_params: Base HEU parameters (scaled per layer)
            theta_bounds: Theta bounds (shared across layers)
            safety: Safety config (shared across layers)
        """
        self._note_config = note_config or LayerConfig(
            time_scale_factor=1.0, modulation_strength=0.0
        )
        self._phrase_config = phrase_config or LayerConfig(
            time_scale_factor=20.0, modulation_strength=0.3
        )
        self._form_config = form_config or LayerConfig(
            time_scale_factor=200.0, modulation_strength=0.2
        )

        self._base_params = base_params or HEUParamsRef()
        self._theta_bounds = theta_bounds or HEUThetaBounds()
        self._safety = safety or SafetyConfig()

        # Create scaled params for each layer
        note_params = self._scale_params(self._base_params, self._note_config)
        phrase_params = self._scale_params(self._base_params, self._phrase_config)
        form_params = self._scale_params(self._base_params, self._form_config)

        # Create HEUs for each layer
        self._note_unit = HHEnvelopeUnit(note_params, self._theta_bounds, self._safety)
        self._phrase_unit = HHEnvelopeUnit(phrase_params, self._theta_bounds, self._safety)
        self._form_unit = HHEnvelopeUnit(form_params, self._theta_bounds, self._safety)

        # Spike rate tracking (smoothed for higher layers)
        self._note_spike_ema: float = 0.0
        self._phrase_spike_ema: float = 0.0
        self._spike_ema_tau_ms: float = 100.0

    def _scale_params(self, base: HEUParamsRef, config: LayerConfig) -> HEUParamsRef:
        """Create scaled parameters for a layer.

        Args:
            base: Base parameters
            config: Layer configuration with time scale factor

        Returns:
            Scaled HEUParamsRef
        """
        scale = config.time_scale_factor
        return HEUParamsRef(
            total_conductance_ref=base.total_conductance_ref,
            attack_tau_ref_ms=base.attack_tau_ref_ms * scale,
            recovery_tau_ref_ms=base.recovery_tau_ref_ms * scale,
            refractory_ref_ms=base.refractory_ref_ms * scale,
            spike_threshold=base.spike_threshold,
        )

    def step(
        self,
        note_excitation: float,
        dt_ms: float,
        external_reward: float = 0.0,
    ) -> LayerStackOutput:
        """Advance all layers by one timestep.

        Processing order:
        1. Form layer (slowest, receives phrase spike rate + reward)
        2. Phrase layer (medium, receives note spike rate + form modulation)
        3. Note layer (fastest, receives excitation + phrase modulation)

        Args:
            note_excitation: External excitation to note layer
            dt_ms: Timestep in milliseconds
            external_reward: External reward/novelty signal for form layer

        Returns:
            LayerStackOutput with all layer outputs
        """
        # --- Form layer (slowest) ---
        # Input: smoothed phrase spike rate + external reward signal
        form_excitation = self._phrase_spike_ema + external_reward * 0.5
        form_output = self._form_unit.step(form_excitation, dt_ms, None)

        # Form modulation affects phrase layer parameters
        form_mod = self._form_config.modulation_strength * form_output.envelope

        # --- Phrase layer (medium) ---
        # Input: smoothed note spike rate
        phrase_excitation = self._note_spike_ema
        phrase_modulators = HEUModulators(
            attack_gain=1.0 + form_mod,  # Form boosts phrase attack
            recovery_gain=1.0,
            excitability=1.0 - 0.3 * form_mod,  # Form can dampen phrase excitability
            refractory_scale=1.0,
        )
        phrase_output = self._phrase_unit.step(phrase_excitation, dt_ms, phrase_modulators)

        # Phrase modulation affects note layer parameters
        phrase_mod = self._phrase_config.modulation_strength * phrase_output.envelope

        # --- Note layer (fastest) ---
        note_modulators = HEUModulators(
            attack_gain=1.0 + phrase_mod,  # Phrase boosts note attack for accents
            recovery_gain=1.0 - 0.2 * phrase_mod,  # Phrase slows note recovery
            excitability=1.0 + 0.2 * phrase_mod,  # Phrase increases sensitivity
            refractory_scale=1.0 - 0.1 * phrase_mod,  # Phrase allows faster repeats
        )
        note_output = self._note_unit.step(note_excitation, dt_ms, note_modulators)

        # --- Update spike rate EMAs ---
        self._update_spike_ema(note_output.spike_event, phrase_output.spike_event, dt_ms)

        return LayerStackOutput(
            note_envelope=note_output.envelope,
            note_spike=note_output.spike_event,
            note_on_event=False,  # Not tracked by HEUStep
            note_off_event=note_output.note_off_event,
            phrase_envelope=phrase_output.envelope,
            phrase_spike=phrase_output.spike_event,
            phrase_on_event=False,  # Not tracked by HEUStep
            phrase_off_event=phrase_output.note_off_event,
            form_envelope=form_output.envelope,
            form_spike=form_output.spike_event,
            form_on_event=False,  # Not tracked by HEUStep
            form_off_event=form_output.note_off_event,
        )

    def _update_spike_ema(self, note_spike: bool, phrase_spike: bool, dt_ms: float) -> None:
        """Update spike rate EMAs.

        Args:
            note_spike: Whether note layer spiked
            phrase_spike: Whether phrase layer spiked
            dt_ms: Timestep in milliseconds
        """
        alpha = dt_ms / self._spike_ema_tau_ms

        # Convert spike to instantaneous rate estimate
        note_rate = 1.0 if note_spike else 0.0
        phrase_rate = 1.0 if phrase_spike else 0.0

        self._note_spike_ema += alpha * (note_rate - self._note_spike_ema)
        self._phrase_spike_ema += alpha * (phrase_rate - self._phrase_spike_ema)

    def reset(self) -> None:
        """Reset all layers to initial state."""
        self._note_unit.reset()
        self._phrase_unit.reset()
        self._form_unit.reset()
        self._note_spike_ema = 0.0
        self._phrase_spike_ema = 0.0

    @property
    def note_unit(self) -> HHEnvelopeUnit:
        """Note-level HEU."""
        return self._note_unit

    @property
    def phrase_unit(self) -> HHEnvelopeUnit:
        """Phrase-level HEU."""
        return self._phrase_unit

    @property
    def form_unit(self) -> HHEnvelopeUnit:
        """Form-level HEU."""
        return self._form_unit
