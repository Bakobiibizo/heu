"""Coupling configuration for inter-voice dynamics.

Defines CouplingConfig with configurable signal source, injection site,
strength, and safety clamps.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CouplingConfig(BaseModel):
    """Configuration for inter-voice coupling.

    Attributes:
        signal: Which voice state to read as coupling source.
            - "value": instantaneous envelope value (needs overlap)
            - "activity_ema": smoothed activity (works at any tempo)
            - "spike_impulse": decaying trace triggered by spikes
        site: Where to inject the coupling effect.
            - "excitation_add": add to excitation signal
            - "attack_gain_mul": multiply attack_gain (parameter modulation)
            - "threshold_add": shift spike threshold
        strength: Overall coupling strength multiplier.
        interval_weight_scale: Scales the interval weight table values.
        impulse_kernel_tau_ms: Decay time constant for spike_impulse traces.
        modulation_clip: Symmetric clamp on raw coupling value.
        min_attack_gain_mul: Floor for attack_gain multiplier.
        max_attack_gain_mul: Ceiling for attack_gain multiplier.
    """

    signal: Literal["value", "activity_ema", "spike_impulse"] = "activity_ema"
    site: Literal["excitation_add", "attack_gain_mul", "threshold_add"] = "attack_gain_mul"
    strength: float = Field(default=0.1, ge=0.0, le=2.0)
    interval_weight_scale: float = Field(default=1.0, ge=0.0, le=5.0)
    impulse_kernel_tau_ms: float = Field(default=100.0, gt=0.0, le=2000.0)
    modulation_clip: float = Field(default=0.5, ge=0.0, le=5.0)
    min_attack_gain_mul: float = Field(default=0.5, gt=0.0, le=1.0)
    max_attack_gain_mul: float = Field(default=1.5, ge=1.0, le=3.0)
