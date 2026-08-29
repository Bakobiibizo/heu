"""
HEU Configuration Models (v1.0)
===============================

Pydantic models for all HEU configuration.

Components:
-----------
1. HEUConfig: Unified configuration combining params, bounds, safety, and gate dynamics
2. HEUParamsRef: Reference values for HEU parameters (backwards compatibility)
3. HEUThetaBounds: Bounds for learnable parameters (backwards compatibility)
4. SafetyConfig: Homeostatic safety configuration (backwards compatibility)
5. LayerConfig: Configuration for hierarchy layers
6. ExperimentConfig: Configuration for experiments

Time Unit Convention:
--------------------
- Internal: All code uses milliseconds (tau_ms, dt_ms, etc.)
- Public API: step(dt_sec) accepts seconds, converts to ms at the edge

Usage:
------
    from hh_envelope.core.config import HEUConfig, HEUParamsRef, SafetyConfig

    # New unified config (recommended)
    config = HEUConfig()

    # Legacy configs (backwards compatible)
    params = HEUParamsRef()
    safety = SafetyConfig(activity_ceiling=0.9)
"""

from typing import Literal

from pydantic import BaseModel, Field


class HEUConfig(BaseModel):
    """Unified configuration for HH-like Envelope Unit.

    Combines reference parameters, theta bounds, safety config,
    and gate dynamics into a single configuration object.

    All time units are in MILLISECONDS internally.
    """

    # Reference parameters
    total_conductance_ref: float = Field(
        default=2.5,
        gt=0,
        description="Reference total conductance (attack + recovery)",
    )
    attack_tau_ref_ms: float = Field(
        default=5.0,
        gt=0,
        description="Reference attack time constant in ms (fast, ~5ms)",
    )
    recovery_tau_ref_ms: float = Field(
        default=50.0,
        gt=0,
        description="Reference recovery time constant in ms (slow, ~50ms)",
    )
    refractory_ref_ms: float = Field(
        default=20.0,
        ge=0,
        description="Reference refractory period in ms",
    )
    leak_strength: float = Field(
        default=0.1,
        ge=0,
        description="Leak conductance for passive decay toward rest",
    )

    # Gate dynamics (sigmoid approach)
    attack_slope: float = Field(
        default=5.0,
        gt=0,
        description="Steepness of attack gate sigmoid (lower = better amplitude sensitivity)",
    )
    attack_bias: float = Field(
        default=0.5,
        description="Threshold for attack gate activation (higher = less resting gate buildup)",
    )
    recovery_slope: float = Field(
        default=4.0,
        gt=0,
        description="Steepness of recovery gate sigmoid",
    )
    recovery_bias: float = Field(
        default=0.2,
        description="Threshold for recovery gate activation",
    )
    inhibit_slope: float = Field(
        default=6.0,
        gt=0,
        description="Steepness of inhibit gate sigmoid",
    )
    inhibit_bias: float = Field(
        default=0.5,
        description="Threshold for inhibit gate deactivation",
    )
    enable_inhibit_gate: bool = Field(
        default=False,
        description="Whether to use the inhibit gate (default off for simplicity)",
    )

    # Theta bounds
    theta_total_bounds: tuple[float, float] = Field(
        default=(-2.0, 2.0),
        description="Bounds for total conductance theta (log scale)",
    )
    theta_balance_bounds: tuple[float, float] = Field(
        default=(-3.0, 3.0),
        description="Bounds for attack/recovery balance theta (sigmoid scale)",
    )
    theta_attack_tau_bounds: tuple[float, float] = Field(
        default=(-2.0, 2.0),
        description="Bounds for attack tau theta (log scale)",
    )
    theta_recovery_tau_bounds: tuple[float, float] = Field(
        default=(-2.0, 2.0),
        description="Bounds for recovery tau theta (log scale)",
    )

    # Spike & envelope
    spike_threshold: float = Field(
        default=0.8,
        gt=0,
        le=1.0,
        description="Envelope threshold for spike detection",
    )
    note_off_threshold: float = Field(
        default=0.1,
        ge=0,
        lt=1.0,
        description="Envelope threshold for note-off detection (for MIDI export)",
    )
    envelope_mode: Literal["clip", "sigmoid"] = Field(
        default="clip",
        description="How to normalize envelope: clip to [0,1] or apply sigmoid",
    )

    # Safety / homeostasis
    homeostasis_tau_ms: float = Field(
        default=500.0,
        gt=0,
        description="EMA time constant for activity tracking in ms",
    )
    activity_target: float = Field(
        default=0.3,
        gt=0,
        le=1.0,
        description="Target mean activity level",
    )
    activity_ceiling: float = Field(
        default=0.8,
        gt=0,
        le=1.0,
        description="Hard ceiling - freeze learning above this",
    )
    min_damping_factor: float = Field(
        default=0.1,
        gt=0,
        le=1.0,
        description="Minimum damping factor (never damp below this)",
    )

    # Integrator
    integrator: Literal["euler", "semi_implicit", "rk2"] = Field(
        default="semi_implicit",
        description="Numerical integrator to use",
    )

    # Lookahead (compressor-style)
    lookahead_ms: float = Field(
        default=0.0,
        ge=0,
        description="Lookahead time in ms (0 = disabled). Anticipates peaks for gain control.",
    )
    lookahead_mode: Literal["peak", "rms"] = Field(
        default="peak",
        description="Lookahead detection mode: peak (max) or rms (smoothed)",
    )
    lookahead_ceiling: float = Field(
        default=0.95,
        gt=0,
        le=1.0,
        description="Target ceiling for lookahead gain control (prevents saturation)",
    )

    # Oversampling
    oversample_factor: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Internal oversampling factor (1 = no oversampling)",
    )

    def to_params_ref(self) -> "HEUParamsRef":
        """Convert to legacy HEUParamsRef for backwards compatibility."""
        return HEUParamsRef(
            total_conductance_ref=self.total_conductance_ref,
            attack_tau_ref_ms=self.attack_tau_ref_ms,
            recovery_tau_ref_ms=self.recovery_tau_ref_ms,
            refractory_ref_ms=self.refractory_ref_ms,
            spike_threshold=self.spike_threshold,
        )

    def to_theta_bounds(self) -> "HEUThetaBounds":
        """Convert to legacy HEUThetaBounds for backwards compatibility."""
        return HEUThetaBounds(
            theta_total=self.theta_total_bounds,
            theta_balance=self.theta_balance_bounds,
            theta_attack_tau=self.theta_attack_tau_bounds,
            theta_recovery_tau=self.theta_recovery_tau_bounds,
        )

    def to_safety_config(self) -> "SafetyConfig":
        """Convert to legacy SafetyConfig for backwards compatibility."""
        return SafetyConfig(
            homeostasis_tau_ms=self.homeostasis_tau_ms,
            activity_target=self.activity_target,
            activity_ceiling=self.activity_ceiling,
            min_damping_factor=self.min_damping_factor,
        )


class HEUParamsRef(BaseModel):
    """Reference values for HEU parameters.

    These are the baseline values that theta parameters scale.
    Using exp(theta) * ref allows safe reparameterization for learning.
    """

    total_conductance_ref: float = Field(
        default=2.5,
        gt=0,
        description="Reference total conductance (attack + recovery)",
    )
    attack_tau_ref_ms: float = Field(
        default=5.0,
        gt=0,
        description="Reference attack time constant in ms (fast, ~5ms)",
    )
    recovery_tau_ref_ms: float = Field(
        default=50.0,
        gt=0,
        description="Reference recovery time constant in ms (slow, ~50ms)",
    )
    refractory_ref_ms: float = Field(
        default=20.0,
        ge=0,
        description="Reference refractory period in ms",
    )
    spike_threshold: float = Field(
        default=0.8,
        gt=0,
        le=1.0,
        description="Envelope threshold for spike detection",
    )


class HEUThetaBounds(BaseModel):
    """Bounds for learnable theta parameters.

    Parameters are learned in theta-space and transformed:
    - conductance = exp(theta_total) * ref
    - ratio = sigmoid(theta_balance)
    - tau = exp(theta_tau) * ref

    Bounds prevent extreme values during learning.
    """

    theta_total: tuple[float, float] = Field(
        default=(-2.0, 2.0),
        description="Bounds for total conductance theta (log scale)",
    )
    theta_balance: tuple[float, float] = Field(
        default=(-3.0, 3.0),
        description="Bounds for attack/recovery balance theta (sigmoid scale)",
    )
    theta_attack_tau: tuple[float, float] = Field(
        default=(-2.0, 2.0),
        description="Bounds for attack tau theta (log scale)",
    )
    theta_recovery_tau: tuple[float, float] = Field(
        default=(-2.0, 2.0),
        description="Bounds for recovery tau theta (log scale)",
    )


class SafetyConfig(BaseModel):
    """Homeostatic safety configuration.

    Safety is 'do no harm' logic - it only constrains, never adds energy.
    Prevents runaway dynamics during simulation and learning.
    """

    homeostasis_tau_ms: float = Field(
        default=500.0,
        gt=0,
        description="EMA time constant for activity tracking in ms",
    )
    activity_target: float = Field(
        default=0.3,
        gt=0,
        le=1.0,
        description="Target mean activity level",
    )
    activity_ceiling: float = Field(
        default=0.8,
        gt=0,
        le=1.0,
        description="Hard ceiling - freeze learning above this",
    )
    min_damping_factor: float = Field(
        default=0.1,
        gt=0,
        le=1.0,
        description="Minimum damping factor (never damp below this)",
    )


class LayerConfig(BaseModel):
    """Configuration for a single layer in the hierarchy.

    Higher layers have longer time constants and modulate
    parameters of lower layers.
    """

    time_scale_factor: float = Field(
        default=1.0,
        gt=0,
        description="Multiplier for time constants relative to base",
    )
    modulation_strength: float = Field(
        default=0.5,
        ge=0,
        le=1.0,
        description="How strongly this layer modulates the layer below",
    )


class ExperimentConfig(BaseModel):
    """Configuration for running experiments."""

    dt_ms: float = Field(
        default=1.0,
        gt=0,
        description="Timestep in milliseconds (1 kHz control rate recommended)",
    )
    duration_sec: float = Field(
        default=10.0,
        gt=0,
        description="Experiment duration in seconds",
    )
    random_seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility (None = random)",
    )
