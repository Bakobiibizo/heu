"""
Core module: HH-like Envelope Unit (HEU) and supporting components (v1.0).

Components:
    config: Pydantic configuration models (HEUConfig, HEUParamsRef, etc.)
    heu_state: State dataclasses (HEUState, HEUModulators, HEUStep)
    heu_unit: Main HHEnvelopeUnit class
    integrators: Numerical integration (semi-implicit Euler, RK2)
    safety: Homeostatic safety (ActivityMonitor, damping)

V1.0 Changes:
    - Added HEUConfig: unified configuration
    - Renamed Modulators -> HEUModulators (alias kept)
    - Renamed StepOutput -> HEUStep (alias kept)
    - HEUState: added refractory_remaining_ms, activity_ema; renamed inactivation_gate -> inhibit_gate
    - HHEnvelopeUnit.step(): added tonic parameter; simplified dynamics
"""

from .config import (
    ExperimentConfig,
    HEUConfig,
    HEUParamsRef,
    HEUThetaBounds,
    LayerConfig,
    SafetyConfig,
)
from .heu_state import HEUModulators, HEUState, HEUStep, Modulators, StepOutput
from .heu_unit import HHEnvelopeUnit
from .integrators import exponential_decay_step, rk2_step, semi_implicit_euler_step
from .safety import ActivityMonitor, apply_bounds_to_all_theta, clamp_theta

__all__ = [
    # Config (v1.0)
    "HEUConfig",
    # Config (legacy, backwards compatible)
    "HEUParamsRef",
    "HEUThetaBounds",
    "SafetyConfig",
    "LayerConfig",
    "ExperimentConfig",
    # State (v1.0)
    "HEUState",
    "HEUModulators",
    "HEUStep",
    # State (legacy aliases)
    "Modulators",
    "StepOutput",
    # Main unit
    "HHEnvelopeUnit",
    # Integrators
    "semi_implicit_euler_step",
    "rk2_step",
    "exponential_decay_step",
    # Safety
    "ActivityMonitor",
    "clamp_theta",
    "apply_bounds_to_all_theta",
]
