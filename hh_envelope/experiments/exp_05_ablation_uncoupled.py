"""Experiment 05: Uncoupled-Parameter Ablation
===============================================

Tests whether the expressivity separation from Exp 04 persists when the
balance coupling is removed.

The coupled HEU constrains attack and recovery strengths via:
    alpha = g_total * sigmoid(theta_balance)
    beta  = g_total * (1 - sigmoid(theta_balance))

The uncoupled HEU removes this constraint:
    alpha = exp(theta_attack_strength) * alpha_ref
    beta  = exp(theta_recovery_strength) * beta_ref

Both have 4 learnable parameters and identical dynamics otherwise.

Three conditions tested against the same two-voice target (from Exp 04):
    1. Coupled single-voice (4 params, balance constraint)
    2. Uncoupled single-voice (4 params, independent alpha/beta)
    3. Two-voice coupled (8 params, baseline)

Possible outcomes:
    A. Uncoupled still fails → separation is due to single-state limit
    B. Uncoupled succeeds → separation is specifically about balance coupling
    C. Partial improvement → both factors contribute

Success criteria:
    - All three conditions optimized with identical CMA-ES settings
    - Clear quantitative comparison (loss, regional MSE, params at bounds)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hh_envelope.core import HEUParamsRef, HEUThetaBounds, HHEnvelopeUnit, SafetyConfig
from hh_envelope.core.heu_unit import sigmoid
from hh_envelope.core.safety import clamp_theta
from hh_envelope.learning.loss import combined_loss
from hh_envelope.polyphony.articulation import (
    ArticulatedNote,
    ArticulationConfig,
    CombinationMode,
    TonicConfig,
    VoiceRole,
)
from hh_envelope.utils.signals import target_envelope_grace_note_two_voice


# ============================================================================
# Uncoupled HEU
# ============================================================================


class UncoupledHEU(HHEnvelopeUnit):
    """HEU variant with independent attack and recovery strengths.

    Removes the balance coupling constraint. Instead of:
        alpha = g_total * sigmoid(theta_balance)
        beta  = g_total * (1 - sigmoid(theta_balance))

    Uses:
        alpha = exp(theta_attack_strength) * alpha_ref
        beta  = exp(theta_recovery_strength) * beta_ref

    This keeps the total parameter count at 4 (same as coupled)
    while allowing alpha and beta to vary independently.
    """

    PARAM_NAMES = [
        "theta_attack_strength",
        "theta_recovery_strength",
        "theta_attack_tau",
        "theta_recovery_tau",
    ]

    # Bounds for the uncoupled parameters
    THETA_ATTACK_BOUNDS = (-3.0, 3.0)
    THETA_RECOVERY_BOUNDS = (-3.0, 3.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Independent strength parameters (replace theta_total + theta_balance)
        self._theta_attack_strength: float = 0.0
        self._theta_recovery_strength: float = 0.0
        # Reference values: half of total_conductance_ref each
        # At theta=0, alpha=beta=1.25 (same as coupled with theta_total=0, theta_balance=0)
        self._alpha_ref = self.params_ref.total_conductance_ref / 2.0
        self._beta_ref = self.params_ref.total_conductance_ref / 2.0

    @property
    def attack_strength(self) -> float:
        """Independent attack conductance (no balance coupling)."""
        return math.exp(self._theta_attack_strength) * self._alpha_ref

    @property
    def recovery_strength(self) -> float:
        """Independent recovery conductance (no balance coupling)."""
        return math.exp(self._theta_recovery_strength) * self._beta_ref

    def set_uncoupled_theta(
        self,
        theta_attack_strength: float | None = None,
        theta_recovery_strength: float | None = None,
        theta_attack_tau: float | None = None,
        theta_recovery_tau: float | None = None,
    ) -> None:
        """Set uncoupled theta parameters with bounds clamping."""
        if theta_attack_strength is not None:
            self._theta_attack_strength = clamp_theta(
                theta_attack_strength, self.THETA_ATTACK_BOUNDS
            )
        if theta_recovery_strength is not None:
            self._theta_recovery_strength = clamp_theta(
                theta_recovery_strength, self.THETA_RECOVERY_BOUNDS
            )
        if theta_attack_tau is not None:
            self._theta_attack_tau = clamp_theta(
                theta_attack_tau, self.theta_bounds.theta_attack_tau
            )
        if theta_recovery_tau is not None:
            self._theta_recovery_tau = clamp_theta(
                theta_recovery_tau, self.theta_bounds.theta_recovery_tau
            )

    def set_uncoupled_theta_from_dict(self, d: dict[str, float]) -> None:
        """Set uncoupled theta from dictionary."""
        self.set_uncoupled_theta(
            theta_attack_strength=d.get("theta_attack_strength"),
            theta_recovery_strength=d.get("theta_recovery_strength"),
            theta_attack_tau=d.get("theta_attack_tau"),
            theta_recovery_tau=d.get("theta_recovery_tau"),
        )

    def get_uncoupled_theta(self) -> dict[str, float]:
        """Get current uncoupled theta parameters."""
        return {
            "theta_attack_strength": self._theta_attack_strength,
            "theta_recovery_strength": self._theta_recovery_strength,
            "theta_attack_tau": self._theta_attack_tau,
            "theta_recovery_tau": self._theta_recovery_tau,
        }

    def get_uncoupled_bounds(self) -> tuple[list[float], list[float]]:
        """Get lower and upper bounds for CMA-ES."""
        lower = [
            self.THETA_ATTACK_BOUNDS[0],
            self.THETA_RECOVERY_BOUNDS[0],
            self.theta_bounds.theta_attack_tau[0],
            self.theta_bounds.theta_recovery_tau[0],
        ]
        upper = [
            self.THETA_ATTACK_BOUNDS[1],
            self.THETA_RECOVERY_BOUNDS[1],
            self.theta_bounds.theta_attack_tau[1],
            self.theta_bounds.theta_recovery_tau[1],
        ]
        return lower, upper


# ============================================================================
# Shared utilities (same as exp_04)
# ============================================================================


@dataclass
class RegionalMSE:
    """MSE broken down by region of the envelope."""
    grace_mse: float
    gap_mse: float
    main_attack_mse: float
    main_sustain_mse: float
    total_mse: float


@dataclass
class AblationResult:
    """Result from one condition of the ablation."""
    condition: str
    best_theta: dict[str, float]
    best_loss: float
    regional_mse: RegionalMSE
    iterations: int
    predicted: np.ndarray
    parameters_at_bounds: int
    parameter_analysis: dict[str, str]
    n_params: int


def create_input_for_two_voice_target(
    main_duration_ms: float,
    dt_ms: float = 1.0,
) -> np.ndarray:
    """Create input signal matching the two-voice target generation."""
    total_duration = 50.0 + main_duration_ms
    num_samples = int(total_duration / dt_ms)
    signal = np.zeros(num_samples)

    # Grace pulse at t=0
    signal[0] = 1.5

    # Grace tonic (t=2 to t=25)
    grace_tonic_start = int(2.0 / dt_ms)
    grace_tonic_end = int(25.0 / dt_ms)
    for i in range(grace_tonic_start, min(grace_tonic_end, num_samples)):
        signal[i] = 0.40

    # Main tonic (t=50 onward, ends 60ms before total)
    main_tonic_start = int(50.0 / dt_ms)
    main_tonic_end = int((total_duration - 60.0) / dt_ms)
    for i in range(main_tonic_start, min(main_tonic_end, num_samples)):
        signal[i] = 0.35

    return signal


def compute_regional_mse(
    predicted: np.ndarray,
    target: np.ndarray,
    dt_ms: float = 1.0,
) -> RegionalMSE:
    """Compute MSE broken down by region."""
    min_len = min(len(predicted), len(target))
    predicted = predicted[:min_len]
    target = target[:min_len]

    grace_end = int(25.0 / dt_ms)
    gap_end = int(50.0 / dt_ms)
    attack_end = int(80.0 / dt_ms)

    grace_mse = float(np.mean((predicted[:grace_end] - target[:grace_end]) ** 2))

    if gap_end <= min_len:
        gap_mse = float(
            np.mean((predicted[grace_end:gap_end] - target[grace_end:gap_end]) ** 2)
        )
    else:
        gap_mse = 0.0

    if attack_end < min_len:
        attack_mse = float(
            np.mean((predicted[gap_end:attack_end] - target[gap_end:attack_end]) ** 2)
        )
    else:
        attack_mse = float(np.mean((predicted[gap_end:] - target[gap_end:]) ** 2))

    if attack_end < min_len:
        sustain_mse = float(np.mean((predicted[attack_end:] - target[attack_end:]) ** 2))
    else:
        sustain_mse = 0.0

    total_mse = float(np.mean((predicted - target) ** 2))

    return RegionalMSE(
        grace_mse=grace_mse,
        gap_mse=gap_mse,
        main_attack_mse=attack_mse,
        main_sustain_mse=sustain_mse,
        total_mse=total_mse,
    )


def analyze_bounds(
    theta: dict[str, float],
    bounds_map: dict[str, tuple[float, float]],
    tolerance: float = 0.1,
) -> tuple[int, dict[str, str]]:
    """Analyze how many parameters are at bounds."""
    analysis = {}
    at_bounds = 0

    for name, (lo, hi) in bounds_map.items():
        value = theta.get(name, 0.0)
        range_size = hi - lo

        if abs(value - lo) < tolerance * range_size:
            analysis[name] = f"{value:.3f} AT LOWER BOUND ({lo})"
            at_bounds += 1
        elif abs(value - hi) < tolerance * range_size:
            analysis[name] = f"{value:.3f} AT UPPER BOUND ({hi})"
            at_bounds += 1
        else:
            analysis[name] = f"{value:.3f} (within bounds)"

    return at_bounds, analysis


# ============================================================================
# Condition 1: Coupled single-voice (same as Exp 04)
# ============================================================================


def run_coupled_single_voice(
    target: np.ndarray,
    input_signal: np.ndarray,
    dt_ms: float,
    max_iterations: int,
) -> AblationResult:
    """Run coupled single-voice optimization (baseline from Exp 04)."""
    print("\n" + "=" * 60)
    print("CONDITION 1: COUPLED SINGLE-VOICE (balance constraint)")
    print("=" * 60)

    import cma

    bounds_obj = HEUThetaBounds()
    lower_bounds = [
        bounds_obj.theta_total[0],
        bounds_obj.theta_balance[0],
        bounds_obj.theta_attack_tau[0],
        bounds_obj.theta_recovery_tau[0],
    ]
    upper_bounds = [
        bounds_obj.theta_total[1],
        bounds_obj.theta_balance[1],
        bounds_obj.theta_attack_tau[1],
        bounds_obj.theta_recovery_tau[1],
    ]

    def objective(arr):
        unit = HHEnvelopeUnit(
            params_ref=HEUParamsRef(),
            theta_bounds=HEUThetaBounds(),
            safety=SafetyConfig(),
        )
        unit.set_theta(
            theta_total=arr[0],
            theta_balance=arr[1],
            theta_attack_tau=arr[2],
            theta_recovery_tau=arr[3],
        )
        unit.reset()
        envelope = np.zeros(len(input_signal))
        for i, exc in enumerate(input_signal):
            output = unit.step(exc, dt_ms)
            envelope[i] = output.envelope
        min_len = min(len(envelope), len(target))
        return combined_loss(envelope[:min_len], target[:min_len], 0.01)

    starting_points = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([-1.0, 2.0, 0.3, 0.5]),
        np.array([-0.5, 1.5, -0.5, -0.5]),
        np.array([0.5, -1.0, 0.5, -0.3]),
    ]

    best_loss = float("inf")
    best_arr = None
    total_iters = 0

    for restart_idx, x0 in enumerate(starting_points):
        print(f"  Restart {restart_idx + 1}/{len(starting_points)}...", end="", flush=True)
        es = cma.CMAEvolutionStrategy(
            x0, 0.5,
            {
                "bounds": [lower_bounds, upper_bounds],
                "maxiter": max_iterations // len(starting_points),
                "seed": 42 + restart_idx,
                "verbose": -9,
            },
        )
        while not es.stop():
            solutions = es.ask()
            losses = [objective(s) for s in solutions]
            es.tell(solutions, losses)
            total_iters += 1
            best_idx = np.argmin(losses)
            if losses[best_idx] < best_loss:
                best_loss = losses[best_idx]
                best_arr = solutions[best_idx].copy()
        print(f" loss={best_loss:.6f}")

    if best_arr is None:
        best_arr = np.zeros(4)

    best_theta = {
        "theta_total": float(best_arr[0]),
        "theta_balance": float(best_arr[1]),
        "theta_attack_tau": float(best_arr[2]),
        "theta_recovery_tau": float(best_arr[3]),
    }

    # Generate final prediction
    unit = HHEnvelopeUnit(
        params_ref=HEUParamsRef(), theta_bounds=HEUThetaBounds(), safety=SafetyConfig()
    )
    unit.set_theta_from_dict(best_theta)
    unit.reset()
    predicted = np.zeros(len(input_signal))
    for i, exc in enumerate(input_signal):
        output = unit.step(exc, dt_ms)
        predicted[i] = output.envelope

    min_len = min(len(predicted), len(target))
    regional_mse = compute_regional_mse(predicted[:min_len], target[:min_len], dt_ms)

    bounds_map = {
        "theta_total": bounds_obj.theta_total,
        "theta_balance": bounds_obj.theta_balance,
        "theta_attack_tau": bounds_obj.theta_attack_tau,
        "theta_recovery_tau": bounds_obj.theta_recovery_tau,
    }
    at_bounds, analysis = analyze_bounds(best_theta, bounds_map)

    # Report physical alpha/beta for comparison
    alpha = math.exp(best_theta["theta_total"]) * 2.5 * sigmoid(best_theta["theta_balance"])
    beta = math.exp(best_theta["theta_total"]) * 2.5 * (1.0 - sigmoid(best_theta["theta_balance"]))
    print(f"  Loss: {best_loss:.6f}")
    print(f"  alpha={alpha:.4f}, beta={beta:.4f} (coupled: alpha+beta={alpha+beta:.4f})")
    print(f"  Parameters at bounds: {at_bounds}/4")
    for name, status in analysis.items():
        print(f"    {name}: {status}")

    return AblationResult(
        condition="coupled_single_voice",
        best_theta=best_theta,
        best_loss=best_loss,
        regional_mse=regional_mse,
        iterations=total_iters,
        predicted=predicted[:min_len],
        parameters_at_bounds=at_bounds,
        parameter_analysis=analysis,
        n_params=4,
    )


# ============================================================================
# Condition 2: Uncoupled single-voice (ablation)
# ============================================================================


def run_uncoupled_single_voice(
    target: np.ndarray,
    input_signal: np.ndarray,
    dt_ms: float,
    max_iterations: int,
) -> AblationResult:
    """Run uncoupled single-voice optimization (the ablation)."""
    print("\n" + "=" * 60)
    print("CONDITION 2: UNCOUPLED SINGLE-VOICE (independent alpha/beta)")
    print("=" * 60)

    import cma

    ref_unit = UncoupledHEU(
        params_ref=HEUParamsRef(), theta_bounds=HEUThetaBounds(), safety=SafetyConfig()
    )
    lower_bounds, upper_bounds = ref_unit.get_uncoupled_bounds()

    def objective(arr):
        unit = UncoupledHEU(
            params_ref=HEUParamsRef(), theta_bounds=HEUThetaBounds(), safety=SafetyConfig()
        )
        unit.set_uncoupled_theta(
            theta_attack_strength=arr[0],
            theta_recovery_strength=arr[1],
            theta_attack_tau=arr[2],
            theta_recovery_tau=arr[3],
        )
        unit.reset()
        envelope = np.zeros(len(input_signal))
        for i, exc in enumerate(input_signal):
            output = unit.step(exc, dt_ms)
            envelope[i] = output.envelope
        min_len = min(len(envelope), len(target))
        return combined_loss(envelope[:min_len], target[:min_len], 0.01)

    starting_points = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, -1.0, 0.3, 0.5]),
        np.array([-0.5, 0.5, -0.5, -0.5]),
        np.array([0.5, -0.5, 0.5, -0.3]),
    ]

    best_loss = float("inf")
    best_arr = None
    total_iters = 0

    for restart_idx, x0 in enumerate(starting_points):
        print(f"  Restart {restart_idx + 1}/{len(starting_points)}...", end="", flush=True)
        es = cma.CMAEvolutionStrategy(
            x0, 0.5,
            {
                "bounds": [lower_bounds, upper_bounds],
                "maxiter": max_iterations // len(starting_points),
                "seed": 42 + restart_idx,
                "verbose": -9,
            },
        )
        while not es.stop():
            solutions = es.ask()
            losses = [objective(s) for s in solutions]
            es.tell(solutions, losses)
            total_iters += 1
            best_idx = np.argmin(losses)
            if losses[best_idx] < best_loss:
                best_loss = losses[best_idx]
                best_arr = solutions[best_idx].copy()
        print(f" loss={best_loss:.6f}")

    if best_arr is None:
        best_arr = np.zeros(4)

    best_theta = {
        "theta_attack_strength": float(best_arr[0]),
        "theta_recovery_strength": float(best_arr[1]),
        "theta_attack_tau": float(best_arr[2]),
        "theta_recovery_tau": float(best_arr[3]),
    }

    # Generate final prediction
    unit = UncoupledHEU(
        params_ref=HEUParamsRef(), theta_bounds=HEUThetaBounds(), safety=SafetyConfig()
    )
    unit.set_uncoupled_theta_from_dict(best_theta)
    unit.reset()
    predicted = np.zeros(len(input_signal))
    for i, exc in enumerate(input_signal):
        output = unit.step(exc, dt_ms)
        predicted[i] = output.envelope

    min_len = min(len(predicted), len(target))
    regional_mse = compute_regional_mse(predicted[:min_len], target[:min_len], dt_ms)

    bounds_map = {
        "theta_attack_strength": UncoupledHEU.THETA_ATTACK_BOUNDS,
        "theta_recovery_strength": UncoupledHEU.THETA_RECOVERY_BOUNDS,
        "theta_attack_tau": ref_unit.theta_bounds.theta_attack_tau,
        "theta_recovery_tau": ref_unit.theta_bounds.theta_recovery_tau,
    }
    at_bounds, analysis = analyze_bounds(best_theta, bounds_map)

    # Report physical alpha/beta for comparison
    alpha = math.exp(best_theta["theta_attack_strength"]) * 1.25
    beta = math.exp(best_theta["theta_recovery_strength"]) * 1.25
    print(f"  Loss: {best_loss:.6f}")
    print(f"  alpha={alpha:.4f}, beta={beta:.4f} (uncoupled: independent)")
    print(f"  Parameters at bounds: {at_bounds}/4")
    for name, status in analysis.items():
        print(f"    {name}: {status}")

    return AblationResult(
        condition="uncoupled_single_voice",
        best_theta=best_theta,
        best_loss=best_loss,
        regional_mse=regional_mse,
        iterations=total_iters,
        predicted=predicted[:min_len],
        parameters_at_bounds=at_bounds,
        parameter_analysis=analysis,
        n_params=4,
    )


# ============================================================================
# Condition 3: Two-voice coupled (same as Exp 04)
# ============================================================================


def run_two_voice(
    target: np.ndarray,
    main_duration_ms: float,
    dt_ms: float,
    max_iterations: int,
) -> AblationResult:
    """Run two-voice optimization (capability baseline from Exp 04)."""
    print("\n" + "=" * 60)
    print("CONDITION 3: TWO-VOICE COUPLED (8 params, baseline)")
    print("=" * 60)

    import cma

    total_duration = 50.0 + main_duration_ms

    grace_role = VoiceRole(
        name="grace",
        onset_time_ms=0.0,
        pulse_amplitude=1.5,
        active_duration_ms=50.0,
        tonic_config=TonicConfig(level=0.40, start_ms=2.0, end_before_ms=25.0),
    )
    main_role = VoiceRole(
        name="main",
        onset_time_ms=50.0,
        pulse_amplitude=0.0,
        active_duration_ms=None,
        tonic_config=TonicConfig(level=0.35, start_ms=0.0, end_before_ms=60.0),
    )
    config = ArticulationConfig(
        name="two_voice_ablation",
        voice_roles=[grace_role, main_role],
        total_duration_ms=total_duration,
        combination_mode=CombinationMode.MAX,
    )

    bounds_obj = HEUThetaBounds()
    lower_bounds = [
        bounds_obj.theta_total[0], bounds_obj.theta_balance[0],
        bounds_obj.theta_attack_tau[0], bounds_obj.theta_recovery_tau[0],
    ] * 2
    upper_bounds = [
        bounds_obj.theta_total[1], bounds_obj.theta_balance[1],
        bounds_obj.theta_attack_tau[1], bounds_obj.theta_recovery_tau[1],
    ] * 2

    def objective(arr):
        note = ArticulatedNote(config)
        grace_theta = {
            "theta_total": arr[0], "theta_balance": arr[1],
            "theta_attack_tau": arr[2], "theta_recovery_tau": arr[3],
        }
        main_theta = {
            "theta_total": arr[4], "theta_balance": arr[5],
            "theta_attack_tau": arr[6], "theta_recovery_tau": arr[7],
        }
        note.set_voice_theta("grace", grace_theta)
        note.set_voice_theta("main", main_theta)
        envelope = note.run_full(dt_ms)
        min_len = min(len(envelope), len(target))
        return combined_loss(envelope[:min_len], target[:min_len], 0.01)

    starting_points = [
        np.array([-0.5, 1.0, -0.5, -0.3, -0.5, 2.0, 0.3, 0.0]),
        np.array([0.2, -1.5, -0.8, 0.3, -0.3, 2.5, 0.5, -0.2]),
        np.array([-0.3, 0.5, 0.0, 0.5, -1.0, 1.5, 0.0, 0.3]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    best_loss = float("inf")
    best_arr = None
    total_iters = 0

    for restart_idx, x0 in enumerate(starting_points):
        print(f"  Restart {restart_idx + 1}/{len(starting_points)}...", end="", flush=True)
        es = cma.CMAEvolutionStrategy(
            x0, 0.5,
            {
                "bounds": [lower_bounds, upper_bounds],
                "maxiter": max_iterations // len(starting_points),
                "seed": 42 + restart_idx,
                "verbose": -9,
                "popsize": 20,
            },
        )
        while not es.stop():
            solutions = es.ask()
            losses = [objective(s) for s in solutions]
            es.tell(solutions, losses)
            total_iters += 1
            best_idx = np.argmin(losses)
            if losses[best_idx] < best_loss:
                best_loss = losses[best_idx]
                best_arr = solutions[best_idx].copy()
            if best_loss < 0.002:
                break
        print(f" loss={best_loss:.6f}")
        if best_loss < 0.002:
            break

    if best_arr is None:
        best_arr = np.zeros(8)

    best_theta = {
        "grace_theta_total": float(best_arr[0]),
        "grace_theta_balance": float(best_arr[1]),
        "grace_theta_attack_tau": float(best_arr[2]),
        "grace_theta_recovery_tau": float(best_arr[3]),
        "main_theta_total": float(best_arr[4]),
        "main_theta_balance": float(best_arr[5]),
        "main_theta_attack_tau": float(best_arr[6]),
        "main_theta_recovery_tau": float(best_arr[7]),
    }

    # Generate final prediction
    note = ArticulatedNote(config)
    grace_theta = {k.replace("grace_", ""): v for k, v in best_theta.items() if k.startswith("grace_")}
    main_theta = {k.replace("main_", ""): v for k, v in best_theta.items() if k.startswith("main_")}
    note.set_voice_theta("grace", grace_theta)
    note.set_voice_theta("main", main_theta)
    envelope = np.array(note.run_full(dt_ms))

    min_len = min(len(envelope), len(target))
    regional_mse = compute_regional_mse(envelope[:min_len], target[:min_len], dt_ms)

    # Count params at bounds
    at_bounds = 0
    analysis = {}
    param_bounds = {
        "theta_total": bounds_obj.theta_total,
        "theta_balance": bounds_obj.theta_balance,
        "theta_attack_tau": bounds_obj.theta_attack_tau,
        "theta_recovery_tau": bounds_obj.theta_recovery_tau,
    }
    for key, value in best_theta.items():
        for base_name, (lo, hi) in param_bounds.items():
            if base_name in key:
                range_size = hi - lo
                if abs(value - lo) < 0.1 * range_size:
                    analysis[key] = f"{value:.3f} AT LOWER BOUND ({lo})"
                    at_bounds += 1
                elif abs(value - hi) < 0.1 * range_size:
                    analysis[key] = f"{value:.3f} AT UPPER BOUND ({hi})"
                    at_bounds += 1
                else:
                    analysis[key] = f"{value:.3f} (within bounds)"
                break

    print(f"  Loss: {best_loss:.6f}")
    print(f"  Parameters at bounds: {at_bounds}/8")

    return AblationResult(
        condition="two_voice",
        best_theta=best_theta,
        best_loss=best_loss,
        regional_mse=regional_mse,
        iterations=total_iters,
        predicted=envelope[:min_len],
        parameters_at_bounds=at_bounds,
        parameter_analysis=analysis,
        n_params=8,
    )


# ============================================================================
# Comparison and reporting
# ============================================================================


def report_comparison(
    coupled: AblationResult,
    uncoupled: AblationResult,
    two_voice: AblationResult,
) -> dict:
    """Print full comparison of all three conditions."""
    print("\n" + "#" * 60)
    print("ABLATION RESULTS")
    print("#" * 60)

    # Overall loss comparison
    print("\n  OVERALL LOSS")
    print("  " + "-" * 50)
    print(f"  {'Condition':<30} {'Loss':>10} {'Params@Bounds':>15}")
    print(f"  {'Coupled single-voice':<30} {coupled.best_loss:>10.6f} {coupled.parameters_at_bounds}/{coupled.n_params:>11}")
    print(f"  {'Uncoupled single-voice':<30} {uncoupled.best_loss:>10.6f} {uncoupled.parameters_at_bounds}/{uncoupled.n_params:>11}")
    print(f"  {'Two-voice':<30} {two_voice.best_loss:>10.6f} {two_voice.parameters_at_bounds}/{two_voice.n_params:>11}")

    # Improvement ratios
    coupled_vs_two = coupled.best_loss / two_voice.best_loss if two_voice.best_loss > 0 else float("inf")
    uncoupled_vs_two = uncoupled.best_loss / two_voice.best_loss if two_voice.best_loss > 0 else float("inf")
    coupled_vs_uncoupled = coupled.best_loss / uncoupled.best_loss if uncoupled.best_loss > 0 else float("inf")

    print(f"\n  IMPROVEMENT RATIOS")
    print("  " + "-" * 50)
    print(f"  Coupled / Two-voice:     {coupled_vs_two:>8.2f}x")
    print(f"  Uncoupled / Two-voice:   {uncoupled_vs_two:>8.2f}x")
    print(f"  Coupled / Uncoupled:     {coupled_vs_uncoupled:>8.2f}x")

    # Regional MSE comparison
    print(f"\n  REGIONAL MSE BREAKDOWN")
    print("  " + "-" * 50)
    regions = [
        ("Grace (0-25ms)", "grace_mse"),
        ("Gap (25-50ms)", "gap_mse"),
        ("Main attack (50-80ms)", "main_attack_mse"),
        ("Main sustain (80ms+)", "main_sustain_mse"),
    ]

    print(f"  {'Region':<25} {'Coupled':>10} {'Uncoupled':>10} {'Two-voice':>10} {'Unc/Two':>8}")
    for name, attr in regions:
        c = getattr(coupled.regional_mse, attr)
        u = getattr(uncoupled.regional_mse, attr)
        t = getattr(two_voice.regional_mse, attr)
        ratio = u / t if t > 0 else float("inf")
        print(f"  {name:<25} {c:>10.6f} {u:>10.6f} {t:>10.6f} {ratio:>7.1f}x")

    # Parameter analysis
    print(f"\n  PARAMETER ANALYSIS")
    print("  " + "-" * 50)
    print(f"\n  Coupled single-voice ({coupled.parameters_at_bounds}/{coupled.n_params} at bounds):")
    for name, status in coupled.parameter_analysis.items():
        print(f"    {name}: {status}")

    print(f"\n  Uncoupled single-voice ({uncoupled.parameters_at_bounds}/{uncoupled.n_params} at bounds):")
    for name, status in uncoupled.parameter_analysis.items():
        print(f"    {name}: {status}")

    # Interpretation
    print(f"\n  INTERPRETATION")
    print("  " + "-" * 50)

    if uncoupled.best_loss > 0.005 and uncoupled_vs_two > 3.0:
        interpretation = "A"
        print("  Outcome A: UNCOUPLED STILL FAILS")
        print("  The separation is due to the single-state limitation itself,")
        print("  not just the balance coupling. Even with independent alpha/beta,")
        print("  a single envelope state cannot represent conflicting temporal regimes.")
        print(f"  Uncoupled loss ({uncoupled.best_loss:.6f}) still >> two-voice ({two_voice.best_loss:.6f})")
    elif uncoupled.best_loss < 0.005:
        interpretation = "B"
        print("  Outcome B: UNCOUPLED SUCCEEDS")
        print("  The separation is specifically about the balance coupling.")
        print("  Removing the constraint that alpha + beta = g_total allows")
        print("  the single-voice to satisfy both temporal regimes.")
        print(f"  Uncoupled loss ({uncoupled.best_loss:.6f}) matches two-voice ({two_voice.best_loss:.6f})")
    else:
        interpretation = "C"
        print("  Outcome C: PARTIAL IMPROVEMENT")
        print("  Both the coupling AND the single-state constraint contribute.")
        print("  Removing the coupling helps but doesn't fully resolve the gap.")
        improvement_pct = (1.0 - uncoupled.best_loss / coupled.best_loss) * 100
        remaining_pct = (uncoupled.best_loss / coupled.best_loss) * 100
        print(f"  Coupling accounts for ~{improvement_pct:.0f}% of the gap.")
        print(f"  Single-state constraint accounts for ~{remaining_pct:.0f}%.")

    # Gap region specific analysis
    gap_coupled = coupled.regional_mse.gap_mse
    gap_uncoupled = uncoupled.regional_mse.gap_mse
    gap_two = two_voice.regional_mse.gap_mse
    if gap_two > 0:
        print(f"\n  Gap region (critical conflict zone):")
        print(f"    Coupled:   {gap_coupled:.6f}")
        print(f"    Uncoupled: {gap_uncoupled:.6f}")
        print(f"    Two-voice: {gap_two:.6f}")
        if gap_uncoupled > 0:
            print(f"    Uncoupled/Two ratio: {gap_uncoupled / gap_two:.1f}x")

    return {
        "interpretation": interpretation,
        "coupled_loss": coupled.best_loss,
        "uncoupled_loss": uncoupled.best_loss,
        "two_voice_loss": two_voice.best_loss,
        "coupled_vs_two": coupled_vs_two,
        "uncoupled_vs_two": uncoupled_vs_two,
        "coupled_vs_uncoupled": coupled_vs_uncoupled,
    }


def export_figure_data(
    target: np.ndarray,
    coupled: AblationResult,
    uncoupled: AblationResult,
    two_voice: AblationResult,
    dt_ms: float,
    output_path: str | Path = "results/exp_05/figure_data.npz",
) -> None:
    """Export data for paper figure."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    min_len = min(len(target), len(coupled.predicted), len(uncoupled.predicted), len(two_voice.predicted))
    np.savez_compressed(
        out_path,
        t_ms=np.arange(min_len) * dt_ms,
        target=target[:min_len],
        coupled=coupled.predicted[:min_len],
        uncoupled=uncoupled.predicted[:min_len],
        two_voice=two_voice.predicted[:min_len],
    )
    print(f"\n  Exported figure data to {out_path}")


# ============================================================================
# Main
# ============================================================================


def main():
    """Run the complete ablation experiment."""
    print("\n" + "#" * 60)
    print("EXPERIMENT 05: UNCOUPLED-PARAMETER ABLATION")
    print("#" * 60)
    print("\nQuestion: Does the expressivity separation persist when the")
    print("balance coupling is removed?")
    print("\nConditions:")
    print("  1. Coupled single-voice   (4 params: theta_total, theta_balance, tau_a, tau_r)")
    print("  2. Uncoupled single-voice  (4 params: theta_alpha, theta_beta, tau_a, tau_r)")
    print("  3. Two-voice coupled       (8 params: 4 per voice)")

    dt_ms = 1.0
    main_duration_ms = 360.0
    max_iterations = 500

    # Generate target (same as Exp 04)
    print("\n  Generating two-voice target...")
    target = target_envelope_grace_note_two_voice(main_duration_ms, dt_ms)
    input_signal = create_input_for_two_voice_target(main_duration_ms, dt_ms)
    print(f"  Target: {len(target)} samples ({main_duration_ms + 50}ms)")

    # Run all three conditions
    coupled = run_coupled_single_voice(target, input_signal, dt_ms, max_iterations)
    uncoupled = run_uncoupled_single_voice(target, input_signal, dt_ms, max_iterations)
    two_voice = run_two_voice(target, main_duration_ms, dt_ms, max_iterations)

    # Compare
    summary = report_comparison(coupled, uncoupled, two_voice)

    # Export figure data
    export_figure_data(target, coupled, uncoupled, two_voice, dt_ms)

    # Plot if possible
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        min_len = min(len(target), len(coupled.predicted), len(uncoupled.predicted), len(two_voice.predicted))
        time_ms = np.arange(min_len) * dt_ms

        ax1 = axes[0]
        ax1.plot(time_ms, target[:min_len], "k-", label="Target", linewidth=2, alpha=0.8)
        ax1.plot(time_ms, coupled.predicted[:min_len], "r--", label=f"Coupled (loss={coupled.best_loss:.4f})", linewidth=1.5)
        ax1.plot(time_ms, uncoupled.predicted[:min_len], "b--", label=f"Uncoupled (loss={uncoupled.best_loss:.4f})", linewidth=1.5)
        ax1.plot(time_ms, two_voice.predicted[:min_len], "g--", label=f"Two-voice (loss={two_voice.best_loss:.4f})", linewidth=1.5)
        for x in [25, 50, 80]:
            ax1.axvline(x=x, color="gray", linestyle=":", alpha=0.4)
        ax1.set_ylabel("Envelope")
        ax1.set_title("Ablation: Coupled vs Uncoupled vs Two-Voice")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        err_coupled = (coupled.predicted[:min_len] - target[:min_len]) ** 2
        err_uncoupled = (uncoupled.predicted[:min_len] - target[:min_len]) ** 2
        err_two = (two_voice.predicted[:min_len] - target[:min_len]) ** 2
        ax2.semilogy(time_ms, err_coupled + 1e-8, "r-", label="Coupled error", alpha=0.7)
        ax2.semilogy(time_ms, err_uncoupled + 1e-8, "b-", label="Uncoupled error", alpha=0.7)
        ax2.semilogy(time_ms, err_two + 1e-8, "g-", label="Two-voice error", alpha=0.7)
        for x in [25, 50, 80]:
            ax2.axvline(x=x, color="gray", linestyle=":", alpha=0.4)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Squared Error (log)")
        ax2.set_title("Per-Sample Squared Error")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_dir = Path("results/exp_05")
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "ablation_comparison.png", dpi=150, bbox_inches="tight")
        print(f"\n  Saved plot to {out_dir / 'ablation_comparison.png'}")
        plt.show()
    except Exception as e:
        print(f"\n  Plotting skipped: {e}")

    print("\n" + "=" * 60)
    print(f"ABLATION COMPLETE — Interpretation: Outcome {summary['interpretation']}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
