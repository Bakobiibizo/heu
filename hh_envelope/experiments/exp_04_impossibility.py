"""Experiment 04: Single-Voice Impossibility Proof
==================================================

Proves that single-voice HEU cannot achieve low loss against the two-voice
perceptual target, while two-voice architecture achieves better results.

The parameter conflict:
    The two-voice target encodes conflicting dynamics:
    - Grace voice: recovery-biased, fast decay after tonic stops at 25ms
    - Main voice: attack-biased, flat tonic-driven sustain from 50ms onward
    - Single HEU can't have both fast decay AND flat sustain

Success Criteria:
    - Single-voice loss (two-voice target) > 0.005
    - Two-voice loss (two-voice target) < 0.005
    - Improvement ratio > 3x
    - Parameters at bounds (single) >= 2 of 4
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hh_envelope.core import HEUParamsRef, HEUThetaBounds, HHEnvelopeUnit, SafetyConfig
from hh_envelope.learning.loss import combined_loss
from hh_envelope.polyphony.articulation import (
    ArticulatedNote,
    ArticulationConfig,
    CombinationMode,
    TonicConfig,
    VoiceRole,
)
from hh_envelope.utils.signals import (
    target_envelope_grace_note_two_voice,
)
from hh_transcriber.experiments.diagnostics import (
    build_diagnostics_bundle,
    report_diagnostics,
)


@dataclass
class RegionalMSE:
    """MSE broken down by region of the envelope."""

    grace_mse: float  # 0-25ms (grace tonic region)
    gap_mse: float  # 25-50ms (grace decay / gap)
    main_attack_mse: float  # 50-80ms (main ramp-up)
    main_sustain_mse: float  # 80ms+ (main sustain)
    total_mse: float


@dataclass
class ImpossibilityResult:
    """Result from single-voice impossibility test."""

    best_theta: dict[str, float]
    best_loss: float
    regional_mse: RegionalMSE
    iterations: int
    loss_history: list[float]
    target: np.ndarray
    predicted: np.ndarray
    spike_bool: np.ndarray
    parameters_at_bounds: int
    parameter_analysis: dict[str, str]


@dataclass
class TwoVoiceResult:
    """Result from two-voice capability test."""

    grace_theta: dict[str, float]
    main_theta: dict[str, float]
    best_loss: float
    regional_mse: RegionalMSE
    iterations: int
    loss_history: list[float]
    target: np.ndarray
    predicted: np.ndarray
    spike_bool: np.ndarray
    grace_envelope: np.ndarray
    main_envelope: np.ndarray
    parameters_at_bounds: int


def create_unit() -> HHEnvelopeUnit:
    """Create an HEU with default parameters."""
    return HHEnvelopeUnit(
        params_ref=HEUParamsRef(),
        theta_bounds=HEUThetaBounds(),
        safety=SafetyConfig(),
    )


def create_input_for_two_voice_target(
    main_duration_ms: float,
    dt_ms: float = 1.0,
) -> np.ndarray:
    """Create input signal matching the two-voice target generation.

    Must match exactly what the target_envelope_grace_note_two_voice uses:
    - Grace: pulse=1.5 at t=0, tonic=0.40 from t=2 to t=25
    - Main: no pulse, tonic=0.35 from t=50 onward

    Args:
        main_duration_ms: Duration of main note portion
        dt_ms: Timestep

    Returns:
        Input signal array (combined grace+main input for single-voice testing)
    """
    total_duration = 50.0 + main_duration_ms  # Grace+gap=50ms, then main

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
    """Compute MSE broken down by region.

    Region boundaries match the HEU-generated target:
    - Grace (0-25ms): tonic-driven grace peak
    - Decay/Gap (25-50ms): fast decay after tonic stops - key conflict region
    - Main attack (50-80ms): main voice ramp-up
    - Main sustain (80ms+): flat tonic-driven sustain

    Args:
        predicted: Predicted envelope
        target: Target envelope
        dt_ms: Timestep

    Returns:
        RegionalMSE with per-region breakdown
    """
    min_len = min(len(predicted), len(target))
    predicted = predicted[:min_len]
    target = target[:min_len]

    # Region boundaries in samples (matching HEU-generated target)
    grace_end = int(25.0 / dt_ms)  # Grace tonic stops at 25ms
    gap_end = int(50.0 / dt_ms)  # Main starts at 50ms
    attack_end = int(80.0 / dt_ms)  # Main settles by 80ms

    # Compute per-region MSE
    grace_mse = float(np.mean((predicted[:grace_end] - target[:grace_end]) ** 2))

    if gap_end <= min_len:
        gap_mse = float(np.mean((predicted[grace_end:gap_end] - target[grace_end:gap_end]) ** 2))
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


def analyze_parameter_bounds(
    theta: dict[str, float],
    bounds: HEUThetaBounds,
    tolerance: float = 0.1,
) -> tuple[int, dict[str, str]]:
    """Analyze how many parameters are at bounds.

    Args:
        theta: Theta parameter dictionary
        bounds: Theta bounds
        tolerance: How close to bound counts as "at bound"

    Returns:
        (count_at_bounds, analysis_dict)
    """
    analysis = {}
    at_bounds = 0

    param_bounds = {
        "theta_total": bounds.theta_total,
        "theta_balance": bounds.theta_balance,
        "theta_attack_tau": bounds.theta_attack_tau,
        "theta_recovery_tau": bounds.theta_recovery_tau,
    }

    for name, (lo, hi) in param_bounds.items():
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


def prove_single_voice_impossibility(
    main_duration_ms: float = 360.0,
    dt_ms: float = 1.0,
    max_iterations: int = 500,
) -> ImpossibilityResult:
    """Prove that single-voice HEU cannot fit the two-voice target.

    Runs extensive CMA-ES optimization and shows that even the best
    parameter configuration achieves loss > 0.005 due to conflicting
    parameter requirements (grace decay vs main sustain).

    Args:
        main_duration_ms: Duration of main note
        dt_ms: Timestep
        max_iterations: Maximum CMA-ES iterations

    Returns:
        ImpossibilityResult with detailed analysis
    """
    print("\n" + "=" * 60)
    print("SINGLE-VOICE IMPOSSIBILITY PROOF")
    print("=" * 60)

    # Generate two-voice target
    target = target_envelope_grace_note_two_voice(main_duration_ms, dt_ms)

    # Generate input signal optimized for two-voice target
    input_signal = create_input_for_two_voice_target(main_duration_ms, dt_ms)

    print(f"  Target length: {len(target)} samples ({main_duration_ms + 50}ms)")
    print(f"  Input length: {len(input_signal)} samples")

    # Create objective function
    def objective(theta_dict: dict[str, float]) -> float:
        unit = create_unit()
        unit.set_theta_from_dict(theta_dict)

        unit.reset()
        envelope = np.zeros(len(input_signal))
        for i, exc in enumerate(input_signal):
            output = unit.step(exc, dt_ms)
            envelope[i] = output.envelope

        min_len = min(len(envelope), len(target))
        return combined_loss(envelope[:min_len], target[:min_len], 0.01)

    # Run CMA-ES with multiple restarts to ensure we find the global minimum
    print("\n  Running CMA-ES optimization (multiple restarts)...")

    import cma  # type: ignore[import-untyped]

    bounds = HEUThetaBounds()
    lower_bounds = [
        bounds.theta_total[0],
        bounds.theta_balance[0],
        bounds.theta_attack_tau[0],
        bounds.theta_recovery_tau[0],
    ]
    upper_bounds = [
        bounds.theta_total[1],
        bounds.theta_balance[1],
        bounds.theta_attack_tau[1],
        bounds.theta_recovery_tau[1],
    ]

    # Wrapper to convert array to dict for objective
    def objective_array(arr):
        theta_dict = {
            "theta_total": arr[0],
            "theta_balance": arr[1],
            "theta_attack_tau": arr[2],
            "theta_recovery_tau": arr[3],
        }
        return objective(theta_dict)

    # Multiple starting points
    starting_points = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([-1.0, 2.0, 0.3, 0.5]),
        np.array([-0.5, 1.5, -0.5, -0.5]),
        np.array([0.5, -1.0, 0.5, -0.3]),
    ]

    best_loss = float("inf")
    best_theta = None
    all_loss_history = []

    for restart_idx, x0 in enumerate(starting_points):
        print(f"    Restart {restart_idx + 1}/{len(starting_points)}...", end="", flush=True)

        es = cma.CMAEvolutionStrategy(
            x0,
            0.5,
            {
                "bounds": [lower_bounds, upper_bounds],
                "maxiter": max_iterations // len(starting_points),
                "seed": 42 + restart_idx,
                "verbose": -9,
            },
        )

        while not es.stop():
            solutions = es.ask()
            losses = [objective_array(s) for s in solutions]
            es.tell(solutions, losses)

            gen_best_idx = np.argmin(losses)
            if losses[gen_best_idx] < best_loss:
                best_loss = losses[gen_best_idx]
                best_theta = {
                    "theta_total": solutions[gen_best_idx][0],
                    "theta_balance": solutions[gen_best_idx][1],
                    "theta_attack_tau": solutions[gen_best_idx][2],
                    "theta_recovery_tau": solutions[gen_best_idx][3],
                }
            all_loss_history.append(min(losses))

        print(f" loss={best_loss:.6f}")

    # Generate final prediction with best parameters
    unit = create_unit()
    if best_theta is None:
        best_theta = {
            "theta_total": 0.0,
            "theta_balance": 0.0,
            "theta_attack_tau": 0.0,
            "theta_recovery_tau": 0.0,
        }
    unit.set_theta_from_dict(best_theta)
    unit.reset()
    predicted = np.zeros(len(input_signal))
    spike_bool = np.zeros(len(input_signal), dtype=bool)
    for i, exc in enumerate(input_signal):
        output = unit.step(exc, dt_ms)
        predicted[i] = output.envelope
        if output.spike_event:
            spike_bool[i] = True

    # Compute regional MSE
    min_len = min(len(predicted), len(target))
    regional_mse = compute_regional_mse(predicted[:min_len], target[:min_len], dt_ms)

    # Analyze parameter bounds
    at_bounds, analysis = analyze_parameter_bounds(best_theta, bounds)

    print(f"\n  Best single-voice loss: {best_loss:.6f}")
    print(f"  Parameters at bounds: {at_bounds}/4")
    for name, status in analysis.items():
        print(f"    {name}: {status}")

    print("\n  Regional MSE breakdown:")
    print(f"    Grace (0-25ms):        {regional_mse.grace_mse:.6f}")
    print(f"    Decay/Gap (25-50ms):   {regional_mse.gap_mse:.6f}")
    print(f"    Main attack (50-80ms): {regional_mse.main_attack_mse:.6f}")
    print(f"    Main sustain (80ms+):  {regional_mse.main_sustain_mse:.6f}")

    return ImpossibilityResult(
        best_theta=best_theta,
        best_loss=best_loss,
        regional_mse=regional_mse,
        iterations=len(all_loss_history),
        loss_history=all_loss_history,
        target=target[:min_len],
        predicted=predicted[:min_len],
        spike_bool=spike_bool[:min_len],
        parameters_at_bounds=at_bounds,
        parameter_analysis=analysis,
    )


def create_two_voice_config_for_target(
    main_duration_ms: float = 360.0,
) -> ArticulationConfig:
    """Create two-voice config matching the HEU-generated target.

    Must exactly match target_envelope_grace_note_two_voice configuration:
    - Grace: pulse=1.5 at t=0, tonic=0.40 from t=2 to t=25, active 50ms
    - Main: no pulse, tonic=0.35 from t=50 onward

    Args:
        main_duration_ms: Duration of main note

    Returns:
        ArticulationConfig
    """
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
        pulse_amplitude=0.0,  # Pure tonic drive
        active_duration_ms=None,
        tonic_config=TonicConfig(level=0.35, start_ms=0.0, end_before_ms=60.0),
    )

    return ArticulationConfig(
        name="two_voice_impossibility",
        voice_roles=[grace_role, main_role],
        total_duration_ms=total_duration,
        combination_mode=CombinationMode.MAX,
    )


def demonstrate_two_voice_capability(
    main_duration_ms: float = 360.0,
    dt_ms: float = 1.0,
    max_iterations: int = 400,
) -> TwoVoiceResult:
    """Demonstrate that two-voice architecture achieves low loss.

    Uses CMA-ES to optimize 8 parameters (4 per voice) and shows
    that loss < 0.015 is achievable with the two-voice architecture.

    Args:
        main_duration_ms: Duration of main note
        dt_ms: Timestep
        max_iterations: Maximum CMA-ES iterations

    Returns:
        TwoVoiceResult with detailed analysis
    """
    print("\n" + "=" * 60)
    print("TWO-VOICE CAPABILITY DEMONSTRATION")
    print("=" * 60)

    # Generate two-voice target
    target = target_envelope_grace_note_two_voice(main_duration_ms, dt_ms)

    # Create articulation config
    config = create_two_voice_config_for_target(main_duration_ms)

    print(f"  Target length: {len(target)} samples ({main_duration_ms + 50}ms)")

    # Create objective function for 8 parameters
    def objective(theta_flat: np.ndarray) -> float:
        note = ArticulatedNote(config)

        grace_theta = {
            "theta_total": theta_flat[0],
            "theta_balance": theta_flat[1],
            "theta_attack_tau": theta_flat[2],
            "theta_recovery_tau": theta_flat[3],
        }
        main_theta = {
            "theta_total": theta_flat[4],
            "theta_balance": theta_flat[5],
            "theta_attack_tau": theta_flat[6],
            "theta_recovery_tau": theta_flat[7],
        }

        note.set_voice_theta("grace", grace_theta)
        note.set_voice_theta("main", main_theta)

        envelope = note.run_full(dt_ms)
        min_len = min(len(envelope), len(target))
        return combined_loss(envelope[:min_len], target[:min_len], 0.01)

    # CMA-ES with 8 dimensions
    import cma  # type: ignore[import-untyped]

    bounds = HEUThetaBounds()
    lower_bounds = [
        bounds.theta_total[0],
        bounds.theta_balance[0],
        bounds.theta_attack_tau[0],
        bounds.theta_recovery_tau[0],
    ] * 2
    upper_bounds = [
        bounds.theta_total[1],
        bounds.theta_balance[1],
        bounds.theta_attack_tau[1],
        bounds.theta_recovery_tau[1],
    ] * 2

    # Multiple starting points
    starting_points = [
        np.array([-0.5, 1.0, -0.5, -0.3, -0.5, 2.0, 0.3, 0.0]),
        np.array([0.2, -1.5, -0.8, 0.3, -0.3, 2.5, 0.5, -0.2]),
        np.array([-0.3, 0.5, 0.0, 0.5, -1.0, 1.5, 0.0, 0.3]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    loss_history = []
    best_loss = float("inf")
    best_theta = None

    print("\n  Running CMA-ES optimization (multiple restarts)...")

    for restart_idx, x0 in enumerate(starting_points):
        print(f"    Restart {restart_idx + 1}/{len(starting_points)}...", end="", flush=True)

        es = cma.CMAEvolutionStrategy(
            x0,
            0.5,
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

            gen_best_idx = np.argmin(losses)
            if losses[gen_best_idx] < best_loss:
                best_loss = losses[gen_best_idx]
                best_theta = solutions[gen_best_idx].copy()
            loss_history.append(min(losses))

            if best_loss < 0.008:
                break

        print(f" loss={best_loss:.6f}")
        if best_loss < 0.008:
            break

    # Extract per-voice theta
    if best_theta is None:
        best_theta = np.zeros(8)
    grace_theta = {
        "theta_total": float(best_theta[0]),
        "theta_balance": float(best_theta[1]),
        "theta_attack_tau": float(best_theta[2]),
        "theta_recovery_tau": float(best_theta[3]),
    }
    main_theta = {
        "theta_total": float(best_theta[4]),
        "theta_balance": float(best_theta[5]),
        "theta_attack_tau": float(best_theta[6]),
        "theta_recovery_tau": float(best_theta[7]),
    }

    # Generate final envelopes
    note = ArticulatedNote(config)
    note.set_voice_theta("grace", grace_theta)
    note.set_voice_theta("main", main_theta)

    note.reset()
    num_samples = int(config.total_duration_ms / dt_ms)
    predicted = np.zeros(num_samples)
    grace_envelope = np.zeros(num_samples)
    main_envelope = np.zeros(num_samples)
    spike_bool = np.zeros(num_samples, dtype=bool)
    spike_threshold = 0.8
    prev_below = True

    for i in range(num_samples):
        output = note.step(dt_ms)
        predicted[i] = output.combined_envelope
        grace_envelope[i] = output.voice_envelopes.get("grace", 0.0)
        main_envelope[i] = output.voice_envelopes.get("main", 0.0)
        # Detect threshold crossings as spike events
        if predicted[i] >= spike_threshold and prev_below:
            spike_bool[i] = True
        prev_below = predicted[i] < spike_threshold

    # Compute regional MSE
    min_len = min(len(predicted), len(target))
    regional_mse = compute_regional_mse(predicted[:min_len], target[:min_len], dt_ms)

    # Analyze parameter bounds
    all_theta = {
        "grace_theta_total": grace_theta["theta_total"],
        "grace_theta_balance": grace_theta["theta_balance"],
        "grace_theta_attack_tau": grace_theta["theta_attack_tau"],
        "grace_theta_recovery_tau": grace_theta["theta_recovery_tau"],
        "main_theta_total": main_theta["theta_total"],
        "main_theta_balance": main_theta["theta_balance"],
        "main_theta_attack_tau": main_theta["theta_attack_tau"],
        "main_theta_recovery_tau": main_theta["theta_recovery_tau"],
    }

    at_bounds = 0
    param_bounds = {
        "theta_total": bounds.theta_total,
        "theta_balance": bounds.theta_balance,
        "theta_attack_tau": bounds.theta_attack_tau,
        "theta_recovery_tau": bounds.theta_recovery_tau,
    }
    tolerance = 0.1

    for key, value in all_theta.items():
        # Extract the base parameter name
        for base_name, (lo, hi) in param_bounds.items():
            if base_name in key:
                range_size = hi - lo
                if (
                    abs(value - lo) < tolerance * range_size
                    or abs(value - hi) < tolerance * range_size
                ):
                    at_bounds += 1
                break

    print(f"\n  Best two-voice loss: {best_loss:.6f}")
    print(f"  Parameters at bounds: {at_bounds}/8")

    print("\n  Grace voice theta:")
    for k, v in grace_theta.items():
        print(f"    {k}: {v:.4f}")
    print("  Main voice theta:")
    for k, v in main_theta.items():
        print(f"    {k}: {v:.4f}")

    print("\n  Regional MSE breakdown:")
    print(f"    Grace (0-25ms):        {regional_mse.grace_mse:.6f}")
    print(f"    Decay/Gap (25-50ms):   {regional_mse.gap_mse:.6f}")
    print(f"    Main attack (50-80ms): {regional_mse.main_attack_mse:.6f}")
    print(f"    Main sustain (80ms+):  {regional_mse.main_sustain_mse:.6f}")

    return TwoVoiceResult(
        grace_theta=grace_theta,
        main_theta=main_theta,
        best_loss=best_loss,
        regional_mse=regional_mse,
        iterations=len(loss_history),
        loss_history=loss_history,
        target=target[:min_len],
        predicted=predicted[:min_len],
        spike_bool=spike_bool[:min_len],
        grace_envelope=grace_envelope[:min_len],
        main_envelope=main_envelope[:min_len],
        parameters_at_bounds=at_bounds,
    )


def compute_theoretical_bound_analysis(
    single_result: ImpossibilityResult,
    two_result: TwoVoiceResult,
) -> dict:
    """Analyze parameter space to explain WHY single-voice fails.

    Args:
        single_result: Single-voice impossibility result
        two_result: Two-voice capability result

    Returns:
        Dictionary with theoretical analysis
    """
    print("\n" + "=" * 60)
    print("THEORETICAL BOUND ANALYSIS")
    print("=" * 60)

    analysis = {}

    # 1. Parameter conflict analysis
    print("\n  Parameter Conflict Analysis:")
    print("  " + "-" * 40)

    # Single voice has to compromise between grace decay and main sustain
    single_balance = single_result.best_theta.get("theta_balance", 0.0)
    single_recovery_tau = single_result.best_theta.get("theta_recovery_tau", 0.0)

    print(f"  Single-voice theta_balance: {single_balance:.4f}")
    print(f"  Single-voice theta_recovery_tau: {single_recovery_tau:.4f}")

    # Two-voice can have different parameters per voice
    grace_balance = two_result.grace_theta.get("theta_balance", 0.0)
    grace_recovery_tau = two_result.grace_theta.get("theta_recovery_tau", 0.0)
    main_balance = two_result.main_theta.get("theta_balance", 0.0)
    main_recovery_tau = two_result.main_theta.get("theta_recovery_tau", 0.0)

    print(f"  Two-voice grace theta_balance: {grace_balance:.4f}")
    print(f"  Two-voice main theta_balance: {main_balance:.4f}")
    print(f"  Two-voice grace theta_recovery_tau: {grace_recovery_tau:.4f}")
    print(f"  Two-voice main theta_recovery_tau: {main_recovery_tau:.4f}")

    # Check if there's significant parameter divergence
    balance_divergence = abs(grace_balance - main_balance)
    recovery_divergence = abs(grace_recovery_tau - main_recovery_tau)

    print("\n  Parameter divergence (grace vs main):")
    print(f"    theta_balance: {balance_divergence:.4f}")
    print(f"    theta_recovery_tau: {recovery_divergence:.4f}")

    analysis["balance_divergence"] = balance_divergence
    analysis["recovery_divergence"] = recovery_divergence

    # 2. Regional MSE comparison
    print("\n  Regional MSE Improvement:")
    print("  " + "-" * 40)

    regions = ["grace_mse", "gap_mse", "main_attack_mse", "main_sustain_mse"]
    region_names = [
        "Grace (0-25ms)",
        "Decay/Gap (25-50ms)",
        "Main attack (50-80ms)",
        "Main sustain (80ms+)",
    ]

    for region, name in zip(regions, region_names):
        single_mse = getattr(single_result.regional_mse, region)
        two_mse = getattr(two_result.regional_mse, region)
        ratio = single_mse / two_mse if two_mse > 0 else float("inf")
        print(f"    {name}: {single_mse:.6f} -> {two_mse:.6f} ({ratio:.1f}x improvement)")
        analysis[f"{region}_improvement"] = ratio

    # 3. Bound saturation
    print("\n  Bound Saturation:")
    print("  " + "-" * 40)
    print(f"    Single-voice: {single_result.parameters_at_bounds}/4 parameters at bounds")
    print(f"    Two-voice: {two_result.parameters_at_bounds}/8 parameters at bounds")

    analysis["single_at_bounds"] = single_result.parameters_at_bounds
    analysis["two_at_bounds"] = two_result.parameters_at_bounds

    return analysis


def check_pass_conditions(
    single_result: ImpossibilityResult,
    two_result: TwoVoiceResult,
) -> bool:
    """Check if experiment passes H3.2 criteria.

    With HEU-generated target:
    - Single-voice loss > 0.005 (constrained by parameter conflicts)
    - Two-voice loss < 0.005 (achievable by construction)
    - Improvement > 3x (architectural advantage)
    - Single at bounds >= 2 (constrained)

    Args:
        single_result: Single-voice result
        two_result: Two-voice result

    Returns:
        True if all criteria pass
    """
    print("\n" + "=" * 60)
    print("H3.2 PASS CONDITION CHECKS")
    print("=" * 60)

    all_pass = True

    # 1. Single-voice loss > 0.005 (can't match two-voice-generated target)
    single_loss = single_result.best_loss
    single_check = single_loss > 0.005
    status = "PASS" if single_check else "FAIL"
    print("\n  1. Single-voice loss > 0.005")
    print(f"     Actual: {single_loss:.6f} [{status}]")
    if not single_check:
        all_pass = False

    # 2. Two-voice loss < 0.005 (target is HEU-generated, achievable)
    two_loss = two_result.best_loss
    two_check = two_loss < 0.005
    status = "PASS" if two_check else "FAIL"
    print("\n  2. Two-voice loss < 0.005")
    print(f"     Actual: {two_loss:.6f} [{status}]")
    if not two_check:
        all_pass = False

    # 3. Improvement > 3x
    improvement = single_loss / two_loss if two_loss > 0 else float("inf")
    improvement_check = improvement > 3.0
    status = "PASS" if improvement_check else "FAIL"
    print("\n  3. Improvement ratio > 3x")
    print(f"     Actual: {improvement:.2f}x [{status}]")
    if not improvement_check:
        all_pass = False

    # 4. Single-voice parameters at bounds >= 2
    single_bounds = single_result.parameters_at_bounds
    bounds_check = single_bounds >= 2
    status = "PASS" if bounds_check else "FAIL"
    print("\n  4. Single-voice parameters at bounds >= 2")
    print(f"     Actual: {single_bounds}/4 [{status}]")
    if not bounds_check:
        all_pass = False

    # Info only: Two-voice parameters at bounds
    two_bounds = two_result.parameters_at_bounds
    print("\n  5. Two-voice parameters at bounds (info)")
    print(f"     Actual: {two_bounds}/8")

    # Info only: Regional improvement in decay/gap region
    if two_result.regional_mse.gap_mse > 0:
        gap_improvement = single_result.regional_mse.gap_mse / two_result.regional_mse.gap_mse
        print("\n  6. Decay/Gap region improvement (info)")
        print(f"     Actual: {gap_improvement:.1f}x")

    print("\n" + "-" * 60)
    overall = "PASS" if all_pass else "FAIL"
    print(f"OVERALL: {overall}")
    print("-" * 60)

    return all_pass


def plot_comparison(
    single_result: ImpossibilityResult,
    two_result: TwoVoiceResult,
    dt_ms: float = 1.0,
    show: bool = True,
) -> None:
    """Plot comparison of single-voice vs two-voice fits.

    Args:
        single_result: Single-voice result
        two_result: Two-voice result
        dt_ms: Timestep
        show: Whether to show the plot
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    time_ms = np.arange(len(single_result.target)) * dt_ms

    # Plot 1: Single-voice attempt
    ax1 = axes[0]
    ax1.plot(time_ms, single_result.target, "b-", label="Two-Voice Target", linewidth=2, alpha=0.8)
    ax1.plot(time_ms, single_result.predicted, "r--", label="Single-Voice Best Fit", linewidth=2)
    ax1.axvline(x=25, color="gray", linestyle=":", alpha=0.5, label="Region boundaries")
    ax1.axvline(x=50, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(x=80, color="gray", linestyle=":", alpha=0.5)
    ax1.set_ylabel("Envelope")
    ax1.set_title(f"Single-Voice Impossibility (loss={single_result.best_loss:.4f})")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Two-voice success
    ax2 = axes[1]
    ax2.plot(time_ms, two_result.target, "b-", label="Two-Voice Target", linewidth=2, alpha=0.8)
    ax2.plot(time_ms, two_result.predicted, "g--", label="Two-Voice Fit", linewidth=2)
    ax2.axvline(x=25, color="gray", linestyle=":", alpha=0.5)
    ax2.axvline(x=50, color="gray", linestyle=":", alpha=0.5)
    ax2.axvline(x=80, color="gray", linestyle=":", alpha=0.5)
    ax2.set_ylabel("Envelope")
    ax2.set_title(f"Two-Voice Success (loss={two_result.best_loss:.4f})")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Per-voice envelopes
    ax3 = axes[2]
    ax3.plot(time_ms, two_result.grace_envelope, "m-", label="Grace Voice", linewidth=1.5)
    ax3.plot(time_ms, two_result.main_envelope, "c-", label="Main Voice", linewidth=1.5)
    ax3.plot(time_ms, two_result.predicted, "k--", label="Combined (max)", linewidth=1, alpha=0.5)
    ax3.axvline(x=25, color="gray", linestyle=":", alpha=0.5)
    ax3.axvline(x=50, color="gray", linestyle=":", alpha=0.5)
    ax3.axvline(x=80, color="gray", linestyle=":", alpha=0.5)
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Envelope")
    ax3.set_title("Per-Voice Contribution")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()


def run_diagnostics(
    single_result: ImpossibilityResult,
    two_result: TwoVoiceResult,
    dt_ms: float = 1.0,
) -> bool:
    """Build and report diagnostics for both single and two-voice results.

    Returns:
        True if all diagnostics bundles are VALID.
    """
    output_dir = Path("results/exp_04")
    spike_threshold = 0.8
    all_valid = True

    # Expected events: grace pulse at t=0
    expected_events = [0.0]

    print("\n" + "-" * 60)
    print("DIAGNOSTICS:")
    print("-" * 60)

    # Single-voice diagnostics
    envelope = single_result.predicted
    n_frames = len(envelope)
    duration_ms = n_frames * dt_ms

    bundle_single = build_diagnostics_bundle(
        envelope=envelope,
        spike_bool=single_result.spike_bool,
        dt_ms=dt_ms,
        window_ms=(0.0, duration_ms),
        expected_event_times_ms=expected_events,
        spike_threshold=spike_threshold,
        expected_spike_count=(1, 5),  # Grace pulse should trigger at least 1 spike
    )
    report_diagnostics(bundle_single, "single_voice", output_dir / "single_voice_diagnostics.json")
    if not bundle_single.is_valid:
        all_valid = False

    # Two-voice diagnostics
    envelope = two_result.predicted
    n_frames = len(envelope)
    duration_ms = n_frames * dt_ms

    bundle_two = build_diagnostics_bundle(
        envelope=envelope,
        spike_bool=two_result.spike_bool,
        dt_ms=dt_ms,
        window_ms=(0.0, duration_ms),
        expected_event_times_ms=expected_events,
        spike_threshold=spike_threshold,
        expected_spike_count=(1, 5),
    )
    report_diagnostics(bundle_two, "two_voice", output_dir / "two_voice_diagnostics.json")
    if not bundle_two.is_valid:
        all_valid = False

    print("-" * 60)
    print(f"DIAGNOSTICS OVERALL: {'ALL VALID' if all_valid else 'SOME INVALID'}")
    return all_valid


def export_figure_data(
    *,
    t_ms: np.ndarray,
    target: np.ndarray,
    single_best: np.ndarray,
    two_best: np.ndarray,
    output_path: str | Path = "results/exp_04/figure_data.npz",
) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        t_ms=t_ms,
        T_target=target,
        v_single=single_best,
        v_two=two_best,
    )
    print(f"Exported figure data to {out_path}")


def main():
    """Run the complete impossibility proof experiment."""
    print("\n" + "#" * 60)
    print("EXPERIMENT 04: SINGLE-VOICE IMPOSSIBILITY PROOF")
    print("#" * 60)
    print("\nHypothesis H3.2 (V1.1 - HEU-Generated Target):")
    print("  - Single-voice loss > 0.005 (constrained by parameter conflicts)")
    print("  - Two-voice loss < 0.005 (achievable by construction)")
    print("  - Improvement > 3x (meaningful architectural advantage)")
    print("  - Single-voice parameters at bounds >= 2/4")

    dt_ms = 1.0
    main_duration_ms = 360.0

    # 1. Prove single-voice impossibility
    single_result = prove_single_voice_impossibility(
        main_duration_ms=main_duration_ms,
        dt_ms=dt_ms,
        max_iterations=500,
    )

    # 2. Demonstrate two-voice capability
    two_result = demonstrate_two_voice_capability(
        main_duration_ms=main_duration_ms,
        dt_ms=dt_ms,
        max_iterations=400,
    )

    # 3. Theoretical analysis
    analysis = compute_theoretical_bound_analysis(single_result, two_result)
    print("\nKey theoretical insights:")
    print(
        "  Balance divergence: {0:.4f}, Recovery divergence: {1:.4f}".format(
            analysis.get("balance_divergence", 0.0),
            analysis.get("recovery_divergence", 0.0),
        )
    )
    for region in [
        "grace_mse_improvement",
        "gap_mse_improvement",
        "main_attack_mse_improvement",
        "main_sustain_mse_improvement",
    ]:
        if region in analysis:
            print(f"  {region.replace('_', ' ').title()}: {analysis[region]:.1f}x")

    # 4. Check pass conditions
    passed = check_pass_conditions(single_result, two_result)

    # 5. Run diagnostics
    diagnostics_valid = run_diagnostics(single_result, two_result, dt_ms)
    if not diagnostics_valid:
        passed = False

    # 5b. Export figure trajectories (for paper figure)
    export_figure_data(
        t_ms=np.arange(len(single_result.target)) * dt_ms,
        target=single_result.target,
        single_best=single_result.predicted,
        two_best=two_result.predicted,
        output_path="results/exp_04/figure_data.npz",
    )

    # 6. Plot comparison
    try:
        plot_comparison(single_result, two_result, dt_ms, show=True)
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Single-voice best loss: {single_result.best_loss:.6f}")
    print(f"  Two-voice best loss:    {two_result.best_loss:.6f}")
    improvement = single_result.best_loss / two_result.best_loss
    print(f"  Improvement:            {improvement:.2f}x")
    print("\n  Parameters at bounds:")
    print(f"    Single-voice: {single_result.parameters_at_bounds}/4")
    print(f"    Two-voice:    {two_result.parameters_at_bounds}/8")
    print("\n  Conclusion:")
    if passed:
        print("    Two-voice excitable dynamics can reproduce articulated musical")
        print("    gestures that single-voice dynamics FUNDAMENTALLY CANNOT.")
    else:
        print("    Some criteria not met - review results.")

    return passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
