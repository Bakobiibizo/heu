"""
Diagnostic analysis of two-voice grace note performance.

Investigates why the two-voice architecture achieves 0.019 loss
instead of the target 0.01, and 1.76x improvement instead of 3x.
"""

from __future__ import annotations

import numpy as np

from hh_envelope.core import HEUThetaBounds, HHEnvelopeUnit
from hh_envelope.polyphony.articulation import (
    ArticulatedNote,
    CombinationMode,
    create_grace_note_config,
)
from hh_envelope.utils.signals import target_envelope_grace_note


def analyze_target_envelope(duration_ms: float = 360.0, dt_ms: float = 1.0) -> dict:
    """Analyze the target envelope characteristics."""
    target = target_envelope_grace_note(duration_ms, dt_ms)

    # Key regions (from target_envelope_grace_note source)
    grace_duration = 30.0
    gap_duration = 10.0
    main_start = grace_duration + gap_duration  # 40ms

    grace_region = target[: int(grace_duration / dt_ms)]
    gap_region = target[int(grace_duration / dt_ms) : int(main_start / dt_ms)]
    main_region = target[int(main_start / dt_ms) :]

    return {
        "total_samples": len(target),
        "total_duration_ms": len(target) * dt_ms,
        "grace_peak": np.max(grace_region),
        "grace_peak_time_ms": np.argmax(grace_region) * dt_ms,
        "gap_mean": np.mean(gap_region),
        "gap_min": np.min(gap_region),
        "main_peak": np.max(main_region),
        "main_peak_time_ms": main_start + np.argmax(main_region) * dt_ms,
        "main_sustain_mean": np.mean(main_region[50:]),  # After 50ms into main
        "grace_region": grace_region,
        "gap_region": gap_region,
        "main_region": main_region,
        "full_target": target,
    }


def analyze_two_voice_output(
    grace_theta: dict,
    main_theta: dict,
    duration_ms: float = 360.0,
    dt_ms: float = 1.0,
) -> dict:
    """Analyze two-voice output characteristics."""
    config = create_grace_note_config(main_duration_ms=duration_ms)
    note = ArticulatedNote(config)

    note.set_voice_theta("grace", grace_theta)
    note.set_voice_theta("main", main_theta)

    # Run and collect per-voice data
    note.reset()
    num_samples = int(config.total_duration_ms / dt_ms)

    grace_env = np.zeros(num_samples)
    main_env = np.zeros(num_samples)
    combined_env = np.zeros(num_samples)

    for i in range(num_samples):
        output = note.step(dt_ms)
        grace_env[i] = output.voice_envelopes.get("grace", 0.0)
        main_env[i] = output.voice_envelopes.get("main", 0.0)
        combined_env[i] = output.combined_envelope

    # Analyze regions
    grace_duration = 30.0
    main_start = 40.0

    grace_region_idx = int(grace_duration / dt_ms)
    main_start_idx = int(main_start / dt_ms)

    return {
        "grace_voice_peak": np.max(grace_env[:grace_region_idx]),
        "grace_voice_at_gap": np.mean(grace_env[grace_region_idx:main_start_idx]),
        "main_voice_peak": np.max(main_env[main_start_idx:]),
        "main_voice_at_start": main_env[main_start_idx] if main_start_idx < len(main_env) else 0,
        "combined_peak_grace": np.max(combined_env[:grace_region_idx]),
        "combined_at_gap": np.mean(combined_env[grace_region_idx:main_start_idx]),
        "combined_peak_main": np.max(combined_env[main_start_idx:]),
        "grace_envelope": grace_env,
        "main_envelope": main_env,
        "combined_envelope": combined_env,
    }


def compute_regional_losses(
    target: np.ndarray,
    predicted: np.ndarray,
    dt_ms: float = 1.0,
) -> dict:
    """Compute MSE loss for each region of the envelope."""
    grace_duration = 30.0
    gap_duration = 10.0
    main_start = grace_duration + gap_duration

    grace_end_idx = int(grace_duration / dt_ms)
    main_start_idx = int(main_start / dt_ms)

    min_len = min(len(target), len(predicted))
    target = target[:min_len]
    predicted = predicted[:min_len]

    # Regional losses
    grace_loss = np.mean((target[:grace_end_idx] - predicted[:grace_end_idx]) ** 2)
    gap_loss = np.mean(
        (target[grace_end_idx:main_start_idx] - predicted[grace_end_idx:main_start_idx]) ** 2
    )
    main_attack_end = main_start_idx + 30  # First 30ms of main
    main_attack_loss = np.mean(
        (target[main_start_idx:main_attack_end] - predicted[main_start_idx:main_attack_end]) ** 2
    )
    main_sustain_loss = np.mean((target[main_attack_end:] - predicted[main_attack_end:]) ** 2)

    total_loss = np.mean((target - predicted) ** 2)

    # Contribution to total loss (weighted by region length)
    grace_weight = grace_end_idx / min_len
    gap_weight = (main_start_idx - grace_end_idx) / min_len
    main_attack_weight = (main_attack_end - main_start_idx) / min_len
    main_sustain_weight = (min_len - main_attack_end) / min_len

    return {
        "total_loss": total_loss,
        "grace_loss": grace_loss,
        "gap_loss": gap_loss,
        "main_attack_loss": main_attack_loss,
        "main_sustain_loss": main_sustain_loss,
        "grace_weight": grace_weight,
        "gap_weight": gap_weight,
        "main_attack_weight": main_attack_weight,
        "main_sustain_weight": main_sustain_weight,
        "grace_contribution": grace_loss * grace_weight,
        "gap_contribution": gap_loss * gap_weight,
        "main_attack_contribution": main_attack_loss * main_attack_weight,
        "main_sustain_contribution": main_sustain_loss * main_sustain_weight,
    }


def analyze_parameter_bounds() -> dict:
    """Check if optimized parameters hit bounds."""
    bounds = HEUThetaBounds()

    # Best parameters from experiment
    grace_theta = {
        "theta_total": 0.2151,
        "theta_balance": -1.9128,
        "theta_attack_tau": -0.9876,
        "theta_recovery_tau": 0.5507,
    }
    main_theta = {
        "theta_total": -0.3215,
        "theta_balance": 2.9999,
        "theta_attack_tau": 0.9992,
        "theta_recovery_tau": -0.2093,
    }

    bound_hits: dict[str, dict[str, dict[str, float | bool]]] = {
        "grace": {},
        "main": {},
    }

    for name, value in grace_theta.items():
        bound_name = name  # e.g., "theta_total"
        low, high = getattr(bounds, bound_name)
        at_low = abs(value - low) < 0.05
        at_high = abs(value - high) < 0.05
        if at_low or at_high:
            bound_hits["grace"][name] = {"value": value, "at_low": at_low, "at_high": at_high}

    for name, value in main_theta.items():
        bound_name = name
        low, high = getattr(bounds, bound_name)
        at_low = abs(value - low) < 0.05
        at_high = abs(value - high) < 0.05
        if at_low or at_high:
            bound_hits["main"][name] = {"value": value, "at_low": at_low, "at_high": at_high}

    return {
        "bounds": {
            "theta_total": bounds.theta_total,
            "theta_balance": bounds.theta_balance,
            "theta_attack_tau": bounds.theta_attack_tau,
            "theta_recovery_tau": bounds.theta_recovery_tau,
        },
        "bound_hits": bound_hits,
        "grace_theta": grace_theta,
        "main_theta": main_theta,
    }


def analyze_combination_mode_impact(
    grace_theta: dict,
    main_theta: dict,
    duration_ms: float = 360.0,
    dt_ms: float = 1.0,
) -> dict:
    """Compare different combination modes."""
    target = target_envelope_grace_note(duration_ms, dt_ms)

    results = {}
    for mode in [CombinationMode.MAX, CombinationMode.SUM, CombinationMode.MEAN]:
        config = create_grace_note_config(main_duration_ms=duration_ms)
        config = config.__class__(
            name=config.name,
            voice_roles=config.voice_roles,
            total_duration_ms=config.total_duration_ms,
            combination_mode=mode,
        )

        note = ArticulatedNote(config)
        note.set_voice_theta("grace", grace_theta)
        note.set_voice_theta("main", main_theta)

        envelope = note.run_full(dt_ms)
        min_len = min(len(envelope), len(target))
        loss = np.mean((envelope[:min_len] - target[:min_len]) ** 2)

        results[mode.value] = {
            "loss": loss,
            "envelope": envelope[:min_len],
        }

    return results


def analyze_heu_dynamics_limits() -> dict:
    """Analyze what envelope shapes a single HEU can produce."""
    # Test different theta configurations
    configs: dict[str, dict[str, float]] = {
        "neutral": {
            "theta_total": 0,
            "theta_balance": 0,
            "theta_attack_tau": 0,
            "theta_recovery_tau": 0,
        },
        "fast_transient": {
            "theta_total": -0.5,
            "theta_balance": 1.5,
            "theta_attack_tau": -0.5,
            "theta_recovery_tau": -0.5,
        },
        "slow_sustain": {
            "theta_total": -1.0,
            "theta_balance": 2.0,
            "theta_attack_tau": 0.5,
            "theta_recovery_tau": 0.5,
        },
        "max_attack": {
            "theta_total": 0,
            "theta_balance": 3.0,
            "theta_attack_tau": -1.0,
            "theta_recovery_tau": -1.0,
        },
    }

    results = {}
    for name, theta in configs.items():
        unit = HHEnvelopeUnit()
        unit.set_theta_from_dict(theta)

        # Run with single pulse
        unit.reset()
        envelope = []
        for i in range(200):
            excitation = 1.0 if i == 0 else 0.0
            output = unit.step(excitation, dt_ms=1.0)
            envelope.append(output.envelope)

        envelope_arr = np.array(envelope)
        results[name] = {
            "peak": np.max(envelope_arr),
            "peak_time_ms": np.argmax(envelope_arr),
            "half_decay_time_ms": np.argmax(envelope_arr < np.max(envelope_arr) / 2)
            if np.any(envelope_arr < np.max(envelope_arr) / 2)
            else 200,
            "envelope": envelope_arr,
        }

    return results


def main():
    """Run full diagnostic analysis."""
    print("=" * 70)
    print("DIAGNOSTIC ANALYSIS: TWO-VOICE GRACE NOTE PERFORMANCE")
    print("=" * 70)

    # Best parameters from experiment
    grace_theta = {
        "theta_total": 0.2151,
        "theta_balance": -1.9128,
        "theta_attack_tau": -0.9876,
        "theta_recovery_tau": 0.5507,
    }
    main_theta = {
        "theta_total": -0.3215,
        "theta_balance": 2.9999,
        "theta_attack_tau": 0.9992,
        "theta_recovery_tau": -0.2093,
    }

    # 1. Target envelope analysis
    print("\n1. TARGET ENVELOPE CHARACTERISTICS")
    print("-" * 50)
    target_analysis = analyze_target_envelope()
    print(
        f"   Grace peak: {target_analysis['grace_peak']:.3f} at {target_analysis['grace_peak_time_ms']:.1f}ms"
    )
    print(
        f"   Gap mean:   {target_analysis['gap_mean']:.3f} (min: {target_analysis['gap_min']:.3f})"
    )
    print(
        f"   Main peak:  {target_analysis['main_peak']:.3f} at {target_analysis['main_peak_time_ms']:.1f}ms"
    )
    print(f"   Main sustain mean: {target_analysis['main_sustain_mean']:.3f}")

    # 2. Two-voice output analysis
    print("\n2. TWO-VOICE OUTPUT CHARACTERISTICS")
    print("-" * 50)
    output_analysis = analyze_two_voice_output(grace_theta, main_theta)
    print(f"   Grace voice peak:     {output_analysis['grace_voice_peak']:.3f}")
    print(f"   Grace voice at gap:   {output_analysis['grace_voice_at_gap']:.3f}")
    print(f"   Main voice peak:      {output_analysis['main_voice_peak']:.3f}")
    print(f"   Combined grace peak:  {output_analysis['combined_peak_grace']:.3f}")
    print(f"   Combined at gap:      {output_analysis['combined_at_gap']:.3f}")
    print(f"   Combined main peak:   {output_analysis['combined_peak_main']:.3f}")

    # 3. Regional loss analysis
    print("\n3. REGIONAL LOSS BREAKDOWN")
    print("-" * 50)
    regional_losses = compute_regional_losses(
        target_analysis["full_target"],
        output_analysis["combined_envelope"],
    )
    print(f"   Total loss: {regional_losses['total_loss']:.6f}")
    print("\n   Region-specific losses:")
    print(
        f"   Grace region:     MSE={regional_losses['grace_loss']:.6f} (weight={regional_losses['grace_weight']:.2%}, contrib={regional_losses['grace_contribution']:.6f})"
    )
    print(
        f"   Gap region:       MSE={regional_losses['gap_loss']:.6f} (weight={regional_losses['gap_weight']:.2%}, contrib={regional_losses['gap_contribution']:.6f})"
    )
    print(
        f"   Main attack:      MSE={regional_losses['main_attack_loss']:.6f} (weight={regional_losses['main_attack_weight']:.2%}, contrib={regional_losses['main_attack_contribution']:.6f})"
    )
    print(
        f"   Main sustain:     MSE={regional_losses['main_sustain_loss']:.6f} (weight={regional_losses['main_sustain_weight']:.2%}, contrib={regional_losses['main_sustain_contribution']:.6f})"
    )

    # 4. Parameter bounds analysis
    print("\n4. PARAMETER BOUNDS ANALYSIS")
    print("-" * 50)
    bounds_analysis = analyze_parameter_bounds()
    print("   Bounds:")
    for name, (low, high) in bounds_analysis["bounds"].items():
        print(f"     {name}: [{low}, {high}]")
    print("\n   Parameters hitting bounds:")
    for voice, hits in bounds_analysis["bound_hits"].items():
        if hits:
            for param, info in hits.items():
                bound_type = "LOW" if info["at_low"] else "HIGH"
                print(f"     {voice}.{param} = {info['value']:.4f} -> AT {bound_type} BOUND")
        else:
            print(f"     {voice}: no bounds hit")

    # 5. Combination mode comparison
    print("\n5. COMBINATION MODE COMPARISON")
    print("-" * 50)
    mode_analysis = analyze_combination_mode_impact(grace_theta, main_theta)
    for mode, data in mode_analysis.items():
        print(f"   {mode}: loss = {data['loss']:.6f}")

    # 6. HEU dynamics limits
    print("\n6. HEU DYNAMICS LIMITS (single pulse response)")
    print("-" * 50)
    dynamics_analysis = analyze_heu_dynamics_limits()
    for config_name, data in dynamics_analysis.items():
        print(f"   {config_name}:")
        print(
            f"     peak={data['peak']:.3f} at {data['peak_time_ms']}ms, half-decay={data['half_decay_time_ms']}ms"
        )

    # 7. Summary of issues
    print("\n" + "=" * 70)
    print("IDENTIFIED PERFORMANCE ISSUES")
    print("=" * 70)

    issues = []

    # Check grace peak mismatch
    grace_peak_error = abs(output_analysis["combined_peak_grace"] - target_analysis["grace_peak"])
    if grace_peak_error > 0.1:
        issues.append(
            f"GRACE PEAK MISMATCH: Target={target_analysis['grace_peak']:.3f}, Got={output_analysis['combined_peak_grace']:.3f}"
        )

    # Check gap floor mismatch
    gap_error = abs(output_analysis["combined_at_gap"] - target_analysis["gap_mean"])
    if gap_error > 0.05:
        issues.append(
            f"GAP FLOOR MISMATCH: Target={target_analysis['gap_mean']:.3f}, Got={output_analysis['combined_at_gap']:.3f}"
        )

    # Check main peak mismatch
    main_peak_error = abs(output_analysis["combined_peak_main"] - target_analysis["main_peak"])
    if main_peak_error > 0.1:
        issues.append(
            f"MAIN PEAK MISMATCH: Target={target_analysis['main_peak']:.3f}, Got={output_analysis['combined_peak_main']:.3f}"
        )

    # Check parameters at bounds
    for voice, hits in bounds_analysis["bound_hits"].items():
        for param, info in hits.items():
            bound_type = "low" if info["at_low"] else "high"
            issues.append(
                f"PARAM AT BOUND: {voice}.{param}={info['value']:.4f} at {bound_type} bound (optimizer constrained)"
            )

    # Check regional loss dominance
    max_contrib = max(
        regional_losses["grace_contribution"],
        regional_losses["gap_contribution"],
        regional_losses["main_attack_contribution"],
        regional_losses["main_sustain_contribution"],
    )
    if regional_losses["grace_contribution"] == max_contrib:
        issues.append(f"GRACE REGION DOMINATES LOSS: {regional_losses['grace_contribution']:.6f}")
    if regional_losses["gap_contribution"] == max_contrib:
        issues.append(f"GAP REGION DOMINATES LOSS: {regional_losses['gap_contribution']:.6f}")
    if regional_losses["main_attack_contribution"] == max_contrib:
        issues.append(
            f"MAIN ATTACK DOMINATES LOSS: {regional_losses['main_attack_contribution']:.6f}"
        )

    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")

    if not issues:
        print("   No major issues identified.")

    return {
        "target_analysis": target_analysis,
        "output_analysis": output_analysis,
        "regional_losses": regional_losses,
        "bounds_analysis": bounds_analysis,
        "mode_analysis": mode_analysis,
        "dynamics_analysis": dynamics_analysis,
        "issues": issues,
    }


if __name__ == "__main__":
    results = main()
