"""Experiment 02: Two-Voice Coupling
====================================

Validates that coupled voices show consonance-biased behavior.

Hypothesis:
    Two HEUs with weak coupling maintain stable relative phase,
exhibit stronger co-activation for consonant intervals, and avoid
chaotic runaway under moderate coupling.

Metrics:
    - Cross-correlation peak and lag
    - Phase coherence of spike times
    - Activity EMA stability

Pass Conditions:
    - Unison cross-correlation > fifth > tritone
    - No activity_ema exceeds ceiling
    - Both voices remain responsive
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hh_envelope.core import HEUParamsRef, HEUThetaBounds, HHEnvelopeUnit, SafetyConfig
from hh_envelope.polyphony import (
    SharedContext,
    Voice,
    compute_coupling_input,
    get_interval_weight,
)
from hh_envelope.utils.metrics import (
    cross_correlation,
    spike_phase_coherence,
)
from hh_envelope.utils.plotting import plot_two_voice_comparison
from hh_envelope.utils.signals import get_pulse_times, pulse_train
from hh_transcriber.experiments.diagnostics import (
    build_diagnostics_bundle,
    check_coupling_changes_state,
    report_diagnostics,
    spike_times_to_bool,
)


def create_voice(pitch_hint: int, voice_id: int) -> Voice:
    """Create a voice with default parameters.

    Args:
        pitch_hint: MIDI note number
        voice_id: Voice identifier

    Returns:
        Configured Voice instance
    """
    unit = HHEnvelopeUnit(
        params_ref=HEUParamsRef(),
        theta_bounds=HEUThetaBounds(),
        safety=SafetyConfig(),
    )
    return Voice(unit=unit, pitch_hint=pitch_hint, voice_id=voice_id)


def run_two_voice_test(
    interval: int,
    coupling_strength: float,
    bpm: float = 120.0,
    duration_sec: float = 10.0,
    dt_ms: float = 1.0,
) -> dict:
    """Run two-voice coupling test.

    Args:
        interval: Pitch interval in semitones (0=unison, 7=fifth, 6=tritone)
        coupling_strength: Coupling strength (0.05-0.2 recommended)
        bpm: Beats per minute for shared pulse drive
        duration_sec: Duration in seconds
        dt_ms: Timestep in milliseconds

    Returns:
        Dictionary with test results
    """
    # Create voices
    base_pitch = 60  # Middle C
    voice_a = create_voice(pitch_hint=base_pitch, voice_id=0)
    voice_b = create_voice(pitch_hint=base_pitch + interval, voice_id=1)
    voices = [voice_a, voice_b]

    # Generate shared pulse train
    time_ms, signal = pulse_train(bpm, duration_sec, dt_ms)
    num_samples = len(signal)

    # Storage for envelopes
    envelope_a = np.zeros(num_samples)
    envelope_b = np.zeros(num_samples)
    activity_ema_max_a = 0.0
    activity_ema_max_b = 0.0

    # Shared context (neutral for this test)
    context = SharedContext(
        pulse_phase=0.0,
        harmonic_bias=0.5,
        density_pressure=0.0,
        global_modulator=0.0,
    )

    # Run simulation
    for i, excitation in enumerate(signal):
        # Compute coupling inputs
        coupling_a = compute_coupling_input(voice_a, voices, coupling_strength)
        coupling_b = compute_coupling_input(voice_b, voices, coupling_strength)

        # Step both voices with same excitation but individual coupling
        output_a = voice_a.step(excitation, context, coupling_a, dt_ms)
        output_b = voice_b.step(excitation, context, coupling_b, dt_ms)

        envelope_a[i] = output_a.envelope
        envelope_b[i] = output_b.envelope

        # Track max activity EMA
        activity_ema_max_a = max(activity_ema_max_a, voice_a.unit.activity_ema)
        activity_ema_max_b = max(activity_ema_max_b, voice_b.unit.activity_ema)

    # Get spike times
    spike_times_a = voice_a.get_spike_times_array()
    spike_times_b = voice_b.get_spike_times_array()

    # Compute metrics
    lags, corr = cross_correlation(envelope_a, envelope_b, max_lag_samples=100)
    max_corr = float(np.max(np.abs(corr)))
    max_corr_lag = int(lags[np.argmax(np.abs(corr))])

    beat_period_ms = 60000.0 / bpm
    phase_coh = spike_phase_coherence(spike_times_a, spike_times_b, beat_period_ms)

    interval_weight = get_interval_weight(base_pitch, base_pitch + interval)

    return {
        "interval": interval,
        "interval_weight": interval_weight,
        "coupling_strength": coupling_strength,
        "max_correlation": max_corr,
        "max_corr_lag_ms": max_corr_lag * dt_ms,
        "phase_coherence": phase_coh,
        "spike_count_a": len(spike_times_a),
        "spike_count_b": len(spike_times_b),
        "activity_ema_max_a": activity_ema_max_a,
        "activity_ema_max_b": activity_ema_max_b,
        "time_ms": time_ms,
        "envelope_a": envelope_a,
        "envelope_b": envelope_b,
        "spike_times_a": spike_times_a,
        "spike_times_b": spike_times_b,
    }


def analyze_consonance_effect(results: dict) -> dict:
    """Analyze whether consonant intervals show stronger coupling.

    Args:
        results: Dictionary of test results keyed by test name

    Returns:
        Analysis summary
    """
    # Group by interval
    intervals: dict[int, list[dict]] = {}
    for key, r in results.items():
        if "interval" in r:
            interval = r["interval"]
            if interval not in intervals:
                intervals[interval] = []
            intervals[interval].append(r)

    # Compute mean correlation for each interval
    interval_correlations = {}
    for interval, runs in intervals.items():
        mean_corr = np.mean([r["max_correlation"] for r in runs])
        interval_correlations[interval] = mean_corr

    return {
        "interval_correlations": interval_correlations,
        "ordering_correct": (
            interval_correlations.get(0, 0)
            >= interval_correlations.get(7, 0)
            >= interval_correlations.get(6, 0)
        ),
    }


def check_stability(results: dict) -> dict:
    """Check if all runs remained stable.

    Args:
        results: Dictionary of test results

    Returns:
        Stability analysis
    """
    ceiling = 0.8  # From SafetyConfig default
    unstable_runs = []

    for key, r in results.items():
        if "activity_ema_max_a" in r:
            if r["activity_ema_max_a"] > ceiling or r["activity_ema_max_b"] > ceiling:
                unstable_runs.append(key)

    return {
        "all_stable": len(unstable_runs) == 0,
        "unstable_runs": unstable_runs,
    }


def print_results_summary(results: dict) -> None:
    """Print summary of experiment results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 02: TWO-VOICE COUPLING RESULTS")
    print("=" * 60)

    # Print individual results
    for key in sorted(results.keys()):
        r = results[key]
        if "interval" in r:
            print(f"\n{key}:")
            print(f"  Interval: {r['interval']} semitones (weight: {r['interval_weight']:.2f})")
            print(f"  Coupling strength: {r['coupling_strength']}")
            print(f"  Max correlation: {r['max_correlation']:.3f}")
            print(f"  Phase coherence: {r['phase_coherence']:.3f}")
            print(f"  Spikes: A={r['spike_count_a']}, B={r['spike_count_b']}")
            print(
                f"  Max activity EMA: A={r['activity_ema_max_a']:.3f}, B={r['activity_ema_max_b']:.3f}"
            )


def check_pass_conditions(results: dict) -> bool:
    """Check if experiment passes acceptance criteria.

    Returns:
        True if all conditions pass
    """
    passed = True

    print("\n" + "-" * 60)
    print("PASS CONDITION CHECKS:")
    print("-" * 60)

    # Analyze consonance effect
    consonance = analyze_consonance_effect(results)
    ordering_pass = consonance["ordering_correct"]
    status = "PASS" if ordering_pass else "FAIL"
    print(f"  Consonance ordering (unison >= fifth >= tritone): [{status}]")
    corrs = consonance["interval_correlations"]
    for interval in sorted(corrs.keys()):
        print(f"    Interval {interval}: correlation = {corrs[interval]:.3f}")
    if not ordering_pass:
        passed = False

    # Check stability
    stability = check_stability(results)
    stability_pass = stability["all_stable"]
    status = "PASS" if stability_pass else "FAIL"
    print(f"  All runs stable (activity < ceiling): [{status}]")
    if not stability_pass:
        print(f"    Unstable: {stability['unstable_runs']}")
        passed = False

    # Check responsiveness (both voices should spike)
    responsive = True
    for key, r in results.items():
        if "spike_count_a" in r:
            if r["spike_count_a"] == 0 or r["spike_count_b"] == 0:
                responsive = False
                print(f"  Non-responsive: {key}")
    status = "PASS" if responsive else "FAIL"
    print(f"  All voices responsive: [{status}]")
    if not responsive:
        passed = False

    print("-" * 60)
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")

    return passed


def plot_coupling_results(results: dict, show: bool = True) -> None:
    """Plot coupling experiment results."""
    import matplotlib.pyplot as plt

    output_dir = Path("results/image")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot comparison for each interval (at coupling 0.1)
    for interval in [0, 7, 6]:
        key = f"interval_{interval}_coupling_0.1"
        if key in results:
            r = results[key]
            fig = plot_two_voice_comparison(
                r["time_ms"][:3000],  # First 3 seconds
                r["envelope_a"][:3000],
                r["envelope_b"][:3000],
                r["spike_times_a"][r["spike_times_a"] < 3000],
                r["spike_times_b"][r["spike_times_b"] < 3000],
                title=f"Interval {interval} semitones (coupling=0.1)",
            )
            fig.savefig(output_dir / f"exp_02_coupling_interval_{interval}.png", dpi=200)

    if show:
        plt.show()


def run_diagnostics(results: dict, baselines: dict) -> bool:
    """Build and report diagnostics including coupling-effect checks.

    For each coupled run, compares against the uncoupled baseline to verify
    that coupling actually changes system state.

    Args:
        results: Coupled test results.
        baselines: Uncoupled baseline results keyed by interval.

    Returns:
        True if all diagnostics bundles are VALID.
    """
    output_dir = Path("results/exp_02")
    dt_ms = 1.0
    spike_threshold = 0.8
    all_valid = True

    print("\n" + "-" * 60)
    print("DIAGNOSTICS:")
    print("-" * 60)

    for key in sorted(results.keys()):
        r = results[key]
        if "envelope_a" not in r:
            continue

        interval = r["interval"]
        n_frames = len(r["envelope_a"])
        duration_ms = n_frames * dt_ms

        # Per-voice basic diagnostics (finite, quantization, spike count)
        # Expected: ~19 spikes for 10s at 120 BPM
        pulse_times_list = get_pulse_times(120.0, 10.0).tolist()
        n_pulses = len(pulse_times_list)

        for voice_label, env_key, spk_key in [
            ("voice_a", "envelope_a", "spike_times_a"),
            ("voice_b", "envelope_b", "spike_times_b"),
        ]:
            envelope = r[env_key]
            spike_bool = spike_times_to_bool(r[spk_key].tolist(), n_frames, dt_ms)

            bundle = build_diagnostics_bundle(
                envelope=envelope,
                spike_bool=spike_bool,
                dt_ms=dt_ms,
                window_ms=(0.0, duration_ms),
                expected_event_times_ms=pulse_times_list,
                spike_threshold=spike_threshold,
                expected_spike_count=(n_pulses - 1, n_pulses + 1),
            )
            label = f"{key}_{voice_label}"
            report_diagnostics(bundle, label, output_dir / f"{label}_diagnostics.json")
            if not bundle.is_valid:
                all_valid = False

        # Coupling-effect invariant: compare voice_a against uncoupled baseline
        baseline_key = f"baseline_{interval}"
        if baseline_key in baselines:
            baseline = baselines[baseline_key]
            baseline_envelope = baseline["envelope_a"]
            baseline_spike_bool = spike_times_to_bool(
                baseline["spike_times_a"].tolist(), n_frames, dt_ms
            )
            coupled_envelope = r["envelope_a"]
            coupled_spike_bool = spike_times_to_bool(r["spike_times_a"].tolist(), n_frames, dt_ms)

            coupling_inv = check_coupling_changes_state(
                envelope_coupled=coupled_envelope,
                envelope_uncoupled=baseline_envelope,
                spike_bool_coupled=coupled_spike_bool,
                spike_bool_uncoupled=baseline_spike_bool,
                dt_ms=dt_ms,
            )
            label = f"{key}_coupling_effect"
            validity = "PASS" if coupling_inv.passed else "FAIL"
            print(f"  DIAGNOSTICS [{label}]: {validity}")
            if not coupling_inv.passed:
                print(f"    FAILED: {coupling_inv.id}; {coupling_inv.description}")
                for k, v in coupling_inv.details.items():
                    print(f"      {k}: {v}")
                all_valid = False

            # Save coupling invariant as a standalone bundle
            from hh_transcriber.experiments.diagnostics.types import (
                DiagnosticsBundle as _Bundle,
            )
            from hh_transcriber.experiments.diagnostics.types import (
                DiagnosticsReport as _Report,
            )
            from hh_transcriber.experiments.diagnostics.types import (
                QuantizationStats as _QStats,
            )
            from hh_transcriber.experiments.diagnostics.types import (
                SignalStats as _SStats,
            )
            from hh_transcriber.experiments.diagnostics.types import (
                SpikeStats as _SpStats,
            )

            coupling_bundle = _Bundle(
                report=_Report(
                    analysis_window_ms=(0.0, duration_ms),
                    signal=_SStats(
                        peak_value=0, peak_time_ms=0, auc=0, mean_value=0, std_value=0, max_slope=0
                    ),
                    spikes=_SpStats(spike_count=0),
                    quantization=_QStats(dt_ms=dt_ms),
                ),
                invariants=[coupling_inv],
            )
            save_path = output_dir / f"{label}_diagnostics.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(coupling_bundle.model_dump_json(indent=2) + "\n")

    print("-" * 60)
    print(f"DIAGNOSTICS OVERALL: {'ALL VALID' if all_valid else 'SOME INVALID'}")
    return all_valid


def main():
    """Run the complete experiment."""
    print("Starting Experiment 02: Two-Voice Coupling")
    print("=" * 60)

    intervals = [0, 7, 6]  # unison, fifth, tritone
    coupling_strengths = [0.05, 0.1, 0.2]

    # Run uncoupled baselines for each interval (coupling_strength=0)
    baselines = {}
    for interval in intervals:
        print(f"Running baseline (uncoupled) for interval {interval}...")
        baselines[f"baseline_{interval}"] = run_two_voice_test(
            interval=interval,
            coupling_strength=0.0,
            bpm=120,
            duration_sec=10.0,
        )

    results = {}
    for interval in intervals:
        for strength in coupling_strengths:
            print(f"Running interval {interval}, coupling {strength}...")
            key = f"interval_{interval}_coupling_{strength}"
            results[key] = run_two_voice_test(
                interval=interval,
                coupling_strength=strength,
                bpm=120,
                duration_sec=10.0,
            )

    # Print summary
    print_results_summary(results)

    # Check pass conditions
    passed = check_pass_conditions(results)

    # Run diagnostics (with baselines for coupling-effect checks)
    diagnostics_valid = run_diagnostics(results, baselines)
    if not diagnostics_valid:
        passed = False

    # Plot results
    try:
        plot_coupling_results(results, show=True)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    return passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
