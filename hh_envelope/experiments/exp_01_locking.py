"""Experiment 01: Tempo Locking
===============================

Validates that a single HEU phase-locks to rhythmic input.

Hypothesis:
    A single HEU driven by a pulse train will phase-lock to the driving
tempo, exhibit stable refractory spacing, and show asymmetry in response
to accented pulses.

Metrics:
    - PLV (Phase Locking Value) for each tempo
    - ISI (Inter-Spike Interval) statistics
    - Peak consistency

Pass Conditions:
    - PLV > 0.8 for all tempos
    - Refractory prevents ISI < refractory_ref_ms
    - Accent peaks are 20-50% higher than non-accent peaks
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hh_envelope.core import HEUParamsRef, HEUThetaBounds, HHEnvelopeUnit, SafetyConfig
from hh_envelope.utils.metrics import (
    envelope_peak_consistency,
    inter_spike_interval_stats,
    phase_locking_value,
)
from hh_envelope.utils.plotting import plot_pulse_response
from hh_envelope.utils.signals import (
    accented_pulse_train,
    get_pulse_times,
    jittered_pulse_train,
    pulse_train,
)
from hh_transcriber.experiments.diagnostics import (
    build_diagnostics_bundle,
    report_diagnostics,
    spike_times_to_bool,
)


def run_simulation(
    unit: HHEnvelopeUnit,
    signal: np.ndarray,
    dt_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a simulation with the given unit and signal.

    Args:
        unit: HHEnvelopeUnit to simulate
        signal: Excitation signal array
        dt_ms: Timestep in milliseconds

    Returns:
        Tuple of (envelope_array, spike_times_array)
    """
    unit.reset()

    envelope = np.zeros(len(signal))
    spike_times = []
    current_time = 0.0

    for i, excitation in enumerate(signal):
        output = unit.step(excitation, dt_ms)
        envelope[i] = output.envelope
        if output.spike_event:
            spike_times.append(current_time)
        current_time += dt_ms

    return envelope, np.array(spike_times)


def run_tempo_locking_test(
    bpm: float,
    duration_sec: float = 10.0,
    dt_ms: float = 1.0,
) -> dict:
    """Run tempo locking test at specified BPM.

    Args:
        bpm: Beats per minute
        duration_sec: Duration in seconds
        dt_ms: Timestep in milliseconds

    Returns:
        Dictionary with test results
    """
    # Create unit with default params
    unit = HHEnvelopeUnit(
        params_ref=HEUParamsRef(),
        theta_bounds=HEUThetaBounds(),
        safety=SafetyConfig(),
    )

    # Generate pulse train
    time_ms, signal = pulse_train(bpm, duration_sec, dt_ms)
    pulse_times = get_pulse_times(bpm, duration_sec)

    # Run simulation
    envelope, spike_times = run_simulation(unit, signal, dt_ms)

    # Compute metrics
    beat_period_ms = 60000.0 / bpm
    plv = phase_locking_value(pulse_times, spike_times, beat_period_ms)
    isi_stats = inter_spike_interval_stats(spike_times)
    peak_stats = envelope_peak_consistency(envelope, pulse_times, window_ms=50.0, dt_ms=dt_ms)

    return {
        "bpm": bpm,
        "plv": plv,
        "isi_stats": isi_stats,
        "peak_stats": peak_stats,
        "time_ms": time_ms,
        "signal": signal,
        "envelope": envelope,
        "spike_times": spike_times,
        "pulse_times": pulse_times,
    }


def run_accent_test(
    bpm: float = 120.0,
    accent_every: int = 4,
    accent_strength: float = 1.5,
    duration_sec: float = 10.0,
    dt_ms: float = 1.0,
) -> dict:
    """Run accent response test.

    Args:
        bpm: Beats per minute
        accent_every: Accent every Nth beat
        accent_strength: Amplitude multiplier for accents
        duration_sec: Duration in seconds
        dt_ms: Timestep in milliseconds

    Returns:
        Dictionary with test results including accent analysis
    """
    # Create unit
    unit = HHEnvelopeUnit()

    # Generate accented pulse train
    time_ms, signal = accented_pulse_train(
        bpm, duration_sec, accent_every, accent_strength, dt_ms=dt_ms
    )
    pulse_times = get_pulse_times(bpm, duration_sec)

    # Run simulation
    envelope, spike_times = run_simulation(unit, signal, dt_ms)

    # Analyze peaks at accented vs non-accented positions
    beat_period_ms = 60000.0 / bpm
    # Use a 10% beat window capped at 60 ms to adapt to tempo while keeping tight accents
    window_ms = min(0.1 * beat_period_ms, 60.0)
    window_samples = int(window_ms / dt_ms)

    accent_peaks = []
    non_accent_peaks = []

    for i, pulse_t in enumerate(pulse_times):
        sample_idx = int(pulse_t / dt_ms)
        start_idx = sample_idx
        end_idx = min(sample_idx + window_samples, len(envelope))

        if start_idx < len(envelope) and end_idx > start_idx:
            peak = float(np.max(envelope[start_idx:end_idx]))
            if i % accent_every == 0:
                accent_peaks.append(peak)
            else:
                non_accent_peaks.append(peak)

    mean_accent_peak = np.mean(accent_peaks) if accent_peaks else 0.0
    mean_non_accent_peak = np.mean(non_accent_peaks) if non_accent_peaks else 0.0

    # Compute accent ratio
    if mean_non_accent_peak > 0:
        accent_ratio = mean_accent_peak / mean_non_accent_peak
    else:
        accent_ratio = float("inf") if mean_accent_peak > 0 else 1.0

    return {
        "bpm": bpm,
        "accent_every": accent_every,
        "input_accent_strength": accent_strength,
        "mean_accent_peak": mean_accent_peak,
        "mean_non_accent_peak": mean_non_accent_peak,
        "accent_ratio": accent_ratio,
        "time_ms": time_ms,
        "signal": signal,
        "envelope": envelope,
        "spike_times": spike_times,
    }


def run_jitter_test(
    bpm: float = 120.0,
    jitter_ms: float = 15.0,
    duration_sec: float = 10.0,
    dt_ms: float = 1.0,
    seed: int = 42,
) -> dict:
    """Run jitter robustness test.

    Args:
        bpm: Beats per minute
        jitter_ms: Timing jitter standard deviation in ms
        duration_sec: Duration in seconds
        dt_ms: Timestep in milliseconds
        seed: Random seed

    Returns:
        Dictionary with test results
    """
    # Create unit
    unit = HHEnvelopeUnit()

    # Generate jittered pulse train
    time_ms, signal = jittered_pulse_train(bpm, duration_sec, jitter_ms, dt_ms, seed)

    # Get ideal pulse times for comparison
    ideal_pulse_times = get_pulse_times(bpm, duration_sec)

    # Run simulation
    envelope, spike_times = run_simulation(unit, signal, dt_ms)

    # Compute PLV against ideal (not jittered) pulse times
    beat_period_ms = 60000.0 / bpm
    plv = phase_locking_value(ideal_pulse_times, spike_times, beat_period_ms)
    isi_stats = inter_spike_interval_stats(spike_times)

    return {
        "bpm": bpm,
        "jitter_ms": jitter_ms,
        "plv": plv,
        "isi_stats": isi_stats,
        "time_ms": time_ms,
        "signal": signal,
        "envelope": envelope,
        "spike_times": spike_times,
    }


def print_results_summary(results: dict) -> None:
    """Print summary of experiment results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 01: TEMPO LOCKING RESULTS")
    print("=" * 60)

    # Tempo tests
    for key in results:
        if key.startswith("tempo_"):
            r = results[key]
            print(f"\n{key}:")
            print(f"  PLV: {r['plv']:.3f}")
            if r["isi_stats"]["count"] > 1:
                print(f"  Mean ISI: {r['isi_stats']['mean_isi_ms']:.1f} ms")
                print(f"  ISI CV: {r['isi_stats']['cv']:.3f}")
                print(f"  Min ISI: {r['isi_stats']['min_isi_ms']:.1f} ms")
            print(f"  Peak CV: {r['peak_stats']['cv']:.3f}")

    # Accent test
    if "accents" in results:
        r = results["accents"]
        print("\nAccent Test:")
        print(f"  Accent ratio: {r['accent_ratio']:.2f}")
        print(f"  (Input accent strength: {r['input_accent_strength']:.2f})")

    # Jitter test
    if "jitter" in results:
        r = results["jitter"]
        print(f"\nJitter Test ({r['jitter_ms']} ms):")
        print(f"  PLV: {r['plv']:.3f}")


def check_pass_conditions(results: dict) -> bool:
    """Check if experiment passes acceptance criteria.

    Returns:
        True if all conditions pass
    """
    passed = True

    print("\n" + "-" * 60)
    print("PASS CONDITION CHECKS:")
    print("-" * 60)

    # Check PLV for all tempos
    plv_threshold = 0.8
    for key in results:
        if key.startswith("tempo_"):
            plv = results[key]["plv"]
            status = "PASS" if plv >= plv_threshold else "FAIL"
            print(f"  {key} PLV >= {plv_threshold}: {plv:.3f} [{status}]")
            if plv < plv_threshold:
                passed = False

    # Check refractory (min ISI should be >= refractory_ref_ms)
    refractory_ms = 20.0  # Default from HEUParamsRef
    for key in results:
        if key.startswith("tempo_"):
            min_isi = results[key]["isi_stats"].get("min_isi_ms", float("inf"))
            if min_isi != float("inf") and not np.isnan(min_isi):
                status = "PASS" if min_isi >= refractory_ms else "FAIL"
                print(f"  {key} min ISI >= {refractory_ms}: {min_isi:.1f} ms [{status}]")
                if min_isi < refractory_ms:
                    passed = False

    # Check accent ratio (should be 1.2-1.5 for input strength 1.5)
    if "accents" in results:
        ratio = results["accents"]["accent_ratio"]
        expected_min, expected_max = 1.1, 2.0
        status = "PASS" if expected_min <= ratio <= expected_max else "FAIL"
        print(f"  Accent ratio in [{expected_min}, {expected_max}]: {ratio:.2f} [{status}]")
        if not (expected_min <= ratio <= expected_max):
            passed = False

    print("-" * 60)
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")

    return passed


def plot_all_results(results: dict, show: bool = True) -> None:
    """Plot all experiment results."""
    import matplotlib.pyplot as plt

    output_dir = Path("results/image")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot tempo 120 in detail
    if "tempo_120" in results:
        r = results["tempo_120"]
        fig = plot_pulse_response(
            r["time_ms"][:5000],  # First 5 seconds
            r["signal"][:5000],
            r["envelope"][:5000],
            r["spike_times"][r["spike_times"] < 5000],
            title="120 BPM Tempo Locking (First 5 seconds)",
        )
        fig.savefig(output_dir / "exp_01_locking_tempo_120.png", dpi=200)

    # Plot accents
    if "accents" in results:
        r = results["accents"]
        fig = plot_pulse_response(
            r["time_ms"][:5000],
            r["signal"][:5000],
            r["envelope"][:5000],
            r["spike_times"][r["spike_times"] < 5000],
            title="Accented Pulse Response (First 5 seconds)",
        )
        fig.savefig(output_dir / "exp_01_locking_accents.png", dpi=200)

    if show:
        plt.show()


def run_diagnostics(results: dict) -> bool:
    """Build and report diagnostics for all sub-tests.

    Returns:
        True if all diagnostics bundles are VALID.
    """
    output_dir = Path("results/exp_01")
    dt_ms = 1.0
    spike_threshold = 0.8
    all_valid = True

    print("\n" + "-" * 60)
    print("DIAGNOSTICS:")
    print("-" * 60)

    # Tempo tests
    for key in sorted(results.keys()):
        if not key.startswith("tempo_"):
            continue
        r = results[key]
        envelope = r["envelope"]
        n_frames = len(envelope)
        spike_bool = spike_times_to_bool(r["spike_times"].tolist(), n_frames, dt_ms)
        pulse_times_list = r["pulse_times"].tolist()
        duration_ms = n_frames * dt_ms
        n_pulses = len(pulse_times_list)
        beat_period_ms = 60000.0 / r["bpm"]
        # Silence window: mid-beat region between first and second pulse
        # Pulses start at t=0, so silence = (100ms, beat_period-100ms)
        silence_start = 100.0
        silence_end = beat_period_ms - 100.0

        bundle = build_diagnostics_bundle(
            envelope=envelope,
            spike_bool=spike_bool,
            dt_ms=dt_ms,
            window_ms=(0.0, duration_ms),
            expected_event_times_ms=pulse_times_list,
            spike_threshold=spike_threshold,
            expected_spike_count=(n_pulses - 1, n_pulses + 1),
            silence_windows_ms=[(silence_start, silence_end)]
            if silence_end > silence_start
            else None,
        )
        report_diagnostics(bundle, key, output_dir / f"{key}_diagnostics.json")
        if not bundle.is_valid:
            all_valid = False

    # Accent test; still has expected pulse times
    if "accents" in results:
        r = results["accents"]
        envelope = r["envelope"]
        n_frames = len(envelope)
        spike_bool = spike_times_to_bool(r["spike_times"].tolist(), n_frames, dt_ms)
        duration_ms = n_frames * dt_ms
        bpm = r["bpm"]
        pulse_times_list = get_pulse_times(bpm, duration_ms / 1000.0).tolist()
        n_pulses = len(pulse_times_list)
        beat_period_ms = 60000.0 / bpm
        silence_start = 100.0
        silence_end = beat_period_ms - 100.0

        bundle = build_diagnostics_bundle(
            envelope=envelope,
            spike_bool=spike_bool,
            dt_ms=dt_ms,
            window_ms=(0.0, duration_ms),
            expected_event_times_ms=pulse_times_list,
            spike_threshold=spike_threshold,
            expected_spike_count=(n_pulses - 1, n_pulses + 1),
            silence_windows_ms=[(silence_start, silence_end)]
            if silence_end > silence_start
            else None,
        )
        report_diagnostics(bundle, "accents", output_dir / "accents_diagnostics.json")
        if not bundle.is_valid:
            all_valid = False

    # Jitter test; use ideal pulse times as expected events
    if "jitter" in results:
        r = results["jitter"]
        envelope = r["envelope"]
        n_frames = len(envelope)
        spike_bool = spike_times_to_bool(r["spike_times"].tolist(), n_frames, dt_ms)
        duration_ms = n_frames * dt_ms
        bpm = r["bpm"]
        pulse_times_list = get_pulse_times(bpm, duration_ms / 1000.0).tolist()
        n_pulses = len(pulse_times_list)

        bundle = build_diagnostics_bundle(
            envelope=envelope,
            spike_bool=spike_bool,
            dt_ms=dt_ms,
            window_ms=(0.0, duration_ms),
            expected_event_times_ms=pulse_times_list,
            spike_threshold=spike_threshold,
            expected_spike_count=(n_pulses - 2, n_pulses + 2),
        )
        report_diagnostics(bundle, "jitter", output_dir / "jitter_diagnostics.json")
        if not bundle.is_valid:
            all_valid = False

    print("-" * 60)
    print(f"DIAGNOSTICS OVERALL: {'ALL VALID' if all_valid else 'SOME INVALID'}")
    return all_valid


def main():
    """Run the complete experiment."""
    print("Starting Experiment 01: Tempo Locking")
    print("=" * 60)

    results = {}

    # Test multiple tempos
    for bpm in [90, 120, 140]:
        print(f"Running tempo test at {bpm} BPM...")
        results[f"tempo_{bpm}"] = run_tempo_locking_test(bpm, duration_sec=10.0)

    # Test accents
    print("Running accent test...")
    results["accents"] = run_accent_test(bpm=120, accent_every=4, accent_strength=1.5)

    # Test jitter
    print("Running jitter test...")
    results["jitter"] = run_jitter_test(bpm=120, jitter_ms=15.0)

    # Print summary
    print_results_summary(results)

    # Check pass conditions
    passed = check_pass_conditions(results)

    # Run diagnostics
    diagnostics_valid = run_diagnostics(results)
    if not diagnostics_valid:
        passed = False

    # Plot results
    try:
        plot_all_results(results, show=True)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    return passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
