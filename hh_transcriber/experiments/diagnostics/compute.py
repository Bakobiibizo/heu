"""Core diagnostic computation utilities.

Computes signal stats, spike stats, timing accuracy, operating point
margin, and quantization artifact detection from raw envelope/spike data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import (
    DiagnosticInvariant,
    DiagnosticsBundle,
    DiagnosticsReport,
    OperatingPointStats,
    QuantizationStats,
    SignalStats,
    SpikeStats,
    TimingStats,
)


def spike_times_to_bool(
    spike_times_ms: list[float] | np.ndarray,
    n_frames: int,
    dt_ms: float,
) -> np.ndarray:
    """Convert spike times (ms) to a boolean frame array.

    Args:
        spike_times_ms: Spike event times in milliseconds.
        n_frames: Total number of frames in the output array.
        dt_ms: Timestep in milliseconds.

    Returns:
        Boolean array of length n_frames with True at spike frames.
    """
    spike_bool = np.zeros(n_frames, dtype=bool)
    for spike_time in spike_times_ms:
        frame_idx = int(round(float(spike_time) / dt_ms))
        if 0 <= frame_idx < n_frames:
            spike_bool[frame_idx] = True
    return spike_bool


def compute_signal_stats(
    series: np.ndarray,
    time_ms: np.ndarray,
    window: tuple[float, float],
) -> SignalStats:
    """Compute signal statistics over an analysis window."""
    start_ms, end_ms = window
    mask = (time_ms >= start_ms) & (time_ms <= end_ms)
    segment = series[mask]
    segment_time = time_ms[mask]

    if segment.size == 0:
        raise ValueError("Analysis window produced empty segment")

    peak_idx = int(np.argmax(segment))
    peak_value = float(segment[peak_idx])
    peak_time_ms = float(segment_time[peak_idx])

    # AUC via trapezoid
    auc = float(np.trapz(segment, segment_time))

    # Slope in units per ms
    dt_array = np.diff(segment_time)
    ds_array = np.diff(segment)
    safe_dt = np.where(dt_array == 0, 1e-9, dt_array)
    slopes = ds_array / safe_dt
    max_slope = float(np.max(slopes)) if slopes.size else 0.0

    return SignalStats(
        peak_value=peak_value,
        peak_time_ms=peak_time_ms,
        auc=auc,
        mean_value=float(np.mean(segment)),
        std_value=float(np.std(segment)),
        max_slope=max_slope,
    )


def compute_spike_stats(
    spike_bool: np.ndarray,
    time_ms: np.ndarray,
    window: tuple[float, float],
) -> SpikeStats:
    """Compute spike event statistics over an analysis window."""
    start_ms, end_ms = window
    mask = (time_ms >= start_ms) & (time_ms <= end_ms)
    spikes_in_window = spike_bool[mask]
    time_in_window = time_ms[mask]

    spike_times = time_in_window[spikes_in_window.astype(bool)]
    spike_times_ms = [float(val) for val in spike_times.tolist()]
    spike_count = len(spike_times_ms)

    isi = np.diff(spike_times) if spike_count >= 2 else np.array([], dtype=float)
    isi_ms = [float(val) for val in isi.tolist()]

    if isi.size >= 2:
        isi_mean = float(np.mean(isi))
        isi_std = float(np.std(isi))
        isi_cv = float(isi_std / (isi_mean + 1e-9))
    elif isi.size == 1:
        isi_mean = float(isi[0])
        isi_cv = 0.0
    else:
        isi_mean = None
        isi_cv = None

    min_isi = float(np.min(isi)) if isi.size else None

    return SpikeStats(
        spike_count=spike_count,
        spike_times_ms=spike_times_ms,
        isi_ms=isi_ms,
        isi_mean_ms=isi_mean,
        isi_cv=isi_cv,
        min_isi_ms=min_isi,
    )


def compute_timing_stats(
    observed_spike_times_ms: list[float],
    expected_event_times_ms: list[float],
) -> TimingStats:
    """Compute timing accuracy via greedy nearest-match.

    Each expected event time is matched to the nearest unmatched spike.
    """
    if not expected_event_times_ms:
        return TimingStats()

    spikes = np.array(observed_spike_times_ms, dtype=float)
    expected = np.array(expected_event_times_ms, dtype=float)

    errors: list[float] = []
    used = np.zeros(len(spikes), dtype=bool)

    for expected_time in expected:
        if spikes.size == 0:
            break
        distances = np.abs(spikes - expected_time)
        distances[used] = np.inf
        nearest_idx = int(np.argmin(distances))
        if not np.isfinite(distances[nearest_idx]):
            break
        used[nearest_idx] = True
        errors.append(float(spikes[nearest_idx] - expected_time))

    abs_err = np.abs(np.array(errors, dtype=float)) if errors else np.array([], dtype=float)

    median_abs = float(np.median(abs_err)) if abs_err.size else None
    mad_abs = float(np.median(np.abs(abs_err - np.median(abs_err)))) if abs_err.size else None
    max_abs = float(np.max(abs_err)) if abs_err.size else None

    # Drift slope: timing error vs event time (linear fit)
    drift = None
    if len(errors) >= 3:
        matched_expected = expected[: len(errors)]
        coeff = np.polyfit(matched_expected / 1000.0, np.array(errors), deg=1)
        drift = float(coeff[0])

    return TimingStats(
        expected_event_times_ms=[float(val) for val in expected.tolist()],
        timing_errors_ms=errors,
        median_abs_error_ms=median_abs,
        mad_abs_error_ms=mad_abs,
        max_abs_error_ms=max_abs,
        drift_slope_ms_per_s=drift,
    )


def compute_operating_point(
    peak_value: float,
    spike_threshold: float | None,
    margin_target: float | None,
) -> OperatingPointStats | None:
    """Compute operating point margin (peak minus threshold)."""
    if spike_threshold is None:
        return None
    margin = float(peak_value - spike_threshold)
    margin_ok = None
    if margin_target is not None:
        margin_ok = bool(margin >= margin_target)
    return OperatingPointStats(
        spike_threshold=float(spike_threshold),
        peak_minus_threshold=margin,
        margin_ok=margin_ok,
        margin_target=float(margin_target) if margin_target is not None else None,
    )


def compute_quantization_stats(
    spike_times_ms: list[float],
    dt_ms: float,
) -> QuantizationStats:
    """Detect quantization artifacts in spike timing.

    Heuristic: if many spikes exist but very few unique timestep indices,
    spikes are snapped to the discrete grid.
    """
    if not spike_times_ms:
        return QuantizationStats(
            dt_ms=float(dt_ms),
            unique_spike_time_steps=0,
            quantization_suspected=False,
        )

    steps = [int(round(spike_time / dt_ms)) for spike_time in spike_times_ms]
    unique_steps = len(set(steps))

    # Heuristic: many spikes but very few unique grid positions
    suspected = False
    if len(spike_times_ms) >= 8 and unique_steps <= 2:
        suspected = True

    return QuantizationStats(
        dt_ms=float(dt_ms),
        unique_spike_time_steps=unique_steps,
        quantization_suspected=suspected,
    )


def build_diagnostics_bundle(
    *,
    envelope: np.ndarray,
    spike_bool: np.ndarray,
    dt_ms: float,
    window_ms: tuple[float, float],
    expected_event_times_ms: list[float] | None = None,
    spike_threshold: float | None = None,
    margin_target: float | None = None,
    expected_spike_count: tuple[int, int] | None = None,
    silence_windows_ms: list[tuple[float, float]] | None = None,
) -> DiagnosticsBundle:
    """Build a complete diagnostics bundle from raw envelope and spike data.

    This is the main entry point. Experiments call this with their HEU output
    and get back a full DiagnosticsBundle with report + invariant checks.

    Args:
        envelope: Envelope time series.
        spike_bool: Boolean spike array (same length as envelope).
        dt_ms: Timestep in milliseconds.
        window_ms: Analysis window (start_ms, end_ms).
        expected_event_times_ms: Expected pulse/event times for timing checks.
        spike_threshold: Spike threshold for margin checks.
        margin_target: Minimum acceptable margin above threshold (default 0.02
            when spike_threshold is provided).
        expected_spike_count: Tuple (min, max) for expected spike count range.
        silence_windows_ms: List of (start, end) windows where no spikes
            should occur. Used for false-positive detection.
    """
    n_frames = len(envelope)
    time_ms = np.arange(n_frames, dtype=float) * float(dt_ms)

    signal = compute_signal_stats(envelope, time_ms, window_ms)
    spikes = compute_spike_stats(spike_bool, time_ms, window_ms)

    timing = None
    if expected_event_times_ms is not None:
        timing = compute_timing_stats(spikes.spike_times_ms, expected_event_times_ms)

    operating_point = compute_operating_point(signal.peak_value, spike_threshold, margin_target)
    quantization = compute_quantization_stats(spikes.spike_times_ms, dt_ms)

    report = DiagnosticsReport(
        analysis_window_ms=window_ms,
        signal=signal,
        spikes=spikes,
        timing=timing,
        operating_point=operating_point,
        quantization=quantization,
        coupling=None,
    )

    invariants: list[DiagnosticInvariant] = []

    # Invariant 1: Envelope contains only finite values
    finite_ok = bool(np.isfinite(envelope).all())
    invariants.append(
        DiagnosticInvariant(
            id="INV_FINITE",
            description="Envelope contains only finite values",
            passed=finite_ok,
            details={"finite": finite_ok},
        )
    )

    # Invariant 2: Spike times not overly quantized
    invariants.append(
        DiagnosticInvariant(
            id="INV_NO_QUANTIZATION_ARTIFACT",
            description="Spike times are not overly quantized (heuristic)",
            passed=not report.quantization.quantization_suspected,
            details={
                "spike_count": report.spikes.spike_count,
                "unique_steps": report.quantization.unique_spike_time_steps,
                "dt_ms": report.quantization.dt_ms,
            },
        )
    )

    # Invariant 3: If expected events provided, at least 50% must match
    if expected_event_times_ms:
        matched = len(report.timing.timing_errors_ms) if report.timing else 0
        invariants.append(
            DiagnosticInvariant(
                id="INV_TIMING_HAS_MATCHES",
                description="Timing stats must match at least 50% of expected events",
                passed=(matched >= int(0.5 * len(expected_event_times_ms))),
                details={"matched": matched, "expected": len(expected_event_times_ms)},
            )
        )

    # Invariant 4: Operating point margin meets target
    # Default margin_target to 0.02 when spike_threshold is provided
    effective_margin_target = margin_target
    if effective_margin_target is None and spike_threshold is not None:
        effective_margin_target = 0.02
    if spike_threshold is not None and effective_margin_target is not None:
        op = compute_operating_point(signal.peak_value, spike_threshold, effective_margin_target)
        if op is not None:
            invariants.append(
                DiagnosticInvariant(
                    id="INV_MARGIN_OK",
                    description="Operating point margin meets configured target",
                    passed=bool(op.margin_ok),
                    details={
                        "peak": signal.peak_value,
                        "threshold": op.spike_threshold,
                        "margin": op.peak_minus_threshold,
                        "target": effective_margin_target,
                    },
                )
            )
            # Update report with the effective operating point
            report = DiagnosticsReport(
                analysis_window_ms=report.analysis_window_ms,
                signal=report.signal,
                spikes=report.spikes,
                timing=report.timing,
                operating_point=op,
                quantization=report.quantization,
                coupling=report.coupling,
            )

    # Invariant 5: Spike count in expected range
    if expected_spike_count is not None:
        min_count, max_count = expected_spike_count
        actual = report.spikes.spike_count
        in_range = min_count <= actual <= max_count
        invariants.append(
            DiagnosticInvariant(
                id="INV_SPIKE_COUNT_EXPECTED_RANGE",
                description="Spike count within expected range",
                passed=in_range,
                details={
                    "actual": actual,
                    "expected_min": min_count,
                    "expected_max": max_count,
                },
            )
        )

    # Invariant 6: No false positives during silence windows
    if silence_windows_ms:
        time_ms = np.arange(len(envelope), dtype=float) * float(dt_ms)
        total_false_positives = 0
        for s_start, s_end in silence_windows_ms:
            mask = (time_ms >= s_start) & (time_ms <= s_end)
            total_false_positives += int(spike_bool[mask].sum())
        invariants.append(
            DiagnosticInvariant(
                id="INV_FALSE_POSITIVES_DURING_SILENCE",
                description="No spikes during designated silence windows",
                passed=(total_false_positives == 0),
                details={
                    "false_positive_count": total_false_positives,
                    "silence_windows": silence_windows_ms,
                },
            )
        )

    return DiagnosticsBundle(report=report, invariants=invariants)


def check_coupling_changes_state(
    envelope_coupled: np.ndarray,
    envelope_uncoupled: np.ndarray,
    spike_bool_coupled: np.ndarray,
    spike_bool_uncoupled: np.ndarray,
    dt_ms: float,
    epsilon_envelope: float = 1e-4,
    epsilon_peak: float = 1e-3,
) -> DiagnosticInvariant:
    """Check that coupling actually changes system state.

    At least one of these must hold:
    - Mean absolute envelope difference > epsilon_envelope
    - Peak envelope difference > epsilon_peak
    - Spike count differs
    - Any spike time differs by > 0.5 * dt_ms

    Args:
        envelope_coupled: Envelope with coupling applied.
        envelope_uncoupled: Envelope without coupling (baseline).
        spike_bool_coupled: Spike array with coupling.
        spike_bool_uncoupled: Spike array without coupling.
        dt_ms: Timestep.
        epsilon_envelope: Minimum mean absolute difference.
        epsilon_peak: Minimum peak difference.

    Returns:
        DiagnosticInvariant with pass/fail and detail metrics.
    """
    min_len = min(len(envelope_coupled), len(envelope_uncoupled))
    env_c = envelope_coupled[:min_len]
    env_u = envelope_uncoupled[:min_len]

    mean_abs_diff = float(np.mean(np.abs(env_c - env_u)))
    peak_diff = abs(float(np.max(env_c)) - float(np.max(env_u)))

    spike_count_coupled = int(spike_bool_coupled[:min_len].sum())
    spike_count_uncoupled = int(spike_bool_uncoupled[:min_len].sum())
    spike_count_differs = spike_count_coupled != spike_count_uncoupled

    # Check spike time shifts
    time_ms = np.arange(min_len, dtype=float) * dt_ms
    times_c = time_ms[spike_bool_coupled[:min_len].astype(bool)]
    times_u = time_ms[spike_bool_uncoupled[:min_len].astype(bool)]
    spike_time_shift = False
    if len(times_c) > 0 and len(times_u) > 0 and len(times_c) == len(times_u):
        max_shift = float(np.max(np.abs(times_c - times_u)))
        spike_time_shift = max_shift > 0.5 * dt_ms

    passed = (
        mean_abs_diff > epsilon_envelope
        or peak_diff > epsilon_peak
        or spike_count_differs
        or spike_time_shift
    )

    return DiagnosticInvariant(
        id="INV_COUPLING_CHANGES_STATE",
        description="Coupling must produce measurable state change vs uncoupled baseline",
        passed=passed,
        details={
            "mean_abs_envelope_diff": mean_abs_diff,
            "peak_diff": peak_diff,
            "spike_count_coupled": spike_count_coupled,
            "spike_count_uncoupled": spike_count_uncoupled,
            "spike_count_differs": spike_count_differs,
            "spike_time_shift": spike_time_shift,
            "epsilon_envelope": epsilon_envelope,
            "epsilon_peak": epsilon_peak,
        },
    )


def check_symmetry_not_collapsed(
    envelope_a: np.ndarray,
    envelope_b: np.ndarray,
    eps: float = 1e-4,
) -> DiagnosticInvariant:
    """Check that two coupled voices have NOT collapsed to identical output.

    Passes when the envelopes are distinguishable (correlation < 0.999 OR
    max absolute difference > eps). Fails when coupled voices produce
    identical output despite asymmetric input.

    Args:
        envelope_a: Envelope from voice A.
        envelope_b: Envelope from voice B.
        eps: Minimum max absolute difference to consider non-collapsed.

    Returns:
        DiagnosticInvariant with id="INV_SYMMETRY_NOT_COLLAPSED".
    """
    min_len = min(len(envelope_a), len(envelope_b))
    env_a = envelope_a[:min_len]
    env_b = envelope_b[:min_len]

    max_abs_diff = float(np.max(np.abs(env_a - env_b)))

    # Compute correlation (handle constant signals)
    std_a = float(np.std(env_a))
    std_b = float(np.std(env_b))
    if std_a < 1e-10 or std_b < 1e-10:
        # If either is constant, check if they're the same constant
        corr = 1.0 if max_abs_diff < eps else 0.0
    else:
        corr = float(np.corrcoef(env_a, env_b)[0, 1])

    passed = corr < 0.999 or max_abs_diff > eps

    return DiagnosticInvariant(
        id="INV_SYMMETRY_NOT_COLLAPSED",
        description="Coupled voices must not collapse to identical output",
        passed=passed,
        details={
            "correlation": corr,
            "max_abs_diff": max_abs_diff,
            "eps": eps,
        },
    )


def report_coupling_summary(
    coupling_raws: list[list[float]],
    modulations: list[list[float]],
    label: str,
) -> dict:
    """Print and return summary statistics of coupling values.

    Args:
        coupling_raws: Per-voice list of raw coupling values over time.
            Shape: [num_voices][num_timesteps].
        modulations: Per-voice list of applied modulation values over time.
            Shape: [num_voices][num_timesteps].
        label: Human-readable label for this summary.

    Returns:
        JSON-serializable dict with min/mean/max per voice.
    """
    summary: dict = {"label": label, "voices": []}

    for voice_idx in range(len(coupling_raws)):
        raws = np.array(coupling_raws[voice_idx])
        mods = np.array(modulations[voice_idx])

        voice_stats = {
            "voice_idx": voice_idx,
            "raw_min": float(np.min(raws)) if raws.size else 0.0,
            "raw_mean": float(np.mean(raws)) if raws.size else 0.0,
            "raw_max": float(np.max(raws)) if raws.size else 0.0,
            "mod_min": float(np.min(mods)) if mods.size else 1.0,
            "mod_mean": float(np.mean(mods)) if mods.size else 1.0,
            "mod_max": float(np.max(mods)) if mods.size else 1.0,
        }
        summary["voices"].append(voice_stats)

    print(f"  COUPLING SUMMARY [{label}]:")
    for vs in summary["voices"]:
        print(
            f"    Voice {vs['voice_idx']}: "
            f"raw=[{vs['raw_min']:.4f}, {vs['raw_mean']:.4f}, {vs['raw_max']:.4f}] "
            f"mod=[{vs['mod_min']:.4f}, {vs['mod_mean']:.4f}, {vs['mod_max']:.4f}]"
        )

    return summary


def report_diagnostics(
    bundle: DiagnosticsBundle,
    label: str,
    save_path: Path | None = None,
) -> None:
    """Print diagnostics summary and optionally save JSON.

    Args:
        bundle: Completed diagnostics bundle.
        label: Human-readable label for this check.
        save_path: If provided, write bundle JSON to this path.
    """
    validity = "VALID" if bundle.is_valid else "INVALID"
    print(f"  DIAGNOSTICS [{label}]: {validity}")
    if not bundle.is_valid:
        for inv in bundle.invariants:
            if not inv.passed:
                print(f"    FAILED: {inv.id}; {inv.description}")
                if inv.details:
                    for k, v in inv.details.items():
                        print(f"      {k}: {v}")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(bundle.model_dump_json(indent=2) + "\n")
