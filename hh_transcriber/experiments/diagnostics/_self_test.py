"""Self-test for diagnostics compute layer; no pytest."""

from __future__ import annotations

import numpy as np

from hh_transcriber.experiments.diagnostics.compute import (
    build_diagnostics_bundle,
    compute_quantization_stats,
    compute_signal_stats,
    compute_spike_stats,
    compute_timing_stats,
)


def main() -> None:
    failures: list[str] = []

    # --- Test 1: Basic diagnostics bundle with known data ---
    dt_ms = 1.0
    n_frames = 1000
    time_axis = np.arange(n_frames) * dt_ms

    # Gaussian envelope peaking at 100ms
    envelope = np.exp(-0.5 * ((time_axis - 100.0) / 10.0) ** 2).astype(np.float32)

    # Spikes at 100ms and 600ms
    spike_bool = np.zeros(n_frames, dtype=bool)
    spike_bool[100] = True
    spike_bool[600] = True

    expected_events = [100.0, 600.0]

    diag = build_diagnostics_bundle(
        envelope=envelope,
        spike_bool=spike_bool,
        dt_ms=dt_ms,
        window_ms=(0.0, 999.0),
        expected_event_times_ms=expected_events,
        spike_threshold=0.5,
        margin_target=0.05,
    )

    # Check spike count
    if diag.report.spikes.spike_count == 2:
        print("PASS: spike_count == 2")
    else:
        print(f"FAIL: spike_count = {diag.report.spikes.spike_count}, expected 2")
        failures.append("spike_count")

    # Check timing errors are near zero (spikes at exact expected times)
    if diag.report.timing is not None and len(diag.report.timing.timing_errors_ms) == 2:
        print("PASS: timing matched 2 events")
    else:
        print("FAIL: timing match count wrong")
        failures.append("timing_match_count")

    if diag.report.timing and abs(diag.report.timing.timing_errors_ms[0]) < 1e-6:
        print("PASS: first timing error ~0 (spike on expected time)")
    else:
        err = diag.report.timing.timing_errors_ms[0] if diag.report.timing else "N/A"
        print(f"FAIL: first timing error = {err}")
        failures.append("timing_error_zero")

    # Check operating point margin
    if diag.report.operating_point is not None:
        margin = diag.report.operating_point.peak_minus_threshold
        if margin is not None and margin > 0.05:
            print(f"PASS: margin = {margin:.3f} > 0.05")
        else:
            print(f"FAIL: margin = {margin}")
            failures.append("margin_value")
    else:
        print("FAIL: operating_point is None")
        failures.append("operating_point_none")

    # Check quantization: 2 spikes at unique steps, should NOT be suspected
    if not diag.report.quantization.quantization_suspected:
        print("PASS: quantization not suspected (2 unique spikes)")
    else:
        print("FAIL: quantization incorrectly suspected")
        failures.append("quantization_false_positive")

    # Check validity (all invariants should pass)
    if diag.is_valid:
        print("PASS: bundle is_valid = True")
    else:
        failed_invs = [inv.id for inv in diag.invariants if not inv.passed]
        print(f"FAIL: is_valid=False, failed: {failed_invs}")
        failures.append("is_valid")

    # --- Test 2: Quantization detection ---
    # 20 spikes all at the same grid step → suspected
    many_same_spikes = [100.0] * 20
    quant_stats = compute_quantization_stats(many_same_spikes, dt_ms=1.0)
    if quant_stats.quantization_suspected:
        print("PASS: quantization suspected for 20 spikes at same step")
    else:
        print("FAIL: quantization not detected for degenerate case")
        failures.append("quantization_detection")

    # Few spikes at same step → NOT suspected (below threshold)
    few_spikes = [100.0, 100.0, 100.0]
    quant_few = compute_quantization_stats(few_spikes, dt_ms=1.0)
    if not quant_few.quantization_suspected:
        print("PASS: quantization not suspected for 3 spikes (below threshold)")
    else:
        print("FAIL: quantization incorrectly suspected for 3 spikes")
        failures.append("quantization_few")

    # --- Test 3: Signal stats ---
    signal = compute_signal_stats(envelope, time_axis, window=(50.0, 150.0))
    if abs(signal.peak_value - 1.0) < 0.01:
        print(f"PASS: peak_value = {signal.peak_value:.4f} (~1.0 for Gaussian center)")
    else:
        print(f"FAIL: peak_value = {signal.peak_value}")
        failures.append("signal_peak")

    if abs(signal.peak_time_ms - 100.0) < 1.0:
        print(f"PASS: peak_time_ms = {signal.peak_time_ms} (~100ms)")
    else:
        print(f"FAIL: peak_time_ms = {signal.peak_time_ms}")
        failures.append("signal_peak_time")

    if signal.auc > 0:
        print(f"PASS: auc = {signal.auc:.2f} (positive)")
    else:
        print(f"FAIL: auc = {signal.auc}")
        failures.append("signal_auc")

    if signal.max_slope > 0:
        print(f"PASS: max_slope = {signal.max_slope:.4f} (positive)")
    else:
        print(f"FAIL: max_slope = {signal.max_slope}")
        failures.append("signal_slope")

    # --- Test 4: Spike stats with ISI ---
    spike_array = np.zeros(2000, dtype=bool)
    spike_array[100] = True
    spike_array[600] = True
    spike_array[1100] = True
    time_2k = np.arange(2000, dtype=float)

    spike_stats = compute_spike_stats(spike_array, time_2k, window=(0.0, 1999.0))
    if spike_stats.spike_count == 3:
        print("PASS: spike_count = 3 in extended window")
    else:
        print(f"FAIL: spike_count = {spike_stats.spike_count}")
        failures.append("spike_count_3")

    if spike_stats.isi_mean_ms is not None and abs(spike_stats.isi_mean_ms - 500.0) < 1.0:
        print(f"PASS: isi_mean_ms = {spike_stats.isi_mean_ms:.1f} (~500ms)")
    else:
        print(f"FAIL: isi_mean_ms = {spike_stats.isi_mean_ms}")
        failures.append("isi_mean")

    if spike_stats.isi_cv is not None and spike_stats.isi_cv < 0.01:
        print(f"PASS: isi_cv = {spike_stats.isi_cv:.4f} (~0, uniform spacing)")
    else:
        print(f"FAIL: isi_cv = {spike_stats.isi_cv}")
        failures.append("isi_cv")

    # --- Test 5: Timing stats with offset ---
    # Spikes at 102, 502; expected at 100, 500 → errors = +2, +2
    timing = compute_timing_stats(
        observed_spike_times_ms=[102.0, 502.0],
        expected_event_times_ms=[100.0, 500.0],
    )
    if len(timing.timing_errors_ms) == 2:
        print("PASS: timing matched 2 events with offset")
    else:
        print(f"FAIL: timing matched {len(timing.timing_errors_ms)}")
        failures.append("timing_offset_count")

    if timing.median_abs_error_ms is not None and abs(timing.median_abs_error_ms - 2.0) < 0.1:
        print(f"PASS: median_abs_error = {timing.median_abs_error_ms:.1f}ms (~2ms)")
    else:
        print(f"FAIL: median_abs_error = {timing.median_abs_error_ms}")
        failures.append("timing_offset_median")

    # --- Test 6: Empty spikes produces valid bundle ---
    empty_envelope = np.zeros(100, dtype=np.float32)
    empty_spikes = np.zeros(100, dtype=bool)
    diag_empty = build_diagnostics_bundle(
        envelope=empty_envelope,
        spike_bool=empty_spikes,
        dt_ms=1.0,
        window_ms=(0.0, 99.0),
    )
    if diag_empty.report.spikes.spike_count == 0:
        print("PASS: empty spikes → spike_count=0")
    else:
        print(f"FAIL: empty spikes → spike_count={diag_empty.report.spikes.spike_count}")
        failures.append("empty_spikes")

    # --- Test 7: JSON serialization round-trip ---
    json_str = diag.model_dump_json()
    if len(json_str) > 100:
        print(f"PASS: model_dump_json() produces {len(json_str)} chars")
    else:
        print(f"FAIL: JSON too short: {len(json_str)}")
        failures.append("json_dump")

    from hh_transcriber.experiments.diagnostics.types import DiagnosticsBundle as DB

    roundtrip = DB.model_validate_json(json_str)
    if roundtrip.report.spikes.spike_count == diag.report.spikes.spike_count:
        print("PASS: JSON round-trip preserves data")
    else:
        print("FAIL: JSON round-trip data mismatch")
        failures.append("json_roundtrip")

    # --- Test 8: INV_FINITE fails for NaN envelope ---
    nan_envelope = np.ones(100, dtype=np.float32)
    nan_envelope[50] = np.nan
    diag_nan = build_diagnostics_bundle(
        envelope=nan_envelope,
        spike_bool=np.zeros(100, dtype=bool),
        dt_ms=1.0,
        window_ms=(0.0, 99.0),
    )
    finite_inv = next(inv for inv in diag_nan.invariants if inv.id == "INV_FINITE")
    if not finite_inv.passed:
        print("PASS: INV_FINITE fails for NaN envelope")
    else:
        print("FAIL: INV_FINITE passed despite NaN")
        failures.append("inv_finite_nan")

    if not diag_nan.is_valid:
        print("PASS: is_valid=False when INV_FINITE fails")
    else:
        print("FAIL: is_valid=True despite failed invariant")
        failures.append("is_valid_nan")

    # Summary
    if failures:
        print(f"\n{len(failures)} test(s) FAILED: {failures}")
        raise SystemExit(1)
    else:
        print("\nAll diagnostics self-tests PASSED")


if __name__ == "__main__":
    main()
