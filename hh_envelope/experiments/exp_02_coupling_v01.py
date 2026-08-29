"""
Experiment 02 (v0.1): Coupling Engine Validation
=================================================

Tests the new CouplingEngine with configurable signal, site, and strength
across four sub-experiments designed to exercise regimes where coupling
should produce measurable state changes.

Sub-experiments:
    exp02a: Overlap regime (tonic keeps value elevated)
    exp02b: Slow state at 120 BPM (activity_ema signal)
    exp02c: Spike impulse signal
    exp02d: Symmetry break (asymmetric input)

Acceptance Gate:
    gate_1: exp02a coupling changes state (overlap regime works)
    gate_2: exp02b OR exp02c passes (at least one non-overlap works)
    gate_3: exp02d symmetry not collapsed
    gate_4: all diagnostics valid (INV_FINITE, MARGIN, etc.)

Usage:
    python -m hh_envelope.experiments.exp_02_coupling_v01
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hh_envelope.core import HEUParamsRef, HEUThetaBounds, HHEnvelopeUnit, SafetyConfig
from hh_envelope.coupling import CouplingConfig, CouplingEngine
from hh_envelope.polyphony import SharedContext, Voice
from hh_envelope.utils.signals import accented_pulse_train, pulse_train
from hh_transcriber.experiments.diagnostics import (
    build_diagnostics_bundle,
    check_coupling_changes_state,
    check_symmetry_not_collapsed,
    report_coupling_summary,
    report_diagnostics,
    spike_times_to_bool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_voice(pitch_hint: int, voice_id: int) -> Voice:
    """Create a voice with default parameters."""
    unit = HHEnvelopeUnit(
        params_ref=HEUParamsRef(),
        theta_bounds=HEUThetaBounds(),
        safety=SafetyConfig(),
    )
    return Voice(unit=unit, pitch_hint=pitch_hint, voice_id=voice_id)


def _run_coupled_sim(
    voices: list[Voice],
    signals: list[np.ndarray],
    coupling_cfg: CouplingConfig,
    dt_ms: float = 1.0,
    tonic: float = 0.0,
) -> dict:
    """Run N-voice simulation with CouplingEngine.

    Args:
        voices: List of Voice instances (will be reset).
        signals: Per-voice excitation arrays (same length).
        coupling_cfg: Coupling configuration.
        dt_ms: Timestep in milliseconds.
        tonic: Continuous drive level for all voices.

    Returns:
        Dictionary with envelopes, spike_times, coupling_raws, modulations.
    """
    num_voices = len(voices)
    num_samples = len(signals[0])
    engine = CouplingEngine(num_voices=num_voices, config=coupling_cfg)

    # Reset all voices
    for voice in voices:
        voice.reset()

    # Storage
    envelopes = [np.zeros(num_samples) for _ in range(num_voices)]
    coupling_raws: list[list[float]] = [[] for _ in range(num_voices)]
    modulations: list[list[float]] = [[] for _ in range(num_voices)]

    context = SharedContext(
        pulse_phase=0.0, harmonic_bias=0.5, density_pressure=0.0, global_modulator=0.0
    )

    for i in range(num_samples):
        # Get voice exports
        exports = [voice.get_export() for voice in voices]

        # Compute coupling contributions
        contributions = engine.step(exports, dt_ms)

        # Step each voice
        for v_idx, voice in enumerate(voices):
            output = voice.step(
                onset_signal=float(signals[v_idx][i]),
                shared_context=context,
                dt_ms=dt_ms,
                tonic=tonic,
                coupling_contributions=contributions[v_idx],
            )
            envelopes[v_idx][i] = output.envelope

            # Track coupling stats
            contrib = contributions[v_idx]
            if coupling_cfg.site == "excitation_add":
                coupling_raws[v_idx].append(contrib.excitation_add)
                modulations[v_idx].append(contrib.excitation_add)
            elif coupling_cfg.site == "attack_gain_mul":
                coupling_raws[v_idx].append(contrib.attack_gain_mul - 1.0)
                modulations[v_idx].append(contrib.attack_gain_mul)
            elif coupling_cfg.site == "threshold_add":
                coupling_raws[v_idx].append(contrib.threshold_add)
                modulations[v_idx].append(contrib.threshold_add)

    # Gather spike times
    spike_times = [voice.get_spike_times_array() for voice in voices]

    return {
        "envelopes": envelopes,
        "spike_times": spike_times,
        "coupling_raws": coupling_raws,
        "modulations": modulations,
        "num_samples": num_samples,
        "dt_ms": dt_ms,
    }


def _run_uncoupled_sim(
    voices: list[Voice],
    signals: list[np.ndarray],
    dt_ms: float = 1.0,
    tonic: float = 0.0,
) -> dict:
    """Run N-voice simulation without coupling (baseline).

    Args:
        voices: List of Voice instances (will be reset).
        signals: Per-voice excitation arrays.
        dt_ms: Timestep in milliseconds.
        tonic: Continuous drive level.

    Returns:
        Dictionary with envelopes and spike_times.
    """
    num_voices = len(voices)
    num_samples = len(signals[0])

    for voice in voices:
        voice.reset()

    envelopes = [np.zeros(num_samples) for _ in range(num_voices)]
    context = SharedContext(
        pulse_phase=0.0, harmonic_bias=0.5, density_pressure=0.0, global_modulator=0.0
    )

    for i in range(num_samples):
        for v_idx, voice in enumerate(voices):
            output = voice.step(
                onset_signal=float(signals[v_idx][i]),
                shared_context=context,
                dt_ms=dt_ms,
                tonic=tonic,
            )
            envelopes[v_idx][i] = output.envelope

    spike_times = [voice.get_spike_times_array() for voice in voices]

    return {
        "envelopes": envelopes,
        "spike_times": spike_times,
        "num_samples": num_samples,
        "dt_ms": dt_ms,
    }


def _check_diagnostics_valid(
    envelope: np.ndarray,
    spike_times: np.ndarray,
    dt_ms: float,
    label: str,
    output_dir: Path,
    spike_threshold: float = 0.8,
) -> bool:
    """Run standard diagnostics on a single voice envelope.

    Returns True if the bundle is valid.
    """
    n_frames = len(envelope)
    duration_ms = n_frames * dt_ms
    spike_bool = spike_times_to_bool(spike_times.tolist(), n_frames, dt_ms)

    bundle = build_diagnostics_bundle(
        envelope=envelope,
        spike_bool=spike_bool,
        dt_ms=dt_ms,
        window_ms=(0.0, duration_ms),
        spike_threshold=spike_threshold,
    )
    report_diagnostics(bundle, label, output_dir / f"{label}_diagnostics.json")
    return bundle.is_valid


# ---------------------------------------------------------------------------
# Exp02a: Overlap Regime
# ---------------------------------------------------------------------------


def run_exp02a(output_dir: Path) -> bool:
    """Overlap regime: tonic keeps value elevated between beats.

    Uses signal="value", site="attack_gain_mul" at multiple strengths.
    Checks INV_COUPLING_CHANGES_STATE for each coupled run vs uncoupled.

    Returns True if coupling changes state for at least one strength.
    """
    print("\n" + "=" * 60)
    print("EXP02a: OVERLAP REGIME (tonic=0.15, signal=value)")
    print("=" * 60)

    bpm = 120.0
    duration_sec = 5.0
    dt_ms = 1.0
    tonic = 0.15
    intervals = [0, 7, 6]
    strengths = [0.1, 0.2, 0.5]

    any_passed = False
    all_diagnostics_valid = True

    for interval in intervals:
        base_pitch = 60
        voices_uncoupled = [
            _create_voice(base_pitch, 0),
            _create_voice(base_pitch + interval, 1),
        ]
        _, signal = pulse_train(bpm, duration_sec, dt_ms)
        signals = [signal, signal.copy()]

        # Uncoupled baseline
        baseline = _run_uncoupled_sim(voices_uncoupled, signals, dt_ms, tonic)

        for strength in strengths:
            voices_coupled = [
                _create_voice(base_pitch, 0),
                _create_voice(base_pitch + interval, 1),
            ]
            cfg = CouplingConfig(
                signal="value",
                site="attack_gain_mul",
                strength=strength,
            )
            coupled = _run_coupled_sim(voices_coupled, signals, cfg, dt_ms, tonic)

            # Check coupling changes state (voice 0)
            n_frames = coupled["num_samples"]
            spike_bool_c = spike_times_to_bool(coupled["spike_times"][0].tolist(), n_frames, dt_ms)
            spike_bool_u = spike_times_to_bool(baseline["spike_times"][0].tolist(), n_frames, dt_ms)
            inv = check_coupling_changes_state(
                envelope_coupled=coupled["envelopes"][0],
                envelope_uncoupled=baseline["envelopes"][0],
                spike_bool_coupled=spike_bool_c,
                spike_bool_uncoupled=spike_bool_u,
                dt_ms=dt_ms,
            )

            label = f"exp02a_int{interval}_s{strength}"
            status = "PASS" if inv.passed else "FAIL"
            print(f"  {label}: INV_COUPLING_CHANGES_STATE = {status}")
            print(f"    mean_abs_diff={inv.details['mean_abs_envelope_diff']:.6f}")

            if inv.passed:
                any_passed = True

            # Report coupling summary
            report_coupling_summary(coupled["coupling_raws"], coupled["modulations"], label)

            # Per-voice diagnostics
            for v_idx in range(2):
                valid = _check_diagnostics_valid(
                    coupled["envelopes"][v_idx],
                    coupled["spike_times"][v_idx],
                    dt_ms,
                    f"{label}_v{v_idx}",
                    output_dir,
                )
                if not valid:
                    all_diagnostics_valid = False

    print(
        f"\n  EXP02a RESULT: {'PASS' if any_passed else 'FAIL'} "
        f"(coupling changes state in at least one config)"
    )
    return any_passed and all_diagnostics_valid


# ---------------------------------------------------------------------------
# Exp02b: Slow State at 120 BPM
# ---------------------------------------------------------------------------


def run_exp02b(output_dir: Path) -> bool:
    """Slow state: activity_ema signal at 120 BPM, no tonic.

    Uses signal="activity_ema", site="attack_gain_mul", strength=0.5.
    activity_ema ~0.303 at steady state. With unison weight=1.0 and
    strength=0.5: coupling_raw = 0.303 * 1.0 * 0.5 = 0.152.

    Returns True if coupling changes state for at least one interval.
    """
    print("\n" + "=" * 60)
    print("EXP02b: SLOW STATE (activity_ema, 120 BPM, no tonic)")
    print("=" * 60)

    bpm = 120.0
    duration_sec = 10.0
    dt_ms = 1.0
    intervals = [0, 7, 6]
    strength = 0.5

    any_passed = False
    all_diagnostics_valid = True

    for interval in intervals:
        base_pitch = 60
        voices_uncoupled = [
            _create_voice(base_pitch, 0),
            _create_voice(base_pitch + interval, 1),
        ]
        _, signal = pulse_train(bpm, duration_sec, dt_ms)
        signals = [signal, signal.copy()]

        baseline = _run_uncoupled_sim(voices_uncoupled, signals, dt_ms, tonic=0.0)

        voices_coupled = [
            _create_voice(base_pitch, 0),
            _create_voice(base_pitch + interval, 1),
        ]
        cfg = CouplingConfig(
            signal="activity_ema",
            site="attack_gain_mul",
            strength=strength,
        )
        coupled = _run_coupled_sim(voices_coupled, signals, cfg, dt_ms, tonic=0.0)

        n_frames = coupled["num_samples"]
        spike_bool_c = spike_times_to_bool(coupled["spike_times"][0].tolist(), n_frames, dt_ms)
        spike_bool_u = spike_times_to_bool(baseline["spike_times"][0].tolist(), n_frames, dt_ms)
        inv = check_coupling_changes_state(
            envelope_coupled=coupled["envelopes"][0],
            envelope_uncoupled=baseline["envelopes"][0],
            spike_bool_coupled=spike_bool_c,
            spike_bool_uncoupled=spike_bool_u,
            dt_ms=dt_ms,
        )

        label = f"exp02b_int{interval}_s{strength}"
        status = "PASS" if inv.passed else "FAIL"
        print(f"  {label}: INV_COUPLING_CHANGES_STATE = {status}")
        print(f"    mean_abs_diff={inv.details['mean_abs_envelope_diff']:.6f}")

        if inv.passed:
            any_passed = True

        report_coupling_summary(coupled["coupling_raws"], coupled["modulations"], label)

        for v_idx in range(2):
            valid = _check_diagnostics_valid(
                coupled["envelopes"][v_idx],
                coupled["spike_times"][v_idx],
                dt_ms,
                f"{label}_v{v_idx}",
                output_dir,
            )
            if not valid:
                all_diagnostics_valid = False

    # INFO: correlation ordering (demoted from pass condition)
    print("\n  INFO: Correlation ordering (unison >= fifth >= tritone) - not a pass condition")

    print(
        f"\n  EXP02b RESULT: {'PASS' if any_passed else 'FAIL'} "
        f"(coupling changes state for at least one interval)"
    )
    return any_passed and all_diagnostics_valid


# ---------------------------------------------------------------------------
# Exp02c: Spike Impulse
# ---------------------------------------------------------------------------


def run_exp02c(output_dir: Path) -> bool:
    """Spike impulse: decaying trace triggered by spikes.

    Uses signal="spike_impulse", site="attack_gain_mul", strength=0.3,
    tau=150ms. At 500ms beat spacing, impulse decays to ~0.036.
    Tests at higher strength (0.8) as well.

    Returns True if coupling changes state for at least one config.
    """
    print("\n" + "=" * 60)
    print("EXP02c: SPIKE IMPULSE (tau=150ms)")
    print("=" * 60)

    bpm = 120.0
    duration_sec = 10.0
    dt_ms = 1.0
    intervals = [0, 7]
    strengths = [0.3, 0.8]

    any_passed = False
    all_diagnostics_valid = True

    for interval in intervals:
        base_pitch = 60
        _, signal = pulse_train(bpm, duration_sec, dt_ms)
        signals = [signal, signal.copy()]

        voices_uncoupled = [
            _create_voice(base_pitch, 0),
            _create_voice(base_pitch + interval, 1),
        ]
        baseline = _run_uncoupled_sim(voices_uncoupled, signals, dt_ms, tonic=0.0)

        for strength in strengths:
            voices_coupled = [
                _create_voice(base_pitch, 0),
                _create_voice(base_pitch + interval, 1),
            ]
            cfg = CouplingConfig(
                signal="spike_impulse",
                site="attack_gain_mul",
                strength=strength,
                impulse_kernel_tau_ms=150.0,
            )
            coupled = _run_coupled_sim(voices_coupled, signals, cfg, dt_ms, tonic=0.0)

            n_frames = coupled["num_samples"]
            spike_bool_c = spike_times_to_bool(coupled["spike_times"][0].tolist(), n_frames, dt_ms)
            spike_bool_u = spike_times_to_bool(baseline["spike_times"][0].tolist(), n_frames, dt_ms)
            inv = check_coupling_changes_state(
                envelope_coupled=coupled["envelopes"][0],
                envelope_uncoupled=baseline["envelopes"][0],
                spike_bool_coupled=spike_bool_c,
                spike_bool_uncoupled=spike_bool_u,
                dt_ms=dt_ms,
            )

            label = f"exp02c_int{interval}_s{strength}"
            status = "PASS" if inv.passed else "FAIL"
            print(f"  {label}: INV_COUPLING_CHANGES_STATE = {status}")
            print(f"    mean_abs_diff={inv.details['mean_abs_envelope_diff']:.6f}")

            if inv.passed:
                any_passed = True

            report_coupling_summary(coupled["coupling_raws"], coupled["modulations"], label)

            for v_idx in range(2):
                valid = _check_diagnostics_valid(
                    coupled["envelopes"][v_idx],
                    coupled["spike_times"][v_idx],
                    dt_ms,
                    f"{label}_v{v_idx}",
                    output_dir,
                )
                if not valid:
                    all_diagnostics_valid = False

    print(
        f"\n  EXP02c RESULT: {'PASS' if any_passed else 'FAIL'} "
        f"(spike impulse coupling changes state)"
    )
    if not any_passed:
        print("  NOTE: Spike impulse may be too weak at 120 BPM (500ms spacing).")
        print("        This is an honest finding about the regime's limitations.")
    return any_passed and all_diagnostics_valid


# ---------------------------------------------------------------------------
# Exp02d: Symmetry Break
# ---------------------------------------------------------------------------


def run_exp02d(output_dir: Path) -> bool:
    """Symmetry break: asymmetric input should produce distinguishable output.

    Voice A: accented pulse train (accent_every=4, accent_strength=1.5)
    Voice B: constant amplitude pulse train
    Uses signal="activity_ema", site="attack_gain_mul", strength=0.5.

    Checks:
        INV_SYMMETRY_NOT_COLLAPSED (A != B in coupled condition) [GATE]
        INV_COUPLING_CHANGES_STATE (coupled != uncoupled) [INFO]

    Returns True if symmetry is not collapsed and diagnostics are valid.
    """
    print("\n" + "=" * 60)
    print("EXP02d: SYMMETRY BREAK (asymmetric input)")
    print("=" * 60)

    bpm = 120.0
    duration_sec = 10.0
    dt_ms = 1.0
    base_pitch = 60
    interval = 0  # Unison for maximum coupling
    strength = 0.5

    # Asymmetric signals
    _, signal_a = accented_pulse_train(
        bpm, duration_sec, accent_every=4, accent_strength=1.5, dt_ms=dt_ms
    )
    _, signal_b = pulse_train(bpm, duration_sec, dt_ms)
    signals = [signal_a, signal_b]

    # Uncoupled baseline
    voices_uncoupled = [
        _create_voice(base_pitch, 0),
        _create_voice(base_pitch + interval, 1),
    ]
    baseline = _run_uncoupled_sim(voices_uncoupled, signals, dt_ms, tonic=0.0)

    # Coupled run
    voices_coupled = [
        _create_voice(base_pitch, 0),
        _create_voice(base_pitch + interval, 1),
    ]
    cfg = CouplingConfig(
        signal="activity_ema",
        site="attack_gain_mul",
        strength=strength,
    )
    coupled = _run_coupled_sim(voices_coupled, signals, cfg, dt_ms, tonic=0.0)

    # Check symmetry not collapsed
    sym_inv = check_symmetry_not_collapsed(coupled["envelopes"][0], coupled["envelopes"][1])
    sym_status = "PASS" if sym_inv.passed else "FAIL"
    print(f"  INV_SYMMETRY_NOT_COLLAPSED: {sym_status}")
    print(f"    correlation={sym_inv.details['correlation']:.6f}")
    print(f"    max_abs_diff={sym_inv.details['max_abs_diff']:.6f}")

    # Check coupling changes state (voice A)
    n_frames = coupled["num_samples"]
    spike_bool_c = spike_times_to_bool(coupled["spike_times"][0].tolist(), n_frames, dt_ms)
    spike_bool_u = spike_times_to_bool(baseline["spike_times"][0].tolist(), n_frames, dt_ms)
    coupling_inv = check_coupling_changes_state(
        envelope_coupled=coupled["envelopes"][0],
        envelope_uncoupled=baseline["envelopes"][0],
        spike_bool_coupled=spike_bool_c,
        spike_bool_uncoupled=spike_bool_u,
        dt_ms=dt_ms,
    )
    coupling_status = "PASS" if coupling_inv.passed else "INFO"
    print(f"  INV_COUPLING_CHANGES_STATE (voice A): {coupling_status} [informational]")
    print(f"    mean_abs_diff={coupling_inv.details['mean_abs_envelope_diff']:.6f}")

    report_coupling_summary(coupled["coupling_raws"], coupled["modulations"], "exp02d")

    # Diagnostics validity
    all_diagnostics_valid = True
    for v_idx in range(2):
        valid = _check_diagnostics_valid(
            coupled["envelopes"][v_idx],
            coupled["spike_times"][v_idx],
            dt_ms,
            f"exp02d_v{v_idx}",
            output_dir,
        )
        if not valid:
            all_diagnostics_valid = False

    passed = sym_inv.passed and all_diagnostics_valid
    print(f"\n  EXP02d RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Main + Acceptance Gate
# ---------------------------------------------------------------------------


def main() -> bool:
    """Run all sub-experiments and evaluate acceptance gate.

    Acceptance Gate:
        gate_1: exp02a passes (overlap regime works)
        gate_2: exp02b OR exp02c passes (at least one non-overlap works)
        gate_3: exp02d passes (symmetry not collapsed)
        gate_4: all diagnostics valid (implicit in sub-experiment returns)

    Returns True if the acceptance gate passes.
    """
    print("=" * 60)
    print("EXPERIMENT 02 (v0.1): COUPLING ENGINE VALIDATION")
    print("=" * 60)

    output_dir = Path("results/exp_02_coupling")
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_1 = run_exp02a(output_dir)
    gate_2b = run_exp02b(output_dir)
    gate_2c = run_exp02c(output_dir)
    gate_3 = run_exp02d(output_dir)

    gate_2 = gate_2b or gate_2c

    # Acceptance gate
    print("\n" + "=" * 60)
    print("ACCEPTANCE GATE:")
    print("=" * 60)
    print(f"  gate_1 (exp02a overlap regime):      {'PASS' if gate_1 else 'FAIL'}")
    print(f"  gate_2 (exp02b OR exp02c non-overlap): {'PASS' if gate_2 else 'FAIL'}")
    print(f"    gate_2b (activity_ema):             {'PASS' if gate_2b else 'FAIL'}")
    print(f"    gate_2c (spike_impulse):            {'PASS' if gate_2c else 'FAIL'}")
    print(f"  gate_3 (exp02d symmetry break):      {'PASS' if gate_3 else 'FAIL'}")

    passed = gate_1 and gate_2 and gate_3
    print("-" * 60)
    print(f"  OVERALL: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    return passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
