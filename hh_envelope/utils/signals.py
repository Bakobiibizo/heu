"""Signal Generation
===================

Input signal generators for experiments and validation.

Pulse Trains:
- pulse_train: Regular pulses at specified BPM
- accented_pulse_train: Periodic accented beats
- jittered_pulse_train: Humanized timing with random jitter

Target Envelopes:
- target_envelope_staccato: Sharp attack, quick decay
- target_envelope_legato: Sustained plateau, gentle release
- target_envelope_grace_note: Grace note followed by main note
- target_envelope_accent: Initial peak above sustain level

Input Signals:
- input_signal_staccato: Single impulse trigger
- input_signal_legato: Onset pulse + sustained tonic
- input_signal_grace_note: Two pulses for grace note articulation
"""

from __future__ import annotations

import math

import numpy as np


def pulse_train(
    bpm: float,
    duration_sec: float,
    dt_ms: float = 1.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a regular pulse train at specified BPM.

    Args:
        bpm: Beats per minute
        duration_sec: Total duration in seconds
        dt_ms: Timestep in milliseconds
        amplitude: Pulse amplitude

    Returns:
        Tuple of (time_array_ms, signal_array)
    """
    # Calculate timing
    beat_period_ms = 60000.0 / bpm  # ms per beat
    duration_ms = duration_sec * 1000.0
    num_samples = int(duration_ms / dt_ms)

    # Create time array
    time_ms = np.arange(num_samples) * dt_ms

    # Create signal array
    signal = np.zeros(num_samples)

    # Place pulses at beat times
    beat_time_ms = 0.0
    while beat_time_ms < duration_ms:
        sample_idx = int(beat_time_ms / dt_ms)
        if 0 <= sample_idx < num_samples:
            signal[sample_idx] = amplitude
        beat_time_ms += beat_period_ms

    return time_ms, signal


def accented_pulse_train(
    bpm: float,
    duration_sec: float,
    accent_every: int = 4,
    accent_strength: float = 1.5,
    base_amplitude: float = 1.0,
    dt_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a pulse train with periodic accents.

    Args:
        bpm: Beats per minute
        duration_sec: Total duration in seconds
        accent_every: Accent every Nth beat (e.g., 4 for downbeats in 4/4)
        accent_strength: Multiplier for accented beats
        base_amplitude: Base pulse amplitude
        dt_ms: Timestep in milliseconds

    Returns:
        Tuple of (time_array_ms, signal_array)
    """
    beat_period_ms = 60000.0 / bpm
    duration_ms = duration_sec * 1000.0
    num_samples = int(duration_ms / dt_ms)

    time_ms = np.arange(num_samples) * dt_ms
    signal = np.zeros(num_samples)

    beat_time_ms = 0.0
    beat_count = 0
    while beat_time_ms < duration_ms:
        sample_idx = int(beat_time_ms / dt_ms)
        if 0 <= sample_idx < num_samples:
            # Accent every Nth beat (0-indexed, so first beat is accented)
            if beat_count % accent_every == 0:
                signal[sample_idx] = base_amplitude * accent_strength
            else:
                signal[sample_idx] = base_amplitude
        beat_time_ms += beat_period_ms
        beat_count += 1

    return time_ms, signal


def jittered_pulse_train(
    bpm: float,
    duration_sec: float,
    jitter_ms: float = 10.0,
    dt_ms: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a pulse train with random timing jitter (humanization).

    Args:
        bpm: Beats per minute
        duration_sec: Total duration in seconds
        jitter_ms: Standard deviation of timing jitter in ms
        dt_ms: Timestep in milliseconds
        seed: Random seed for reproducibility

    Returns:
        Tuple of (time_array_ms, signal_array)
    """
    rng = np.random.default_rng(seed)

    beat_period_ms = 60000.0 / bpm
    duration_ms = duration_sec * 1000.0
    num_samples = int(duration_ms / dt_ms)

    time_ms = np.arange(num_samples) * dt_ms
    signal = np.zeros(num_samples)

    # Generate jittered pulse times
    beat_time_ms = 0.0
    while beat_time_ms < duration_ms:
        # Add Gaussian jitter
        jittered_time = beat_time_ms + rng.normal(0, jitter_ms)
        jittered_time = max(0, jittered_time)  # Don't go negative

        sample_idx = int(jittered_time / dt_ms)
        if 0 <= sample_idx < num_samples:
            signal[sample_idx] = 1.0
        beat_time_ms += beat_period_ms

    return time_ms, signal


def get_pulse_times(
    bpm: float,
    duration_sec: float,
) -> np.ndarray:
    """Get array of pulse times in milliseconds.

    Args:
        bpm: Beats per minute
        duration_sec: Total duration in seconds

    Returns:
        Array of pulse times in ms
    """
    beat_period_ms = 60000.0 / bpm
    duration_ms = duration_sec * 1000.0

    pulse_times = []
    beat_time_ms = 0.0
    while beat_time_ms < duration_ms:
        pulse_times.append(beat_time_ms)
        beat_time_ms += beat_period_ms

    return np.array(pulse_times)


def target_envelope_staccato(
    note_duration_ms: float,
    dt_ms: float = 1.0,
) -> np.ndarray:
    """Generate staccato target by running HEU at reference theta.

    The target IS what the HEU produces at a known parameter set.
    This ensures the target is achievable and musically meaningful.

    Musical criteria: sharp attack, fast decay (~20ms to 10%), near-zero sustain.

    Reference theta: total=-1.0, balance=1.5, attack_tau=-1.0, recovery_tau=0.0

    Args:
        note_duration_ms: Total note duration
        dt_ms: Timestep in milliseconds

    Returns:
        Envelope array
    """
    from hh_envelope.core import HHEnvelopeUnit

    unit = HHEnvelopeUnit()
    unit.set_theta(
        theta_total=-1.0,
        theta_balance=1.5,
        theta_attack_tau=-1.0,
        theta_recovery_tau=0.0,
    )
    unit.reset()

    inp = input_signal_staccato(note_duration_ms, dt_ms)
    num_samples = len(inp)
    envelope = np.zeros(num_samples)
    for i in range(num_samples):
        out = unit.step(inp[i], dt_ms)
        envelope[i] = out.envelope

    return envelope


def target_envelope_legato(
    note_duration_ms: float,
    dt_ms: float = 1.0,
) -> np.ndarray:
    """Generate legato target by running HEU at reference theta.

    The target IS what the HEU produces at a known parameter set.
    This ensures the target is achievable and musically meaningful.

    Musical criteria: fast attack, sustained plateau (~0.80-0.95),
    natural decay slope, release when tonic stops.

    Reference theta: total=-2.0, balance=2.0, attack_tau=0.0, recovery_tau=-1.0

    Args:
        note_duration_ms: Total note duration
        dt_ms: Timestep in milliseconds

    Returns:
        Envelope array
    """
    from hh_envelope.core import HHEnvelopeUnit

    unit = HHEnvelopeUnit()
    unit.set_theta(
        theta_total=-2.0,
        theta_balance=2.0,
        theta_attack_tau=0.0,
        theta_recovery_tau=-1.0,
    )
    unit.reset()

    inp = input_signal_legato(note_duration_ms, dt_ms)
    num_samples = len(inp)
    envelope = np.zeros(num_samples)
    for i in range(num_samples):
        out = unit.step(inp[i], dt_ms)
        envelope[i] = out.envelope

    return envelope


def target_envelope_grace_note(
    main_duration_ms: float,
    dt_ms: float = 1.0,
) -> np.ndarray:
    """Generate grace note target by running HEU at reference theta.

    The target IS what the HEU produces at a known parameter set.
    This ensures the target is achievable and musically meaningful.

    Musical criteria: grace peak (~0.93), clean gap (near-zero),
    main peak (1.0), sustained tail with natural decay.

    Reference theta: total=-0.5, balance=1.5, attack_tau=-0.5, recovery_tau=0.5

    Args:
        main_duration_ms: Duration of main note (grace note adds ~40ms)
        dt_ms: Timestep in milliseconds

    Returns:
        Envelope array
    """
    from hh_envelope.core import HHEnvelopeUnit

    unit = HHEnvelopeUnit()
    unit.set_theta(
        theta_total=-0.5,
        theta_balance=1.5,
        theta_attack_tau=-0.5,
        theta_recovery_tau=0.5,
    )
    unit.reset()

    inp = input_signal_grace_note(main_duration_ms, dt_ms)
    num_samples = len(inp)
    envelope = np.zeros(num_samples)
    for i in range(num_samples):
        out = unit.step(inp[i], dt_ms)
        envelope[i] = out.envelope

    return envelope


def input_signal_staccato(
    duration_ms: float,
    dt_ms: float = 1.0,
    amplitude: float = 1.5,
) -> np.ndarray:
    """Generate input signal for staccato articulation.

    A single impulse at t=0 to trigger a transient response.

    Args:
        duration_ms: Total duration
        dt_ms: Timestep
        amplitude: Pulse amplitude (default 1.5 for attack_bias=0.5)

    Returns:
        Input signal array
    """
    num_samples = int(duration_ms / dt_ms)
    signal = np.zeros(num_samples)
    signal[0] = amplitude
    return signal


def input_signal_legato(
    duration_ms: float,
    dt_ms: float = 1.0,
    onset_amplitude: float = 1.5,
    tonic_level: float = 0.35,
    tonic_start_ms: float = 20.0,
    tonic_end_ms: float | None = None,
) -> np.ndarray:
    """Generate input signal for legato articulation.

    Initial onset pulse + sustained tonic drive to maintain envelope.

    Args:
        duration_ms: Total duration
        dt_ms: Timestep
        onset_amplitude: Initial onset pulse amplitude (default 1.5 for attack_bias=0.5)
        tonic_level: Continuous excitation level during sustain (default 0.35 for attack_bias=0.5)
        tonic_start_ms: When tonic drive begins (after attack settles)
        tonic_end_ms: When tonic drive ends (None = 100ms before end)

    Returns:
        Input signal array
    """
    num_samples = int(duration_ms / dt_ms)
    signal = np.zeros(num_samples)

    # Initial onset pulse
    signal[0] = onset_amplitude

    # Tonic drive for sustain
    tonic_start_idx = int(tonic_start_ms / dt_ms)
    if tonic_end_ms is None:
        tonic_end_ms = duration_ms - 100.0  # Stop 100ms before end for release
    tonic_end_idx = int(max(tonic_start_ms, tonic_end_ms) / dt_ms)

    for i in range(tonic_start_idx, min(tonic_end_idx, num_samples)):
        signal[i] = tonic_level

    return signal


def input_signal_grace_note(
    main_duration_ms: float,
    dt_ms: float = 1.0,
    grace_amplitude: float = 1.0,
    main_amplitude: float = 1.5,
    tonic_level: float = 0.25,
) -> np.ndarray:
    """Generate input signal for grace note articulation.

    Two pulses: quick grace note followed by main note.
    Includes tonic drive during main note to sustain the second peak.
    Timing is synchronized with target_envelope_grace_note.

    Args:
        main_duration_ms: Duration of main note portion
        dt_ms: Timestep
        grace_amplitude: Grace note pulse amplitude (default 1.0 for attack_bias=0.5)
        main_amplitude: Main note pulse amplitude (default 1.5 for attack_bias=0.5)
        tonic_level: Tonic drive during main note sustain (default 0.25 for attack_bias=0.5)

    Returns:
        Input signal array
    """
    # Must match target_envelope_grace_note timing
    grace_duration = 30.0  # Duration of grace note region
    gap_duration = 10.0  # Gap between grace and main
    total_duration = grace_duration + gap_duration + main_duration_ms

    num_samples = int(total_duration / dt_ms)
    signal = np.zeros(num_samples)

    # Grace note pulse at t=0
    signal[0] = grace_amplitude

    # Main note pulse at start of main note region
    main_start_ms = grace_duration + gap_duration  # 40ms
    main_idx = int(main_start_ms / dt_ms)
    if 0 <= main_idx < num_samples:
        signal[main_idx] = main_amplitude

    # Tonic drive during main note sustain (after attack settles)
    tonic_start_ms = main_start_ms + 30.0  # 30ms after main onset
    tonic_end_ms = total_duration - 80.0  # Stop 80ms before end for release
    tonic_start_idx = int(tonic_start_ms / dt_ms)
    tonic_end_idx = int(max(tonic_start_ms, tonic_end_ms) / dt_ms)

    for i in range(tonic_start_idx, min(tonic_end_idx, num_samples)):
        signal[i] = tonic_level

    return signal


def target_envelope_grace_note_two_voice(
    main_duration_ms: float = 360.0,
    dt_ms: float = 1.0,
) -> np.ndarray:
    """Generate two-voice impossibility target by running ArticulatedNote.

    The target IS what a two-voice HEU produces at reference theta values.
    This makes it achievable for two-voice (by construction) and impossible
    for single-voice (due to conflicting parameter requirements).

    The conflict:
    - Grace voice: recovery-biased, fast decay after tonic stops at 25ms
    - Main voice: attack-biased, flat tonic-driven sustain
    - Single HEU can't have both fast decay AND flat sustain

    Reference parameters:
    - Grace: total=-1.0, balance=-0.5, attack_tau=-0.5, recovery_tau=-1.0
    - Main: total=-2.0, balance=1.5, attack_tau=0.0, recovery_tau=-1.0

    Args:
        main_duration_ms: Duration of main note (after grace region)
        dt_ms: Timestep in milliseconds

    Returns:
        Envelope array (two-voice combined via max)
    """
    from hh_envelope.polyphony.articulation import (
        ArticulatedNote,
        ArticulationConfig,
        CombinationMode,
        TonicConfig,
        VoiceRole,
    )

    total_duration = 50.0 + main_duration_ms  # Grace+gap=50ms, then main

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
        pulse_amplitude=0.0,  # Pure tonic drive (no onset spike)
        active_duration_ms=None,
        tonic_config=TonicConfig(level=0.35, start_ms=0.0, end_before_ms=60.0),
    )
    config = ArticulationConfig(
        name="two_voice_impossibility",
        voice_roles=[grace_role, main_role],
        total_duration_ms=total_duration,
        combination_mode=CombinationMode.MAX,
    )

    # Reference theta: conflicting dynamics between voices
    grace_theta = {
        "theta_total": -1.0,
        "theta_balance": -0.5,  # Recovery-biased → fast decay
        "theta_attack_tau": -0.5,
        "theta_recovery_tau": -1.0,  # Fast recovery
    }
    main_theta = {
        "theta_total": -2.0,
        "theta_balance": 1.5,  # Attack-biased → flat sustain
        "theta_attack_tau": 0.0,
        "theta_recovery_tau": -1.0,  # Slow recovery relative to conductance
    }

    note = ArticulatedNote(config)
    note.set_voice_theta("grace", grace_theta)
    note.set_voice_theta("main", main_theta)

    envelope = note.run_full(dt_ms)
    return np.array(envelope)


def target_envelope_accent(
    note_duration_ms: float,
    accent_peak: float = 1.2,
    sustain_level: float = 0.7,
    dt_ms: float = 1.0,
) -> np.ndarray:
    """Generate an accented envelope with initial peak above sustain.

    Characteristics:
    - Sharp attack to accent peak
    - Quick drop to sustain level
    - Gradual decay

    Args:
        note_duration_ms: Total note duration
        accent_peak: Peak level (can be > 1.0)
        sustain_level: Sustain level after accent
        dt_ms: Timestep in milliseconds

    Returns:
        Envelope array
    """
    num_samples = int(note_duration_ms / dt_ms)
    time_ms = np.arange(num_samples) * dt_ms

    attack_tau = 2.0  # Very fast attack
    accent_decay_tau = 20.0  # Quick decay from accent
    attack_time = 5.0
    accent_duration = 50.0  # Time to reach sustain
    release_tau = 80.0

    envelope = np.zeros(num_samples)
    for i, t in enumerate(time_ms):
        if t < attack_time:
            # Attack to accent peak
            envelope[i] = accent_peak * (1.0 - math.exp(-t / attack_tau))
        elif t < accent_duration:
            # Decay from accent to sustain
            t_decay = t - attack_time
            # Exponential decay from peak toward sustain
            envelope[i] = sustain_level + (accent_peak - sustain_level) * math.exp(
                -t_decay / accent_decay_tau
            )
        else:
            # Gradual release from sustain
            t_release = t - accent_duration
            envelope[i] = sustain_level * math.exp(-t_release / release_tau)

    return envelope
