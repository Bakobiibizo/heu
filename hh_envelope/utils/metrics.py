"""Measurement Functions
=======================

Metrics for experiment evaluation and validation.

Functions:
    - phase_locking_value: PLV for tempo locking assessment
    - inter_spike_interval_stats: ISI mean, std, CV, min, max
    - envelope_peak_consistency: Peak statistics at pulse times
    - cross_correlation: Cross-correlation between two signals
    - spike_phase_coherence: Phase coherence of spike times
"""

from __future__ import annotations

import numpy as np


def phase_locking_value(
    pulse_times: np.ndarray,
    spike_times: np.ndarray,
    pulse_period_ms: float | None = None,
) -> float:
    """Compute phase locking value (PLV) between spikes and pulses.

    PLV measures how consistently spikes occur at the same phase
    relative to the driving pulse train. Values range from 0 (random)
    to 1 (perfect locking).

    Args:
        pulse_times: Array of pulse times in ms
        spike_times: Array of spike times in ms
        pulse_period_ms: Period between pulses (computed if None)

    Returns:
        PLV value in [0, 1]
    """
    if len(spike_times) == 0:
        return 0.0

    if len(pulse_times) < 2:
        return 0.0

    # Compute pulse period if not provided
    if pulse_period_ms is None:
        pulse_period_ms = float(np.median(np.diff(pulse_times)))

    # For each spike, find phase relative to nearest pulse
    phases = []
    for spike_t in spike_times:
        # Find nearest pulse before spike
        pulse_idx = np.searchsorted(pulse_times, spike_t) - 1
        if pulse_idx < 0:
            pulse_idx = 0
        if pulse_idx >= len(pulse_times):
            continue

        pulse_t = pulse_times[pulse_idx]
        # Phase is time since pulse, normalized to [0, 1)
        phase = (spike_t - pulse_t) / pulse_period_ms
        phase = phase % 1.0  # Wrap to [0, 1)
        phases.append(phase)

    if len(phases) == 0:
        return 0.0

    phases_arr = np.array(phases)

    # PLV = magnitude of mean unit vector on circle
    # phase -> angle in [0, 2*pi)
    angles = 2 * np.pi * phases_arr
    mean_vector = np.mean(np.exp(1j * angles))
    plv = np.abs(mean_vector)

    return float(plv)


def inter_spike_interval_stats(spike_times: np.ndarray) -> dict[str, float]:
    """Compute statistics on inter-spike intervals.

    Args:
        spike_times: Array of spike times in ms

    Returns:
        Dictionary with ISI statistics
    """
    if len(spike_times) < 2:
        return {
            "mean_isi_ms": float("nan"),
            "std_isi_ms": float("nan"),
            "cv": float("nan"),
            "min_isi_ms": float("nan"),
            "max_isi_ms": float("nan"),
            "count": len(spike_times),
        }

    isis = np.diff(spike_times)

    mean_isi = float(np.mean(isis))
    std_isi = float(np.std(isis))
    cv = std_isi / mean_isi if mean_isi > 0 else float("nan")

    return {
        "mean_isi_ms": mean_isi,
        "std_isi_ms": std_isi,
        "cv": cv,
        "min_isi_ms": float(np.min(isis)),
        "max_isi_ms": float(np.max(isis)),
        "count": len(spike_times),
    }


def envelope_peak_consistency(
    envelope: np.ndarray,
    pulse_times: np.ndarray,
    window_ms: float = 50.0,
    dt_ms: float = 1.0,
) -> dict[str, float]:
    """Measure consistency of envelope peaks following pulses.

    Args:
        envelope: Envelope time series
        pulse_times: Array of pulse times in ms
        window_ms: Window after each pulse to search for peak
        dt_ms: Timestep in milliseconds

    Returns:
        Dictionary with peak statistics
    """
    window_samples = int(window_ms / dt_ms)
    peaks = []

    for pulse_t in pulse_times:
        sample_idx = int(pulse_t / dt_ms)
        start_idx = sample_idx
        end_idx = min(sample_idx + window_samples, len(envelope))

        if start_idx < len(envelope) and end_idx > start_idx:
            window = envelope[start_idx:end_idx]
            peak = float(np.max(window))
            peaks.append(peak)

    if len(peaks) == 0:
        return {
            "mean_peak": float("nan"),
            "std_peak": float("nan"),
            "cv": float("nan"),
        }

    peaks_arr = np.array(peaks)
    mean_peak = float(np.mean(peaks_arr))
    std_peak = float(np.std(peaks_arr))
    cv = std_peak / mean_peak if mean_peak > 0 else float("nan")

    return {
        "mean_peak": mean_peak,
        "std_peak": std_peak,
        "cv": cv,
    }


def cross_correlation(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    max_lag_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized cross-correlation between two signals.

    Args:
        signal_a: First signal
        signal_b: Second signal
        max_lag_samples: Maximum lag to compute

    Returns:
        Tuple of (lag_array, correlation_array)
    """
    # Normalize signals
    a_norm = signal_a - np.mean(signal_a)
    b_norm = signal_b - np.mean(signal_b)

    a_std = np.std(a_norm)
    b_std = np.std(b_norm)

    if a_std == 0 or b_std == 0:
        lags = np.arange(-max_lag_samples, max_lag_samples + 1)
        return lags, np.zeros_like(lags, dtype=float)

    a_norm = a_norm / a_std
    b_norm = b_norm / b_std

    # Compute cross-correlation
    full_corr = np.correlate(a_norm, b_norm, mode="full")
    full_corr = full_corr / len(signal_a)  # Normalize

    # Extract relevant lags
    center = len(full_corr) // 2
    start = max(0, center - max_lag_samples)
    end = min(len(full_corr), center + max_lag_samples + 1)

    corr = full_corr[start:end]
    lags = np.arange(start - center, end - center)

    return lags, corr


def spike_phase_coherence(
    spikes_a: np.ndarray,
    spikes_b: np.ndarray,
    reference_period_ms: float,
) -> float:
    """Measure phase coherence between two spike trains.

    Computes how consistently the relative phase between
    spikes_a and spikes_b is maintained.

    Args:
        spikes_a: First spike train times in ms
        spikes_b: Second spike train times in ms
        reference_period_ms: Reference period for phase computation

    Returns:
        Coherence value in [0, 1]
    """
    if len(spikes_a) == 0 or len(spikes_b) == 0:
        return 0.0

    # For each spike in A, find nearest spike in B and compute phase difference
    phase_diffs = []

    for spike_a in spikes_a:
        # Find nearest spike in B
        if len(spikes_b) == 0:
            continue

        idx = np.argmin(np.abs(spikes_b - spike_a))
        spike_b = spikes_b[idx]

        # Phase difference normalized to reference period
        phase_diff = (spike_a - spike_b) / reference_period_ms
        phase_diff = phase_diff % 1.0
        phase_diffs.append(phase_diff)

    if len(phase_diffs) == 0:
        return 0.0

    phase_diffs_arr = np.array(phase_diffs)

    # Coherence = concentration of phase differences
    angles = 2 * np.pi * phase_diffs_arr
    mean_vector = np.mean(np.exp(1j * angles))
    coherence = np.abs(mean_vector)

    return float(coherence)


def detect_spikes(
    envelope: np.ndarray,
    threshold: float = 0.5,
    dt_ms: float = 1.0,
    min_spacing_ms: float = 20.0,
) -> np.ndarray:
    """Detect spike times from an envelope signal.

    Args:
        envelope: Envelope time series
        threshold: Threshold for spike detection
        dt_ms: Timestep in milliseconds
        min_spacing_ms: Minimum spacing between spikes

    Returns:
        Array of spike times in ms
    """
    min_spacing_samples = int(min_spacing_ms / dt_ms)
    spike_times = []
    last_spike_sample = -min_spacing_samples - 1

    # Find threshold crossings (rising edge)
    for i in range(1, len(envelope)):
        if (
            envelope[i] >= threshold
            and envelope[i - 1] < threshold
            and (i - last_spike_sample) >= min_spacing_samples
        ):
            spike_times.append(i * dt_ms)
            last_spike_sample = i

    return np.array(spike_times)
