"""Loss Functions
================

Loss functions for envelope fitting and optimization.

Functions:
- envelope_mse: Mean squared error between predicted and target envelopes
- smoothness_penalty: Penalty for jitter/oscillation in envelopes
- combined_loss: MSE + smoothness penalty with configurable weight
- temporal_alignment_loss: Loss based on spike timing alignment
- peak_timing_loss: Loss based on envelope peak timing
- shape_correlation_loss: Loss based on shape correlation (1 - correlation)

All losses return float values (for gradient-free optimization).
"""

from __future__ import annotations

import numpy as np


def envelope_mse(predicted: np.ndarray, target: np.ndarray) -> float:
    """Compute mean squared error between predicted and target envelopes.

    Args:
        predicted: Predicted envelope array
        target: Target envelope array

    Returns:
        Mean squared error
    """
    if len(predicted) != len(target):
        # Handle length mismatch by truncating to shorter
        min_len = min(len(predicted), len(target))
        predicted = predicted[:min_len]
        target = target[:min_len]

    return float(np.mean((predicted - target) ** 2))


def smoothness_penalty(envelope: np.ndarray) -> float:
    """Compute smoothness penalty to discourage jittery envelopes.

    Penalizes rapid changes in the envelope, which can indicate
    unstable dynamics or overfitting to noise.

    Args:
        envelope: Envelope array

    Returns:
        Mean squared difference (lower = smoother)
    """
    if len(envelope) < 2:
        return 0.0

    diff = np.diff(envelope)
    return float(np.mean(diff**2))


def combined_loss(
    predicted: np.ndarray,
    target: np.ndarray,
    lambda_smooth: float = 0.01,
) -> float:
    """Compute combined loss with MSE and smoothness penalty.

    Args:
        predicted: Predicted envelope array
        target: Target envelope array
        lambda_smooth: Weight for smoothness penalty

    Returns:
        Combined loss value
    """
    mse = envelope_mse(predicted, target)
    smooth = smoothness_penalty(predicted)
    return mse + lambda_smooth * smooth


def temporal_alignment_loss(
    predicted_spikes: list[float] | np.ndarray,
    target_spikes: list[float] | np.ndarray,
    tolerance_ms: float = 10.0,
    missing_penalty: float = 1.0,
    extra_penalty: float = 0.5,
) -> float:
    """Compute loss based on spike timing alignment.

    For each target spike, finds the nearest predicted spike and
    penalizes timing errors. Also penalizes missing and extra spikes.

    Args:
        predicted_spikes: Array of predicted spike times in ms
        target_spikes: Array of target spike times in ms
        tolerance_ms: Timing tolerance before penalty increases
        missing_penalty: Penalty for each missing spike
        extra_penalty: Penalty for each extra spike

    Returns:
        Temporal alignment loss
    """
    predicted = np.array(predicted_spikes) if len(predicted_spikes) > 0 else np.array([])
    target = np.array(target_spikes) if len(target_spikes) > 0 else np.array([])

    if len(target) == 0:
        # No target spikes - penalize any predicted spikes
        return extra_penalty * len(predicted)

    if len(predicted) == 0:
        # No predicted spikes - penalize all missing
        return missing_penalty * len(target)

    # For each target spike, find nearest predicted spike
    timing_errors = []
    matched_predicted = set()

    for target_t in target:
        distances = np.abs(predicted - target_t)
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]

        if nearest_dist <= tolerance_ms:
            # Within tolerance - small penalty proportional to distance
            timing_errors.append((nearest_dist / tolerance_ms) ** 2)
            matched_predicted.add(nearest_idx)
        else:
            # Outside tolerance - penalty increases with distance
            timing_errors.append(1.0 + (nearest_dist - tolerance_ms) / tolerance_ms)
            # Still count as matched if it's the closest
            if nearest_dist < 2 * tolerance_ms:
                matched_predicted.add(nearest_idx)

    # Compute base timing loss
    timing_loss = sum(timing_errors) / len(target)

    # Add penalty for extra (unmatched) predicted spikes
    num_extra = len(predicted) - len(matched_predicted)
    extra_loss = extra_penalty * max(0, num_extra) / max(1, len(target))

    return timing_loss + extra_loss


def peak_timing_loss(
    predicted: np.ndarray,
    target: np.ndarray,
    dt_ms: float = 1.0,
    threshold: float = 0.5,
) -> float:
    """Compute loss based on envelope peak timing alignment.

    Finds peaks in both envelopes and computes timing error.

    Args:
        predicted: Predicted envelope array
        target: Target envelope array
        dt_ms: Timestep in milliseconds
        threshold: Minimum height to count as a peak

    Returns:
        Peak timing loss
    """
    from scipy.signal import find_peaks  # type: ignore[import-untyped]

    # Find peaks
    pred_peaks, _ = find_peaks(predicted, height=threshold)
    target_peaks, _ = find_peaks(target, height=threshold)

    # Convert to times
    pred_times = pred_peaks * dt_ms
    target_times = target_peaks * dt_ms

    return temporal_alignment_loss(
        pred_times,
        target_times,
        tolerance_ms=dt_ms * 10,
    )


def shape_correlation_loss(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute loss based on shape correlation (1 - correlation).

    Measures how well the shape matches regardless of amplitude.

    Args:
        predicted: Predicted envelope array
        target: Target envelope array

    Returns:
        1 - correlation (0 = perfect match, 2 = anti-correlated)
    """
    if len(predicted) != len(target):
        min_len = min(len(predicted), len(target))
        predicted = predicted[:min_len]
        target = target[:min_len]

    # Normalize
    pred_norm = predicted - np.mean(predicted)
    target_norm = target - np.mean(target)

    pred_std = np.std(pred_norm)
    target_std = np.std(target_norm)

    if pred_std == 0 or target_std == 0:
        return 1.0  # No correlation possible

    correlation = np.sum(pred_norm * target_norm) / (len(predicted) * pred_std * target_std)

    return float(1.0 - correlation)
