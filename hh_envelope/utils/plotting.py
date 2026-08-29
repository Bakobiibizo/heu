"""Visualization
===============

Plotting functions for experiments and analysis.

Functions:
    - plot_envelope_with_spikes: Envelope with spike markers
    - plot_pulse_response: Pulse input and envelope response
    - plot_two_voice_comparison: Side-by-side voice comparison
    - plot_learning_curve: Loss vs iterations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# Style constants
ENVELOPE_COLOR = "#2563eb"  # Blue
PULSE_COLOR = "#dc2626"  # Red
SPIKE_COLOR = "#16a34a"  # Green
TARGET_COLOR = "#9333ea"  # Purple
PREDICTED_COLOR = "#2563eb"  # Blue
VOICE_A_COLOR = "#2563eb"  # Blue
VOICE_B_COLOR = "#dc2626"  # Red


def plot_envelope_with_spikes(
    time_ms: np.ndarray,
    envelope: np.ndarray,
    spike_times: np.ndarray | None = None,
    title: str = "",
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot envelope with optional spike markers.

    Args:
        time_ms: Time array in milliseconds
        envelope: Envelope values
        spike_times: Optional array of spike times in ms
        title: Plot title
        ax: Optional matplotlib axes (creates new if None)

    Returns:
        The matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    # Plot envelope
    ax.plot(time_ms, envelope, color=ENVELOPE_COLOR, linewidth=1, label="Envelope")

    # Mark spikes
    if spike_times is not None and len(spike_times) > 0:
        for spike_t in spike_times:
            ax.axvline(spike_t, color=SPIKE_COLOR, alpha=0.5, linewidth=0.8)
        # Add to legend (single entry) - use invisible line for legend
        ax.axvline(x=0, color=SPIKE_COLOR, alpha=0.0, linewidth=0.8, label="Spikes")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Envelope")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    return ax


def plot_pulse_response(
    time_ms: np.ndarray,
    pulse_signal: np.ndarray,
    envelope: np.ndarray,
    spike_times: np.ndarray | None = None,
    title: str = "",
) -> "Figure":
    """Plot pulse input and envelope response in two panels.

    Args:
        time_ms: Time array in milliseconds
        pulse_signal: Pulse/excitation signal
        envelope: Envelope response
        spike_times: Optional array of spike times in ms
        title: Figure title

    Returns:
        The matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Top: Pulse input
    ax_pulse = axes[0]
    ax_pulse.stem(
        time_ms,
        pulse_signal,
        linefmt=PULSE_COLOR,
        markerfmt="ro",
        basefmt=" ",
    )
    ax_pulse.set_ylabel("Excitation")
    ax_pulse.set_title("Input Pulses")
    ax_pulse.grid(True, alpha=0.3)

    # Bottom: Envelope response
    ax_env = axes[1]
    ax_env.plot(time_ms, envelope, color=ENVELOPE_COLOR, linewidth=1, label="Envelope")

    if spike_times is not None and len(spike_times) > 0:
        for spike_t in spike_times:
            ax_env.axvline(spike_t, color=SPIKE_COLOR, alpha=0.5, linewidth=0.8)

    ax_env.set_xlabel("Time (ms)")
    ax_env.set_ylabel("Envelope")
    ax_env.set_ylim(-0.05, 1.1)
    ax_env.grid(True, alpha=0.3)
    ax_env.set_title("Envelope Response")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    return fig


def plot_two_voice_comparison(
    time_ms: np.ndarray,
    envelope_a: np.ndarray,
    envelope_b: np.ndarray,
    spikes_a: np.ndarray | None = None,
    spikes_b: np.ndarray | None = None,
    title: str = "",
) -> "Figure":
    """Plot two voice envelopes for comparison.

    Args:
        time_ms: Time array in milliseconds
        envelope_a: First voice envelope
        envelope_b: Second voice envelope
        spikes_a: Optional spike times for voice A
        spikes_b: Optional spike times for voice B
        title: Figure title

    Returns:
        The matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Voice A
    ax_a = axes[0]
    ax_a.plot(time_ms, envelope_a, color=VOICE_A_COLOR, linewidth=1, label="Voice A")
    if spikes_a is not None and len(spikes_a) > 0:
        for spike_t in spikes_a:
            ax_a.axvline(spike_t, color=SPIKE_COLOR, alpha=0.3, linewidth=0.8)
    ax_a.set_ylabel("Envelope A")
    ax_a.set_ylim(-0.05, 1.1)
    ax_a.legend(loc="upper right")
    ax_a.grid(True, alpha=0.3)

    # Voice B
    ax_b = axes[1]
    ax_b.plot(time_ms, envelope_b, color=VOICE_B_COLOR, linewidth=1, label="Voice B")
    if spikes_b is not None and len(spikes_b) > 0:
        for spike_t in spikes_b:
            ax_b.axvline(spike_t, color=SPIKE_COLOR, alpha=0.3, linewidth=0.8)
    ax_b.set_ylabel("Envelope B")
    ax_b.set_ylim(-0.05, 1.1)
    ax_b.legend(loc="upper right")
    ax_b.grid(True, alpha=0.3)

    # Overlay
    ax_overlay = axes[2]
    ax_overlay.plot(
        time_ms, envelope_a, color=VOICE_A_COLOR, linewidth=1, alpha=0.7, label="Voice A"
    )
    ax_overlay.plot(
        time_ms, envelope_b, color=VOICE_B_COLOR, linewidth=1, alpha=0.7, label="Voice B"
    )
    ax_overlay.set_xlabel("Time (ms)")
    ax_overlay.set_ylabel("Envelope")
    ax_overlay.set_ylim(-0.05, 1.1)
    ax_overlay.legend(loc="upper right")
    ax_overlay.grid(True, alpha=0.3)
    ax_overlay.set_title("Overlay")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    return fig


def plot_learning_curve(
    iterations: np.ndarray,
    losses: np.ndarray,
    target_loss: float | None = None,
    title: str = "",
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot learning curve showing loss over iterations.

    Args:
        iterations: Iteration numbers
        losses: Loss values
        target_loss: Optional target loss to show as horizontal line
        title: Plot title
        ax: Optional matplotlib axes

    Returns:
        The matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(iterations, losses, color=ENVELOPE_COLOR, linewidth=1.5, label="Loss")

    if target_loss is not None:
        ax.axhline(
            target_loss,
            color=TARGET_COLOR,
            linestyle="--",
            linewidth=1,
            label=f"Target ({target_loss:.4f})",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    return ax


def plot_envelope_fit(
    time_ms: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    title: str = "",
) -> "Figure":
    """Plot target vs predicted envelope with residual.

    Args:
        time_ms: Time array in milliseconds
        target: Target envelope
        predicted: Predicted envelope
        title: Figure title

    Returns:
        The matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[3, 1])

    # Main plot: overlay
    ax_main = axes[0]
    ax_main.plot(time_ms, target, color=TARGET_COLOR, linewidth=2, label="Target", alpha=0.8)
    ax_main.plot(
        time_ms,
        predicted,
        color=PREDICTED_COLOR,
        linewidth=1.5,
        label="Predicted",
        linestyle="--",
    )
    ax_main.set_ylabel("Envelope")
    ax_main.set_ylim(-0.05, 1.1)
    ax_main.legend(loc="upper right")
    ax_main.grid(True, alpha=0.3)

    # Residual plot
    ax_residual = axes[1]
    residual = predicted - target
    ax_residual.fill_between(
        time_ms, residual, 0, color=PREDICTED_COLOR, alpha=0.3, label="Residual"
    )
    ax_residual.axhline(0, color="black", linewidth=0.5)
    ax_residual.set_xlabel("Time (ms)")
    ax_residual.set_ylabel("Residual")
    ax_residual.set_ylim(-0.3, 0.3)
    ax_residual.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    return fig


def plot_phase_distribution(
    phases: np.ndarray,
    title: str = "Phase Distribution",
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot circular histogram of phases.

    Args:
        phases: Phase values in [0, 1)
        title: Plot title
        ax: Optional polar axes

    Returns:
        The matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))

    # Convert to radians
    angles = 2 * np.pi * phases

    # Histogram
    n_bins = 24
    hist, bin_edges = np.histogram(angles, bins=n_bins, range=(0, 2 * np.pi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 2 * np.pi / n_bins

    ax.bar(
        bin_centers,
        hist,
        width=width,
        color=ENVELOPE_COLOR,
        alpha=0.7,
        edgecolor="white",
    )

    # Mean vector
    mean_angle = np.angle(np.mean(np.exp(1j * angles)))
    mean_length = np.abs(np.mean(np.exp(1j * angles)))
    ax.arrow(
        mean_angle,
        0,
        0,
        mean_length * max(hist),
        head_width=0.1,
        head_length=max(hist) * 0.1,
        color=SPIKE_COLOR,
        linewidth=2,
    )

    ax.set_title(title)

    return ax
