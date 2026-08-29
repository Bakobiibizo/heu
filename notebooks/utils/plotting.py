"""Plotting utilities for HH-Envelope notebooks.

Provides consistent, publication-quality plots for demonstrating
HEU dynamics, coupling, and optimization results.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def setup_notebook_style():
    """Configure matplotlib for notebook-friendly display."""
    plt.rcParams.update(
        {
            "figure.figsize": (12, 5),
            "figure.dpi": 100,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "lines.linewidth": 1.8,
        }
    )


def plot_gate_dynamics(
    voltage: np.ndarray,
    attack_gate: np.ndarray,
    recovery_gate: np.ndarray,
    time_ms: np.ndarray | None = None,
    input_signal: np.ndarray | None = None,
    spike_times: np.ndarray | None = None,
    title: str = "HEU Gate Dynamics",
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Plot HEU internal state: voltage, gates, and optional input.

    Args:
        voltage: Envelope/voltage signal
        attack_gate: Fast gate values
        recovery_gate: Slow gate values
        time_ms: Time axis (auto-generated if None)
        input_signal: Optional input excitation
        spike_times: Optional spike event times
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_panels = 3 + (1 if input_signal is not None else 0)
    if time_ms is None:
        time_ms = np.arange(len(voltage), dtype=float)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    panel_idx = 0

    # Input signal (if provided)
    if input_signal is not None:
        axes[panel_idx].fill_between(time_ms, 0, input_signal, alpha=0.4, color="gray")
        axes[panel_idx].set_ylabel("Input")
        axes[panel_idx].set_ylim(bottom=0)
        panel_idx += 1

    # Voltage / envelope
    axes[panel_idx].plot(time_ms, voltage, color="#2196F3", label="Envelope")
    axes[panel_idx].axhline(0.8, color="red", linestyle="--", alpha=0.4, label="Spike threshold")
    if spike_times is not None and len(spike_times) > 0:
        for st in spike_times:
            axes[panel_idx].axvline(st, color="red", alpha=0.2, linewidth=0.8)
    axes[panel_idx].set_ylabel("Envelope")
    axes[panel_idx].legend(loc="upper right")
    axes[panel_idx].set_ylim([-0.05, 1.1])
    panel_idx += 1

    # Attack gate
    axes[panel_idx].plot(time_ms, attack_gate, color="#4CAF50", label="Attack gate")
    axes[panel_idx].set_ylabel("Attack Gate")
    axes[panel_idx].set_ylim([0, 1])
    axes[panel_idx].legend(loc="upper right")
    panel_idx += 1

    # Recovery gate
    axes[panel_idx].plot(time_ms, recovery_gate, color="#FF9800", label="Recovery gate")
    axes[panel_idx].set_ylabel("Recovery Gate")
    axes[panel_idx].set_ylim([0, 1])
    axes[panel_idx].legend(loc="upper right")

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout()
    return fig


def plot_envelope_comparison(
    target: np.ndarray,
    output: np.ndarray,
    time_ms: np.ndarray | None = None,
    title: str = "Envelope Comparison",
    labels: tuple[str, str] = ("Target", "Output"),
    show_error: bool = True,
    figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """Plot target vs output envelope with optional error panel.

    Args:
        target: Target envelope
        output: Output envelope
        time_ms: Time axis
        title: Plot title
        labels: Labels for target and output
        show_error: Show error subplot
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if time_ms is None:
        time_ms = np.arange(len(target), dtype=float)

    n_rows = 2 if show_error else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = [axes]

    # Envelopes
    axes[0].plot(time_ms, target, label=labels[0], linewidth=2.5, alpha=0.6, color="#1976D2")
    axes[0].plot(time_ms, output, label=labels[1], linewidth=2, alpha=0.8, color="#E64A19")
    axes[0].set_ylabel("Envelope")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(bottom=-0.02)

    # Error
    if show_error:
        error = target - output
        axes[1].fill_between(time_ms, 0, error, alpha=0.4, color="#E53935")
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_ylabel("Error")
        axes[1].set_xlabel("Time (ms)")

        mse = float(np.mean(error**2))
        axes[1].set_title(f"MSE = {mse:.6f}", fontsize=10, loc="right")
    else:
        axes[0].set_xlabel("Time (ms)")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def create_phase_polar_plot(
    spike_times: np.ndarray,
    pulse_period_ms: float,
    title: str = "Phase Locking",
    figsize: tuple[float, float] = (6, 6),
) -> Figure:
    """Create polar plot showing spike phase relative to pulse period.

    Points clustered at one angle = strong phase locking.

    Args:
        spike_times: Array of spike times in ms
        pulse_period_ms: Period of driving pulse in ms
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Compute phases
    phases = (spike_times % pulse_period_ms) / pulse_period_ms * 2 * np.pi

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"projection": "polar"})

    # Plot individual phases
    ax.scatter(phases, np.ones_like(phases), alpha=0.5, s=30, color="#2196F3")

    # Plot mean vector (PLV)
    mean_phase = np.angle(np.mean(np.exp(1j * phases)))
    plv = abs(np.mean(np.exp(1j * phases)))
    ax.annotate(
        "",
        xy=(mean_phase, plv),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
    )

    ax.set_ylim(0, 1.2)
    ax.set_title(f"{title}\nPLV = {plv:.3f}", pad=20, fontsize=13)
    ax.set_rticks([0.5, 1.0])
    return fig


def plot_coupling_comparison(
    time_ms: np.ndarray,
    envelopes: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "Voice Coupling Comparison",
    figsize: tuple[float, float] = (14, 8),
) -> Figure:
    """Plot multiple coupling scenarios side by side.

    Args:
        time_ms: Shared time axis
        envelopes: Dict mapping interval name to (voice_a_env, voice_b_env)
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_intervals = len(envelopes)
    fig, axes = plt.subplots(n_intervals, 1, figsize=figsize, sharex=True)
    if n_intervals == 1:
        axes = [axes]

    colors_a, colors_b = "#1976D2", "#E64A19"

    for idx, (interval_name, (env_a, env_b)) in enumerate(envelopes.items()):
        axes[idx].plot(time_ms, env_a, color=colors_a, label="Voice A", alpha=0.8)
        axes[idx].plot(time_ms, env_b, color=colors_b, label="Voice B", alpha=0.8)
        axes[idx].set_ylabel(interval_name)
        axes[idx].set_ylim([-0.02, 1.1])
        if idx == 0:
            axes[idx].legend(loc="upper right")

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_optimization_trajectory(
    loss_history: list[float] | np.ndarray,
    target_loss: float | None = None,
    title: str = "Optimization Progress",
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Plot optimization loss over iterations.

    Args:
        loss_history: Loss values per iteration
        target_loss: Optional target threshold line
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    iterations = np.arange(len(loss_history))
    ax.semilogy(iterations, loss_history, color="#1976D2", linewidth=2)

    if target_loss is not None:
        ax.axhline(
            target_loss, color="green", linestyle="--", alpha=0.7, label=f"Target: {target_loss}"
        )
        ax.legend()

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_parameter_space(
    theta_history: list[dict[str, float]],
    reference_theta: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    title: str = "Parameter Trajectory",
    figsize: tuple[float, float] = (12, 4),
) -> Figure:
    """Plot parameter values over optimization iterations.

    Args:
        theta_history: List of theta dicts per iteration
        reference_theta: Optional ground-truth theta
        bounds: Optional parameter bounds
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    param_names = list(theta_history[0].keys())
    n_params = len(param_names)

    fig, axes = plt.subplots(1, n_params, figsize=figsize)
    if n_params == 1:
        axes = [axes]

    colors = ["#1976D2", "#E64A19", "#4CAF50", "#9C27B0"]

    for idx, name in enumerate(param_names):
        values = [th[name] for th in theta_history]
        axes[idx].plot(values, color=colors[idx % len(colors)], linewidth=1.5)

        if reference_theta is not None and name in reference_theta:
            axes[idx].axhline(
                reference_theta[name], color="green", linestyle="--", alpha=0.7, label="Reference"
            )

        if bounds is not None and name in bounds:
            lo, hi = bounds[name]
            axes[idx].axhline(lo, color="red", linestyle=":", alpha=0.4)
            axes[idx].axhline(hi, color="red", linestyle=":", alpha=0.4)

        axes[idx].set_title(name, fontsize=10)
        axes[idx].set_xlabel("Iteration")
        if idx == 0:
            axes[idx].set_ylabel("Value")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_regional_mse(
    regions: dict[str, tuple[float, float]],
    single_mse: dict[str, float],
    two_voice_mse: dict[str, float],
    title: str = "Regional MSE Comparison",
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Bar chart comparing single-voice vs two-voice MSE by region.

    Args:
        regions: Dict mapping region name to (start_ms, end_ms)
        single_mse: MSE per region for single-voice
        two_voice_mse: MSE per region for two-voice
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    region_names = list(regions.keys())
    x = np.arange(len(region_names))
    width = 0.35

    single_vals = [single_mse[r] for r in region_names]
    two_vals = [two_voice_mse[r] for r in region_names]

    ax.bar(x - width / 2, single_vals, width, label="Single-voice", color="#E64A19", alpha=0.8)
    ax.bar(x + width / 2, two_vals, width, label="Two-voice", color="#1976D2", alpha=0.8)

    ax.set_ylabel("MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(region_names)
    ax.legend()
    ax.set_yscale("log")
    ax.set_title(title)

    # Add improvement annotations
    for i, (sv, tv) in enumerate(zip(single_vals, two_vals)):
        if tv > 0:
            improvement = sv / tv
            ax.annotate(
                f"{improvement:.0f}x",
                xy=(x[i] + width / 2, tv),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#1976D2",
                fontweight="bold",
            )

    fig.tight_layout()
    return fig
