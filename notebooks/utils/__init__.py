"""Notebook utilities for HH-Envelope Model demonstrations."""

from .audio import envelope_to_audio, sonify_pulse_train
from .plotting import (
    create_phase_polar_plot,
    plot_coupling_comparison,
    plot_envelope_comparison,
    plot_gate_dynamics,
    plot_optimization_trajectory,
    plot_parameter_space,
    plot_regional_mse,
    setup_notebook_style,
)
from .signals import run_heu_simulation, run_heu_with_gates

__all__ = [
    # Plotting
    "setup_notebook_style",
    "plot_gate_dynamics",
    "plot_envelope_comparison",
    "create_phase_polar_plot",
    "plot_coupling_comparison",
    "plot_optimization_trajectory",
    "plot_parameter_space",
    "plot_regional_mse",
    # Audio
    "envelope_to_audio",
    "sonify_pulse_train",
    # Signals
    "run_heu_simulation",
    "run_heu_with_gates",
]
