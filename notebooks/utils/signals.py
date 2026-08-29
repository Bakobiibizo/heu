"""Signal utilities for HH-Envelope notebooks.

Provides convenience wrappers for running HEU simulations
and extracting gate dynamics for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationResult:
    """Complete results from an HEU simulation."""

    time_ms: np.ndarray
    envelope: np.ndarray
    spike_times: np.ndarray
    attack_gate: np.ndarray
    recovery_gate: np.ndarray
    input_signal: np.ndarray
    raw_values: np.ndarray = field(default_factory=lambda: np.array([]))


def run_heu_simulation(
    unit,
    signal: np.ndarray,
    dt_ms: float = 1.0,
    tonic: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run HEU simulation and return envelope + spike times.

    Simple wrapper for basic simulations.

    Args:
        unit: HHEnvelopeUnit instance
        signal: Excitation signal array
        dt_ms: Timestep in milliseconds
        tonic: Continuous tonic drive

    Returns:
        Tuple of (envelope_array, spike_times_array)
    """
    unit.reset()

    envelope = np.zeros(len(signal))
    spike_times = []
    current_time = 0.0

    for i, excitation in enumerate(signal):
        output = unit.step(excitation, dt_ms, tonic=tonic)
        envelope[i] = output.envelope
        if output.spike_event:
            spike_times.append(current_time)
        current_time += dt_ms

    return envelope, np.array(spike_times)


def run_heu_with_gates(
    unit,
    signal: np.ndarray,
    dt_ms: float = 1.0,
    tonic: float = 0.0,
) -> SimulationResult:
    """Run HEU simulation capturing all internal gate dynamics.

    Use this when you want to visualize the full internal state.

    Args:
        unit: HHEnvelopeUnit instance
        signal: Excitation signal array
        dt_ms: Timestep in milliseconds
        tonic: Continuous tonic drive

    Returns:
        SimulationResult with all gate values
    """
    unit.reset()

    n_samples = len(signal)
    time_ms = np.arange(n_samples) * dt_ms
    envelope = np.zeros(n_samples)
    attack_gate = np.zeros(n_samples)
    recovery_gate = np.zeros(n_samples)
    raw_values = np.zeros(n_samples)
    spike_times = []
    current_time = 0.0

    for i, excitation in enumerate(signal):
        output = unit.step(excitation, dt_ms, tonic=tonic)
        envelope[i] = output.envelope
        raw_values[i] = output.raw_value
        attack_gate[i] = output.state.attack_gate
        recovery_gate[i] = output.state.recovery_gate
        if output.spike_event:
            spike_times.append(current_time)
        current_time += dt_ms

    return SimulationResult(
        time_ms=time_ms,
        envelope=envelope,
        spike_times=np.array(spike_times),
        attack_gate=attack_gate,
        recovery_gate=recovery_gate,
        input_signal=signal.copy(),
        raw_values=raw_values,
    )
