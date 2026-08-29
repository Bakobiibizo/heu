"""Pydantic models for experiment diagnostics.

Provides stable, JSON-serializable types for signal stats, spike stats,
timing accuracy, operating point margin, quantization detection, and
coupling effects.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SignalStats(BaseModel):
    """Envelope signal statistics over an analysis window."""

    peak_value: float
    peak_time_ms: float
    auc: float  # area under curve over the analysis window
    mean_value: float
    std_value: float
    max_slope: float  # max derivative in units per ms


class SpikeStats(BaseModel):
    """Spike event statistics over an analysis window."""

    spike_count: int
    spike_times_ms: list[float] = Field(default_factory=list)
    isi_ms: list[float] = Field(default_factory=list)
    isi_mean_ms: float | None = None
    isi_cv: float | None = None  # coefficient of variation
    min_isi_ms: float | None = None


class TimingStats(BaseModel):
    """Timing accuracy: observed spike times vs expected event times."""

    expected_event_times_ms: list[float] = Field(default_factory=list)
    timing_errors_ms: list[float] = Field(default_factory=list)  # observed - expected
    median_abs_error_ms: float | None = None
    mad_abs_error_ms: float | None = None
    max_abs_error_ms: float | None = None
    drift_slope_ms_per_s: float | None = None  # linear drift of error vs time


class OperatingPointStats(BaseModel):
    """Operating point margin: how close peaks are to spike threshold."""

    spike_threshold: float | None = None
    peak_minus_threshold: float | None = None
    margin_ok: bool | None = None
    margin_target: float | None = None


class QuantizationStats(BaseModel):
    """Quantization artifact detection for spike times."""

    dt_ms: float
    unique_spike_time_steps: int | None = None
    quantization_suspected: bool | None = None  # too few unique time steps


class CouplingStats(BaseModel):
    """Coupling effect summary for multi-voice experiments."""

    coupling_strength: float | None = None
    coupling_mode: (
        Literal["excitation_add", "threshold_mod", "attack_gain_mod", "refractory_mod"] | None
    ) = None
    mean_coupling_effect: float | None = None  # experiment-defined scalar summary


class DiagnosticsReport(BaseModel):
    """Standard diagnostics block that every experiment emits."""

    analysis_window_ms: tuple[float, float]
    signal: SignalStats
    spikes: SpikeStats
    timing: TimingStats | None = None
    operating_point: OperatingPointStats | None = None
    quantization: QuantizationStats
    coupling: CouplingStats | None = None


class DiagnosticInvariant(BaseModel):
    """Invariant check that decides whether an experiment run is VALID."""

    id: str
    description: str
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class DiagnosticsBundle(BaseModel):
    """Complete diagnostics output: report + invariant checks."""

    report: DiagnosticsReport
    invariants: list[DiagnosticInvariant] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Run is valid only if all invariants pass."""
        return all(inv.passed for inv in self.invariants)
