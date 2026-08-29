"""Experiment diagnostics: standard metrics, invariants, and validity checks."""

from .compute import (
    build_diagnostics_bundle,
    check_coupling_changes_state,
    check_symmetry_not_collapsed,
    report_coupling_summary,
    report_diagnostics,
    spike_times_to_bool,
)
from .types import DiagnosticsBundle

__all__ = [
    "DiagnosticsBundle",
    "build_diagnostics_bundle",
    "check_coupling_changes_state",
    "check_symmetry_not_collapsed",
    "report_coupling_summary",
    "report_diagnostics",
    "spike_times_to_bool",
]
