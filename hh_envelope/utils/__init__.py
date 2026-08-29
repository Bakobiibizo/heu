"""
Utils module: Signal generation, metrics, and visualization.

Priority: P0 (Validation)

Components:
    signals: Pulse train generators and target envelope shapes
    metrics: PLV, ISI stats, peak consistency, cross-correlation
    plotting: Envelope, pulse response, multi-voice, and learning curve plots
"""

from .metrics import (
    cross_correlation,
    envelope_peak_consistency,
    inter_spike_interval_stats,
    phase_locking_value,
    spike_phase_coherence,
)
from .plotting import (
    plot_envelope_fit,
    plot_envelope_with_spikes,
    plot_learning_curve,
    plot_pulse_response,
    plot_two_voice_comparison,
)
from .signals import (
    accented_pulse_train,
    jittered_pulse_train,
    pulse_train,
    target_envelope_accent,
    target_envelope_grace_note,
    target_envelope_legato,
    target_envelope_staccato,
)

__all__ = [
    "pulse_train",
    "accented_pulse_train",
    "jittered_pulse_train",
    "target_envelope_staccato",
    "target_envelope_legato",
    "target_envelope_grace_note",
    "target_envelope_accent",
    "phase_locking_value",
    "inter_spike_interval_stats",
    "envelope_peak_consistency",
    "cross_correlation",
    "spike_phase_coherence",
    "plot_envelope_with_spikes",
    "plot_pulse_response",
    "plot_two_voice_comparison",
    "plot_learning_curve",
    "plot_envelope_fit",
]
