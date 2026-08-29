"""Inter-voice coupling module.

Provides configurable coupling between voices with selectable
signal sources, injection sites, and safety clamps.

Components:
    CouplingConfig: Configuration for coupling behavior
    CouplingContribution: Effects to apply to a voice's HEU
    CouplingEngine: Computes coupling contributions between voices
    VoiceExport: Voice state snapshot for coupling computation

    # Interval-based coupling (legacy, maintained for compatibility)
    INTERVAL_WEIGHTS: Consonance bias table for pitch intervals
    get_interval_weight: Lookup interval coupling weight
    compute_coupling_input: Calculate coupling from other voices
    CouplingManager: Legacy coupling manager class
"""

from .engine import CouplingContribution, CouplingEngine, VoiceExport
from .intervals import (
    INTERVAL_WEIGHTS,
    CouplingManager,
    compute_coupling_input,
    get_interval_weight,
)
from .spec import CouplingConfig

__all__ = [
    "CouplingConfig",
    "CouplingContribution",
    "CouplingEngine",
    "CouplingManager",
    "VoiceExport",
    "INTERVAL_WEIGHTS",
    "get_interval_weight",
    "compute_coupling_input",
]
