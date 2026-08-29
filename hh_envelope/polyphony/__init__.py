"""
Polyphony module: Voice objects and inter-voice coupling.

Priority: P1

Components:
    voice: Voice class wrapping HEU with pitch and context
    coupling: Inter-voice coupling based on musical intervals (now in hh_envelope.coupling)
    articulation: Multi-voice articulation patterns (grace notes, trills, etc.)
    voice_allocator: Voice allocation (P3, optional for v0.1)
"""

from hh_envelope.coupling import (
    INTERVAL_WEIGHTS,
    CouplingManager,
    compute_coupling_input,
    get_interval_weight,
)

from .articulation import (
    ArticulatedNote,
    ArticulatedNoteOutput,
    ArticulationConfig,
    CombinationMode,
    TonicConfig,
    VoiceRole,
    create_grace_note_config,
    create_mordent_config,
    create_trill_config,
)
from .voice import SharedContext, Voice, VoiceOutput

__all__ = [
    # Voice
    "Voice",
    "SharedContext",
    "VoiceOutput",
    # Coupling
    "INTERVAL_WEIGHTS",
    "get_interval_weight",
    "compute_coupling_input",
    "CouplingManager",
    # Articulation (V0.3)
    "ArticulatedNote",
    "ArticulatedNoteOutput",
    "ArticulationConfig",
    "CombinationMode",
    "TonicConfig",
    "VoiceRole",
    "create_grace_note_config",
    "create_trill_config",
    "create_mordent_config",
]
