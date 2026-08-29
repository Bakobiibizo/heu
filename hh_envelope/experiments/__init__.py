"""
Experiments module: Hypothesis-driven validation experiments.

These are designed to be quick and falsifiable. If any fail hard, we learn something real.

Experiments:
    exp_01_locking: Tempo locking and groove-like asymmetry (P0)
    exp_02_two_voice_coupling: Two-voice coupling stability (P1)
    exp_03_fit_envelope: Envelope-alignment learning (P2)

Acceptance Criteria for v0.1:
- Single unit locks and behaves musically under rhythmic drive (Exp 01 passes)
- Two units couple without chaos and show bias effects (Exp 02 passes)
- Learning can fit at least two envelope archetypes stably (Exp 03 passes)
"""

from . import exp_01_locking, exp_02_two_voice_coupling, exp_03_fit_envelope

__all__ = ["exp_01_locking", "exp_02_two_voice_coupling", "exp_03_fit_envelope"]
