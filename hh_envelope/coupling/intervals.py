"""Interval-based coupling for inter-voice dynamics.

Provides consonance-biased coupling weights based on musical intervals
and legacy coupling computation functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hh_envelope.polyphony.voice import Voice


# Consonance bias table: pitch class interval -> coupling weight
# Positive = encourage co-activation (consonant)
# Negative = mild inhibition (dissonant)
# Zero = neutral
INTERVAL_WEIGHTS: dict[int, float] = {
    0: 1.0,  # Unison - strongest coupling
    1: -0.3,  # Minor second - dissonant
    2: 0.1,  # Major second - mild consonance
    3: 0.4,  # Minor third - consonant
    4: 0.5,  # Major third - consonant
    5: 0.7,  # Perfect fourth - consonant
    6: -0.2,  # Tritone - dissonant
    7: 0.8,  # Perfect fifth - very consonant
    8: 0.4,  # Minor sixth - consonant
    9: 0.5,  # Major sixth - consonant
    10: 0.1,  # Minor seventh - mild
    11: -0.3,  # Major seventh - dissonant
}


def get_interval_weight(pitch_a: int | None, pitch_b: int | None) -> float:
    """Get coupling weight for an interval between two pitches.

    Args:
        pitch_a: First pitch (MIDI note number)
        pitch_b: Second pitch (MIDI note number)

    Returns:
        Coupling weight (-0.3 to 1.0), 0.0 if either pitch is None
    """
    if pitch_a is None or pitch_b is None:
        return 0.0

    # Compute interval as pitch class difference (0-11)
    interval = abs(pitch_a - pitch_b) % 12
    return INTERVAL_WEIGHTS.get(interval, 0.0)


def compute_coupling_input(
    target_voice: "Voice",
    other_voices: list["Voice"],
    coupling_strength: float = 0.1,
) -> float:
    """Compute coupling input to a target voice from other voices.

    Args:
        target_voice: The voice receiving coupling input
        other_voices: List of voices to couple from
        coupling_strength: Overall coupling strength (0.05-0.2 recommended)

    Returns:
        Total coupling input (sum of weighted envelopes)
    """
    total = 0.0

    for other in other_voices:
        # Skip self-coupling
        if other.voice_id == target_voice.voice_id:
            continue

        # Get interval-based weight
        weight = get_interval_weight(target_voice.pitch_hint, other.pitch_hint)

        # Add weighted envelope contribution
        total += weight * other.last_envelope * coupling_strength

    return total


class CouplingManager:
    """Manages coupling between multiple voices.

    Efficiently computes coupling inputs for all voices
    in a polyphonic system.
    """

    def __init__(
        self,
        voices: list["Voice"],
        coupling_strength: float = 0.1,
    ):
        """Initialize the coupling manager.

        Args:
            voices: List of voices to manage
            coupling_strength: Overall coupling strength
        """
        self._voices = voices
        self._coupling_strength = coupling_strength

    @property
    def coupling_strength(self) -> float:
        """Current coupling strength."""
        return self._coupling_strength

    @coupling_strength.setter
    def coupling_strength(self, value: float) -> None:
        self._coupling_strength = value

    def compute_all_coupling(self) -> dict[int, float]:
        """Compute coupling inputs for all voices.

        Returns:
            Dictionary mapping voice_id to coupling input
        """
        result = {}
        for target in self._voices:
            coupling = compute_coupling_input(target, self._voices, self._coupling_strength)
            result[target.voice_id] = coupling
        return result

    def compute_coupling_matrix(self) -> dict[tuple[int, int], float]:
        """Compute full coupling weight matrix between all voice pairs.

        Returns:
            Dictionary mapping (source_id, target_id) to coupling weight
        """
        matrix = {}
        for source in self._voices:
            for target in self._voices:
                if source.voice_id != target.voice_id:
                    weight = get_interval_weight(source.pitch_hint, target.pitch_hint)
                    matrix[(source.voice_id, target.voice_id)] = weight
        return matrix
