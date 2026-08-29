"""Audio utilities for HH-Envelope notebooks.

Converts envelope signals to audio for auditory demonstration
of the HEU dynamics.
"""

from __future__ import annotations

import numpy as np


def envelope_to_audio(
    envelope: np.ndarray,
    sample_rate: int = 44100,
    carrier_freq: float = 440.0,
    envelope_rate_hz: float = 1000.0,
) -> tuple[np.ndarray, int]:
    """Convert an envelope to audio by amplitude-modulating a carrier tone.

    Args:
        envelope: Envelope signal (at envelope_rate_hz)
        sample_rate: Output audio sample rate
        carrier_freq: Carrier tone frequency in Hz
        envelope_rate_hz: Rate of the input envelope (default 1kHz = 1ms steps)

    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    # Upsample envelope to audio rate
    envelope_duration_sec = len(envelope) / envelope_rate_hz
    n_audio_samples = int(envelope_duration_sec * sample_rate)

    envelope_upsampled = np.interp(
        np.linspace(0, len(envelope) - 1, n_audio_samples),
        np.arange(len(envelope)),
        envelope,
    )

    # Generate carrier
    t = np.arange(n_audio_samples) / sample_rate
    carrier = np.sin(2 * np.pi * carrier_freq * t)

    # Amplitude modulation
    audio = carrier * envelope_upsampled

    # Normalize to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * 0.8 / peak

    return audio, sample_rate


def sonify_pulse_train(
    envelope: np.ndarray,
    n_beats: int = 8,
    tempo_bpm: float = 120.0,
    carrier_freq: float = 440.0,
    sample_rate: int = 44100,
    envelope_rate_hz: float = 1000.0,
) -> tuple[np.ndarray, int]:
    """Create audio from a repeating envelope at specified tempo.

    Useful for hearing rhythmic articulation patterns.

    Args:
        envelope: Single-beat envelope (at envelope_rate_hz)
        n_beats: Number of beats to generate
        tempo_bpm: Tempo in BPM
        carrier_freq: Carrier tone frequency
        sample_rate: Output sample rate
        envelope_rate_hz: Rate of envelope

    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    beat_duration_sec = 60.0 / tempo_bpm
    beat_duration_samples = int(beat_duration_sec * envelope_rate_hz)

    # Repeat envelope for n_beats, padding each to beat duration
    full_envelope = np.zeros(beat_duration_samples * n_beats)
    for i in range(n_beats):
        start = i * beat_duration_samples
        end = start + min(len(envelope), beat_duration_samples)
        copy_len = end - start
        full_envelope[start : start + copy_len] = envelope[:copy_len]

    return envelope_to_audio(full_envelope, sample_rate, carrier_freq, envelope_rate_hz)


def display_audio(audio: np.ndarray, sample_rate: int = 44100):
    """Display audio in a Jupyter notebook using IPython.display.Audio.

    Args:
        audio: Audio samples
        sample_rate: Sample rate
    """
    from IPython.display import Audio
    from IPython.display import display as ipy_display

    ipy_display(Audio(audio, rate=sample_rate))
