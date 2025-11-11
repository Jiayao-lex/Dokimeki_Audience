from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import librosa
import mido
import numpy as np

NOTE_NAMES: Tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJOR_TEMPLATE = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_MINOR_TEMPLATE = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


@dataclass
class ChordPrediction:
    root: str
    quality: str
    confidence: float

    @property
    def label(self) -> str:
        return f"{self.root}{'' if self.quality == 'major' else 'm'}"


def load_audio_samples(file_path: str | Path, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
    """Load mono audio samples resampled to the requested rate."""
    samples, sr = librosa.load(Path(file_path).expanduser().resolve().as_posix(), sr=sample_rate, mono=True)
    return samples, sr


def compute_chroma(samples: np.ndarray, sample_rate: int, hop_length: int = 512) -> np.ndarray:
    """Derive a time-averaged chroma vector from raw audio samples."""
    chroma = librosa.feature.chroma_cqt(y=samples, sr=sample_rate, hop_length=hop_length)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_norm = chroma_mean / np.clip(np.linalg.norm(chroma_mean), a_min=1e-6, a_max=None)
    return chroma_norm


def derive_chroma_from_midi(file_path: str | Path) -> Optional[np.ndarray]:
    """Aggregate MIDI note activity into a normalized pitch-class profile."""
    midi_path = Path(file_path).expanduser().resolve()
    if not midi_path.exists():
        return None

    midi = mido.MidiFile(midi_path)
    profile = np.zeros(12, dtype=float)
    active: dict[int, float] = {}
    current_time = 0.0

    for msg in midi:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note % 12] = current_time
        elif msg.type in {"note_off", "control_change"} or (msg.type == "note_on" and msg.velocity == 0):
            pitch_class = msg.note % 12 if hasattr(msg, "note") else None
            if pitch_class is not None and pitch_class in active:
                start_time = active.pop(pitch_class)
                profile[pitch_class] += max(current_time - start_time, 1e-3)

    for pitch_class, start_time in list(active.items()):
        profile[pitch_class] += max(current_time - start_time, 1e-3)

    if np.allclose(profile, 0.0):
        return None

    profile_norm = profile / np.clip(np.linalg.norm(profile), a_min=1e-6, a_max=None)
    return profile_norm


def _score_templates(chroma_vector: np.ndarray, template: np.ndarray) -> np.ndarray:
    scores = np.empty(12, dtype=float)
    for shift in range(12):
        shifted_template = np.roll(template, shift)
        scores[shift] = float(np.dot(chroma_vector, shifted_template))
    return scores


def estimate_chord(chroma_vector: np.ndarray) -> Optional[ChordPrediction]:
    """Estimate a simple major/minor chord from a chroma vector."""
    if chroma_vector.ndim != 1 or chroma_vector.shape[0] != 12:
        return None

    major_scores = _score_templates(chroma_vector, _MAJOR_TEMPLATE)
    minor_scores = _score_templates(chroma_vector, _MINOR_TEMPLATE)

    best_major_root = int(np.argmax(major_scores))
    best_minor_root = int(np.argmax(minor_scores))

    best_major = major_scores[best_major_root]
    best_minor = minor_scores[best_minor_root]

    if best_major <= 0.0 and best_minor <= 0.0:
        return None

    if best_major >= best_minor:
        return ChordPrediction(root=NOTE_NAMES[best_major_root], quality="major", confidence=float(best_major))
    return ChordPrediction(root=NOTE_NAMES[best_minor_root], quality="minor", confidence=float(best_minor))


def notes_to_pitch_classes(notes: Iterable[int]) -> np.ndarray:
    """Derive an instantaneous chroma vector from an arbitrary note collection."""
    chroma = np.zeros(12, dtype=float)
    for note in notes:
        chroma[int(note) % 12] += 1.0
    if np.allclose(chroma, 0.0):
        return chroma
    return chroma / np.linalg.norm(chroma)
