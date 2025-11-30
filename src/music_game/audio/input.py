from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import librosa
import mido
import numpy as np

NOTE_NAMES: Tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJOR_TEMPLATE = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_MINOR_TEMPLATE = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_AUGMENTED_TEMPLATE = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
_DIMINISHED_TEMPLATE = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@dataclass
class ChordPrediction:
    root: str
    quality: str
    confidence: float

    @property
    def label(self) -> str:
        if self.quality == "major":
            return self.root
        elif self.quality == "minor":
            return f"{self.root}m"
        elif self.quality == "augmented":
            return f"{self.root}aug"
        elif self.quality == "diminished":
            return f"{self.root}dim"
        return f"{self.root} {self.quality}"


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


def compute_chroma_frames(samples: np.ndarray, sample_rate: int, hop_length: int = 512) -> np.ndarray:
    """Derive chroma vectors for each frame from raw audio samples."""
    return librosa.feature.chroma_cqt(y=samples, sr=sample_rate, hop_length=hop_length)


def estimate_chords_over_time(chroma_frames: np.ndarray, hop_length: int, sample_rate: int, window_sec: float = 0.5) -> List[Tuple[float, str]]:
    """Estimate chords for each time segment."""
    # chroma_frames shape: (12, T)
    num_frames = chroma_frames.shape[1]
    chords = []
    
    frames_per_sec = sample_rate / hop_length
    window_size = int(frames_per_sec * window_sec)
    if window_size < 1:
        window_size = 1
        
    for i in range(0, num_frames, window_size):
        window = chroma_frames[:, i:i+window_size]
        # Average chroma over the window
        chroma_mean = np.mean(window, axis=1)
        chroma_norm = chroma_mean / np.clip(np.linalg.norm(chroma_mean), a_min=1e-6, a_max=None)
        
        chord = estimate_chord(chroma_norm)
        if chord:
            time_sec = i * hop_length / sample_rate
            chords.append((time_sec, chord.label))
            
    return chords


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
    """Estimate a simple major/minor/augmented/diminished chord from a chroma vector."""
    if chroma_vector.ndim != 1 or chroma_vector.shape[0] != 12:
        return None

    major_scores = _score_templates(chroma_vector, _MAJOR_TEMPLATE)
    minor_scores = _score_templates(chroma_vector, _MINOR_TEMPLATE)
    augmented_scores = _score_templates(chroma_vector, _AUGMENTED_TEMPLATE)
    diminished_scores = _score_templates(chroma_vector, _DIMINISHED_TEMPLATE)

    best_major_root = int(np.argmax(major_scores))
    best_minor_root = int(np.argmax(minor_scores))
    best_augmented_root = int(np.argmax(augmented_scores))
    best_diminished_root = int(np.argmax(diminished_scores))

    best_major = major_scores[best_major_root]
    best_minor = minor_scores[best_minor_root]
    best_augmented = augmented_scores[best_augmented_root]
    best_diminished = diminished_scores[best_diminished_root]

    if max(best_major, best_minor, best_augmented, best_diminished) <= 0.0:
        return None

    # Find the best match among all types
    candidates = [
        (best_major, best_major_root, "major"),
        (best_minor, best_minor_root, "minor"),
        (best_augmented, best_augmented_root, "augmented"),
        (best_diminished, best_diminished_root, "diminished"),
    ]
    
    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_score, best_root, best_quality = candidates[0]
    
    return ChordPrediction(root=NOTE_NAMES[best_root], quality=best_quality, confidence=float(best_score))


def estimate_key(chroma_vector: np.ndarray) -> Optional[ChordPrediction]:
    """Estimate the global key using Krumhansl-Kessler profiles."""
    if chroma_vector.ndim != 1 or chroma_vector.shape[0] != 12:
        return None

    # Krumhansl-Kessler Profiles (Standard for Key Detection)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    # Normalize profiles
    major_profile /= np.linalg.norm(major_profile)
    minor_profile /= np.linalg.norm(minor_profile)

    # Calculate correlation for all 12 shifts
    major_scores = _score_templates(chroma_vector, major_profile)
    minor_scores = _score_templates(chroma_vector, minor_profile)

    best_major_idx = int(np.argmax(major_scores))
    best_minor_idx = int(np.argmax(minor_scores))

    if major_scores[best_major_idx] >= minor_scores[best_minor_idx]:
        return ChordPrediction(root=NOTE_NAMES[best_major_idx], quality="major", confidence=float(major_scores[best_major_idx]))
    else:
        return ChordPrediction(root=NOTE_NAMES[best_minor_idx], quality="minor", confidence=float(minor_scores[best_minor_idx]))


def notes_to_pitch_classes(notes: Iterable[int]) -> np.ndarray:
    """Derive an instantaneous chroma vector from an arbitrary note collection."""
    chroma = np.zeros(12, dtype=float)
    for note in notes:
        chroma[int(note) % 12] += 1.0
    if np.allclose(chroma, 0.0):
        return chroma
    return chroma / np.linalg.norm(chroma)
