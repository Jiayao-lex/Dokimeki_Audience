from __future__ import annotations

from typing import Dict, Union

import numpy as np

try:
    from essentia.standard import (  # type: ignore
        DynamicComplexity,
        KeyExtractor,
        LPC,
        Loudness,
        RhythmExtractor2013,
        SpectralCentroidTime,
        SpectralComplexity,
    )

    ESSENTIA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency during local dev
    ESSENTIA_AVAILABLE = False


def extract_essentia_descriptors(samples: np.ndarray, sample_rate: int) -> Dict[str, Union[float, str]]:
    """Compute a compact feature dictionary using Essentia if available."""
    if not ESSENTIA_AVAILABLE:
        energy = float(np.mean(np.abs(samples)))
        centroid = float(np.mean(np.abs(np.fft.rfft(samples)))) if samples.size else 0.0
        return {
            "dynamic_complexity": energy,
            "loudness": energy,
            "spectral_centroid": centroid,
            "spectral_complexity": 0.0,
            "bpm": 0.0,
            "key": "C",
            "scale": "major",
        }

    loudness_extractor = Loudness()
    spectral_centroid = SpectralCentroidTime(sampleRate=sample_rate)
    spectral_complexity = SpectralComplexity(sampleRate=sample_rate)
    dynamic_complexity = DynamicComplexity(frameSize=2048)
    rhythm_extractor = RhythmExtractor2013(method="multifeature")
    key_extractor = KeyExtractor()
    lpc = LPC(order=12)

    loudness = float(loudness_extractor(samples))
    centroid = float(np.mean(spectral_centroid(samples)))
    complexity = float(np.mean(spectral_complexity(samples)))
    dyn_complexity, _ = dynamic_complexity(samples)
    bpm, _, _, _, _ = rhythm_extractor(samples)
    key, scale, _ = key_extractor(samples)
    _, prediction_error = lpc(samples)

    return {
        "dynamic_complexity": float(dyn_complexity),
        "loudness": loudness,
        "spectral_centroid": centroid,
        "spectral_complexity": complexity,
        "bpm": float(bpm),
        "key": key,
        "scale": scale,
        "prediction_error": float(np.mean(prediction_error)),
    }
