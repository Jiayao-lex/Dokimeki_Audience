from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..audio.analysis import extract_essentia_descriptors
from ..audio.input import (
    ChordPrediction,
    compute_chroma,
    derive_chroma_from_midi,
    estimate_chord,
    load_audio_samples,
)
from ..emotion.model import EmotionClassifier, EmotionPrediction, FEATURE_KEYS
from ..llm.dialogue import DEFAULT_BASE_URL, DialogueTurn, OllamaClient


@dataclass
class GameResult:
    chord: Optional[ChordPrediction]
    emotion: Optional[EmotionPrediction]
    descriptors: Dict[str, float | str]
    dialogue: Optional[DialogueTurn]


@dataclass
class GameConfig:
    sample_rate: int = 22050
    hop_length: int = 512
    confidence_threshold: float = 0.5
    emotion_labels: List[str] = field(default_factory=lambda: ["joyful", "melancholic", "tense", "calm"])
    ollama_model: str = "llama3"
    history_limit: int = 6

    @classmethod
    def from_file(cls, path: str | Path) -> "GameConfig":
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(
            sample_rate=int(raw.get("sample_rate", cls.sample_rate)),
            hop_length=int(raw.get("hop_length", cls.hop_length)),
            confidence_threshold=float(raw.get("confidence_threshold", cls.confidence_threshold)),
            emotion_labels=list(raw.get("emotion_labels", cls().emotion_labels)),
            ollama_model=str(raw.get("ollama", {}).get("model", cls.ollama_model)),
            history_limit=int(raw.get("history_limit", cls.history_limit)),
        )


class MusicEmotionGame:
    def __init__(
        self,
        config: GameConfig,
        emotion_model_path: str | Path | None = None,
        ollama_base_url: Optional[str] = None,
    ) -> None:
        self.config = config
        self.history: List[DialogueTurn] = []
        self.emotion_classifier = EmotionClassifier(
            labels=config.emotion_labels,
            model_path=emotion_model_path,
        )
        self.ollama = OllamaClient(base_url=ollama_base_url or DEFAULT_BASE_URL, model=config.ollama_model)

    def process_audio_file(self, file_path: str | Path) -> GameResult:
        samples, sr = load_audio_samples(file_path, sample_rate=self.config.sample_rate)
        chroma = compute_chroma(samples, sr, hop_length=self.config.hop_length)
        chord = estimate_chord(chroma)
        descriptors = extract_essentia_descriptors(samples, sr)
        emotion = self._infer_emotion(descriptors)
        dialogue = self._generate_dialogue(chord, emotion, descriptors)
        return GameResult(chord=chord, emotion=emotion, descriptors=descriptors, dialogue=dialogue)

    def process_midi_file(self, file_path: str | Path) -> GameResult:
        chroma = derive_chroma_from_midi(file_path)
        chord = estimate_chord(chroma) if chroma is not None else None
        descriptors: Dict[str, float | str] = {key: 0.0 for key in FEATURE_KEYS}
        emotion = self._infer_emotion({key: float(descriptors[key]) for key in FEATURE_KEYS}) if chord else None
        dialogue = self._generate_dialogue(chord, emotion, descriptors)
        return GameResult(chord=chord, emotion=emotion, descriptors=descriptors, dialogue=dialogue)

    def _infer_emotion(self, descriptors: Dict[str, float | str]) -> Optional[EmotionPrediction]:
        numeric_features = {key: float(descriptors.get(key, 0.0)) for key in FEATURE_KEYS}
        prediction = self.emotion_classifier.predict(numeric_features)
        if prediction.confidence < self.config.confidence_threshold:
            return None
        return prediction

    def _generate_dialogue(
        self,
        chord: Optional[ChordPrediction],
        emotion: Optional[EmotionPrediction],
        descriptors: Dict[str, float | str],
    ) -> Optional[DialogueTurn]:
        if not chord or not emotion:
            return None

        chord_label = chord.label
        turn = self.ollama.generate(
            emotion_label=emotion.label,
            chord_label=chord_label,
            descriptors=descriptors,
            history=self.history,
        )
        self.history.append(turn)
        self.history = self.history[-self.config.history_limit :]
        return turn
