from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..audio.input import ChordPrediction
from ..emotion.model import EmotionPrediction
from ..llm.dialogue import DialogueTurn


@dataclass
class GameResult:
    chord: Optional[ChordPrediction]
    key: Optional[ChordPrediction]
    emotion: Optional[EmotionPrediction]
    descriptors: Dict[str, float | str]
    dialogue: Optional[DialogueTurn]
    chord_sequence: List[tuple[float, str]] = field(default_factory=list)


@dataclass
class GameConfig:
    sample_rate: int = 22050
    hop_length: int = 512
    confidence_threshold: float = 0.5
    emotion_labels: List[str] = field(default_factory=lambda: ["joyful", "melancholic", "tense", "calm"])
    ollama_model: str = "llama3"
    history_limit: int = 6
    
    # Unreal Engine Settings
    unreal_enabled: bool = False
    unreal_ip: str = "127.0.0.1"
    unreal_port: int = 7000

    @classmethod
    def from_file(cls, path: str | Path) -> "GameConfig":
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        
        unreal_config = raw.get("unreal", {})
        
        return cls(
            sample_rate=int(raw.get("sample_rate", cls.sample_rate)),
            hop_length=int(raw.get("hop_length", cls.hop_length)),
            confidence_threshold=float(raw.get("confidence_threshold", cls.confidence_threshold)),
            emotion_labels=list(raw.get("emotion_labels", cls().emotion_labels)),
            ollama_model=str(raw.get("ollama", {}).get("model", cls.ollama_model)),
            history_limit=int(raw.get("history_limit", cls.history_limit)),
            unreal_enabled=bool(unreal_config.get("enabled", False)),
            unreal_ip=str(unreal_config.get("ip", "127.0.0.1")),
            unreal_port=int(unreal_config.get("port", 7000)),
        )
