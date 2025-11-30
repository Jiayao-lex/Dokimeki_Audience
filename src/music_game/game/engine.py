from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..audio.analysis import extract_essentia_descriptors
from ..audio.input import (
    ChordPrediction,
    compute_chroma,
    compute_chroma_frames,
    derive_chroma_from_midi,
    estimate_chord,
    estimate_chords_over_time,
    estimate_key,
    load_audio_samples,
)
from ..emotion.model import EmotionClassifier, EmotionPrediction, FEATURE_KEYS
from ..llm.dialogue import DEFAULT_BASE_URL, DialogueTurn, OllamaClient
from .common import GameConfig, GameResult
from .unreal_client import UnrealClient


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
        
        self.unreal_client: Optional[UnrealClient] = None
        if config.unreal_enabled:
            self.unreal_client = UnrealClient(ip=config.unreal_ip, port=config.unreal_port)

    def process_audio_file(self, file_path: str | Path) -> GameResult:
        samples, sr = load_audio_samples(file_path, sample_rate=self.config.sample_rate)
        chroma = compute_chroma(samples, sr, hop_length=self.config.hop_length)
        chord = estimate_chord(chroma)
        key = estimate_key(chroma)
        
        # Compute chord sequence
        chroma_frames = compute_chroma_frames(samples, sr, hop_length=self.config.hop_length)
        chord_sequence = estimate_chords_over_time(chroma_frames, self.config.hop_length, sr)
        
        descriptors = extract_essentia_descriptors(samples, sr)
        emotion = self._infer_emotion(descriptors)
        
        # Check for consistency between Chord and Emotion
        # If the model predicts an emotion that conflicts with the chord quality (e.g. Minor -> Joyful),
        # we override it with the heuristic.
        should_use_heuristic = emotion is None
        if emotion and chord:
            is_positive_chord = chord.quality in ("major", "augmented")
            is_positive_emotion = emotion.label in ("joyful", "calm")
            if is_positive_chord != is_positive_emotion:
                should_use_heuristic = True

        # Fallback heuristic based on Russell's Circumplex Model (Valence/Arousal)
        if should_use_heuristic and chord:
            bpm = float(descriptors.get("bpm", 0))
            is_fast = bpm > 95
            is_positive_valence = chord.quality in ("major", "augmented")
            
            if is_positive_valence:
                fallback_label = "joyful" if is_fast else "calm"
            else:
                fallback_label = "tense" if is_fast else "melancholic"
                
            emotion = EmotionPrediction(label=fallback_label, confidence=0.6, probabilities={})
        
        dialogue = self._generate_dialogue(chord, key, emotion, descriptors)
        result = GameResult(chord=chord, key=key, emotion=emotion, descriptors=descriptors, dialogue=dialogue, chord_sequence=chord_sequence)
        
        if self.unreal_client:
            self.unreal_client.send_game_result(result)
            
        return result

    def process_midi_file(self, file_path: str | Path) -> GameResult:
        chroma = derive_chroma_from_midi(file_path)
        chord = estimate_chord(chroma) if chroma is not None else None
        key = estimate_key(chroma) if chroma is not None else None
        descriptors: Dict[str, float | str] = {key: 0.0 for key in FEATURE_KEYS}
        emotion = self._infer_emotion({key: float(descriptors[key]) for key in FEATURE_KEYS})

        # Check for consistency between Chord and Emotion
        should_use_heuristic = emotion is None
        if emotion and chord:
            is_positive_chord = chord.quality in ("major", "augmented")
            is_positive_emotion = emotion.label in ("joyful", "calm")
            if is_positive_chord != is_positive_emotion:
                should_use_heuristic = True

        # Fallback heuristic based on Russell's Circumplex Model (Valence/Arousal)
        if should_use_heuristic and chord:
            # MIDI descriptors are currently placeholders, so BPM is 0 (Slow)
            is_fast = False 
            is_positive_valence = chord.quality in ("major", "augmented")
            
            if is_positive_valence:
                fallback_label = "calm"
            else:
                fallback_label = "melancholic"
                
            emotion = EmotionPrediction(label=fallback_label, confidence=0.6, probabilities={})

        dialogue = self._generate_dialogue(chord, key, emotion, descriptors)
        result = GameResult(chord=chord, key=key, emotion=emotion, descriptors=descriptors, dialogue=dialogue)
        
        if self.unreal_client:
            self.unreal_client.send_game_result(result)
            
        return result

    def _infer_emotion(self, descriptors: Dict[str, float | str]) -> Optional[EmotionPrediction]:
        numeric_features = {key: float(descriptors.get(key, 0.0)) for key in FEATURE_KEYS}
        prediction = self.emotion_classifier.predict(numeric_features)
        if prediction.confidence < self.config.confidence_threshold:
            return None
        return prediction

    def _generate_dialogue(
        self,
        chord: Optional[ChordPrediction],
        key: Optional[ChordPrediction],
        emotion: Optional[EmotionPrediction],
        descriptors: Dict[str, float | str],
    ) -> Optional[DialogueTurn]:
        if not chord or not emotion:
            return None

        chord_label = chord.label
        key_label = key.label if key else None
        turn = self.ollama.generate(
            emotion_label=emotion.label,
            chord_label=chord_label,
            key_label=key_label,
            descriptors=descriptors,
            history=self.history,
        )
        self.history.append(turn)
        self.history = self.history[-self.config.history_limit :]
        return turn
