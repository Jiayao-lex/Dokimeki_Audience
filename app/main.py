from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from src.music_game.audio.input import ChordPrediction
from src.music_game.game.common import GameConfig, GameResult
from src.music_game.game.engine import MusicEmotionGame

load_dotenv()

APP_TITLE = "Music Emotion Detector"
SUPPORTED_AUDIO_TYPES = ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/ogg", "audio/webm"]
SUPPORTED_MIDI_TYPES = ["audio/midi", "audio/x-midi", "application/octet-stream"]


@st.cache_resource
def load_game() -> MusicEmotionGame:
    config_path = Path("config/app_settings.yaml")
    config = GameConfig.from_file(config_path) if config_path.exists() else GameConfig()
    model_path = os.getenv("EMOTION_MODEL_PATH")
    ollama_host = os.getenv("OLLAMA_HOST")
    return MusicEmotionGame(config=config, emotion_model_path=model_path, ollama_base_url=ollama_host)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Play a chord, feel the emotion, hear the story.")

    game = load_game()

    uploaded = st.file_uploader(
        "Upload a piano chord recording (WAV/MP3/OGG) or MIDI file",
        type=["wav", "mp3", "ogg", "midi", "mid"],
    )

    audio_input = st.audio_input(
        "Record a piano chord",
        help="To change the microphone, click the camera/lock icon in your browser's address bar.",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Latest Response")
        placeholder = st.empty()
    with col2:
        st.header("Chord & Emotion")
        chord_placeholder = st.empty()
        key_placeholder = st.empty()
        emotion_placeholder = st.empty()

    if uploaded is None and audio_input is None:
        placeholder.info("Upload an audio or MIDI file or record audio to generate dialogue.")
        return

    file_to_process = uploaded if uploaded is not None else audio_input

    if file_to_process:
        # Playback the input audio (skip MIDI)
        is_midi = hasattr(file_to_process, "name") and file_to_process.name.lower().endswith((".mid", ".midi"))
        if not is_midi:
            st.audio(file_to_process)

    result = _handle_upload(game, file_to_process)
    if result is None:
        placeholder.warning("Could not decode the submission. Try another file.")
        return

    _render_result(placeholder, chord_placeholder, key_placeholder, emotion_placeholder, result)


def _handle_upload(game: MusicEmotionGame, uploaded_file) -> Optional[GameResult]:  # type: ignore[no-untyped-def]
    # Ensure we start reading from the beginning of the file
    uploaded_file.seek(0)

    if uploaded_file.size == 0:
        st.error("The received audio file is empty. Please check your microphone settings.")
        return None

    suffix = Path(uploaded_file.name).suffix.lower()
    media_type = uploaded_file.type.lower() if uploaded_file.type else ""

    with NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(uploaded_file.read())
        temp_path = Path(temp.name)

    try:
        if media_type in SUPPORTED_MIDI_TYPES or suffix in {".midi", ".mid"}:
            return game.process_midi_file(temp_path)
        
        # Broader check for audio types
        if (
            media_type in SUPPORTED_AUDIO_TYPES 
            or suffix in {".wav", ".mp3", ".ogg", ".webm"} 
            or media_type.startswith("audio/")
        ):
            return game.process_audio_file(temp_path)
            
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

    return None


def _render_result(placeholder, chord_placeholder, key_placeholder, emotion_placeholder, result: GameResult) -> None:  # type: ignore[no-untyped-def]
    chord_label = result.chord.label if result.chord else "Unknown"
    key_label = result.key.label if result.key else "Unknown"
    emotion_label = result.emotion.label if result.emotion else "Undetermined"

    if result.dialogue:
        placeholder.success(result.dialogue.content)
    else:
        placeholder.info("No dialogue generated for this chord yet.")

    chord_placeholder.metric(label="Chord", value=chord_label)
    key_placeholder.metric(label="Key", value=key_label)

    if result.emotion:
        emotion_placeholder.metric(label="Emotion", value=emotion_label, delta=f"{result.emotion.confidence:.2f}")
        probs = result.emotion.probabilities
        chart_data = {label: value for label, value in probs.items()}
        st.bar_chart(chart_data)

    if result.chord_sequence:
        st.subheader("Chord Progression")
        # Format as a simple timeline string or table
        sequence_data = [{"Time": f"{t:.1f}s", "Chord": c} for t, c in result.chord_sequence]
        st.dataframe(sequence_data, use_container_width=True)
    else:
        emotion_placeholder.metric(label="Emotion", value=emotion_label)


if __name__ == "__main__":
    main()
