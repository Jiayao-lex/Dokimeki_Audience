# Piano Emotion Quest

An interactive Streamlit prototype where players perform piano chords, an AI classifier infers emotional tone, and an Ollama-hosted LLM narrates the experience in real time.

## Features
- **Audio & MIDI ingestion** with `librosa` and `mido` for chord recognition.
- **Essentia-powered descriptors** to capture tonal and rhythmic cues.
- **PyTorch emotion model** (placeholder network ready for custom weights).
- **Ollama integration** for local LLM dialogue generation.
- **Streamlit UI** delivering rapid feedback loops during prototyping.

## Getting Started

### 1. Prerequisites
- Docker Desktop (or Docker Engine) with Compose v2 support.
- Optional: GPU-enabled PyTorch build if you plan to supply your own accelerated model weights.

### 2. Configure Environment
```powershell
Copy-Item .env.example .env
# Edit .env to match your Ollama host/model and emotion model path
```

If you already have an Ollama instance running elsewhere, update `OLLAMA_HOST` accordingly. Otherwise the provided Compose file launches a local Ollama container.

### 3. Launch with Docker
```powershell
# Build images and start both services
docker compose up --build
```

Streamlit becomes available at `http://localhost:8501` once the image finishes installing dependencies. Ollama listens on port `11434` for the LLM requests.

### 4. Upload Chords
1. Visit the Streamlit UI.
2. Upload a short piano chord recording (`.wav`, `.mp3`, `.ogg`) or a MIDI file (`.mid`).
3. Observe detected chord, emotion prediction, and generated narration.

## Local Development (without Docker)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/main.py
```
Ensure Ollama is accessible (install via [https://ollama.ai](https://ollama.ai) and run `ollama serve`).

## Custom Emotion Model
1. Train or fine-tune a PyTorch classifier that ingests the feature vector defined in `FEATURE_KEYS` (`src/music_game/emotion/model.py`).
2. Export weights with `torch.save(model.state_dict(), "models/emotion_cnn.pt")`.
3. Point `EMOTION_MODEL_PATH` (env var) to the new weights before launch.

## Project Structure
```
app/                  Streamlit entry point
config/               YAML configuration presets
src/music_game/       Core audio, emotion, LLM, and game orchestration modules
models/               Placeholder for trained PyTorch weights
Dockerfile            Container recipe for the Streamlit app
docker-compose.yml    App + Ollama service definition
```

## Testing Ideas
- Supply curated MIDI chords representing major/minor emotional palettes.
- Extend `EmotionClassifier` with your own architecture and dataset.
- Plug in alternative Ollama models (e.g., `phi3`, `mistral`) via `.env`.

## Next Steps
- Integrate real-time audio capture for live play.
- Expand emotion ontology and retrain classifier with Essentia feature sets.
- Persist dialogue history to weave longer narrative arcs.
