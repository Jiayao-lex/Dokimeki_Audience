# Music Emotion Detector 🎹✨

An interactive AI-powered music analysis application that listens to your playing, understands the musical context (Chord, Key, Emotion), and generates a real-time narrative response using a local Large Language Model (LLM).

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=for-the-badge)

## 🌟 Features

### 🎵 Advanced Music Analysis
*   **Audio Input**: Record directly from your browser microphone or upload files (WAV, MP3, OGG, MIDI).
*   **Chord Detection**: Identifies **Major**, **Minor**, **Augmented**, and **Diminished** chords.
*   **Key Detection**: Estimates the global key of the piece using Krumhansl-Kessler profiles.
*   **Chord Progression**: Visualizes the timeline of chords detected throughout the recording.

### 🧠 Emotion Intelligence
*   **Hybrid Analysis**: Combines a PyTorch Neural Network with psychological heuristics (Russell's Circumplex Model).
*   **Consistency Checks**: Ensures the detected emotion aligns with the musical theory (e.g., Minor chords $\rightarrow$ Melancholic/Tense).

### 📖 AI Storyteller
*   **Local LLM**: Uses **Llama 3** (via Ollama) running locally to generate narrative responses.
*   **Context Aware**: The narrator reacts to the specific Chord, Key, and Emotion of your performance.

### 🎮 Integrations
*   **Unreal Engine Bridge**: Sends analysis results to a local Unreal Engine instance via UDP (Default: `127.0.0.1:7000`).

---

## 🚀 Getting Started

### Prerequisites
*   **Docker Desktop** (Recommended)
*   *OR* Python 3.9+ and [Ollama](https://ollama.com/) installed locally.

### Option 1: Run with Docker (Recommended)

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Music-Detector
    ```

2.  **Start the application**:
    ```powershell
    docker compose up --build
    ```
    *Note: The first run will automatically download the Llama 3 model (approx. 4.7GB). This may take a few minutes.*

3.  **Access the App**:
    Open your browser to **[http://localhost:8501](http://localhost:8501)**.

### Option 2: Local Development

1.  **Install Ollama**:
    Download from [ollama.com](https://ollama.com/) and run:
    ```bash
    ollama serve
    ollama pull llama3
    ```

2.  **Set up Python Environment**:
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```powershell
    streamlit run app/main.py
    ```

---

## ⚙️ Configuration

You can tweak the game settings in `config/app_settings.yaml`:

```yaml
sample_rate: 22050      # Audio processing quality
confidence_threshold: 0.2 # Minimum confidence for emotion model
emotion_labels:         # Defined emotion categories
  - joyful
  - melancholic
  - tense
  - calm
ollama:
  model: llama3         # LLM model to use
unreal:
  enabled: true         # Enable UDP messaging to Unreal
  ip: "127.0.0.1"
  port: 7000
```

## 📂 Project Structure

```
├── app/
│   └── main.py             # Streamlit Frontend & UI Logic
├── config/
│   └── app_settings.yaml   # Configuration file
├── src/
│   └── music_game/
│       ├── audio/          # Audio processing (Librosa/Essentia)
│       ├── emotion/        # PyTorch Emotion Model
│       ├── game/           # Game Engine & Logic
│       └── llm/            # Ollama Client & Prompt Engineering
├── docker-compose.yml      # Docker services definition
└── requirements.txt        # Python dependencies
```

## 🛠️ Troubleshooting

*   **"Model not found" error**: Ensure the `ollama-init` container has finished pulling the model, or run `docker compose exec ollama ollama pull llama3` manually.
*   **Microphone not working**: Check your browser permissions. Click the lock/camera icon in the address bar to allow microphone access.
*   **Build fails**: Try running `docker compose build --no-cache` to ensure fresh dependencies.
