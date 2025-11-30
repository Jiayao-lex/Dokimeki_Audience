from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import httpx

DEFAULT_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


@dataclass
class DialogueTurn:
    role: str
    content: str


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        emotion_label: str,
        chord_label: str,
        key_label: Optional[str] = None,
        descriptors: Optional[dict[str, float | str]] = None,
        history: Optional[Iterable[DialogueTurn]] = None,
    ) -> DialogueTurn:
        payload = {
            "model": self.model,
            "prompt": _build_prompt(emotion_label, chord_label, key_label, descriptors, history),
            "stream": False,
        }

        with httpx.Client(timeout=self.timeout) as client:
            try:
                response = client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return DialogueTurn(
                        role="assistant",
                        content=f"Error: Model '{self.model}' not found. Please ensure it is pulled in Ollama (e.g., `ollama pull {self.model}`).",
                    )
                raise e
            except httpx.RequestError:
                return DialogueTurn(
                    role="assistant",
                    content="Error: Could not connect to Ollama. Is the service running?",
                )

        content = data.get("response") or data.get("text") or "I need more data to respond."
        return DialogueTurn(role="assistant", content=content.strip())


def _build_prompt(
    emotion_label: str,
    chord_label: str,
    key_label: Optional[str],
    descriptors: Optional[dict[str, float | str]],
    history: Optional[Iterable[DialogueTurn]],
) -> str:
    dialogue_context: List[str] = []
    if history:
        for turn in history:
            dialogue_context.append(f"{turn.role.title()}: {turn.content}")

    descriptor_lines: List[str] = []
    if descriptors:
        for key, value in descriptors.items():
            descriptor_lines.append(f"- {key}: {value}")

    descriptor_block = "\n".join(descriptor_lines) if descriptor_lines else "- No detailed descriptors"
    context_block = "\n".join(dialogue_context)
    key_info = f"Detected key: {key_label}.\n" if key_label else ""

    return (
        "You are the in-game narrator for an interactive piano-driven adventure.\n"
        "Craft a single short line reacting to the player's latest chord.\n"
        f"Detected chord: {chord_label}.\n"
        f"{key_info}"
        f"Predicted emotion: {emotion_label}.\n"
        "Audio descriptors:\n"
        f"{descriptor_block}\n"
        "Previous dialogue (most recent last):\n"
        f"{context_block}\n"
        "Respond with empathetic, emotionally supportive words, as if from a deeply attentive partner. (max 2 sentences)."
    )
