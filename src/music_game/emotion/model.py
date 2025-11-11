from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_KEYS: List[str] = [
    "dynamic_complexity",
    "loudness",
    "spectral_centroid",
    "spectral_complexity",
    "bpm",
    "prediction_error",
]


class EmotionNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.output = nn.Linear(16, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.hidden(features)
        return self.output(x)


@dataclass
class EmotionPrediction:
    label: str
    confidence: float
    probabilities: Dict[str, float]


class EmotionClassifier:
    def __init__(
        self,
        labels: Sequence[str],
        model_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        if not labels:
            raise ValueError("At least one emotion label is required")

        self.labels = list(labels)
        self.device = torch.device(device)
        self.model = EmotionNet(len(FEATURE_KEYS), len(self.labels)).to(self.device)

        if model_path:
            weight_path = Path(model_path)
            if weight_path.exists():
                state_dict = torch.load(weight_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"Model weights not found at {weight_path}")

        self.model.eval()

    def _vectorize(self, features: Dict[str, float]) -> torch.Tensor:
        vector = [float(features.get(key, 0.0)) for key in FEATURE_KEYS]
        return torch.tensor(vector, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict(self, features: Dict[str, float]) -> EmotionPrediction:
        logits = self.model(self._vectorize(features))
        probabilities = F.softmax(logits, dim=0)
        best_idx = int(torch.argmax(probabilities))
        probs_dict = {
            label: float(probabilities[idx].detach().cpu())
            for idx, label in enumerate(self.labels)
        }
        return EmotionPrediction(
            label=self.labels[best_idx],
            confidence=float(probabilities[best_idx].detach().cpu()),
            probabilities=probs_dict,
        )
