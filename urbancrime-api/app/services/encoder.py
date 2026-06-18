"""
Wraps the trained autoencoder encoder half for online inference.

Architecture (recovered from encoder_weights.pt state dict):
    Linear(22 → 16) → BatchNorm1d(16) → ReLU → Linear(16 → 8)

We replicate the architecture in pure PyTorch so we only need the weights file,
not the original training notebook.  The scaler (StandardScaler) is also loaded
so raw feature vectors are normalized before encoding.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(22, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Identity(),  # index 3 placeholder so net.4 matches state dict
            nn.Linear(16, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderService:
    def __init__(self, models_dir: Path, scaler_path: Path) -> None:
        self._model = _Encoder()
        weights = torch.load(
            models_dir / "encoder_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        self._model.load_state_dict(weights)
        self._model.eval()

        self._scaler = joblib.load(scaler_path)

    def encode(self, features: list[float]) -> list[float]:
        """Normalize features and return 8-dim latent embedding."""
        x = np.array(features, dtype=np.float32).reshape(1, -1)
        x_scaled = self._scaler.transform(x)
        with torch.no_grad():
            z = self._model(torch.from_numpy(x_scaled))
        return z.squeeze().tolist()
