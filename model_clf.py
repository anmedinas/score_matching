"""Clasificador auxiliar (CNN) usado como juez automatico en evaluate.py.

Entrenado por separado sobre los datos reales (ver train_clf.py), de forma
independiente a las redes de difusion. Importable como
`from model_clf import Clasificador`.
"""

import torch
import torch.nn as nn


class Clasificador(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        **config,
    ):
        super().__init__()
        self.config = dict(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            **config,
        )

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna logits sobre las clases."""
        return self.classifier(self.features(x))
