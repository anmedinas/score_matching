"""Red de difusion: UNet condicional compartida por modelo_cond y modelo_cfg.

Importable como `from model import Modelo`. Las dos redes de difusion del
proyecto (condicional pura y con CFG) usan esta misma arquitectura; solo
difieren en como se entrenan (ver train.py, --label-dropout).

Convenciones de tensores: x tiene forma (B, C, H, W) (MNIST: C=1, H=W=28);
t tiene forma (B,) con valores en [0, 1]; y (etiqueta) es opcional, e incluye
el token nulo (None / indice reservado) para la red con CFG.
"""

import torch
import torch.nn as nn


class Modelo(nn.Module):
    def __init__(self, **config):
        super().__init__()
        # TODO: construir la UNet condicional (encoder-bottleneck-decoder
        # con skip connections), el embedding temporal sinusoidal y el
        # embedding de etiqueta (incluyendo token nulo para CFG).
        self.config = config

    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None) -> torch.Tensor:
        """Retorna s_theta / eps_theta. y=None usa el token nulo."""
        raise NotImplementedError

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Embedding temporal sinusoidal (auxiliar)."""
        raise NotImplementedError
