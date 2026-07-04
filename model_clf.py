"""Clasificador auxiliar (CNN) usado como juez automatico en evaluate.py.

Entrenado por separado sobre los datos reales (ver train_clf.py), de forma
independiente a las redes de difusion. Importable como
`from model_clf import Clasificador`.
"""

import torch
import torch.nn as nn


class Clasificador(nn.Module):
    def __init__(self, **config):
        super().__init__()
        # TODO: definir una CNN estandar para clasificacion sobre MNIST.
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna logits sobre las clases."""
        raise NotImplementedError
