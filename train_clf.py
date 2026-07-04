"""Entrena el clasificador auxiliar (Clasificador, ver model_clf.py) sobre
los datos reales (independiente de las redes de difusion).

Guarda checkpoints/clasificador.pt. Misma estructura de train / main que
train.py.
"""

import argparse

from model_clf import Clasificador


def train(model, data, *, n_epochs, batch_size, lr, device, seed, **kwargs):
    """Entrena 'model' con mini-batches. Retorna un dict con el historial."""
    raise NotImplementedError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena el clasificador auxiliar.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, dest="n_epochs", default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="checkpoints")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    # TODO: cargar MNIST (data/), instanciar Clasificador, entrenar y
    # guardar clasificador.pt.
    raise NotImplementedError
