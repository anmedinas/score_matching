"""Entrena una red de difusion (Modelo, ver model.py).

--label-dropout controla cual de las dos redes se obtiene:
  0.0   -> red condicional pura   -> checkpoints/modelo_cond.pt
  > 0.0 -> red con CFG (dropout)  -> checkpoints/modelo_cfg.pt

Guarda ademas la curva de perdida (.png) en figures/loss/.
"""

import argparse

from model import Modelo


def train(model, data, *, n_epochs, batch_size, lr, device, seed,
          label_dropout=0.0, **kwargs):
    """Entrena 'model' con mini-batches. Retorna un dict con el historial."""
    raise NotImplementedError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena una red de difusion.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, dest="n_epochs", default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-dropout", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="checkpoints")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    # TODO: cargar MNIST (data/), instanciar Modelo, entrenar y guardar
    # el checkpoint (modelo_cond.pt o modelo_cfg.pt segun --label-dropout)
    # junto con la curva de perdida (.png) en figures/loss/.
    raise NotImplementedError
