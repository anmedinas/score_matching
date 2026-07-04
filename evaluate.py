"""Carga las dos redes de difusion (modelo_cond.pt, modelo_cfg.pt) y el
clasificador (clasificador.pt), genera muestras por clase con cada una
(w=1 y el w != 1 elegido) y reporta fidelidad y diversidad.

Produce las tablas y graficos de la validacion experimental en
figures/evaluate/.
"""

import argparse

from model import Modelo
from model_clf import Clasificador


def evaluate(modelo_cond, modelo_cfg, clasificador, *, w, n_samples, device, **kwargs):
    """Genera muestras con ambas redes y calcula fidelidad condicional y
    diversidad usando el clasificador como juez automatico. Retorna un
    dict con las tablas/metricas calculadas."""
    raise NotImplementedError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compara modelo_cond vs modelo_cfg.")
    parser.add_argument("--modelo-cond", type=str, default="checkpoints/modelo_cond.pt")
    parser.add_argument("--modelo-cfg", type=str, default="checkpoints/modelo_cfg.pt")
    parser.add_argument("--clasificador", type=str, default="checkpoints/clasificador.pt")
    parser.add_argument("--w", type=float, default=3.0, help="w != 1 elegido para CFG")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="figures/evaluate")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    # TODO: cargar los tres checkpoints, correr evaluate() y guardar
    # tablas/graficos de fidelidad y diversidad (w=1 vs w!=1) en --out.
    raise NotImplementedError
