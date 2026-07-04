"""Genera imagenes con una red de difusion dada, integrando la SDE reversa
(Euler-Maruyama) con classifier-free guidance.
"""

import argparse

from model import Modelo


def sample(model, condition, *, n_steps, w, device, **kwargs):
    """
    'condition' contiene la(s) etiqueta(s) y y/o el ruido inicial X0 ~ N(0,I).
    Integra la SDE reversa (Euler-Maruyama) aplicando CFG con escala 'w'
    (w=1 recupera el muestreo condicional puro) y retorna las imagenes
    generadas, forma (B, C, H, W).
    """
    raise NotImplementedError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera imagenes con una red de difusion.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="figures/samples")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    # TODO: cargar el checkpoint indicado, generar de forma reproducible
    # (seed fija) y guardar una grilla .png etiquetada por clase en --out.
    raise NotImplementedError
