"""Carga las dos redes de difusion (modelo_cond.pt, modelo_cfg.pt) y el
clasificador (clasificador.pt), genera muestras por clase con cada una
(w=1 y el w != 1 elegido) y reporta fidelidad y diversidad.

Produce las tablas y graficos de la validacion experimental en
figures/evaluate/.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import Modelo
from model_clf import Clasificador
from sample import sample

N_CLASSES = 10


def _generate_and_classify(model, clasificador, *, w, n_per_class, n_steps, device, seed):
    """Genera n_per_class muestras por cada una de las N_CLASSES clases con
    `model` (CFG con escala `w`) y las clasifica con `clasificador`.
    Retorna dicts {clase: Tensor imagenes} y {clase: Tensor predicciones}."""
    images, preds = {}, {}
    with torch.no_grad():
        for cls in range(N_CLASSES):
            y = torch.full((n_per_class,), cls, dtype=torch.long)
            x = sample(model, {"y": y}, n_steps=n_steps, w=w, device=device, seed=seed + cls)
            logits = clasificador(x.to(device))
            images[cls] = x.cpu()
            preds[cls] = logits.argmax(dim=1).cpu()
    return images, preds


def _fidelity(preds: dict) -> dict:
    """Fraccion de muestras de cada clase que el clasificador asigna a esa
    misma clase (item (i) de la validacion experimental)."""
    return {cls: (p == cls).float().mean().item() for cls, p in preds.items()}


def _diversity(images: dict) -> dict:
    """Distancia media por pares (L2 en espacio de pixeles) entre muestras
    generadas de una misma clase (item (ii)). Se prefiere a la entropia de
    predicciones del clasificador porque mide variabilidad *visual* directa:
    varias muestras identicas del mismo digito dan entropia igual de baja que
    muestras variadas pero inequivocas, mientras que la distancia por pares
    si distingue ambos casos."""
    result = {}
    for cls, x in images.items():
        flat = x.flatten(1)
        n = flat.size(0)
        dists = torch.cdist(flat, flat, p=2)
        mask = ~torch.eye(n, dtype=torch.bool)
        result[cls] = dists[mask].mean().item()
    return result


def evaluate(modelo_cond, modelo_cfg, clasificador, *, w, n_samples, device,
             n_steps=1000, seed=0, **kwargs):
    """Genera muestras con ambas redes y calcula fidelidad condicional y
    diversidad usando el clasificador como juez automatico. Retorna un
    dict con las tablas/metricas calculadas.

    Compara exactamente las dos redes descritas en el enunciado: la red
    condicional pura (modelo_cond, siempre con w=1, sin token nulo) contra
    la red con CFG (modelo_cfg, con la escala `w` != 1 elegida).
    """
    modelo_cond.to(device).eval()
    modelo_cfg.to(device).eval()
    clasificador.to(device).eval()

    n_per_class = max(1, n_samples // N_CLASSES)

    runs = {
        "modelo_cond (w=1)": (modelo_cond, 1.0),
        f"modelo_cfg (w={w:g})": (modelo_cfg, w),
    }

    results = {}
    for name, (model, w_run) in runs.items():
        images, preds = _generate_and_classify(
            model, clasificador, w=w_run, n_per_class=n_per_class,
            n_steps=n_steps, device=device, seed=seed,
        )
        fidelity = _fidelity(preds)
        diversity = _diversity(images)
        results[name] = {
            "w": w_run,
            "fidelity_per_class": fidelity,
            "diversity_per_class": diversity,
            "fidelity_mean": sum(fidelity.values()) / N_CLASSES,
            "diversity_mean": sum(diversity.values()) / N_CLASSES,
        }

    return results


def _save_table(results: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["red", "w", "clase", "fidelidad", "diversidad"])
        for name, r in results.items():
            for cls in range(N_CLASSES):
                writer.writerow([
                    name, r["w"], cls,
                    r["fidelity_per_class"][cls], r["diversity_per_class"][cls],
                ])
            writer.writerow([name, r["w"], "promedio", r["fidelity_mean"], r["diversity_mean"]])


def _plot_bars(results: dict, key: str, ylabel: str, title: str, path: str) -> None:
    names = list(results.keys())
    width = 0.8 / len(names)
    plt.figure(figsize=(8, 4))
    for i, name in enumerate(names):
        values = [results[name][key][cls] for cls in range(N_CLASSES)]
        offsets = [cls + i * width for cls in range(N_CLASSES)]
        plt.bar(offsets, values, width=width, label=name)
    plt.xticks(
        [cls + width * (len(names) - 1) / 2 for cls in range(N_CLASSES)],
        range(N_CLASSES),
    )
    plt.xlabel("clase")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compara modelo_cond vs modelo_cfg.")
    parser.add_argument("--modelo-cond", type=str, default="checkpoints/modelo_cond.pt")
    parser.add_argument("--modelo-cfg", type=str, default="checkpoints/modelo_cfg.pt")
    parser.add_argument("--clasificador", type=str, default="checkpoints/clasificador.pt")
    parser.add_argument("--w", type=float, default=3.0, help="w != 1 elegido para CFG")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="figures/evaluate")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)

    ckpt_cond = torch.load(args.modelo_cond, map_location=args.device, weights_only=False)
    modelo_cond = Modelo(**ckpt_cond["model_config"])
    modelo_cond.load_state_dict(ckpt_cond["model_state"])

    ckpt_cfg = torch.load(args.modelo_cfg, map_location=args.device, weights_only=False)
    modelo_cfg = Modelo(**ckpt_cfg["model_config"])
    modelo_cfg.load_state_dict(ckpt_cfg["model_state"])

    ckpt_clf = torch.load(args.clasificador, map_location=args.device, weights_only=False)
    clasificador = Clasificador(**ckpt_clf["model_config"])
    clasificador.load_state_dict(ckpt_clf["model_state"])

    results = evaluate(
        modelo_cond, modelo_cfg, clasificador,
        w=args.w, n_samples=args.n_samples, n_steps=args.n_steps,
        device=args.device, seed=args.seed,
    )

    os.makedirs(args.out, exist_ok=True)

    table_path = os.path.join(args.out, "fidelidad_diversidad.csv")
    _save_table(results, table_path)
    print(f"tabla guardada en {table_path}")

    fidelity_path = os.path.join(args.out, "fidelidad_por_clase.png")
    _plot_bars(results, "fidelity_per_class", "fidelidad", "Fidelidad condicional por clase", fidelity_path)
    print(f"grafico guardado en {fidelity_path}")

    diversity_path = os.path.join(args.out, "diversidad_por_clase.png")
    _plot_bars(results, "diversity_per_class", "distancia media por pares",
               "Diversidad intra-clase", diversity_path)
    print(f"grafico guardado en {diversity_path}")

    for name, r in results.items():
        print(f"{name}: fidelidad promedio={r['fidelity_mean']:.3f}  "
              f"diversidad promedio={r['diversity_mean']:.3f}")
