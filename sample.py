"""Genera imagenes con una red de difusion dada, integrando la SDE reversa
(Euler-Maruyama) con classifier-free guidance.
"""

import argparse
import os

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from model import Modelo

IMG_SIZE = 28
# Cota inferior de t: alpha_t=t aparece en el denominador de u_theta=(x-eps)/alpha_t
# (item (c) del informe). Cerca de alpha_t=0 el error residual de un eps_theta
# imperfecto se amplifica por 1/alpha_t y la integracion diverge (verificado
# empiricamente: con T_MIN=1e-3 la trayectoria explota a rangos de cientos de
# unidades incluso en ODE puro sin ruido; con T_MIN=0.05 se mantiene acotada
# y cercana a la escala de los datos, std~1). Es el mismo tipo de problema que
# motiva usar T_MIN>0 en cualquier sampler de difusion/flow matching.
T_MIN = 0.05


def sample(model, condition, *, n_steps, w, device, sigma_coef=0.5, seed=None, **kwargs):
    """
    'condition' contiene la(s) etiqueta(s) y y/o el ruido inicial X0 ~ N(0,I):
    un dict con clave "y" (LongTensor (B,), opcional) y/o "x0" (Tensor
    (B,C,H,W), opcional; si falta se muestrea de N(0,I)).

    Integra la SDE reversa (Euler-Maruyama, Algoritmo 2 del apunte MIT p. 12)
    aplicando CFG con escala 'w' (w=1 recupera el muestreo condicional puro,
    Summary 28 p. 42) y retorna las imagenes generadas, forma (B, C, H, W).

    Camino CondOT (alpha_t=t, beta_t=1-t, ver train.py): dado eps_theta, el
    campo de velocidad y el score se recuperan como u_theta=(x_t-eps_theta)/t
    y s_theta=-eps_theta/beta_t; la SDE simulada es
    dX_t = [u_theta(X_t) + (sigma_t^2/2) s_theta(X_t)] dt + sigma_t dW_t,
    con sigma_t = sigma_coef * beta_t (se anula en t=1, dato limpio).
    """
    model.to(device)
    model.eval()

    y = condition.get("y")
    x0 = condition.get("x0")

    if y is not None:
        y = y.to(device)
    batch_size = y.size(0) if y is not None else x0.size(0)

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    if x0 is None:
        shape = (batch_size, model.config["in_channels"], IMG_SIZE, IMG_SIZE)
        x0 = torch.randn(shape, device=device, generator=generator)
    else:
        x0 = x0.to(device)

    h = 1.0 / n_steps
    x, t_val = x0, 0.0

    with torch.no_grad():
        for _ in range(n_steps):
            t_eval = max(t_val, T_MIN)
            t_batch = torch.full((batch_size,), t_eval, device=device)

            if w == 1.0:
                eps_tilde = model(x, t_batch, y)
            else:
                eps_uncond = model(x, t_batch, None)
                eps_cond = model(x, t_batch, y)
                eps_tilde = (1 - w) * eps_uncond + w * eps_cond

            alpha_t = t_eval
            beta_t = max(1.0 - t_eval, T_MIN)
            sigma_t = sigma_coef * beta_t

            u_tilde = (x - eps_tilde) / alpha_t
            s_tilde = -eps_tilde / beta_t
            drift = u_tilde + 0.5 * sigma_t**2 * s_tilde

            noise = torch.randn(x.shape, device=device, generator=generator)
            x = x + h * drift + (h**0.5) * sigma_t * noise

            t_val += h

    return x


def save_labeled_grid(images: torch.Tensor, labels: list[int], path: str, nrow: int) -> None:
    """Guarda una grilla .png con una fila por clase, etiquetada a la izquierda."""
    grid = make_grid(images.clamp(-1, 1).add(1).div(2), nrow=nrow, padding=2)
    grid_img = to_pil_image(grid)

    margin = 24
    canvas = Image.new("RGB", (grid_img.width + margin, grid_img.height), "black")
    canvas.paste(grid_img, (margin, 0))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    n_rows = len(labels)
    row_height = grid_img.height / n_rows
    for i, label in enumerate(labels):
        y_pos = int(i * row_height + row_height / 2 - 5)
        draw.text((4, y_pos), str(label), fill="white")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


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
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Modelo(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])

    n_classes = model.num_classes
    n_per_class = max(1, args.n_samples // n_classes)
    y = torch.arange(n_classes).repeat_interleave(n_per_class)

    images = sample(
        model, {"y": y}, n_steps=args.n_steps, w=args.w, device=args.device, seed=args.seed
    )

    tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_path = os.path.join(args.out, f"{tag}_w{args.w:g}.png")
    save_labeled_grid(images, list(range(n_classes)), out_path, nrow=n_per_class)
    print(f"muestras guardadas en {out_path}")
