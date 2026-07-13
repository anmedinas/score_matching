"""Entrena una red de difusion (Modelo, ver model.py).

--label-dropout controla cual de las dos redes se obtiene:
  0.0   -> red condicional pura   -> checkpoints/modelo_cond.pt
  > 0.0 -> red con CFG (dropout)  -> checkpoints/modelo_cfg.pt

Guarda ademas la curva de perdida (.png) en figures/loss/.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from model import Modelo

DATA_DIR = "data"

# Camino de probabilidad Gaussiano CondOT: p_t(.|z) = N(alpha_t z, beta_t^2 I)
# con alpha_t = t, beta_t = 1 - t (cumple alpha_0=beta_1=0, alpha_1=beta_0=1;
# ver item (a) del informe). x_t = alpha_t*z + beta_t*eps, eps ~ N(0,I).


def alpha(t: torch.Tensor) -> torch.Tensor:
    return t


def beta(t: torch.Tensor) -> torch.Tensor:
    return 1 - t


def train(model, data, *, n_epochs, batch_size, lr, device, seed,
          label_dropout=0.0, **kwargs):
    """Entrena 'model' con mini-batches. Retorna un dict con el historial.

    Perdida DSM con parametrizacion eps-prediction (item (b) del informe):
    L(theta) = E‖eps_theta(x_t, t, y) - eps‖^2, con x_t = alpha_t*z + beta_t*eps
    (Corolario 4.5 de score_matching_teoria.pdf / Algoritmo 4 del apunte MIT).
    Con label_dropout > 0, cada etiqueta se reemplaza por el token nulo con esa
    probabilidad de forma independiente por muestra (Algoritmo 5 del apunte
    MIT, entrenamiento CFG).
    """
    torch.manual_seed(seed)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "epoch_loss": []}
    for epoch in range(n_epochs):
        epoch_losses = []
        for z, y in loader:
            z, y = z.to(device), y.to(device)

            if label_dropout > 0.0:
                drop_mask = torch.rand(y.shape, device=device) < label_dropout
                y = y.masked_fill(drop_mask, model.null_label)

            t = torch.rand(z.size(0), device=device).clamp(min=1e-3)
            eps = torch.randn_like(z)
            x_t = alpha(t).view(-1, 1, 1, 1) * z + beta(t).view(-1, 1, 1, 1) * eps

            eps_pred = model(x_t, t, y)
            loss = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history["loss"].append(loss.item())
            epoch_losses.append(loss.item())

        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        history["epoch_loss"].append(mean_epoch_loss)
        print(f"epoch {epoch + 1}/{n_epochs}  loss {mean_epoch_loss:.4f}")

    return history


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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1),  # [0,1] -> [-1,1]
    ])
    dataset = MNIST(root=DATA_DIR, train=True, download=False, transform=transform)

    model = Modelo()
    history = train(
        model, dataset,
        n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed, label_dropout=args.label_dropout,
    )

    tag = "modelo_cond" if args.label_dropout == 0.0 else "modelo_cfg"

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, f"{tag}.pt")
    torch.save({"model_config": model.config, "model_state": model.state_dict()}, ckpt_path)
    print(f"checkpoint guardado en {ckpt_path}")

    loss_dir = "figures/loss"
    os.makedirs(loss_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(history["loss"])
    plt.xlabel("paso")
    plt.ylabel("loss (MSE ruido)")
    plt.title(f"Curva de perdida — {tag}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(loss_dir, f"{tag}_loss.png")
    plt.savefig(loss_path)
    print(f"curva de perdida guardada en {loss_path}")
