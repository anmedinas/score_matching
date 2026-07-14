"""Entrena el clasificador auxiliar (Clasificador, ver model_clf.py) sobre
los datos reales (independiente de las redes de difusion).

Guarda checkpoints/clasificador.pt. Misma estructura de train / main que
train.py.
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

from model_clf import Clasificador

DATA_DIR = "data"


def train(model, data, *, n_epochs, batch_size, lr, device, seed, **kwargs):
    """Entrena 'model' con mini-batches (cross-entropy). Retorna un dict con
    el historial."""
    torch.manual_seed(seed)
    # Ver train.py: fuerza algoritmos deterministas de cuDNN para que el
    # entrenamiento sea reproducible entre corridas en GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "epoch_loss": []}
    for epoch in range(n_epochs):
        epoch_losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1),  # [0,1] -> [-1,1]
    ])
    dataset = MNIST(root=DATA_DIR, train=True, download=False, transform=transform)

    model = Clasificador()
    history = train(
        model, dataset,
        n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed,
    )

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "clasificador.pt")
    torch.save({"model_config": model.config, "model_state": model.state_dict()}, ckpt_path)
    print(f"checkpoint guardado en {ckpt_path}")

    loss_dir = "figures/loss"
    os.makedirs(loss_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(history["loss"])
    plt.xlabel("paso")
    plt.ylabel("loss (cross-entropy)")
    plt.title("Curva de perdida — clasificador")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(loss_dir, "clasificador_loss.png")
    plt.savefig(loss_path)
    print(f"curva de perdida guardada en {loss_path}")
