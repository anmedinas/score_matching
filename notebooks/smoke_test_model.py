"""Prueba de humo (NO forma parte de la interfaz de la tarea) para validar
model.py end-to-end antes de escribir train.py/sample.py formales:
  1. Carga MNIST real (requiere haber corrido download_data.py antes).
  2. Entrena `Modelo` unos cientos de pasos con DSM simple (path CondOT:
     alpha_t=t, beta_t=1-t; eps-prediction: loss = MSE(eps_theta(x_t,t,y), eps)).
  3. Verifica que la perdida baja.
  4. Genera unas pocas muestras integrando el flujo ODE (Euler) condicionado
     por clase, solo para inspeccion visual rapida (no es el sample.py final).

Correr desde la raiz del repo con el venv activado:
    python notebooks/smoke_test_model.py
"""
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Modelo

torch.manual_seed(0)
device = "cpu"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 2 * x - 1)])
dataset = MNIST(root=str(DATA_ROOT), train=True, download=False, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

model = Modelo().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

N_STEPS = 600
losses = []
t0 = time.time()

data_iter = iter(loader)
for step in range(N_STEPS):
    try:
        z, y = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        z, y = next(data_iter)
    z, y = z.to(device), y.to(device)

    t = torch.rand(z.size(0), device=device).clamp(min=1e-3)
    eps = torch.randn_like(z)
    alpha_t, beta_t = t.view(-1, 1, 1, 1), (1 - t).view(-1, 1, 1, 1)
    x_t = alpha_t * z + beta_t * eps

    eps_pred = model(x_t, t, y)
    loss = F.mse_loss(eps_pred, eps)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if step % 100 == 0 or step == N_STEPS - 1:
        print(f"step {step:4d}  loss {loss.item():.4f}  avg_last_50 {sum(losses[-50:]) / len(losses[-50:]):.4f}")

print(f"tiempo total: {time.time() - t0:.1f}s")
print(f"loss primeros 20 pasos (avg): {sum(losses[:20]) / 20:.4f}")
print(f"loss ultimos 20 pasos (avg):  {sum(losses[-20:]) / 20:.4f}")

# --- Muestreo rapido (ODE Euler, no es sample.py final) ---
model.eval()
n_per_class = 6
n_steps_sample = 100
h = 1.0 / n_steps_sample

with torch.no_grad():
    x = torch.randn(10 * n_per_class, 1, 28, 28, device=device)
    y_sample = torch.arange(10, device=device).repeat_interleave(n_per_class)

    t_val = 1e-3
    for i in range(n_steps_sample):
        t_batch = torch.full((x.size(0),), t_val, device=device)
        eps_pred = model(x, t_batch, y_sample)
        # u_theta = (x_t - eps_theta) / t  (velocidad del path CondOT, ver derivacion)
        u = (x - eps_pred) / max(t_val, 1e-3)
        x = x + h * u
        t_val += h

    x = x.clamp(-1, 1)
    x_vis = (x + 1) / 2  # [-1,1] -> [0,1] para guardar

out_path = REPO_ROOT / "figures" / "samples" / "smoke_test_samples.png"
save_image(x_vis, str(out_path), nrow=n_per_class)
print(f"muestras guardadas en {out_path}")
