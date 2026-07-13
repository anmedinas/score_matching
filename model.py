"""Red de difusion: UNet condicional compartida por modelo_cond y modelo_cfg.

Importable como `from model import Modelo`. Las dos redes de difusion del
proyecto (condicional pura y con CFG) usan esta misma arquitectura; solo
difieren en como se entrenan (ver train.py, --label-dropout).

Convenciones de tensores: x tiene forma (B, C, H, W) (MNIST: C=1, H=W=28);
t tiene forma (B,) con valores en [0, 1]; y (etiqueta) es opcional, e incluye
el token nulo (None / indice reservado) para la red con CFG.
"""

import math

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Bloque residual con inyeccion aditiva de t/y (FiLM sin termino de escala)."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, n_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Modelo(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        time_emb_dim: int = 128,
        cond_dim: int = 128,
        num_classes: int = 10,
        **config,
    ):
        super().__init__()
        self.config = dict(
            in_channels=in_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
            cond_dim=cond_dim,
            num_classes=num_classes,
            **config,
        )
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes
        self.null_label = num_classes  # indice reservado para el token nulo "vacio".

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )
        self.label_embedding = nn.Embedding(num_classes + 1, cond_dim)
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        self.init_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        self.enc1 = ResidualBlock(c1, c1, cond_dim)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)  # 28 -> 14
        self.enc2 = ResidualBlock(c2, c2, cond_dim)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)  # 14 -> 7

        self.mid1 = ResidualBlock(c3, c3, cond_dim)
        self.mid2 = ResidualBlock(c3, c3, cond_dim)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)  # 7 -> 14
        self.dec2 = ResidualBlock(c2 + c2, c2, cond_dim)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)  # 14 -> 28
        self.dec1 = ResidualBlock(c1 + c1, c1, cond_dim)

        self.out_norm = nn.GroupNorm(8, c1)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(c1, in_channels, kernel_size=3, padding=1)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Embedding temporal sinusoidal (auxiliar). t: (B,) con valores en [0, 1]."""
        half_dim = self.time_emb_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        args = t.float()[:, None] * 1000.0 * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.time_emb_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """Retorna eps_theta(x, t, y): ruido predicho, misma forma que x.

        Se parametriza eps_theta (no s_theta directamente) para evitar dividir
        por beta_t durante el entrenamiento (Corolario 4.5 de
        score_matching_teoria.pdf); el score se recupera como
        s_theta = -eps_theta / beta_t (ver item (c) del informe).
        y=None usa el token nulo para todo el batch.
        """
        batch_size = x.size(0)
        if y is None:
            y = torch.full((batch_size,), self.null_label, dtype=torch.long, device=x.device)

        cond = self.cond_mlp(self.time_mlp(self.time_embedding(t)) + self.label_embedding(y))

        h1 = self.enc1(self.init_conv(x), cond)
        h2 = self.enc2(self.down1(h1), cond)
        h = self.mid2(self.mid1(self.down2(h2), cond), cond)

        h = self.dec2(torch.cat([self.up2(h), h2], dim=1), cond)
        h = self.dec1(torch.cat([self.up1(h), h1], dim=1), cond)

        return self.out_conv(self.out_act(self.out_norm(h)))
