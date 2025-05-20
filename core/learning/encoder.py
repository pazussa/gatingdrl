import torch
import torch.nn as nn
import gymnasium as gym 

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------
class _FourierEncoding(nn.Module):
    """Devuelve [x, sin(2^k πx), cos(2^k πx)] para k < num_bands."""
    def __init__(self, num_bands: int = 4):
        super().__init__()
        self.register_buffer("freq_bands", 2 ** torch.arange(num_bands).float() * torch.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expande cada característica con pares seno/coseno a distintas
        frecuencias, preservando la forma (*batch*, features).
        """
        enc = [x]
        for f in self.freq_bands:          # type: ignore[attr-defined]
            enc.append(torch.sin(f * x))
            enc.append(torch.cos(f * x))
        return torch.cat(enc, dim=-1)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

# ------------------------------------------------------------------
# Extractores
# ------------------------------------------------------------------

class SimpleExtractor(BaseFeaturesExtractor):
    """MLP de referencia (64 dims)."""
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)
        inp = observation_space.shape[0]
        self.network = nn.Sequential(
            nn.Linear(inp, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, obs: torch.Tensor):
        return self.network(obs)


class AdvancedExtractor(BaseFeaturesExtractor):
    """
    · Codificación de Fourier (4 bandas) → 36 dims<br>
    · Proyección a 128 + 2 bloques residuales + LayerNorm/SiLU<br>
    Devuelve un vector de 128 dims.
    """
    def __init__(self, observation_space: gym.spaces.Box, num_bands: int = 4):
        super().__init__(observation_space, features_dim=128)
        self.encode = _FourierEncoding(num_bands)
        self.project = nn.Linear((1 + 2 * num_bands) * observation_space.shape[0], 128)
        self.blocks = nn.Sequential(
            _ResidualBlock(128), _ResidualBlock(128),
            nn.LayerNorm(128), nn.SiLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.encode(obs)
        x = self.project(x)
        return self.blocks(x)


# ✔ Hacemos que el entrenamiento utilice el extractor avanzado por defecto.
FeaturesExtractor = AdvancedExtractor
