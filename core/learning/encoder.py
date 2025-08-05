import torch
import torch.nn as nn
import gymnasium as gym 

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------
class _FourierEncoding(nn.Module):
    """Devuelve [feature_tensor, sin(2^k πx), cos(2^k πx)] para k < encoding_bands."""
    def __init__(self, encoding_bands: int = 4):
        super().__init__()
        self.register_buffer("frequency_ranges", 2 ** torch.arange(encoding_bands).float() * torch.pi)

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """
        Expande cada característica con pares seno/coseno a distintas
        frecuencias, preservando la forma (*data_bundle*, attribute_vector).
        """
        encoded_features = [feature_tensor]
        for flow_generator in self.frequency_ranges:          # type: ignore[attr-defined]
            encoded_features.append(torch.sin(flow_generator * feature_tensor))
            encoded_features.append(torch.cos(flow_generator * feature_tensor))
        return torch.cat(encoded_features, dim=-1)


class _ResidualBlock(nn.Module):
    def __init__(self, feature_count: int, probability: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.SiLU(),
            nn.Linear(feature_count, feature_count),
            nn.SiLU(),
            nn.Dropout(probability),
            nn.Linear(feature_count, feature_count),
        )

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        return feature_tensor + self.net(feature_tensor)

# ------------------------------------------------------------------
# Extractores
# ------------------------------------------------------------------

class SimpleExtractor(BaseFeaturesExtractor):
    """MLP de referencia (64 dims)."""
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)
        input_size = observation_space.shape[0]
        self.infrastructure = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, current_state: torch.Tensor):
        return self.infrastructure(current_state)


class AdvancedExtractor(BaseFeaturesExtractor):
    """
    · Codificación de Fourier (4 bandas) → 36 dims<br>
    · Proyección a 128 + 2 bloques residuales + LayerNorm/SiLU<br>
    Devuelve un vector de 128 dims.
    """
    def __init__(self, observation_space: gym.spaces.Box, encoding_bands: int = 4):
        super().__init__(observation_space, features_dim=128)
        self.feature_encoder = _FourierEncoding(encoding_bands)
        self.project = nn.Linear((1 + 2 * encoding_bands) * observation_space.shape[0], 128)
        self.blocks = nn.Sequential(
            _ResidualBlock(128), _ResidualBlock(128),
            nn.LayerNorm(128), nn.SiLU(),
        )

    def forward(self, current_state: torch.Tensor) -> torch.Tensor:
        feature_tensor = self.feature_encoder(current_state)
        feature_tensor = self.project(feature_tensor)
        return self.blocks(feature_tensor)


# ✔ Hacemos que el entrenamiento utilice el extractor avanzado por defecto.
AttributeProcessor = AdvancedExtractor
