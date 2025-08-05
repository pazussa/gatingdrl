"""
Entornos y componentes de aprendizaje por refuerzo
"""

from .environment import NetEnv  # La misma exportación para mantener compatibilidad
# Exporta ambos extractores
from .encoder import AttributeProcessor          # GIN-baseline
from .hats_extractor import HATSExtractor       # NUEVO: HATS
from .maskable_sac import MaskableSAC           # NUEVO: SAC con máscaras




