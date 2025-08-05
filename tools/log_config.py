import logging


# ✅ Añadir función metadata personalizada al módulo logging
if not hasattr(logging, 'metadata'):
    def metadata(msg, *args, **kwargs):
        """Función personalizada para logging de metadatos con formato especial"""
        logging.info(f"[METADATA] {msg}", *args, **kwargs)
    
    # Añadir la función al módulo logging
    logging.metadata = metadata


# 1. ✅ Configura el event_recorder de matplotlib.font_manager ANTES de importar matplotlib
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# 2. ✅ Configura la fuente por defecto
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Asegúrate de que esta fuente esté instalada
# Eliminar el parámetro inválido font.cache_size
# Verificar si el parámetro get_no_warn es válido
try:
    plt.rcParams['font.get_no_warn'] = True
except KeyError:
    # Ignorar si este parámetro tampoco es válido
    pass


# 3. Función para configurar el logging general
def log_config(filename, level=logging.DEBUG):
    # Configuración básica del event_recorder raíz
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename, mode='w'),
            logging.StreamHandler()
        ]
    )
    # 4. 🚨 Evita que matplotlib.font_manager herede el nivel DEBUG
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    