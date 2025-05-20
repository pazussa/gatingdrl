import os
import sys
import logging

# Obtener la ruta del directorio actual (donde está este script)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Añadir al sys.path para permitir importaciones relativas
if PROJECT_ROOT not in sys.path:
    # Insertar al inicio para asegurar que tiene prioridad sobre otras instalaciones
    sys.path.insert(0, PROJECT_ROOT)
    logging.getLogger(__name__).debug(f"Added {PROJECT_ROOT} to PYTHONPATH")

# Diagnóstico: mostrar el path de Python completo

