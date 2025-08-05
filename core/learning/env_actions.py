import math
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from core.network.operation import Operation
from core.network.net import Net

from core.learning.env_utils import SchedulingError, ErrorType, find_next_event_time, check_temp_operations



# --------------------------------------------------------------------------- #
# → NUEVA filosofía de GCL ←                                                 #
#   ▸ Cada vez que un hop *espera* en el egress-switch                       #
#     se añaden **exactamente 2 entradas** (close n).                   #
#   ▸ La longitud del ciclo no cambia (se mantiene en 1 para indicar que     #
#     gestionaremos las reglas como eventos independientes).                 #
# --------------------------------------------------------------------------- #

# Todas las utilidades relacionadas con la reserva dinámica de GCL se han
# eliminado.  A partir de ahora la lista se calcula *a posteriori* en el
# analizador de resultados.

pass  # <-- mantener el archivo, sin lógica de GCL

# --------------------------------------------------------------------------- #
# Funciones para la implementación del método step                            #
# --------------------------------------------------------------------------- #
def process_step_action(self, command):
    """Procesa la acción en el método step"""
    # -------------------------------------------------------------- #
    # 0. Interpretar la acción con componentes reducidos              #
    # -------------------------------------------------------------- #
    # La acción ahora es un array con 3 componentes
    protection_level  = int(command[0])
    spacing_level    = int(command[1])
    stream_choice    = int(command[2])
    
    # Convertir índices en valores reales para usar en el algoritmo
    timing_offset = 0
    protection_options = [0.5, 0.75, 1.0, 1.5, 2.0]
    protection_multiplier = protection_options[protection_level]
    spacing_options = [0.5, 1.0, 1.5, 2.0]
    inter_packet_spacing = spacing_options[spacing_level]
    
    # Registrar las decisiones del agente para visualización
    data_stream = self.current_flow()
    
    # Almacenar todas las decisiones del agente (sin scheduling_policy)
    self.policy_choices = {
        'protection_multiplier': protection_multiplier,
        'inter_packet_spacing': inter_packet_spacing,
        'stream_choice': stream_choice
    }
    
    # NUEVO: Métricas de operación para el análisis
    self.last_operation_info = {
        'bandwidth_used': 0,
        'bandwidth_total': 0,
        'wait_breakdown': {
            'switch': 0,
            'gap': 0,
            'total': 0
        }
    }

    data_stream = self.current_flow()
    segment_index = self.stream_advancement[self.active_stream_id]
    network_connection = self.connection_registry[data_stream.path[segment_index]]
    time_synchronization = True
    transmission_duration = network_connection.transmission_time(data_stream.payload)
    
    # Guard time ahora usa el factor elegido por el agente
    min_protection_interval = network_connection.interference_time()
    safety_interval = min_protection_interval * protection_multiplier
    
    # Registrar las decisiones del agente para visualización posterior
    self.chosen_protection = safety_interval
    self.chosen_spacing = inter_packet_spacing

    # Si el ORIGEN del enlace es un switch ⇒ este hop ES un egress
    def _get_src(node_pair):
        return node_pair[0] if isinstance(node_pair, tuple) \
               else node_pair.split('-')[0]

    sw_src = _get_src(network_connection.link_id)
    outbound_from_switch = sw_src.startswith('S') and not sw_src.startswith('SRV')
    
    # Use fixed conservative GCL strategy
    scheduling_policy = 0
    
    return (data_stream, segment_index, network_connection, time_synchronization, transmission_duration,
            safety_interval, protection_multiplier,               # ➊  NUEVO
            timing_offset, inter_packet_spacing, sw_src,
            outbound_from_switch, scheduling_policy)
