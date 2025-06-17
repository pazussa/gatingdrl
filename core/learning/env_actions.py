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
def process_step_action(self, action):
    """Procesa la acción en el método step"""
    # -------------------------------------------------------------- #
    # 0. Interpretar la acción con componentes reducidos              #
    # -------------------------------------------------------------- #
    # La acción ahora es un array con 3 componentes
    guard_factor_idx  = int(action[0])
    switch_gap_idx    = int(action[1])
    flow_selection    = int(action[2])
    
    # Convertir índices en valores reales para usar en el algoritmo
    offset_us = 0
    guard_factor_values = [0.5, 0.75, 1.0, 1.5, 2.0]
    guard_factor = guard_factor_values[guard_factor_idx]
    switch_gap_values = [0.5, 1.0, 1.5, 2.0]
    switch_gap = switch_gap_values[switch_gap_idx]
    
    # Registrar las decisiones del agente para visualización
    flow = self.current_flow()
    
    # Almacenar todas las decisiones del agente (sin gcl_strategy)
    self.agent_decisions = {
        'guard_factor': guard_factor,
        'switch_gap': switch_gap,
        'flow_selection': flow_selection
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

    flow = self.current_flow()
    hop_idx = self.flow_progress[self.active_flow_idx]
    link = self.link_dict[flow.path[hop_idx]]
    gating = True
    trans_time = link.transmission_time(flow.payload)
    
    # Guard time ahora usa el factor elegido por el agente
    base_guard_time = link.interference_time()
    guard_time = base_guard_time * guard_factor
    
    # Registrar las decisiones del agente para visualización posterior
    self.guard_time_selected = guard_time
    self.switch_gap_selected = switch_gap

    # Si el ORIGEN del enlace es un switch ⇒ este hop ES un egress
    def _get_src(node_pair):
        return node_pair[0] if isinstance(node_pair, tuple) \
               else node_pair.split('-')[0]

    sw_src = _get_src(link.link_id)
    is_egress_from_switch = sw_src.startswith('S') and not sw_src.startswith('SRV')
    
    # Use fixed conservative GCL strategy
    gcl_strategy = 0
    
    return (flow, hop_idx, link, gating, trans_time,
            guard_time, guard_factor,               # ➊  NUEVO
            offset_us, switch_gap, sw_src,
            is_egress_from_switch, gcl_strategy)
