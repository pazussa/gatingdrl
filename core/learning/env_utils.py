import math
import os
import random
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Dict, Any



import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.network.operation import Operation, check_operation_isolation
from core.network.net import Net, Network, generate_flows, generate_simple_topology, FlowGenerator, UniDirectionalFlowGenerator

# --------------------------------------------------------------------------- #
#  Definiciones de error                                                      #
# --------------------------------------------------------------------------- #
class ErrorType(Enum):
    PeriodExceed = auto()

class SchedulingError(Exception):
    def __init__(self, error_type: ErrorType, msg: str):
        super().__init__(f"Error: {msg}")
        self.error_type = error_type
        self.msg = msg

# Funciones auxiliares para NetEnv
def find_next_event_time(link_busy_until, switch_busy_until, current_time):
    """Encuentra el siguiente tiempo de evento programado después de current_time"""
    next_event_time = float('inf')
    
    # Buscar en todos los tiempos de ocupación de enlaces
    for time in link_busy_until.values():
        if time > current_time and time < next_event_time:
            next_event_time = time
            
    # Buscar en todos los tiempos de ocupación de switches
    for time in switch_busy_until.values():
        if time > current_time and time < next_event_time:
            next_event_time = time
    
    return next_event_time if next_event_time < float('inf') else None

def check_valid_link(link, operation, current_flow, links_operations):
    """Comprueba si una operación es válida en un enlace"""
    for f_rhs, op_rhs in links_operations[link]:
        offset = check_operation_isolation(
            (operation, current_flow.period), (op_rhs, f_rhs.period)
        )
        if offset is not None:
            return offset
    return None

def check_temp_operations(temp_operations, links_operations, current_flow):
    """Verifica todas las operaciones temporales"""
    for link, op in temp_operations:
        offset = check_valid_link(link, op, current_flow, links_operations)
        if offset is not None:
            return offset
    return None
