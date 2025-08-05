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
    DEADLINE_VIOLATION = auto()

class SchedulingError(Exception):
    def __init__(self, failure_category: ErrorType, error_message: str):
        super().__init__(f"Error: {error_message}")
        self.failure_category = failure_category
        self.error_message = error_message

# Funciones auxiliares para NetEnv
def find_next_event_time(connection_free_time, node_free_time, system_clock):
    """Encuentra el siguiente tiempo de evento programado después de system_clock"""
    upcoming_event = float('inf')
    
    # Buscar en todos los tiempos de ocupación de enlaces
    for time in connection_free_time.values():
        if time > system_clock and time < upcoming_event:
            upcoming_event = time
            
    # Buscar en todos los tiempos de ocupación de switches
    for time in node_free_time.values():
        if time > system_clock and time < upcoming_event:
            upcoming_event = time
    
    return upcoming_event if upcoming_event < float('inf') else None

def check_valid_link(network_connection, operation, current_flow, connection_activities):
    """Comprueba si una operación es válida en un enlace"""
    for f_rhs, op_rhs in connection_activities[network_connection]:
        time_adjustment = check_operation_isolation(
            (operation, current_flow.period), (op_rhs, f_rhs.period)
        )
        if time_adjustment is not None:
            return time_adjustment
    return None

def check_temp_operations(provisional_activities, connection_activities, current_flow):
    """Verifica todas las operaciones temporales"""
    for network_connection, operation_record in provisional_activities:
        time_adjustment = check_valid_link(network_connection, operation_record, current_flow, connection_activities)
        if time_adjustment is not None:
            return time_adjustment
    return None
