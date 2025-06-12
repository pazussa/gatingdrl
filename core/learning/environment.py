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

from tools.definitions import LOG_DIR
from core.network.operation import Operation, check_operation_isolation
from core.network.net import Net, Network, generate_flows, generate_simple_topology, FlowGenerator, UniDirectionalFlowGenerator

# Importar desde los módulos auxiliares
from core.learning.env_utils import ErrorType, SchedulingError, find_next_event_time
from core.learning.env_utils import check_valid_link, check_temp_operations
from core.learning.env_actions import process_step_action

# --------------------------------------------------------------------------- #
#  Entorno TSN / DRL                                                          #
# --------------------------------------------------------------------------- #
class NetEnv(gym.Env):
    """Entorno de simulación TSN para aprendizaje por refuerzo."""
    
    # Usar la constante centralizada en Net
    MIN_SWITCH_GAP = Net.SWITCH_GAP_MIN

    @dataclass
    class GclInfo:
        gcl_cycle: int = 1
        gcl_length: int = 0

    # --------------------------------------------------------------------- #
    #  Inicialización                                                       #
    # --------------------------------------------------------------------- #
    def __init__(self, network: Optional[Network] = None, 
                curriculum_enabled: bool = True,
                initial_complexity: float = 0.25,
                curriculum_step: float = 0.05) -> None:
        super().__init__()

        # --- Curriculum Learning Adaptativo ---
        self.curriculum_enabled = curriculum_enabled
        self.initial_complexity = initial_complexity 
        self.curriculum_step = curriculum_step
        self.current_complexity = initial_complexity
        self.consecutive_successes = 0
        self.original_flows = []  # Store the original complete set of flows
        
        # Si se proporciona una red, guardar su estructura original
        if network is not None:
            self.total_flows = len(network.flows)
            self.base_graph = network.graph
            # Guardar todos los flujos originales
            self.original_flows = list(network.flows)
            
            # Registro explícito para depuración del modo curriculum
            self.logger = logging.getLogger(f"{__name__}.{os.getpid()}")
            self.logger.setLevel(logging.INFO)
            self.logger.info(f"Inicializando entorno con {self.total_flows} flujos (curriculum: {curriculum_enabled}, complejidad inicial: {initial_complexity})")
            
            if curriculum_enabled and initial_complexity < 1.0:
                # En modo curriculum, reducir el número inicial de flujos
                active_flows = int(self.total_flows * self.initial_complexity)
                active_flows = max(5, active_flows)  # Mínimo 5 flujos para empezar
                
                # Seleccionar subset de flujos para el nivel de complejidad actual
                active_flows_list = self.original_flows[:active_flows]
                network = Network(network.graph, active_flows_list)
                self.logger.info(f"Modo curriculum ACTIVADO: usando {active_flows}/{self.total_flows} flujos inicialmente")
            else:
                # Si curriculum está desactivado o complejidad es 1.0, usar todos los flujos
                self.logger.info(f"Modo curriculum DESACTIVADO: usando todos los {self.total_flows} flujos")
                self.current_complexity = 1.0  # Forzar complejidad completa
        
        # Si no se entrega una red, construir topología y flujos sencillos
        if network is None:
            g = generate_simple_topology()
            f = generate_flows(g, 10)
            network = Network(g, f)
            self.total_flows = len(network.flows)
            self.base_graph = g
            # Crear generador de flujos apropiado
            self.flow_generator = FlowGenerator(g)

        # --- Estructuras base -------------------------------------------- #
        self.graph = network.graph
        self.flows = list(network.flows)               # lista estable
        self.line_graph, self.link_dict = (
            network.line_graph,
            network.links_dict,
        )

        # --- Estados internos ------------------------------------------- #
        self.num_flows: int = len(self.flows)
        # Reloj de referencia global (solo para la observación)
        # Se calcula siempre como el próximo evento más cercano
        self.global_time: int = 0
        self.flow_progress: List[int] = [0] * self.num_flows  # hop en curso de cada flujo
        self.flow_completed: List[bool] = [False] * self.num_flows
        self.flow_first_tx: List[int | None] = [None] * self.num_flows
        self.current_flow_idx: int = 0                 # para round‑robin

        self.links_operations = defaultdict(list)

        #  🔀  Las estructuras ligadas a GCL ya no se utilizan.  Mantener
        #  únicamente la planificación de operaciones; el cálculo de las
        #  tablas se traslada al *ResAnalyzer*.

        self.temp_operations: List[tuple] = []         # op en construcción

        # 🔹 Placeholder para mantener compatibilidad con código heredado.
        #   Ya no se llena ni se usa, pero evita AttributeError.
        self.links_gcl: dict = {}

        # ⏱️  NUEVO: reloj "ocupado‑hasta" por enlace
        #    (cuándo queda libre cada enlace)
        self.link_busy_until = defaultdict(int)
        # ⏱️⏱️ reloj "ocupado‑hasta" por switch **sólo para EGRESOS**
        self.switch_busy_until = defaultdict(int)
        
        # 📊 NUEVO: Registro del último tiempo de llegada por switch
        # Para garantizar separación mínima entre paquetes
        self.switch_last_arrival = defaultdict(int)

        # ⏲️  Último instante en que **se creó** (primer hop) un paquete
        #     – solo se usa para imponer la separación en el PRIMER enlace
        self.last_packet_start = -Net.PACKET_GAP_EXTRA

        # 🚦 NUEVO: sección crítica global – "una sola cola"
        self.global_queue_busy_until = 0

        # --- Espacios de observación y acción --------------------------- #
        self._setup_spaces()

        # --- Logger ------------------------------------------------------ #
        self.logger = logging.getLogger(f"{__name__}.{os.getpid()}")
        self.logger.setLevel(logging.INFO)

        # Cache: «¿es nodo final?»
        self._es_node_cache: Dict[Any, bool] = {}
        
        # Variable para datos de operación
        self.last_operation_info = {}
        self.agent_decisions = {}

        # Orden FIFO inmutable: simplemente el índice de creación del flujo
        # (flows ya está en el mismo orden en que se generaron).
        self._fifo_order = list(range(self.num_flows))

        # ──────────────────────────────────────────────────────────────
        #  📊  Métricas de latencia extremo-a-extremo (µs) por episodio
        # ──────────────────────────────────────────────────────────────
        #   Se irán llenando a medida que cada flujo se completa.
        #   Se resumen al final del episodio en environment_impl.step.
        self._latency_samples: list[int] = []

        # ⏱️  lista para almacenar la latencia de cada flujo completado
        self._flow_latencies: list[int] = []

    # ----------------------------------------------------------------- #
    #  Helper estático (picklable) para muestrear la separación global  #
    # ----------------------------------------------------------------- #
    @staticmethod
    def _next_packet_gap() -> int:
        """
        Devuelve la separación entre paquetes (µs) delegando la lógica
        íntegramente a ``Net.sample_packet_gap``.  
        Esta versión elimina código muerto y evita ramas nunca alcanzadas.
        """
        return Net.sample_packet_gap()

    # --------------------------------------------------------------------- #
    #  Configuración de gymnasium                                           #
    # --------------------------------------------------------------------- #
    def _default_gcl_info(self):
        return self.GclInfo()

    def _setup_spaces(self):
        # MODIFICAR LA OBSERVACIÓN: Incluir información de múltiples flujos (hasta 5)
        # Para cada flujo: [período_norm, payload_norm, progreso, tiempo_espera_norm, es_seleccionable]
        # Además de las características globales originales
        FLUJOS_OBSERVABLES = 5  # Número de flujos que podemos observar a la vez
        FEATURES_POR_FLUJO = 5  # Características por flujo
        FEATURES_GLOBALES = 4   # Características globales (tiempo, ocupación GCL, etc.)
        
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(FEATURES_GLOBALES + FLUJOS_OBSERVABLES * FEATURES_POR_FLUJO,), 
            dtype=np.float32
        )
        
        # NUEVO: Almacenar constantes para uso posterior
        self.NUM_FLUJOS_OBSERVABLES = FLUJOS_OBSERVABLES
        self.FEATURES_POR_FLUJO = FEATURES_POR_FLUJO
        self.FEATURES_GLOBALES = FEATURES_GLOBALES

        # ╔═══════════════════════════════════════════════════════════════════════╗
        # ║  ESPACIO DE ACCIÓN (3 DIMENSIONES)                                    ║
        # ║  0. Guard factor [0-4]                                                ║
        # ║  1. Gap mínimo en switch [0-3]                                        ║
        # ║  2. Selección de flujo candidato [0-4]                                ║
        # ╚═══════════════════════════════════════════════════════════════════════╝
        self.action_space = spaces.MultiDiscrete([
            5,                 # Guard factor
            4,                 # Gap mínimo switch
            FLUJOS_OBSERVABLES # Selección de flujo
        ])

    # --------------------------------------------------------------------- #
    #  Utilidades de selección de flujo / enlace                            #
    # --------------------------------------------------------------------- #
    def select_next_flow_by_agent(self, flow_selection):
        """
        Permite que el agente RL seleccione el próximo flujo a programar.
        
        Args:
            flow_selection: Índice del flujo seleccionado por el agente (0-4)
        """
        # Usar el índice de flujo seleccionado por el agente si está disponible
        if hasattr(self, 'current_candidate_flows') and self.current_candidate_flows:
            if 0 <= flow_selection < len(self.current_candidate_flows):
                selected_idx = self.current_candidate_flows[flow_selection]
                if not self.flow_completed[selected_idx]:
                    self.current_flow_idx = selected_idx
                    return

        # Si la selección directa falla, usar FIFO como fallback
        now = self.global_time
        chosen = None  # (idx, wait_time)

        for idx, done in enumerate(self.flow_completed):
            if done:
                continue
            prog = self.flow_progress[idx]
            if prog == 0 or prog >= len(self.flows[idx].path):
                continue
            next_link = self.flows[idx].path[prog]
            if not next_link[0].startswith('S'):
                continue

            prev_link_id = self.flows[idx].path[prog-1]
            prev_ops = self.links_operations[self.link_dict[prev_link_id]]
            if not prev_ops:
                continue  # el hop previo aún no fue programado
            prev_op = prev_ops[-1][1]
            wait_time = now - prev_op.reception_time
            if chosen is None or wait_time > chosen[1]:
                chosen = (idx, wait_time)

        if chosen:
            self.current_flow_idx = chosen[0]
            return
                
        # Si no se encontró un flujo adecuado, buscar cualquier flujo no completado
        if self.flow_completed[self.current_flow_idx]:
            for idx, completed in enumerate(self.flow_completed):
                if not completed:
                    self.current_flow_idx = idx
                    return

    def current_flow(self):
        return self.flows[self.current_flow_idx]

    def current_link(self):
        idx = self.flow_progress[self.current_flow_idx]
        flow = self.current_flow()
        return self.link_dict[flow.path[idx]]

    # ----------------------------------------------------------------- #
    #  FIFO helper                                                      #
    # ----------------------------------------------------------------- #
    def _next_fifo_idx(self) -> int | None:
        """
        Devuelve el índice del **único** flujo que debe programarse ahora
        según FIFO estricto (el que más tiempo lleva esperando).
        Si no hay flujos pendientes, devuelve None.
        
        Prioridad: 
        1. Paquetes que están esperando en un switch (para transmisión inmediata)
        2. Paquetes con mayor tiempo de espera
        """
        best: tuple[int, int, bool, int] | None = None   # (wait_time, -idx, is_in_switch, hop_idx)
        best_idx: int | None = None

        now = self.global_time
        for idx, done in enumerate(self.flow_completed):
            if done:
                continue
            hop_idx = self.flow_progress[idx]
            if hop_idx >= len(self.flows[idx].path):
                continue

            # --- tiempo desde que el paquete está "listo" ---
            if hop_idx == 0:
                ready_t = 0                          # nunca se ha transmitido
                is_in_switch = False
            else:
                prev_link_id = self.flows[idx].path[hop_idx - 1]
                prev_ops = self.links_operations.get(self.link_dict[prev_link_id], [])
                if not prev_ops:
                    continue        # hop previo aún no programado ⇒ no listo
                
                prev_op = prev_ops[-1][1]
                ready_t = prev_op.reception_time
                
                # Determinar si el paquete está esperando en un switch
                # Si el destino del enlace anterior es un switch y el origen del siguiente enlace
                # coincide con ese switch, entonces el paquete está esperando en un switch
                dst_node = prev_link_id[1] if isinstance(prev_link_id, tuple) else prev_link_id.split('-')[1]
                is_in_switch = dst_node.startswith('S') and not dst_node.startswith('SRV')

            wait = now - ready_t
            fifo_rank = -idx               # menor índice ⇒ más antiguo
            
            # Prioridad: 1) paquetes en switch, 2) tiempo de espera, 3) índice más bajo
            cand = (int(is_in_switch), wait, fifo_rank, hop_idx)

            if (best is None) or (cand > best):
                best = cand
                best_idx = idx

        return best_idx

    # --------------------------------------------------------------------- #
    #  Reinicio                                                             #
    # --------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Si es necesario incrementar la complejidad
        if self.curriculum_enabled and self.consecutive_successes >= 3:
            if self.increase_complexity():
                # Calcular número de flujos según complejidad actual
                num_flows = int(self.total_flows * self.current_complexity)
                num_flows = max(5, min(self.total_flows, num_flows))
                
                if self.original_flows:
                    # Usar un subset consistente de los flujos originales
                    active_flows_list = self.original_flows[:num_flows]
                    network = Network(self.base_graph, active_flows_list)
                    
                    # Actualizar las estructuras con la nueva red
                    self.graph = network.graph
                    self.flows = list(network.flows)
                    self.line_graph, self.link_dict = (
                        network.line_graph,
                        network.links_dict,
                    )
                    self.num_flows = len(self.flows)
                    
                    self.logger.info(f"Curriculum: Incrementando a {num_flows}/{self.total_flows} flujos (complejidad: {self.current_complexity:.2f})")
                
            self.consecutive_successes = 0

        self.global_time = 0
        self.flow_progress = [0] * self.num_flows
        self.flow_completed = [False] * self.num_flows
        self.flow_first_tx = [None] * self.num_flows
        self.current_flow_idx = 0

        self.links_operations.clear()
        self.links_gcl = {}   # reiniciar placeholder
        self.temp_operations.clear()
        self._es_node_cache.clear()
        self.link_busy_until.clear()
        self.switch_busy_until.clear()
        self.switch_last_arrival.clear()    # NUEVO: Limpiar tiempos de llegada
        self.global_queue_busy_until = 0
        self.last_packet_start = -Net.PACKET_GAP_EXTRA

        # Limpiar métricas de latencia para el nuevo episodio
        self._latency_samples.clear()

        # ⏱️  reiniciar latencias e2e acumuladas
        self._flow_latencies.clear()

        return self._get_observation(), {}

    # --------------------------------------------------------------------- #
    #  Observación                                                          #
    # --------------------------------------------------------------------- #
    def _get_observation(self):
        # Creamos una observación que contenga información sobre múltiples flujos
        # La observación tendrá: [características_globales, características_flujo1, características_flujo2, ...]
        
        # 1. Características globales de la red
        global_time_norm = self.global_time / 10000  # Normalizar tiempo global
        
        # Ya no hay GCL dinámico → utilización 0 siempre
        gcl_util_norm = 0.0
        
        # Calcular porcentaje de flujos completados
        completion_rate = sum(self.flow_completed) / self.num_flows
        
        # Nivel de curriculum
        curriculum_norm = self.current_complexity
        
        # Vector de características globales
        global_features = [global_time_norm, gcl_util_norm, completion_rate, curriculum_norm]
        
        # 2. Obtener una lista de flujos candidatos para programar
        candidatos = []
        for idx, completed in enumerate(self.flow_completed):
            if not completed:
                flow = self.flows[idx]
                hop_idx = self.flow_progress[idx]
                
                # Verificar que el flujo tiene un hop válido para programar
                if hop_idx >= len(flow.path):
                    continue
                    
                # Calcular cuánto tiempo ha estado esperando (si aplica)
                wait_time = 0
                if hop_idx > 0:
                    prev_link_id = flow.path[hop_idx-1]
                    prev_ops = self.links_operations.get(self.link_dict[prev_link_id], [])
                    if prev_ops:
                        prev_op = prev_ops[-1][1]
                        wait_time = max(0, self.global_time - prev_op.reception_time)
                
                # Normalizar valores - convertir a características significativas 
                period_norm = flow.period / 10000  # Períodos más cortos → valores más pequeños
                payload_norm = flow.payload / Net.MTU  # Payloads más pequeños → valores más pequeños
                hop_progress = self.flow_progress[idx] / len(flow.path)
                wait_time_norm = min(wait_time / flow.period, 1.0)  # Normalizar por periodo
                
                # Calcular urgencia basada en plazo próximo
                # Cuanto menor sea el tiempo hasta el deadline, mayor urgencia (1.0 = muy urgente)
                deadline_remaining = (flow.period - (self.global_time % flow.period)) / flow.period
                urgency = 1.0 - deadline_remaining  # 0.0 = acaba de empezar, 1.0 = casi vencido
                
                # Log del flujo candidato con sus características para depuración
                self.logger.debug(f"Candidato: {flow.flow_id}, Period: {period_norm:.2f}, Payload: {payload_norm:.2f}, Wait: {wait_time_norm:.2f}, Urgency: {urgency:.2f}")
                
                # Añadir a candidatos con sus características
                candidatos.append((idx, [period_norm, payload_norm, hop_progress, wait_time_norm, urgency]))
        
        # 3. Si no hay candidatos, devolver una observación con ceros
        if not candidatos:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # 4. Ordenar candidatos de forma significativa para el agente
        # Considerar múltiples factores: tiempo espera (FIFO), urgencia, tamaño payload
        candidatos = sorted(candidatos, key=lambda x: (
            x[1][3],  # Tiempo de espera (mayor primero - FIFO)
            x[1][4],  # Urgencia (mayor primero)
            -x[1][1]  # Payload (menor primero - más rápido)
        ), reverse=True)
        
        # Asegurar que tenemos exactamente NUM_FLUJOS_OBSERVABLES candidatos
        if len(candidatos) > self.NUM_FLUJOS_OBSERVABLES:
            candidatos = candidatos[:self.NUM_FLUJOS_OBSERVABLES]
        
        while len(candidatos) < self.NUM_FLUJOS_OBSERVABLES:
            candidatos.append((-1, [0.0, 0.0, 0.0, 0.0, 0.0]))
        
        # 5. Actualizar los índices de flujos candidatos para recuperarlos después
        self.current_candidate_flows = [idx for idx, _ in candidatos if idx >= 0]
        
        # Registro para depuración
        if self.current_candidate_flows:
            self.logger.debug(f"Candidatos ordenados: {[self.flows[idx].flow_id for idx in self.current_candidate_flows]}")
        
        # 6. Construir la observación completa concatenando características
        obs = np.array(global_features + [feat for _, feats in candidatos for feat in feats], dtype=np.float32)
        
        return obs

    # ------------------------------------------------------------------ #
    #  Máscaras de acción                                                #
    # ------------------------------------------------------------------ #
    def action_masks(self):
        """
        Genera máscaras para el espacio de acción MultiDiscreto.
        """
        # Calcular tamaño total de la máscara sumando todas las dimensiones
        mask_size = sum(self.action_space.nvec)
        
        # Crear una máscara plana donde todas las acciones están permitidas (0 = permitida)
        mask = np.zeros(mask_size, dtype=np.int8)
        
        try:
            if self.current_flow_idx < len(self.flows) and not self.flow_completed[self.current_flow_idx]:
                flow = self.current_flow()
                hop_idx = self.flow_progress[self.current_flow_idx]
                
                if hop_idx < len(flow.path):
                    link = self.link_dict[flow.path[hop_idx]]
                    
                    # Si es el primer hop, los factores de guard time altos podrían desperdiciarse
                    if hop_idx == 0:
                        # Índice para guard time alto
                        offset = 0  # No offset needed anymore
                        mask[offset + 3] = 1  # Desactivar valores 3 y 4 (factores altos)
                        mask[offset + 4] = 1
        except Exception as e:
            self.logger.warning(f"Error generando máscaras de acción: {e}")
            
        return mask

    # --------------------------------------------------------------------- #
    #  Comprobaciones de aislamiento y GCL                                  #
    # --------------------------------------------------------------------- #
    def _is_es_source(self, link_id):
        """
        Devuelve True si el **origen** de link_id es una End-Station (ES).
        1) Primero consulta el atributo ``node_type`` del grafo.
        2) Si no existe, usa la convención de nombres:
           E*, C*  = clientes/ES genéricas  
           SRV*    = servidores (también ES)  
           S*      = switches
        """
        if link_id in self._es_node_cache:
            return self._es_node_cache[link_id]

        src = link_id[0] if isinstance(link_id, tuple) else link_id.split('-')[0]

        # a) Metadata del grafo (más robusta)
        if self.graph.nodes.get(src, {}).get("node_type") == "ES":
            self._es_node_cache[link_id] = True
            return True

        # b) Convención de nombres
        is_es = src.startswith(("E", "C", "SRV"))
        self._es_node_cache[link_id] = is_es
        return is_es

    #  ⛔  Retirado.  La agrupación de GCL dejó de ser necesaria.

    def _check_valid_link(self, link, operation):
        return check_valid_link(link, operation, self.current_flow(), self.links_operations)

    def _check_temp_operations(self):
        return check_temp_operations(self.temp_operations, self.links_operations, self.current_flow())

    def _find_next_event_time(self, current_time):
        """Encuentra el siguiente tiempo de evento programado después de current_time"""
        return find_next_event_time(self.link_busy_until, self.switch_busy_until, current_time)

    # Método obsoleto: generación dinámica de GCL eliminada.
    # Se mantiene para compatibilidad pero no hace nada y devuelve False.
    def add_gating_with_grouping(self, *args, **kwargs):
        return False

    def increase_complexity(self):
        """Incrementa el nivel de dificultad del entorno"""
        if not self.curriculum_enabled or self.current_complexity >= 1.0:
            return False
            
        # Incrementar complejidad gradualmente
        self.current_complexity = min(1.0, self.current_complexity + self.curriculum_step)
        return True

    # --------------------------------------------------------------------- #
    #  Método principal step() - delegado al módulo environment_impl         #
    # --------------------------------------------------------------------- #
    from core.learning.environment_impl import step  # (sigue igual, pero sin GCL)

    # --------------------------------------------------------------------- #
    #  gym stubs                                                             #
    # --------------------------------------------------------------------- #
    def render(self):
        pass

    def close(self):
        pass

