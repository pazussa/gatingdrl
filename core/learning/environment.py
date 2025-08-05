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
        gcl_length: int = 0    # --------------------------------------------------------------------- #
    #  Inicialización                                                       #
    # --------------------------------------------------------------------- #
    def __init__(self, infrastructure: Optional[Network] = None, 
                adaptive_learning: bool = True,
                starting_difficulty: float = 0.25,
                advancement_rate: float = 0.05,
                graph_mode_enabled: bool = False) -> None:
        super().__init__()

        # --- Curriculum Learning Adaptativo ---
        self.adaptive_learning = adaptive_learning
        self.starting_difficulty = starting_difficulty 
        self.advancement_rate = advancement_rate
        self.difficulty_level = starting_difficulty
        self.streak_count = 0
        self.complete_stream_set = []  # Store the original complete set of traffic_streams
        
        # --- Graph observation support ---
        self.graph_mode_enabled = graph_mode_enabled
        
        # Si se proporciona una red, guardar su estructura original
        if infrastructure is not None:
            self.complete_stream_count = len(infrastructure.traffic_streams)
            self.network_topology = infrastructure.network_structure
            # Guardar todos los flujos originales
            self.complete_stream_set = list(infrastructure.traffic_streams)
            
            # Registro explícito para depuración del modo curriculum
            self.event_recorder = logging.getLogger(f"{__name__}.{os.getpid()}")
            self.event_recorder.setLevel(logging.INFO)
            self.event_recorder.info(f"Inicializando entorno con {self.complete_stream_count} flujos (curriculum: {adaptive_learning}, complejidad inicial: {starting_difficulty})")
            
            if adaptive_learning and starting_difficulty < 1.0:
                # En modo curriculum, reducir el número inicial de flujos
                enabled_streams = int(self.complete_stream_count * self.starting_difficulty)
                enabled_streams = max(5, enabled_streams)  # Mínimo 5 flujos para empezar
                
                # Seleccionar subset de flujos para el nivel de complejidad actual
                stream_collection = self.complete_stream_set[:enabled_streams]
                infrastructure = Network(infrastructure.network_structure, stream_collection)
                self.event_recorder.info(f"Modo curriculum ACTIVADO: usando {enabled_streams}/{self.complete_stream_count} flujos inicialmente")
            else:
                # Si curriculum está desactivado o complejidad es 1.0, usar todos los flujos
                self.event_recorder.info(f"Modo curriculum DESACTIVADO: usando todos los {self.complete_stream_count} flujos")
                self.difficulty_level = 1.0  # Forzar complejidad completa
        
        # Si no se entrega una red, construir topología y flujos sencillos
        if infrastructure is None:
            network_graph = generate_simple_topology()
            flow_generator = generate_flows(network_graph, 10)
            infrastructure = Network(network_graph, flow_generator)
            self.complete_stream_count = len(infrastructure.traffic_streams)
            self.network_topology = network_graph
            # Crear generador de flujos apropiado
            self.stream_factory = FlowGenerator(network_graph)

        # --- Estructuras base -------------------------------------------- #
        self.network_structure = infrastructure.network_structure
        self.traffic_streams = list(infrastructure.traffic_streams)               # lista estable
        self.line_graph, self.connection_registry = (
            infrastructure.line_graph,
            infrastructure.links_dict,
        )

        # --- Estados internos ------------------------------------------- #
        self.stream_count: int = len(self.traffic_streams)
        # Reloj de referencia global (solo para la observación)
        # Se calcula siempre como el próximo evento más cercano
        self.simulation_clock: int = 0
        self.stream_advancement: List[int] = [0] * self.stream_count  # hop en curso de cada flujo
        self.stream_finished: List[bool] = [False] * self.stream_count
        self.initial_transmission: List[int | None] = [None] * self.stream_count
        self.active_stream_id: int = 0                 # para round‑robin

        self.connection_activities = defaultdict(list)

        #  🔀  Las estructuras ligadas a GCL ya no se utilizan.  Mantener
        #  únicamente la planificación de operaciones; el cálculo de las
        #  tablas se traslada al *ResAnalyzer*.

        self.provisional_activities: List[tuple] = []         # operation_record en construcción

        # 🔹 Placeholder para mantener compatibilidad con código heredado.
        #   Ya no se llena ni se usa, pero evita AttributeError.
        self.connection_schedules: dict = {}

        # ⏱️  NUEVO: reloj "ocupado‑hasta" por enlace
        #    (cuándo queda libre cada enlace)
        self.connection_free_time = defaultdict(int)
        # ⏱️⏱️ reloj "ocupado‑hasta" por switch **sólo para EGRESOS**
        self.node_free_time = defaultdict(int)
        
        # 📊 NUEVO: Registro del último tiempo de llegada por switch
        # Para garantizar separación mínima entre paquetes
        self.node_last_reception = defaultdict(int)

        # ⏲️  Último instante en que **se creó** (primer hop) un paquete
        #     – solo se usa para imponer la separación en el PRIMER enlace
        self.last_packet_start = -Net.PACKET_GAP_EXTRA

        # 🚦 NUEVO: sección crítica global – "una sola cola"
        self.system_busy_time = 0

        # --- Espacios de observación y acción --------------------------- #
        self._setup_spaces()

        # --- Logger ------------------------------------------------------ #
        self.event_recorder = logging.getLogger(f"{__name__}.{os.getpid()}")
        self.event_recorder.setLevel(logging.INFO)

        # Cache: «¿es nodo final?»
        self._es_node_cache: Dict[Any, bool] = {}
        
        # Variable para datos de operación
        self.last_operation_info = {}
        self.policy_choices = {}

        # Orden FIFO inmutable: simplemente el índice de creación del flujo
        # (traffic_streams ya está en el mismo orden en que se generaron).
        self._fifo_order = list(range(self.stream_count))

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
    # --------------------------------------------------------------------- #    def _default_gcl_info(self):
        return self.GclInfo()

    def _setup_spaces(self):
        # MODIFICAR LA OBSERVACIÓN: Incluir información de múltiples flujos (hasta 5)
        # Para cada flujo: [período_norm, normalized_size, progreso, tiempo_espera_norm, es_seleccionable]
        # Además de las características globales originales
        OBSERVABLE_STREAMS = 5  # Número de flujos que podemos observar a la vez
        STREAM_ATTRIBUTES = 5  # Características por flujo
        GLOBAL_ATTRIBUTES = 4   # Características globales (tiempo, ocupación GCL, etc.)
        
        # NUEVO: Almacenar constantes para uso posterior
        self.MAX_OBSERVABLE_STREAMS = OBSERVABLE_STREAMS
        self.STREAM_ATTRIBUTES = STREAM_ATTRIBUTES
        self.GLOBAL_ATTRIBUTES = GLOBAL_ATTRIBUTES
        
        if self.graph_mode_enabled:
            # Dict observation space for network_structure-based extractors (HATSExtractor)
            vertex_count = len(self.network_structure.nodes())
            connection_count = len(self.network_structure.edges())
            path_limit = max(len(data_stream.path) for data_stream in self.traffic_streams) if self.traffic_streams else 10
            
            self.observation_space = spaces.Dict({
                'attribute_matrix': spaces.Box(
                    low=-1.0, high=1.0, 
                    shape=(vertex_count, 8), dtype=np.float32
                ),
                'connection_properties': spaces.Box(
                    low=0.0, high=1.0, 
                    shape=(connection_count, 4), dtype=np.float32
                ),
                'topology_matrix': spaces.Box(
                    low=0, high=1, 
                    shape=(vertex_count, vertex_count), dtype=np.int32
                ),
                'flow_feature': spaces.Box(
                    low=0.0, high=1.0, 
                    shape=(6,), dtype=np.float32
                ),
                'link_feature': spaces.Box(
                    low=0.0, high=1.0, 
                    shape=(4,), dtype=np.float32
                ),
                'remain_hops': spaces.Box(
                    low=0, high=path_limit, 
                    shape=(path_limit,), dtype=np.int32
                )
            })
        else:
            # Box observation space for standard extractors (AttributeProcessor)
            self.observation_space = spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=(GLOBAL_ATTRIBUTES + OBSERVABLE_STREAMS * STREAM_ATTRIBUTES,), 
                dtype=np.float32
            )

        # ╔═══════════════════════════════════════════════════════════════════════╗
        # ║  ESPACIO DE ACCIÓN (3 DIMENSIONES)                                    ║
        # ║  0. Guard factor [0-4]                                                ║
        # ║  1. Gap mínimo en switch [0-3]                                        ║
        # ║  2. Selección de flujo candidato [0-4]                                ║
        # ╚═══════════════════════════════════════════════════════════════════════╝
        self.action_space = spaces.MultiDiscrete([
            5,                 # Guard factor
            4,                 # Gap mínimo switch
            OBSERVABLE_STREAMS # Selección de flujo
        ])

    # --------------------------------------------------------------------- #
    #  Utilidades de selección de flujo / enlace                            #
    # --------------------------------------------------------------------- #
    def select_next_flow_by_agent(self, stream_choice):
        """
        Permite que el agente RL seleccione el próximo flujo a programar.
        
        Args:
            stream_choice: Índice del flujo seleccionado por el agente (0-4)
        """
        # Usar el índice de flujo seleccionado por el agente si está disponible
        if hasattr(self, 'active_nominees') and self.active_nominees:
            if 0 <= stream_choice < len(self.active_nominees):
                chosen_identifier = self.active_nominees[stream_choice]
                if not self.stream_finished[chosen_identifier]:
                    self.active_stream_id = chosen_identifier
                    return

        # Si la selección directa falla, usar FIFO como fallback
        current_instant = self.simulation_clock
        selected_option = None  # (identifier, waiting_duration)

        for identifier, episode_complete in enumerate(self.stream_finished):
            if episode_complete:
                continue
            advancement = self.stream_advancement[identifier]
            if advancement == 0 or advancement >= len(self.traffic_streams[identifier].path):
                continue
            next_link = self.traffic_streams[identifier].path[advancement]
            if not next_link[0].startswith('S'):
                continue

            upstream_identifier = self.traffic_streams[identifier].path[advancement-1]
            prior_activities = self.connection_activities[self.connection_registry[upstream_identifier]]
            if not prior_activities:
                continue  # el hop previo aún no fue programado
            previous_operation = prior_activities[-1][1]
            waiting_duration = current_instant - previous_operation.reception_time
            if selected_option is None or waiting_duration > selected_option[1]:
                selected_option = (identifier, waiting_duration)

        if selected_option:
            self.active_stream_id = selected_option[0]
            return
                
        # Si no se encontró un flujo adecuado, buscar cualquier flujo no completado
        if self.stream_finished[self.active_stream_id]:
            for identifier, completed in enumerate(self.stream_finished):
                if not completed:
                    self.active_stream_id = identifier
                    return

    def current_flow(self):
        return self.traffic_streams[self.active_stream_id]

    def current_link(self):
        identifier = self.stream_advancement[self.active_stream_id]
        data_stream = self.current_flow()
        return self.connection_registry[data_stream.path[identifier]]

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
        optimal_candidate: tuple[int, int, bool, int] | None = None   # (waiting_duration, -identifier, queued_in_switch, segment_index)
        optimal_index: int | None = None

        current_instant = self.simulation_clock
        for identifier, episode_complete in enumerate(self.stream_finished):
            if episode_complete:
                continue
            segment_index = self.stream_advancement[identifier]
            if segment_index >= len(self.traffic_streams[identifier].path):
                continue

            # --- tiempo desde que el paquete está "listo" ---
            if segment_index == 0:
                preparation_time = 0                          # nunca se ha transmitido
                queued_in_switch = False
            else:
                upstream_identifier = self.traffic_streams[identifier].path[segment_index - 1]
                prior_activities = self.connection_activities.get(self.connection_registry[upstream_identifier], [])
                if not prior_activities:
                    continue        # hop previo aún no programado ⇒ no listo
                
                previous_operation = prior_activities[-1][1]
                preparation_time = previous_operation.reception_time
                
                # Determinar si el paquete está esperando en un switch
                # Si el destino del enlace anterior es un switch y el origen del siguiente enlace
                # coincide con ese switch, entonces el paquete está esperando en un switch
                target_endpoint = upstream_identifier[1] if isinstance(upstream_identifier, tuple) else upstream_identifier.split('-')[1]
                queued_in_switch = target_endpoint.startswith('S') and not target_endpoint.startswith('SRV')

            queue_time = current_instant - preparation_time
            priority_order = -identifier               # menor índice ⇒ más antiguo
            
            # Prioridad: 1) paquetes en switch, 2) tiempo de espera, 3) índice más bajo
            candidate = (int(queued_in_switch), queue_time, priority_order, segment_index)

            if (optimal_candidate is None) or (candidate > optimal_candidate):
                optimal_candidate = candidate
                optimal_index = identifier

        return optimal_index

    # --------------------------------------------------------------------- #
    #  Reinicio                                                             #
    # --------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Si es necesario incrementar la complejidad - MÁS CONSERVADOR
        if self.adaptive_learning and self.streak_count >= 5:  # Aumentar de 3 a 5 éxitos
            if self.increase_complexity():
                # Calcular número de flujos según complejidad actual
                stream_count = int(self.complete_stream_count * self.difficulty_level)
                stream_count = max(5, min(self.complete_stream_count, stream_count))
                
                if self.complete_stream_set:
                    # Usar un subset consistente de los flujos originales
                    stream_collection = self.complete_stream_set[:stream_count]
                    infrastructure = Network(self.network_topology, stream_collection)
                    
                    # Actualizar las estructuras con la nueva red
                    self.network_structure = infrastructure.network_structure
                    self.traffic_streams = list(infrastructure.traffic_streams)
                    self.line_graph, self.connection_registry = (
                        infrastructure.line_graph,
                        infrastructure.links_dict,
                    )
                    self.stream_count = len(self.traffic_streams)
                    
                    self.event_recorder.info(f"Curriculum: Incrementando a {stream_count}/{self.complete_stream_count} flujos (complejidad: {self.difficulty_level:.2f})")
                
            self.streak_count = 0

        self.simulation_clock = 0
        self.stream_advancement = [0] * self.stream_count
        self.stream_finished = [False] * self.stream_count
        self.initial_transmission = [None] * self.stream_count
        self.active_stream_id = 0

        self.connection_activities.clear()
        self.connection_schedules = {}   # reiniciar placeholder
        self.provisional_activities.clear()
        self._es_node_cache.clear()
        self.connection_free_time.clear()
        self.node_free_time.clear()
        self.node_last_reception.clear()    # NUEVO: Limpiar tiempos de llegada
        self.system_busy_time = 0
        self.last_packet_start = -Net.PACKET_GAP_EXTRA

        # Limpiar métricas de latencia para el nuevo episodio
        self._latency_samples.clear()

        # ⏱️  reiniciar latencias e2e acumuladas
        self._flow_latencies.clear()

        return self._get_observation(), {}    # --------------------------------------------------------------------- #
    #  Observación                                                          #
    # --------------------------------------------------------------------- #
    def _get_observation(self):
        if self.graph_mode_enabled:
            return self._get_graph_observation()
        else:
            return self._get_vector_observation()
    
    def _get_vector_observation(self):
        """Get traditional Box observation for standard extractors"""
        # Creamos una observación que contenga información sobre múltiples flujos
        # La observación tendrá: [características_globales, características_flujo1, características_flujo2, ...]
        
        # 1. Características globales de la red
        normalized_time = self.simulation_clock / 10000  # Normalizar tiempo global
        
        # Ya no hay GCL dinámico → utilización 0 siempre
        schedule_utilization = 0.0
        
        # Calcular porcentaje de flujos completados
        finish_ratio = sum(self.stream_finished) / self.stream_count
        
        # Nivel de curriculum
        learning_progress = self.difficulty_level
        
        # Vector de características globales
        global_features = [normalized_time, schedule_utilization, finish_ratio, learning_progress]
        
        # 2. Obtener una lista de flujos nominees para programar
        nominees = []
        for identifier, completed in enumerate(self.stream_finished):
            if not completed:
                data_stream = self.traffic_streams[identifier]
                segment_index = self.stream_advancement[identifier]
                
                # Verificar que el flujo tiene un hop válido para programar
                if segment_index >= len(data_stream.path):
                    continue
                    
                # Calcular cuánto tiempo ha estado esperando (si aplica)
                waiting_duration = 0
                if segment_index > 0:
                    upstream_identifier = data_stream.path[segment_index-1]
                    prior_activities = self.connection_activities.get(self.connection_registry[upstream_identifier], [])
                    if prior_activities:
                        previous_operation = prior_activities[-1][1]
                        waiting_duration = max(0, self.simulation_clock - previous_operation.reception_time)
                
                # Normalizar valores - convertir a características significativas 
                normalized_period = data_stream.period / 10000  # Períodos más cortos → valores más pequeños
                normalized_size = data_stream.payload / Net.MTU  # Payloads más pequeños → valores más pequeños
                path_position = self.stream_advancement[identifier] / len(data_stream.path)
                normalized_wait = min(waiting_duration / data_stream.period, 1.0)  # Normalizar por periodo
                
                # Calcular urgencia basada en plazo próximo
                # Cuanto menor sea el tiempo hasta el deadline, mayor urgencia (1.0 = muy urgente)
                time_budget = (data_stream.period - (self.simulation_clock % data_stream.period)) / data_stream.period
                priority_score = 1.0 - time_budget  # 0.0 = acaba de empezar, 1.0 = casi vencido
                
                # Log del flujo candidato con sus características para depuración
                self.event_recorder.debug(f"Candidato: {data_stream.flow_id}, Period: {normalized_period:.2f}, Payload: {normalized_size:.2f}, Wait: {normalized_wait:.2f}, Urgency: {priority_score:.2f}")
                
                # Añadir a nominees con sus características
                nominees.append((identifier, [normalized_period, normalized_size, path_position, normalized_wait, priority_score]))
        
        # 3. Si no hay nominees, devolver una observación con ceros
        if not nominees:
            if self.graph_mode_enabled:
                return self._get_empty_graph_observation()
            else:
                return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # 4. Ordenar nominees de forma significativa para el agente
        # Considerar múltiples factores: tiempo espera (FIFO), urgencia, tamaño payload
        nominees = sorted(nominees, key=lambda feature_tensor: (
            feature_tensor[1][3],  # Tiempo de espera (mayor primero - FIFO)
            feature_tensor[1][4],  # Urgencia (mayor primero)
            -feature_tensor[1][1]  # Payload (menor primero - más rápido)
        ), reverse=True)
        
        # Asegurar que tenemos exactamente MAX_OBSERVABLE_STREAMS nominees
        if len(nominees) > self.MAX_OBSERVABLE_STREAMS:
            nominees = nominees[:self.MAX_OBSERVABLE_STREAMS]
        
        while len(nominees) < self.MAX_OBSERVABLE_STREAMS:
            nominees.append((-1, [0.0, 0.0, 0.0, 0.0, 0.0]))
        
        # 5. Actualizar los índices de flujos nominees para recuperarlos después
        self.active_nominees = [identifier for identifier, _ in nominees if identifier >= 0]
        
        # Registro para depuración
        if self.active_nominees:
            self.event_recorder.debug(f"Candidatos ordenados: {[self.traffic_streams[identifier].flow_id for identifier in self.active_nominees]}")
        
        # 6. Construir la observación completa concatenando características
        current_state = np.array(global_features + [feature_vector for _, feats in nominees for feature_vector in feats], dtype=np.float32)
        
        return current_state
    
    def _get_graph_observation(self):
        """Get Dict observation for network_structure-based extractors (HATSExtractor)"""
        try:
            return self._neighbors_features()
        except Exception as e:
            self.event_recorder.warning(f"Error getting network_structure observation: {e}. Falling back to empty observation.")
            return self._get_empty_graph_observation()
    
    def _get_empty_graph_observation(self):
        """Return empty Dict observation when no candidates available"""
        vertex_count = len(self.network_structure.nodes())
        connection_count = len(self.network_structure.edges())
        path_limit = max(len(data_stream.path) for data_stream in self.traffic_streams) if self.traffic_streams else 10
        
        return {
            'attribute_matrix': np.zeros((vertex_count, 8), dtype=np.float32),
            'connection_properties': np.zeros((connection_count, 4), dtype=np.float32),
            'topology_matrix': np.array([[1 if self.network_structure.has_edge(iterator, j) else 0 
                                         for j in range(vertex_count)] 
                                        for iterator in range(vertex_count)], dtype=np.int32),
            'flow_feature': np.zeros((6,), dtype=np.float32),
            'link_feature': np.zeros((4,), dtype=np.float32),
            'remain_hops': np.zeros((path_limit,), dtype=np.int32)
        }
    
    def _neighbors_features(self):
        """
        Build network_structure-based observation for HATSExtractor.
        Returns Dict with network_structure structure and attribute_vector.
        """
        if not self.active_nominees or not self.traffic_streams:
            return self._get_empty_graph_observation()
        
        # Get current data_stream being scheduled
        flow_idx = self.active_nominees[0] if self.active_nominees else 0
        if flow_idx >= len(self.traffic_streams):
            return self._get_empty_graph_observation()
            
        data_stream = self.traffic_streams[flow_idx]
        segment_index = self.stream_advancement[flow_idx]
        
        if segment_index >= len(data_stream.path):
            return self._get_empty_graph_observation()
        
        # Graph structure
        vertex_count = len(self.network_structure.nodes())
        connection_count = len(self.network_structure.edges())
        
        # Node attribute_vector matrix (vertex_count feature_tensor 8)
        attribute_matrix = np.zeros((vertex_count, 8), dtype=np.float32)
        vertex_catalog = list(self.network_structure.nodes())
        
        for iterator, node in enumerate(vertex_catalog):
            # Basic node attribute_vector
            attribute_matrix[iterator, 0] = len(list(self.network_structure.neighbors(node))) / vertex_count  # Degree normalized
            attribute_matrix[iterator, 1] = 1.0 if node in data_stream.path else 0.0  # In current data_stream path
            attribute_matrix[iterator, 2] = 1.0 if iterator == segment_index and node in data_stream.path else 0.0  # Current hop
            
            # Traffic load attribute_vector
            node_load = sum(1 for f_idx, flow_generator in enumerate(self.traffic_streams) 
                           if not self.stream_finished[f_idx] and node in flow_generator.path)
            attribute_matrix[iterator, 3] = node_load / len(self.traffic_streams)  # Normalized load
            
            # Timing attribute_vector
            attribute_matrix[iterator, 4] = self.simulation_clock / 10000.0  # Normalized time
            attribute_matrix[iterator, 5] = self.difficulty_level
            
            # Buffer/congestion indicators (simplified)
            attribute_matrix[iterator, 6] = 0.5  # Placeholder for buffer utilization
            attribute_matrix[iterator, 7] = 0.0  # Placeholder for congestion
        
        # Edge attribute_vector matrix (connection_count feature_tensor 4)  
        connection_properties = np.zeros((connection_count, 4), dtype=np.float32)
        connection_catalog = list(self.network_structure.edges())
        
        for iterator, (source_node, destination_node) in enumerate(connection_catalog):
            # Edge load and utilization
            link_utilization = sum(1 for f_idx, flow_generator in enumerate(self.traffic_streams)
                           if not self.stream_finished[f_idx] and 
                           any(flow_generator.path[j] == source_node and flow_generator.path[j+1] == destination_node 
                               for j in range(len(flow_generator.path)-1)))
            connection_properties[iterator, 0] = link_utilization / len(self.traffic_streams)  # Load
            connection_properties[iterator, 1] = 1.0 if (source_node, destination_node) in [(data_stream.path[j], data_stream.path[j+1]) 
                                                        for j in range(len(data_stream.path)-1)] else 0.0  # In current path
            connection_properties[iterator, 2] = 0.5  # Bandwidth utilization (placeholder)
            connection_properties[iterator, 3] = 0.0  # Latency (placeholder)
        
        # Adjacency matrix
        topology_matrix = np.array([[1 if self.network_structure.has_edge(iterator, j) else 0 
                                     for j in range(vertex_count)] 
                                    for iterator in range(vertex_count)], dtype=np.int32)
        
        # Flow attribute_vector (6 attribute_vector)
        flow_feature = np.array([
            data_stream.period / 10000.0,  # Normalized period
            data_stream.payload / Net.MTU,  # Normalized payload
            segment_index / len(data_stream.path),  # Progress
            (self.simulation_clock % data_stream.period) / data_stream.period,  # Phase in period
            len(data_stream.path) / vertex_count,  # Path length normalized
            sum(self.stream_finished) / len(self.traffic_streams)  # Completion rate
        ], dtype=np.float32)
        
        # Link attribute_vector (4 attribute_vector) - current network_connection being scheduled
        if segment_index < len(data_stream.path) - 1:
            current_link = (data_stream.path[segment_index], data_stream.path[segment_index + 1])
            link_feature = np.array([
                1.0,  # Link is active
                connection_properties[connection_catalog.index(current_link), 0] if current_link in connection_catalog else 0.0,  # Load
                0.5,  # Bandwidth (placeholder)
                self.simulation_clock / 10000.0  # Current time
            ], dtype=np.float32)
        else:
            link_feature = np.zeros((4,), dtype=np.float32)
        
        # Remaining path_segments
        path_limit = max(len(flow_generator.path) for flow_generator in self.traffic_streams) if self.traffic_streams else 10
        remain_hops = np.zeros((path_limit,), dtype=np.int32)
        pending_segments = len(data_stream.path) - segment_index
        if pending_segments > 0:
            remain_hops[:min(pending_segments, path_limit)] = 1
        
        return {
            'attribute_matrix': attribute_matrix,
            'connection_properties': connection_properties, 
            'topology_matrix': topology_matrix,
            'flow_feature': flow_feature,
            'link_feature': link_feature,
            'remain_hops': remain_hops
        }

    # ------------------------------------------------------------------ #
    #  Máscaras de acción                                                #
    # ------------------------------------------------------------------ #
    def permitted_actions(self):
        """
        Genera máscaras para el espacio de acción MultiDiscreto.
        """
        # Calcular tamaño total de la máscara sumando todas las dimensiones
        constraint_count = sum(self.action_space.nvec)
        
        # Crear una máscara plana donde todas las acciones están permitidas (0 = permitida)
        validity_vector = np.zeros(constraint_count, dtype=np.int8)
        
        try:
            if self.active_stream_id < len(self.traffic_streams) and not self.stream_finished[self.active_stream_id]:
                data_stream = self.current_flow()
                segment_index = self.stream_advancement[self.active_stream_id]
                
                if segment_index < len(data_stream.path):
                    network_connection = self.connection_registry[data_stream.path[segment_index]]
                    
                    # Si es el primer hop, los factores de guard time altos podrían desperdiciarse
                    if segment_index == 0:
                        # Índice para guard time alto
                        time_adjustment = 0  # No time_adjustment needed anymore
                        validity_vector[time_adjustment + 3] = 1  # Desactivar valores 3 y 4 (factores altos)
                        validity_vector[time_adjustment + 4] = 1
        except Exception as e:
            self.event_recorder.warning(f"Error generando máscaras de acción: {e}")
            
        return validity_vector

    def action_masks(self):
        """
        Genera máscaras para el espacio de acción MultiDiscreto.
        Método requerido por MaskablePPO.
        """
        return self.permitted_actions()

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

        source_node = link_id[0] if isinstance(link_id, tuple) else link_id.split('-')[0]

        # a) Metadata del grafo (más robusta)
        if self.network_structure.nodes.get(source_node, {}).get("node_type") == "ES":
            self._es_node_cache[link_id] = True
            return True

        # b) Convención de nombres
        is_es = source_node.startswith(("E", "C", "SRV"))
        self._es_node_cache[link_id] = is_es
        return is_es

    #  ⛔  Retirado.  La agrupación de GCL dejó de ser necesaria.

    def _check_valid_link(self, network_connection, operation):
        return check_valid_link(network_connection, operation, self.current_flow(), self.connection_activities)

    def _check_temp_operations(self):
        return check_temp_operations(self.provisional_activities, self.connection_activities, self.current_flow())

    def _find_next_event_time(self, system_clock):
        """Encuentra el siguiente tiempo de evento programado después de system_clock"""
        return find_next_event_time(self.connection_free_time, self.node_free_time, system_clock)

    # Método obsoleto: generación dinámica de GCL eliminada.
    # Se mantiene para compatibilidad pero no hace nada y devuelve False.
    def add_gating_with_grouping(self, *args, **optional_params):
        return False

    def increase_complexity(self):
        """Incrementa el nivel de dificultad del entorno de forma MÁS CONSERVADORA"""
        if not self.adaptive_learning or self.difficulty_level >= 1.0:
            return False
            
        # Incrementar complejidad más lentamente para evitar colapso
        # Reducir incremento de 0.05 (5%) a 0.02 (2%)
        conservative_increment = self.advancement_rate * 0.4  # 40% del rate original
        self.difficulty_level = min(1.0, self.difficulty_level + conservative_increment)
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

