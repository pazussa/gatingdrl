import logging
import math
import networkx as nx
import numpy as np
import random
import typing

# Conjunto de períodos disponibles
PERIOD_SET = [2000, 4000, 8000]

class Net:
    # Configuración de red y temporizaciones
    MTU = 1518
    # ➡️ Nueva: separación mínima entre llegadas sucesivas al mismo switch (µs)
    SWITCH_GAP_MIN = 1

    # ➡️ **NUEVO** – parámetros para la separación probabilística entre creaciones
    #     de paquetes (sólo se evalúa en el PRIMER hop de cada flujo)
    #
    #     • PACKET_GAP_MODE:
    #         ─ fixed        → separación constante = PACKET_GAP_EXTRA
    #         ─ uniform      → U[min,max]  definido en PACKET_GAP_UNIFORM
    #         ─ exponential  → Exp(λ)  con media = PACKET_GAP_EXTRA
    #         ─ gaussian     → N(μ,σ²)  definido en PACKET_GAP_GAUSS
    #         ─ pareto       → Pareto(α,xm) definido en PACKET_GAP_PARETO
    #     • Los valores se configuran desde CLI en ui/train.py / ui/test.py.
    #
    PACKET_GAP_MODE    = "fixed"
    PACKET_GAP_EXTRA   = 0
    PACKET_GAP_UNIFORM = (0, 0)       # sólo usado si mode == uniform
    # Nuevos contenedores para gaussian y pareto
    PACKET_GAP_GAUSS   = (0.0, 1.0)   # μ, σ
    PACKET_GAP_PARETO  = (2.0, 1.0)   # α, x_m

    # ------------------------------------------------------------------
    #  Configuración centralizada desde CLI
    # ------------------------------------------------------------------
    @staticmethod
    def set_gap_distribution(dist: str, params: list[float] | None = None):
        """
        Ajusta la distribución global del gap entre creaciones de paquetes.

        ─ dist = fixed        → params = [gap]
        ─ dist = uniform      → params = [min, max]
        ─ dist = exponential  → params = [mean]
        ─ dist = gaussian     → params = [mean, sigma]
        ─ dist = pareto       → params = [alpha, x_m]
        """
        params = params or []
        Net.PACKET_GAP_MODE = dist

        if dist == "fixed":
            Net.PACKET_GAP_EXTRA = int(params[0]) if params else 0

        elif dist == "uniform":
            assert len(params) == 2, "--dist-params necesita MIN MAX"
            Net.PACKET_GAP_UNIFORM = (int(params[0]), int(params[1]))

        elif dist == "exponential":
            Net.PACKET_GAP_EXTRA = int(params[0]) if params else 1

        elif dist == "gaussian":
            assert len(params) >= 2, "--dist-params necesita MU SIGMA"
            Net.PACKET_GAP_GAUSS = (float(params[0]), float(params[1]))

        elif dist == "pareto":
            assert len(params) == 2, "--dist-params necesita ALPHA XM"
            Net.PACKET_GAP_PARETO = (float(params[0]), float(params[1]))

        else:
            raise ValueError(f"Distribución desconocida: {dist}")

    @staticmethod
    def sample_packet_gap() -> int:
        """Devuelve la separación (µs) según la configuración global."""
        if Net.PACKET_GAP_MODE == "fixed":
            return Net.PACKET_GAP_EXTRA
        elif Net.PACKET_GAP_MODE == "uniform":
            lo, hi = Net.PACKET_GAP_UNIFORM
            return random.randint(lo, hi) if hi > lo else lo
        elif Net.PACKET_GAP_MODE == "exponential":
            μ = max(Net.PACKET_GAP_EXTRA, 1)      # evita λ=0
            return int(random.expovariate(1/μ))
        elif Net.PACKET_GAP_MODE == "gaussian":
            μ, σ = Net.PACKET_GAP_GAUSS
            # recortamos a ≥0 y redondeamos
            return max(0, int(random.gauss(μ, σ)))
        elif Net.PACKET_GAP_MODE == "pareto":
            α, xm = Net.PACKET_GAP_PARETO
            # random.paretovariate devuelve xm*(1-U)^(-1/α) con xm=1
            return int(random.paretovariate(α)*xm)
        else:
            raise ValueError(f"Modo de gap desconocido: {Net.PACKET_GAP_MODE}")
  
    DELAY_PROC_RX = 1  # Nuevo: tiempo de procesamiento de recepción
    SYNC_ERROR = 0  # Sin incertidumbre: relojes perfectamente alineados
    DELAY_PROP = 1
    GCL_CYCLE_MAX = 128000
    # Longitud máxima típica de un Gate Control List en hardware TSN (open/close)
    GCL_LENGTH_MAX = 256

class Link:
    embedding_length = 3

    def __init__(self, link_id, link_rate):
        self.link_id = link_id
        self.gcl_capacity = Net.GCL_LENGTH_MAX
        self.link_rate = link_rate

    def __hash__(self):
        return hash(self.link_id)

    def __eq__(self, other):
        return isinstance(other, Link) and self.link_id == other.link_id

    def __repr__(self):
        return f"Link{self.link_id}"

    def interference_time(self):
        return self.transmission_time(Net.MTU)

    def transmission_time(self, payload):
        # 12B para IFG, 8B para preámbulo
        return math.ceil((payload + 12 + 8) * 8 / self.link_rate)

class Flow:
    def __init__(self, flow_id, src_id, dst_id, path, period=2000, payload=100, e2e_delay=None):
        self.flow_id = flow_id
        self.src_id = src_id
        self.dst_id = dst_id
        self.path = path
        # Validar período
        assert period in PERIOD_SET, f"Período inválido {period}"
        self.period = period
        self.payload = payload
        self.e2e_delay = period if e2e_delay is None else e2e_delay
        # Jitter completamente eliminado

    def __hash__(self):
        return hash(self.flow_id)

    def __eq__(self, other):
        return isinstance(other, Flow) and self.flow_id == other.flow_id

    def __repr__(self):
        return f"Flow(id='{self.flow_id}', src='{self.src_id}', dst='{self.dst_id}', period={self.period})"

# Funciones para generar topologías simplificadas
def generate_graph(topo, link_rate=100):
    """Genera diferentes topologías de grafo según el tipo especificado"""
    if topo == "SIMPLE":
        return generate_simple_topology(link_rate)
    elif topo == "UNIDIR":
        return generate_unidirectional_topology(link_rate)
    else:
        raise ValueError(f"Topología desconocida: {topo}. Solo se soportan 'SIMPLE' y 'UNIDIR'")

def generate_simple_topology(link_rate: int = 100):
    """Genera una topología S-E-S mínima con identificadores **tuple**.

    Mantener el mismo tipo de ``link_id`` en toda la base de código
    evita las múltiples llamadas a ``split('-')`` y simplifica las
    comprobaciones de si un nodo es *switch* o *end-station*.
    """
    graph = nx.DiGraph()
    graph.add_node("S1", node_type="SW")

    for node in ("C1", "C2", "SRV1"):
        graph.add_node(node, node_type="ES")
        # Enlaces bidireccionales (src, dst)
        graph.add_edge(node, "S1", link_id=(node, "S1"), link_rate=link_rate)
        graph.add_edge("S1", node, link_id=("S1", node), link_rate=link_rate)

    return graph

def generate_unidirectional_topology(link_rate=100):
    """Genera una topología unidireccional donde los datos fluyen solo de clientes a servidor"""
    graph = nx.DiGraph()
    # Crear nodos
    graph.add_node("S1", node_type="SW")
    for node in ["C1", "C2", "SRV1"]:
        graph.add_node(node, node_type="ES")
    # Crear enlaces unidireccionales
    for client in ["C1", "C2"]:
        graph.add_edge(client, "S1", link_id=(client, "S1"), link_rate=link_rate)
    graph.add_edge("S1", "SRV1", link_id=("S1", "SRV1"), link_rate=link_rate)
    return graph

def _transform_line_graph(graph):
    """Transforma un grafo en su línea de grafo y crea diccionario de enlaces"""
    line_graph = nx.line_graph(graph)
    links_dict = {node: Link(node, graph.edges[node]['link_rate']) for node in line_graph.nodes}
    return line_graph, links_dict

class Network:
    def __init__(self, graph, flows):
        self.graph = graph
        self.flows = flows
        # Construir línea de grafo y diccionario de enlaces
        self.line_graph, self.links_dict = _transform_line_graph(graph)

    def disable_gcl(self, num_nodes):
        """Deshabilita la capacidad GCL para un número específico de nodos"""
        list_nodes = random.sample(list(self.graph.nodes), num_nodes)
        list_links = []
        for node in list_nodes:
            list_links.extend([link for link in self.links_dict.values() if node == link.link_id[0]])
        for link in list_links:
            link.gcl_capacity = 0

    def set_gcl(self, num_gcl):
        """Establece la capacidad GCL para todos los enlaces"""
        for link in self.links_dict.values():
            link.gcl_capacity = num_gcl

class FlowGenerator:
    def __init__(self, graph, seed=None, period_set=None, min_payload=1500, max_payload=1518): # Añadir min/max payload
        self.graph = graph
        # Inicializar semilla para reproducibilidad
        if seed is not None:
            random.seed(seed)
        # Inicializar conjunto de períodos usando el global PERIOD_SET como default
        self.period_set = period_set if period_set is not None else PERIOD_SET
        for period in self.period_set:
            assert isinstance(period, int) and period > 0
        # Jitters eliminado
        # Identificar nodos finales
        self.es_nodes = [n for n, d in graph.nodes(data=True) if d['node_type'] == 'ES']
        self.num_generated_flows = 0
        # Guardar rango de payload
        self.min_payload = min_payload
        self.max_payload = max_payload
        assert self.min_payload <= self.max_payload, "min_payload debe ser <= max_payload"

    def _generate_flow(self):
        """Genera un único flujo aleatorio"""
        # Seleccionar dos nodos aleatorios
        src_id, dst_id = random.sample(self.es_nodes, 2)
        # Calcular ruta más corta
        path = nx.shortest_path(self.graph, src_id, dst_id)
        path = [(path[i], path[i+1]) for i in range(len(path)-1)]
        # Seleccionar período aleatorio
        period = random.choice(self.period_set)

        # Crear flujo con payload aleatorio dentro del rango especificado
        flow = Flow(
            f"F{self.num_generated_flows}", src_id, dst_id, path,
            payload=random.randint(self.min_payload, self.max_payload), # Usar el rango
            period=period
        )
        self.num_generated_flows += 1
        return flow

    def __call__(self, num_flows=1):
        """Genera un conjunto de flujos aleatorios"""
        return [self._generate_flow() for _ in range(num_flows)]

class UniDirectionalFlowGenerator(FlowGenerator):
    def _generate_flow(self):
        """Genera un flujo unidireccional desde clientes a servidores"""
        # Identificar nodos cliente y servidor
        client_nodes = [n for n in self.es_nodes if n.startswith('C')]
        server_nodes = [n for n in self.es_nodes if n.startswith('SRV')]
        
        # Si no hay clientes o servidores, usar generación normal
        if not client_nodes or not server_nodes:
            return super()._generate_flow()
            
        # Seleccionar origen y destino
        src_id = random.choice(client_nodes)
        dst_id = random.choice(server_nodes)
        
        # Calcular ruta
        path = nx.shortest_path(self.graph, src_id, dst_id)
        path = [(path[i], path[i+1]) for i in range(len(path)-1)]
        
        # Seleccionar parámetros
        period = random.choice(self.period_set)
        
        # Crear flujo con payload aleatorio dentro del rango especificado
        flow = Flow(
            f"F{self.num_generated_flows}", src_id, dst_id, path,
            payload=random.randint(self.min_payload, self.max_payload), # Usar el rango
            period=period
        )
        self.num_generated_flows += 1
        return flow

def generate_flows(graph, num_flows=50, seed=None, period_set=PERIOD_SET, unidirectional=False, min_payload=1500, max_payload=1518): # Añadir min/max payload
    """Función para generar flujos con configuración específica"""
    generator_class = UniDirectionalFlowGenerator if unidirectional else FlowGenerator
    # Pasar el rango de payload al constructor del generador
    generator = generator_class(graph, seed, period_set, min_payload, max_payload)
    return generator(num_flows)
