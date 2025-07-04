import logging
import math
import networkx as nx
import numpy as np
import random
import typing


# Conjunto de períodos disponibles
PERIOD_SET = [1000]

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
    """
    Devuelve el grafo dirigido para la topología *topo*.

    · SIMPLE              – mínimo S-E-S bidireccional  
    · UNIDIR, UNIDIR2-8   – variantes unidireccionales (solo ES → SRV1)
    """
    # Bidireccional de referencia
    if topo == "SIMPLE":
        return generate_simple_topology(link_rate)

    # Familia unidireccional  (cualquier nombre que empiece por UNIDIR…)
    topo = topo.upper()
    if topo.startswith("UNIDIR"):
        mapping = {
            "UNIDIR":  generate_unidirectional_topology,
            "UNIDIR2": generate_unidirectional_chain_topology2,
            "UNIDIR3": generate_unidirectional_chain_topology3,
            "UNIDIR4": generate_unidirectional_topology4,
            "UNIDIR5": generate_unidirectional_topology5,
            "UNIDIR6": generate_unidirectional_topology6,
            "UNIDIR7": generate_unidirectional_topology7,
            "UNIDIR8": generate_unidirectional_topology8,
            "UNIDIR9": generate_unidirectional_topology9,
            "UNIDIR10": generate_unidirectional_topology10,
            "UNIDIR11": generate_unidirectional_topology11,
            "UNIDIR12": generate_unidirectional_topology12,
            "UNIDIR13": generate_unidirectional_topology13,
            "UNIDIR14": generate_unidirectional_topology14,
            "UNIDIR15": generate_unidirectional_topology15,
        }
        try:
            return mapping[topo](link_rate)
        except KeyError:
            raise ValueError(
                f"Topología desconocida: {topo}. "
                f"Opciones válidas: {', '.join(mapping.keys())}"
            )

    raise ValueError(f"Topología desconocida: {topo}")

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

def generate_unidirectional_chain_topology(num_switches: int = 2,
                                           link_rate: int = 100):
    """
    Genera una topología en cadena unidireccional con el número especificado
    de switches (Edge → Aggregation) y un servidor final.

    Los nodos cliente se asignan automáticamente según la convención
    C1, C2, ..., Cn.
    """
    g = nx.DiGraph()

    # Agregar nodos de switch
    for i in range(1, num_switches + 1):
        g.add_node(f"S{i}", node_type="SW")

    # Nodo servidor
    g.add_node("SRV1", node_type="ES")

    # Clientes y enlaces cliente → switch
    for i in range(1, num_switches + 1):
        cli = f"C{i}"
        g.add_node(cli, node_type="ES")                  # ← ¡aquí el fix!
        g.add_edge(cli, f"S{i}", link_id=(cli, f"S{i}"), link_rate=link_rate)

    # Enlaces entre switches
    for i in range(1, num_switches):
        g.add_edge(f"S{i}", f"S{i+1}", link_id=(f"S{i}", f"S{i+1}"), link_rate=link_rate)

    # Enlace del último switch al servidor
    g.add_edge(f"S{num_switches}", "SRV1", link_id=(f"S{num_switches}", "SRV1"), link_rate=link_rate)

    return g

# Helper functions to maintain backward compatibility
def generate_unidirectional_chain_topology2(link_rate=100):
    return generate_unidirectional_chain_topology(2, link_rate)

def generate_unidirectional_chain_topology3(link_rate=100):
    return generate_unidirectional_chain_topology(3, link_rate)

# ╔═══════════════════════════════════════════════════════════════════╗
# ║  NUEVA  TOPOLOGÍA  –  UNIDIR4                                     ║
# ║  "agregación 1-nivel":                                            ║
# ║        C1,C2,C3,C4 → S1 → S2 → SRV1                               ║
# ╚═══════════════════════════════════════════════════════════════════╝
def generate_unidirectional_topology4(link_rate=100):
    """
    Topología de dos switches en cascada (Edge → Aggregation) con
    cuatro clientes y un servidor.  Todo el tráfico fluye **hacia SRV1**.
    """
    g = nx.DiGraph()

    # switches
    g.add_node("S1", node_type="SW")
    g.add_node("S2", node_type="SW")

    # end-stations
    for c in ["C1", "C2", "C3", "C4", "SRV1"]:
        g.add_node(c, node_type="ES")

    # enlaces cliente → edge-switch
    for c in ["C1", "C2", "C3", "C4"]:
        g.add_edge(c, "S1", link_id=(c, "S1"), link_rate=link_rate)

    # backbone unidireccional
    g.add_edge("S1", "S2",     link_id=("S1", "S2"),     link_rate=link_rate)
    g.add_edge("S2", "SRV1",   link_id=("S2", "SRV1"),   link_rate=link_rate)

    return g

# ╔═══════════════════════════════════════════════════════════════════╗
# ║  NUEVA  TOPOLOGÍA  –  UNIDIR5                                     ║
# ║  "anillo parcial / tres switches":                                ║
# ║   (C1,C2)→S1 → S2 → S3 → SRV1                                     ║
# ║   (C3,C4)→S2                                                     ║
# ║   (C5,C6)→S3                                                     ║
# ╚═══════════════════════════════════════════════════════════════════╝
def generate_unidirectional_topology5(link_rate=100):
    """
    Tres switches en línea (S1→S2→S3) simulando el *spine* de un pequeño
    anillo industrial.  Seis clientes repartidos, todos dirigidos a SRV1.
    """
    g = nx.DiGraph()

    # switches
    for sw in ["S1", "S2", "S3"]:
        g.add_node(sw, node_type="SW")

    # end-stations
    for es in ["C1", "C2", "C3", "C4", "C5", "C6", "SRV1"]:
        g.add_node(es, node_type="ES")

    # clientes → switches locales
    for c in ["C1", "C2"]:
        g.add_edge(c, "S1", link_id=(c, "S1"), link_rate=link_rate)
    for c in ["C3", "C4"]:
        g.add_edge(c, "S2", link_id=(c, "S2"), link_rate=link_rate)
    for c in ["C5", "C6"]:
        g.add_edge(c, "S3", link_id=(c, "S3"), link_rate=link_rate)

    # backbone lineal
    g.add_edge("S1", "S2",   link_id=("S1", "S2"),   link_rate=link_rate)
    g.add_edge("S2", "S3",   link_id=("S2", "S3"),   link_rate=link_rate)
    g.add_edge("S3", "SRV1", link_id=("S3", "SRV1"), link_rate=link_rate)

    return g

# ────────────────────────────────────────────────────────────────────
#  NUEVAS TOPOLOGÍAS UNIDIR 6-8
# ────────────────────────────────────────────────────────────────────

def _add_es(graph, node_list, sw_name, link_rate):
    """Helper: añade ES→switch unidireccional."""
    for n in node_list:
        graph.add_node(n, node_type="ES")
        graph.add_edge(n, sw_name, link_id=(n, sw_name), link_rate=link_rate)


def generate_unidirectional_topology6(link_rate=100):
    """
    UNIDIR6 – 'ring line' de 3 switches  
    C1,C2→S1 → S2 → SRV1  
    C3,C4 conectan a S2; C5,C6 a S3.
    """
    g = nx.DiGraph()
    # Switches
    for s in ("S1", "S2", "S3"):
        g.add_node(s, node_type="SW")
    # Server
    g.add_node("SRV1", node_type="ES")

    # End-stations
    _add_es(g, ("C1", "C2"), "S1", link_rate)
    _add_es(g, ("C3", "C4"), "S2", link_rate)
    _add_es(g, ("C5", "C6"), "S3", link_rate)

    # Backbone line
    g.add_edge("S1", "S2", link_id=("S1", "S2"), link_rate=link_rate)
    g.add_edge("S2", "S3", link_id=("S2", "S3"), link_rate=link_rate)
    g.add_edge("S3", "SRV1", link_id=("S3", "SRV1"), link_rate=link_rate)
    return g


def generate_unidirectional_topology7(link_rate=100):
    """
    UNIDIR7 – 'star-of-switches'  
    Tres leaf-switches (S1-S3) con 2 clientes cada uno → core S0 → SRV1.
    """
    g = nx.DiGraph()
    g.add_node("S0", node_type="SW")   # Core
    for s in ("S1", "S2", "S3"):
        g.add_node(s, node_type="SW")
    g.add_node("SRV1", node_type="ES")

    _add_es(g, ("C1", "C2"), "S1", link_rate)
    _add_es(g, ("C3", "C4"), "S2", link_rate)
    _add_es(g, ("C5", "C6"), "S3", link_rate)

    for leaf in ("S1", "S2", "S3"):
        g.add_edge(leaf, "S0", link_id=(leaf, "S0"), link_rate=link_rate)
    g.add_edge("S0", "SRV1", link_id=("S0", "SRV1"), link_rate=link_rate)
    return g


def generate_unidirectional_topology8(link_rate=100):
    """
    UNIDIR8 – spine/leaf simplificado (2 niveles)  
    4 leafs con 2 clientes cada uno → agg S5 → core S6 → SRV1.
    """
    g = nx.DiGraph()
    # Switches
    for s in ("S1", "S2", "S3", "S4", "S5", "S6"):
        g.add_node(s, node_type="SW")
    g.add_node("SRV1", node_type="ES")

    # Leafs
    _add_es(g, ("C1", "C2"), "S1", link_rate)
    _add_es(g, ("C3", "C4"), "S2", link_rate)
    _add_es(g, ("C5", "C6"), "S3", link_rate)
    _add_es(g, ("C7", "C8"), "S4", link_rate)

    for leaf in ("S1", "S2", "S3", "S4"):
        g.add_edge(leaf, "S5", link_id=(leaf, "S5"), link_rate=link_rate)
    g.add_edge("S5", "S6", link_id=("S5", "S6"), link_rate=link_rate)
    g.add_edge("S6", "SRV1", link_id=("S6", "SRV1"), link_rate=link_rate)
    return g

# ─────────────────────────────────────────────────────────────────────────
#  NUEVAS TOPOLOGÍAS
# ─────────────────────────────────────────────────────────────────────────

def generate_unidirectional_topology9(link_rate=100):
    """Estrella clásica: S1 como hub central."""
    g = nx.DiGraph()
    g.add_node("S1", node_type="SW")
    _add_es(g, [f"C{i}" for i in range(1, 7)], "S1", link_rate)
    g.add_node("SRV1", node_type="ES")
    g.add_edge("S1", "SRV1", link_id=("S1", "SRV1"), link_rate=link_rate)
    return g

def generate_unidirectional_topology10(link_rate=100):
    """Anillo de 3 switches – todos los ES envían a SRV1."""
    g = nx.DiGraph()
    for sw in ("S1", "S2", "S3"):
        g.add_node(sw, node_type="SW")
    # anillo unidireccional (S1→S2→S3→S1)
    g.add_edge("S1", "S2", link_id=("S1", "S2"), link_rate=link_rate)
    g.add_edge("S2", "S3", link_id=("S2", "S3"), link_rate=link_rate)
    g.add_edge("S3", "S1", link_id=("S3", "S1"), link_rate=link_rate)
    # clientes
    _add_es(g, ["C1"], "S1", link_rate)
    _add_es(g, ["C2"], "S2", link_rate)
    _add_es(g, ["C3"], "S3", link_rate)
    # servidor accesible desde cualquier switch
    g.add_node("SRV1", node_type="ES")
    for sw in ("S1", "S2", "S3"):
        g.add_edge(sw, "SRV1", link_id=(sw, "SRV1"), link_rate=link_rate)
    return g

def generate_unidirectional_topology11(link_rate=100):
    """Core-Aggregation-Edge con 8 ES – prueba multi-nivel."""
    g = nx.DiGraph()
    # core
    g.add_node("S0", node_type="SW")
    # aggregation
    for aggr in ("S1", "S2"):
        g.add_node(aggr, node_type="SW")
        g.add_edge(aggr, "S0", link_id=(aggr, "S0"), link_rate=link_rate)
    # edge
    edge_map = {"S1": ("S3", "S4"), "S2": ("S5", "S6")}
    for aggr, edges in edge_map.items():
        for esw in edges:
            g.add_node(esw, node_type="SW")
            g.add_edge(esw, aggr, link_id=(esw, aggr), link_rate=link_rate)
    # end-stations under edge switches
    client_map = {
        "S3": ("C1", "C2"), "S4": ("C3", "C4"),
        "S5": ("C5", "C6"), "S6": ("C7", "C8"),
    }
    for esw, cls in client_map.items():
        _add_es(g, cls, esw, link_rate)
    # server
    g.add_node("SRV1", node_type="ES")
    g.add_edge("S0", "SRV1", link_id=("S0", "SRV1"), link_rate=link_rate)
    return g

# ────────────────────────────────────────────────────────────────────────
#  UNIDIR-12  →  grafo completo (7 nodos)
# ────────────────────────────────────────────────────────────────────────

def generate_unidirectional_topology12(link_rate=100):
    """
    Grafo *todos-con-todos*: dos switches + 4 clientes + 1 servidor.
    Se generan enlaces **dirigidos** entre *cada* par de nodos distinto.
    """
    g = nx.DiGraph()

    switches = ["S1", "S2"]
    clients  = [f"C{i}" for i in range(1, 5)]
    server   = ["SRV1"]
    all_nodes = switches + clients + server

    # anotar tipo de nodo
    for n in switches:
        g.add_node(n, node_type="SW")
    for n in clients + server:
        g.add_node(n, node_type="ES")

    # enlaces dirigidos entre cualquier par u ≠ v
    for u in all_nodes:
        for v in all_nodes:
            if u == v:
                continue
            g.add_edge(u, v, link_id=(u, v), link_rate=link_rate)

    return g

# ──────────────────────────────────────────────────────────────────
#  UNIDIR-13  ·  Cliente + anillo SW + 2 servidores
# ──────────────────────────────────────────────────────────────────

def generate_unidirectional_topology13(link_rate: int = 100):
    """
    Topología:

        C1 → S1 → S2 → S3 → S4 → S1   (anillo CW)
        S1 → SRV2
        S3 → SRV1

    Todos los enlaces son **dirigidos** y de la misma velocidad.
    """
    g = nx.DiGraph()

    # ── nodos ─────────────────────────────────────────────────────
    sw_nodes  = ["S1", "S2", "S3", "S4"]
    cli_nodes = ["C1"]
    srv_nodes = ["SRV1", "SRV2"]

    for n in sw_nodes:
        g.add_node(n, node_type="SW")
    for n in cli_nodes + srv_nodes:
        g.add_node(n, node_type="ES")

    # ── enlaces cliente→anillo ────────────────────────────────────
    g.add_edge("C1", "S1", link_id=("C1", "S1"), link_rate=link_rate)

    # ── anillo unidireccional S1→S2→S3→S4→S1 ─────────────────────
    ring = ["S1", "S2", "S3", "S4", "S1"]
    for u, v in zip(ring, ring[1:]):
        g.add_edge(u, v, link_id=(u, v), link_rate=link_rate)

    # ── salidas a servidores ─────────────────────────────────────
    g.add_edge("S3", "SRV1", link_id=("S3", "SRV1"), link_rate=link_rate)
    g.add_edge("S1", "SRV2", link_id=("S1", "SRV2"), link_rate=link_rate)

    return g

# ──────────────────────────────────────────────────────────────────────
#  UNIDIR-14 :  «rueda» con hub central + anillo de 4 switches
#               ─ 3 clientes en S1-S3, servidor en S4
#               ─ tráfico *unidireccional* desde los clientes → servidor
#                 (paths:  Cx→Si→S0→S4→SRV1)
# ---------------------------------------------------------------------
def generate_unidirectional_topology14(link_rate=100):
    """
    Topología 'rueda':
        • S0  ─ hub central
        • S1,S2,S3,S4 ─ switches de borde formando un *anillo* unidireccional
        • C1,C2,C3   ─ clientes (uno por S1-S3)
        • SRV1       ─ servidor conectado a S4

    Dirección de los enlaces (→):
        C* → S*    (clientes al borde)
        Si → S0    (borde ⇒ hub)
        S0 → S4    (hub ⇒ borde con servidor)
        S1 → S2 → S3 → S4 → S1   (anillo horario)
        S4 → SRV1  (último salto al servidor)
    """
    g = nx.DiGraph()

    # Hub central
    g.add_node("S0", node_type="SW")

    # Switches de borde en el anillo
    edge_sw = ["S1", "S2", "S3", "S4"]
    for sw in edge_sw:
        g.add_node(sw, node_type="SW")

    # Enlaces del anillo (S1→S2→S3→S4→S1)
    for i, src in enumerate(edge_sw):
        dst = edge_sw[(i + 1) % len(edge_sw)]
        g.add_edge(src, dst, link_id=(src, dst), link_rate=link_rate)

    # Spokes (borde → hub) y regreso hub → S4 para llegar al servidor
    for sw in edge_sw:
        g.add_edge(sw, "S0", link_id=(sw, "S0"), link_rate=link_rate)
    g.add_edge("S0", "S4", link_id=("S0", "S4"), link_rate=link_rate)

    # Clientes
    clients = [("C1", "S1"), ("C2", "S2"), ("C3", "S3")]
    for c, sw in clients:
        g.add_node(c, node_type="ES")
        g.add_edge(c, sw, link_id=(c, sw), link_rate=link_rate)

    # Servidor
    g.add_node("SRV1", node_type="ES")
    g.add_edge("S4", "SRV1", link_id=("S4", "SRV1"), link_rate=link_rate)

    return g

# ──────────────────────────────────────────────────────────────────────
#  UNIDIR-15 :  pentágono de 5 switches (S0-S4) con 3 clientes y 1 servidor
#               • ciclo dirigido: S0→S1→S2→S3→S4→S0
#               • C1→S0 ; C2→S1 ; C3→S2
#               • S3→SRV1   (servidor colgado en S3)
# ---------------------------------------------------------------------
def generate_unidirectional_topology15(link_rate=100):
    """
    Topología pentagonal unidireccional:

        C1→S0 → S1 → S2 → S3 → S4 → S0
                  ↑     │
        C2→S1     │     │
                  │     ↓
        C3→S2     SRV1←S3

    Todo el tráfico fluye **hacia delante** siguiendo la orientación de las flechas,
    por lo que cualquier paquete de un cliente alcanzará el servidor en S3
    después de recorrer parte del pentágono.
    """
    g = nx.DiGraph()

    # ----- switches S0-S4 -----
    sw = [f"S{i}" for i in range(5)]
    for s in sw:
        g.add_node(s, node_type="SW")

    # pentágono dirigido (horario)
    for i in range(5):
        src = sw[i]
        dst = sw[(i + 1) % 5]
        g.add_edge(src, dst, link_id=(src, dst), link_rate=link_rate)

    # ----- clientes -----
    clients = [("C1", "S0"), ("C2", "S1"), ("C3", "S2")]
    for c, s in clients:
        g.add_node(c, node_type="ES")
        g.add_edge(c, s, link_id=(c, s), link_rate=link_rate)

    # ----- servidor -----
    g.add_node("SRV1", node_type="ES")
    g.add_edge("S3", "SRV1", link_id=("S3", "SRV1"), link_rate=link_rate)

    return g

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
