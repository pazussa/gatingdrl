import logging
import argparse
import os
import sys
import random                       # payloads
import multiprocessing              # núcleos
import matplotlib.pyplot as plt     # 🔹 para la gráfica
import numpy as np                  # 🔹
import math                         # 🔹
from core.network.net import Net    # 🔹 muestreador oficial

# Configure Qt to use offscreen rendering by default
os.environ["QT_QPA_PLATFORM"] = "offscreen"



# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.execute import execute_from_command_line
from core.network.net import generate_flows, generate_graph, Network
from core.scheduler.scheduler import DrlScheduler, ResAnalyzer
from ui.tsn_visualizer_plotly import visualize_tsn_schedule_plotly  # Nuevo visualizador con Plotly
from core.omnet_export import export_omnet_files                    # ← NUEVO

DEFAULT_MIN_PAYLOAD = 64   # Valor por defecto mínimo razonable
DEFAULT_MAX_PAYLOAD = 1518 # Valor por defecto máximo MTU
DEFAULT_MAX_JITTER  = 0    # ← NUEVO: por defecto sin jitter

# ╔═══════════════════════════════════════════════════════════════════╗
#  HELPERS – gaps reales y gráfica de verificación                   #
# ╚═══════════════════════════════════════════════════════════════════╝

def _is_es(node_name: str) -> bool:
    """Heurística rápida: un nombre que empiece por E, C o SRV es End-Station."""
    return node_name.startswith(("E", "C", "SRV"))


def _extract_packet_gaps(schedule_res) -> list[int]:
    """
    Devuelve los *gaps* entre instantes de creación (**start_time**) de los
    paquetes cuyo primer hop sale de una End-Station.
    """
    starts = []
    for link, ops in schedule_res.items():
        src = link.link_id[0] if isinstance(link.link_id, tuple) else link.link_id.split("-")[0]
        if not _is_es(src):
            continue
        for _flow, op in ops:
            starts.append(op.start_time)          # primer hop ⇒ start_time

    starts.sort()
    return [s2 - s1 for s1, s2 in zip(starts, starts[1:])]

def _pdf_theoretical(dist: str, params: list[float], xs: np.ndarray) -> np.ndarray:
    if dist == "uniform":
        lo, hi = params
        return np.where((xs >= lo) & (xs <= hi), 1 / (hi - lo), 0)
    if dist == "exponential":
        μ, = params
        return (1 / μ) * np.exp(-xs / μ)
    if dist == "gaussian":
        μ, σ = params
        return np.where(
            xs >= 0,
            (1 / (σ * math.sqrt(2 * math.pi))) * np.exp(-(xs - μ) ** 2 / (2 * σ ** 2)),
            0,
        )
    if dist == "pareto":
        α, xm = params
        return np.where(xs >= xm, α * xm ** α / xs ** (α + 1), 0)
    if dist == "fixed":
        pdf = np.zeros_like(xs)
        pdf[np.abs(xs - params[0]) < 0.5] = 1.0
        return pdf
    return np.zeros_like(xs)


def _plot_gap_distribution(
    dist: str,
    params: list[float],
    out_dir: str = ".",
) -> None:
    """Histograma de las muestras crudas según Net.sample_packet_gap()."""

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    raw = [Net.sample_packet_gap() for _ in range(max(500, len(params) * 100))]
    ax.hist(raw, bins=50, density=True, alpha=0.7, label="Muestras")

    ax.set_xlabel("Gap (µs)")
    ax.set_ylabel("Densidad")
    # Formatear los parámetros con nombre según la distribución
    if dist == "fixed":
        param_str = f"gap={params[0]}"
    elif dist == "uniform":
        param_str = f"min={params[0]}, max={params[1]}"
    elif dist == "exponential":
        param_str = f"mean={params[0]}"
    elif dist == "gaussian":
        param_str = f"mu={params[0]}, sigma={params[1]}"
    elif dist == "pareto":
        param_str = f"alpha={params[0]}, xm={params[1]}"
    else:
        param_str = ", ".join(map(str, params))
    ax.set_title(f"FDP fuente('{dist}', {param_str})")
    ax.legend()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"gap_dist_{dist}.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logging.info(f"[test] Gráfico de distribución guardado en → {out_path}")


# --------------------------------------------------------------------------- #
#  Helper – gráfico de verificación de la distribución seleccionada           #
# --------------------------------------------------------------------------- #
# (la versión previa basada en Net.sample_packet_gap se ha retirado)

def get_best_model_file(topo, alg='MaskablePPO'):
    """Retorna la ruta completa al archivo del mejor modelo para la topología y algoritmo dados"""
    from tools.definitions import OUT_DIR
    return os.path.join(OUT_DIR, f"best_model_{topo}_{alg}", "best_model.zip")

def test(topo: str, num_flows: int, num_envs: int = 0,
         best_model_path: str = None, alg: str = 'MaskablePPO', link_rate: int = 100,
         min_payload: int = DEFAULT_MIN_PAYLOAD, max_payload: int = DEFAULT_MAX_PAYLOAD,
         visualize: bool = True, show_log: bool = True,
         gcl_threshold: int = 30, plot_gap_dist: bool = True,
         seed: int | None = None):
    
    # Para la prueba / inferencia forzamos **un solo entorno**
    num_envs = 1
    logging.info("Usando 1 entorno (modo inferencia)")
    
    # Configurar logging: INFO para consola, DEBUG para archivo
    from tools.log_config import log_config
    from tools.definitions import OUT_DIR
    log_config(os.path.join(OUT_DIR, f'test_{topo}_{num_flows}.log'), level=logging.INFO)

    # Set random seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except Exception:
            pass

    # Siempre usar la ruta predeterminada para la topología y algoritmo
    if best_model_path is None:
        best_model_path = get_best_model_file(topo, alg)
        logging.info(f"Usando modelo predeterminado: {best_model_path}")
    
    # Verificar si el archivo existe
    if not os.path.exists(best_model_path):
        logging.error(f"Error: Modelo no encontrado: {best_model_path}")
        return False

    graph = generate_graph(topo, link_rate)

    # Generar flujos usando el rango de payload especificado
    flows = generate_flows(                       # ← SIN jitter
        graph, num_flows,
        unidirectional=topo.startswith("UNIDIR"),
        min_payload=min_payload,
        max_payload=max_payload,
        seed=seed,
    )
    
    # Debug info: mostrar número de flujos generados
    logging.info(f"Generados {len(flows)} flujos para topología {topo} (solicitados: {num_flows})")
    
    # Create network with ALL flows - no curriculum learning in test mode
    network = Network(graph, flows)
    
    # Always use DrlScheduler with explicitly disabled curriculum learning
    scheduler = DrlScheduler(network, num_envs=num_envs, use_curriculum=False)
    
    if best_model_path:
        scheduler.load_model(best_model_path, alg)
    
    scheduler.schedule()                      # ejecuta con early-stop
    schedule_res = scheduler.get_res()

    scheduled_cnt = 0
    if schedule_res:
        scheduled_cnt = {
            f.flow_id
            for link_ops in schedule_res.values()
            for f, _ in link_ops
        }.__len__()

    logging.info(f"Flujos programados: {scheduled_cnt} de {num_flows} solicitados")
    is_scheduled = (scheduled_cnt == num_flows)
    
    # a partir de aquí usa `schedule_res` (si existe) independientemente
    if schedule_res:
        # Analizar y guardar logs detallados del scheduling
        analyzer = ResAnalyzer(network, schedule_res)
        
        # Apply custom GCL threshold if provided
        if gcl_threshold != 30:  # If different from default
            analyzer.gap_threshold_us = gcl_threshold
            analyzer.recalculate_gcl_tables(gcl_threshold)
        
        log_file = f'schedule_res_by_link_{analyzer.analyzer_id}.log'  # Usar el ID almacenado en el analizador
        log_path = os.path.join(OUT_DIR, log_file)
        logging.info(f"Schedule details saved to {log_path}")

        # Imprimir información de flujos y tablas GCL estáticas
        analyzer.print_flow_info()  # Mostrar tabla de flujos independientemente
        
        # Usar el método actualizado que solo muestra la tabla GCL generada
        analyzer.print_gcl_tables()

        # ──────────────────────────────────────────────────────────────
        #  📊  FORZAR CÁLCULO Y MOSTRAR MÉTRICAS DE LATENCIA E2E
        # ──────────────────────────────────────────────────────────────
        print("\n🔥 FORZANDO CÁLCULO DE MÉTRICAS DE LATENCIA...")
        try:
            latency_metrics = analyzer.calculate_latency_metrics()
            print(f"✅ Métricas calculadas exitosamente: {latency_metrics}")
        except Exception as e:
            print(f"❌ ERROR calculando métricas de latencia: {e}")
            import traceback
            traceback.print_exc()

        # ──────────────────────────────────────────────────────────────
        #  🔗  NUEVO: CALCULAR Y MOSTRAR UTILIZACIÓN DE ENLACES
        # ──────────────────────────────────────────────────────────────
        print("\n🔥 CALCULANDO UTILIZACIÓN DE ENLACES...")
        try:
            utilization_metrics = analyzer.calculate_link_utilization()
            print(f"✅ Utilización calculada exitosamente")
        except Exception as e:
            print(f"❌ ERROR calculando utilización de enlaces: {e}")
            import traceback
            traceback.print_exc()

        # ------------------------------------------------------------- #
        #  Verificación gráfica de la distribución antes del scheduling
        # ------------------------------------------------------------- #
        if plot_gap_dist:
            # ---- parámetros efectivos según el modo ya aplicado a Net ----
            dist_mode = Net.PACKET_GAP_MODE
            
            if dist_mode == "fixed":
                eff_params = [Net.PACKET_GAP_EXTRA]
            elif dist_mode == "uniform":
                eff_params = list(Net.PACKET_GAP_UNIFORM)
            elif dist_mode == "exponential":
                eff_params = [Net.PACKET_GAP_EXTRA]
            elif dist_mode == "gaussian":
                eff_params = list(Net.PACKET_GAP_GAUSS)
            elif dist_mode == "pareto":
                eff_params = list(Net.PACKET_GAP_PARETO)
            else:
                eff_params = []

            from tools.definitions import OUT_DIR
            _plot_gap_distribution(dist_mode, eff_params, OUT_DIR)

        # Mostrar el contenido del log de scheduling en la consola si se solicita
        if show_log:
            try:
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        log_content = f.read()
                    print("\n" + "="*80)
                    print("SCHEDULING DETAILS BY LINK:")
                    print("="*80)
                    print(log_content)
                    print("="*80 + "\n")
                else:
                    logging.error(f"Archivo de log no encontrado: {log_path}")
            except Exception as e:
                logging.error(f"Error reading schedule log: {e}")
        
        if visualize:
            # Try to visualize with error handling
            try:
                # Visualizar la programación usando Plotly (más estable)
                save_path = os.path.join(OUT_DIR, f'tsn_schedule_{topo}_{num_flows}.html')
                visualize_tsn_schedule_plotly(schedule_res, save_path)
                
                logging.info(f"Visualización interactiva guardada en {save_path}")
            except Exception as e:
                logging.error(f"Error durante la visualización: {e}")
                logging.info("Continuando sin visualización interactiva.")

        # ────────────────────────────────────────────────────────────────
        #  NUEVO: exportar .ned y .ini cada vez que haya scheduling OK
        # ────────────────────────────────────────────────────────────────
        try:
            ned_path, ini_path = export_omnet_files(
                network,
                schedule_res,
                analyzer._gcl_tables,
                topo,
                OUT_DIR
            )
            logging.info(f"OMNeT++ files escritos:\n  • {ned_path}\n  • {ini_path}")

            # ──────────────────────────────────────────────────────────────
            #  RESUMEN GLOBAL DE FLOWS — última línea del log
            # ──────────────────────────────────────────────────────────────
            logging.info(f"Programados con éxito: {scheduled_cnt}/{num_flows} flujos")

        except Exception as e:
            logging.error(f"Error exportando ficheros OMNeT++: {e}")

    else:
        logging.error("No se pudo programar ningún flujo.")

    return is_scheduled


if __name__ == '__main__':
    # «resolve» evita choques de nombres si algún módulo añade flags
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument('--topo', type=str, required=True)
    parser.add_argument('--num_flows', type=int, required=True)
    parser.add_argument('--alg', type=str, default='MaskablePPO')
    parser.add_argument('--link_rate', type=int, default=100)
    # Añadir argumentos para min/max payload
    parser.add_argument('--min-payload', type=int, default=DEFAULT_MIN_PAYLOAD, help=f"Tamaño mínimo de payload en bytes (default: {DEFAULT_MIN_PAYLOAD})")
    parser.add_argument('--max-payload', type=int, default=DEFAULT_MAX_PAYLOAD, help=f"Tamaño máximo de payload en bytes (default: {DEFAULT_MAX_PAYLOAD})")
    # ── parámetro eliminado: max-jitter ya no existe ──
    # ───────── NUEVA interfaz unificada ─────────
    parser.add_argument('--dist', type=str, default='fixed',
                        choices=['fixed', 'uniform', 'exponential', 'gaussian', 'pareto'],
                        help='Distribución de separación de paquetes')
    parser.add_argument('--dist-params', type=float, nargs='+', default=[],
                        help='Parámetros de la distribución (ver README)')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generar visualización TSN')
    parser.add_argument('--show-log', action='store_true', default=True, help='Mostrar detalles del scheduling en consola')
    parser.add_argument('--gcl-threshold', type=int, default=30, 
                        help='Threshold in µs for GCL entry generation')
    parser.add_argument('--no-gap-plot',
                        action='store_false',
                        dest='plot_gap_dist',
                        help='Desactiva la gráfica de verificación de gaps')
    parser.add_argument('--seed', type=int, default=None,
                        help='Semilla para reproducibilidad (por defecto aleatoria)')
    # Se eliminó completamente el parámetro --best_model_path
    
    args = parser.parse_args()
    
    # Validar rango de payload
    if args.min_payload > args.max_payload:
        logging.error(f"Error: min-payload ({args.min_payload}) no puede ser mayor que max-payload ({args.max_payload})")
        sys.exit(1)
    if args.min_payload < 1 or args.max_payload < 1:
        logging.error("Error: min-payload y max-payload deben ser >= 1")
        sys.exit(1)

    # 👉 Configurar la distribución global de separación entre paquetes
    from core.network.net import Net
    try:
        Net.set_gap_distribution(args.dist, args.dist_params)
    except (AssertionError, ValueError) as e:
        logging.error(e)
        sys.exit(1)

    # ----------------------------------------------------------------- #
    #  La verificación de gaps se hará **después** del scheduling
    # ----------------------------------------------------------------- #

    # La función test ahora determinará automáticamente la ruta del modelo y usará el rango de payload
    test(args.topo, args.num_flows, 0,
         None, args.alg, args.link_rate,
         min_payload=args.min_payload, max_payload=args.max_payload,
         visualize=args.visualize, show_log=args.show_log,
         gcl_threshold=args.gcl_threshold,
         plot_gap_dist=args.plot_gap_dist,
         seed=args.seed)
