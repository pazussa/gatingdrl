import logging
import argparse
import os
import sys
import random  # Añadido para corregir payloads
import multiprocessing  # Añadido para detectar cores disponibles

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

def get_best_model_file(topo, alg='MaskablePPO'):
    """Retorna la ruta completa al archivo del mejor modelo para la topología y algoritmo dados"""
    from tools.definitions import OUT_DIR
    return os.path.join(OUT_DIR, f"best_model_{topo}_{alg}", "best_model.zip")

def test(topo: str, num_flows: int, num_envs: int = 0,
         best_model_path: str = None, alg: str = 'MaskablePPO', link_rate: int = 100,
         min_payload: int = DEFAULT_MIN_PAYLOAD, max_payload: int = DEFAULT_MAX_PAYLOAD,
         visualize: bool = True, show_log: bool = True, gcl_threshold: int = 30):
    
    # Si num_envs es 0 o negativo, usar todos los cores disponibles
    if num_envs <= 0:
        num_envs = max(1, multiprocessing.cpu_count())
        logging.info(f"Usando {num_envs} entornos en paralelo (núcleos CPU detectados)")
    
    # Configurar logging: INFO para consola, DEBUG para archivo
    from tools.log_config import log_config
    from tools.definitions import OUT_DIR
    log_config(os.path.join(OUT_DIR, f'test_{topo}_{num_flows}.log'), level=logging.INFO)

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
    flows = generate_flows(graph, num_flows, unidirectional=(topo == "UNIDIR"), min_payload=min_payload, max_payload=max_payload) # Pasar min/max payload
    
    # Debug info: mostrar número de flujos generados
    logging.info(f"Generados {len(flows)} flujos para topología {topo} (solicitados: {num_flows})")
    
    # Create network with ALL flows - no curriculum learning in test mode
    network = Network(graph, flows)
    
    # Always use DrlScheduler with explicitly disabled curriculum learning
    scheduler = DrlScheduler(network, num_envs=num_envs, use_curriculum=False)
    
    if best_model_path:
        scheduler.load_model(best_model_path, alg)
    
    is_scheduled = scheduler.schedule()
    
    if is_scheduled:
        # Debug info: mostrar número de flujos programados
        schedule_res = scheduler.get_res()
        scheduled_flows = set()
        for link_ops in schedule_res.values():
            for flow, _ in link_ops:
                scheduled_flows.add(flow.flow_id)
        
        logging.info(f"Flujos programados: {len(scheduled_flows)} de {num_flows} solicitados")
        
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
                analyzer._gcl_tables,   # tablas ya calculadas
                topo,
                OUT_DIR
            )
            logging.info(f"OMNeT++ files escritos:\n  • {ned_path}\n  • {ini_path}")
        except Exception as e:
            logging.error(f"Error exportando ficheros OMNeT++: {e}")
    else:
        logging.error("Fail to find a valid solution.")

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

    # La función test ahora determinará automáticamente la ruta del modelo y usará el rango de payload
    test(args.topo, args.num_flows, 0,
         None, args.alg, args.link_rate, 
         min_payload=args.min_payload, max_payload=args.max_payload,
         visualize=args.visualize, show_log=args.show_log,
         gcl_threshold=args.gcl_threshold)  # Pass the threshold
