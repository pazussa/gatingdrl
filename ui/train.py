import argparse
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import os
import sys
import shutil  # Para eliminar directorios recursivamente

# Add Qt platform environment variable before any imports that might use Qt
# This helps Qt find the correct platform plugin
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Use offscreen rendering by default

# Configurar el path antes de cualquier otra importación
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#print(f"Set Python path to include: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

# Importar módulos desde rutas relativas
from ui.test import test
from tools.definitions import OUT_DIR, LOG_DIR  # Add LOG_DIR to import
from core.learning.encoder import FeaturesExtractor
# Importar MaskablePPO directamente
from sb3_contrib import MaskablePPO
from core.scheduler.scheduler import DrlScheduler
from core.learning.environment import NetEnv # Eliminar TrainingNetEnv
from tools.log_config import log_config
from core.network.net import FlowGenerator, UniDirectionalFlowGenerator, generate_graph, Network
from tools.definitions import OUT_DIR, LOG_DIR

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

TOPO = 'SIMPLE'  # Cambiado de 'CEV' a 'SIMPLE' para coincidir con el valor por defecto

# Usar siempre el número máximo de cores disponibles
NUM_ENVS = max(1, multiprocessing.cpu_count())
NUM_FLOWS = 50

# Cambiar la definición del algoritmo a una constante fija
DRL_ALG = 'MaskablePPO'

MONITOR_ROOT_DIR = os.path.join(OUT_DIR, "monitor")

DEFAULT_MIN_PAYLOAD = 64   # Valor por defecto mínimo razonable
DEFAULT_MAX_PAYLOAD = 1518 # Valor por defecto máximo MTU


def get_best_model_path(topo=TOPO, alg=DRL_ALG):
    """Retorna la ruta al modelo entrenado según la topología y algoritmo"""
    return os.path.join(OUT_DIR, f"best_model_{topo}_{alg}")

def get_best_model_file(topo=TOPO, alg=DRL_ALG):
    """Retorna la ruta completa al archivo del modelo (best_model.zip)"""
    return os.path.join(get_best_model_path(topo, alg), "best_model.zip")


def make_env(num_flows, rank: int, topo: str, monitor_dir, training: bool = True, link_rate: int = 100, 
             min_payload: int = DEFAULT_MIN_PAYLOAD, max_payload: int = DEFAULT_MAX_PAYLOAD,
             use_curriculum: bool = True):
    def _init():
        graph = generate_graph(topo, link_rate)

        # Simplificar - eliminar jitters
        # Use UniDirectionalFlowGenerator for UNIDIR topology
        is_unidir = topo == "UNIDIR"
        # Pasar el rango de payload al generador
        if is_unidir:
            flow_generator = UniDirectionalFlowGenerator(graph, min_payload=min_payload, max_payload=max_payload)
        else:
            flow_generator = FlowGenerator(graph, min_payload=min_payload, max_payload=max_payload)

        # Generar todos los flujos - asegurarse de crear exactamente el número solicitado
        flows = flow_generator(num_flows)
        logging.info(f"Generados {len(flows)} flujos para {topo} (solicitados: {num_flows})")
        
        network = Network(graph, flows)
        
        # Crear entorno con curriculum learning adaptativo
        env = NetEnv(
            network, 
            curriculum_enabled=use_curriculum,  
            initial_complexity=0.25 if use_curriculum else 1.0,  # Si no hay curriculum, usar 100% de complejidad
            curriculum_step=0.05      # Incrementar 5% por cada éxito
        )

        # Wrap the environment with Monitor
        env = Monitor(env, os.path.join(monitor_dir, f'{"train" if training else "eval"}_{rank}'))
        return env

    return _init


def train(topo: str, num_time_steps, monitor_dir, num_flows=NUM_FLOWS, pre_trained_model=None, link_rate=100, min_payload: int = DEFAULT_MIN_PAYLOAD, max_payload: int = DEFAULT_MAX_PAYLOAD, use_curriculum: bool = True):
    # ────────────────────────────────────────────────────────────────
    #  NUEVO: Limpiar completamente el directorio de salida
    # ────────────────────────────────────────────────────────────────
    if os.path.exists(OUT_DIR):
        logging.info(f"Limpiando directorio de salida: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    
    # Recrear el directorio vacío
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)  # También recreamos LOG_DIR

    # Siempre usar todos los cores disponibles
    n_envs = NUM_ENVS
    logging.info(f"Usando {n_envs} entornos en paralelo (núcleos CPU detectados: {multiprocessing.cpu_count()})")
    
    env = SubprocVecEnv([
        # Ya no hay distinción entre entornos de entrenamiento y evaluación,
        # ambos usan la configuración completa de num_flows desde el principio
        make_env(num_flows, i, topo, monitor_dir, link_rate=link_rate, min_payload=min_payload, max_payload=max_payload, use_curriculum=use_curriculum)  # Pasar flag de curriculum
        for i in range(n_envs)
        ])

    if pre_trained_model is not None:
        model = MaskablePPO.load(pre_trained_model, env)
    else:
        policy_kwargs = dict(
            features_extractor_class=FeaturesExtractor,
        )

        # Usar siempre MaskablePPO sin condicionales
        model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    eval_env = SubprocVecEnv([
        # El entorno de evaluación también usa la misma configuración
        make_env(num_flows, i, topo, monitor_dir, training=False, link_rate=link_rate, min_payload=min_payload, max_payload=max_payload, use_curriculum=False)  # Siempre desactivar curriculum en evaluación
        for i in range(n_envs)
        ])
    
    # Crear callback de evaluación y métricas
    callbacks = [
        EvalCallback(eval_env, 
                   best_model_save_path=get_best_model_path(topo=topo, alg=DRL_ALG),
                   log_path=OUT_DIR, 
                   eval_freq=max(10000 // n_envs, 1))
    ]

    # Train the agent with just the eval callback
    model.learn(total_timesteps=num_time_steps, callback=callbacks)

    logging.info("Training complete.")

    # NUEVO: Ejecutar un episodio final y guardar los detalles del scheduling
    logging.info("Ejecutando episodio final para generar informe detallado...")
    obs = env.reset()
    done = False
    final_info = None
    
    # Ejecutar un episodio completo con el modelo entrenado
    while not done:
        action_masks = np.vstack(env.env_method('action_masks'))
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks.astype(np.int8))
        obs, _, dones, infos = env.step(action)
        done = any(dones)
        if done:
            # Buscar información de éxito en algún entorno
            for i, is_done in enumerate(dones):
                if is_done and infos[i].get('success'):
                    final_info = infos[i]
                    break
    
    # Remove report generation code
    if final_info and final_info.get('ScheduleRes'):
        logging.info(f"Scheduling final exitoso con {len(final_info['ScheduleRes'])} enlaces programados")

    logging.info("------Finish learning------")
    return None  # Return None instead of metrics


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    try:
        x, y = ts2xy(load_results(log_folder), "timesteps")
        y = moving_average(y, window=50)
        # Truncate x
        x = x[len(x) - len(y):]

        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
        plt.title(title + " Smoothed")
        plt.savefig(os.path.join(log_folder, "reward.png"))
        
        # Use non-blocking display and catch any errors
        try:
            plt.show(block=False)
            plt.pause(3)  # Espera 3 segundos para mostrar la gráfica
            plt.close()
        except Exception as e:
            logging.warning(f"Could not display plot interactively: {e}")
            logging.info("Plot saved to file, continuing without interactive display.")
    except Exception as e:
        logging.error(f"Error plotting results: {e}")
        logging.info("Continuing without plotting.")


def main():
    # Mover todas las declaraciones global al inicio de la función
    global TOPO, NUM_ENVS, DRL_ALG
    
    # specify an existing model to train.
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument('--time_steps', type=int, required=True)
    parser.add_argument('--num_flows', type=int, nargs='?', default=NUM_FLOWS)
    parser.add_argument('--debug', action='store_true')
    # Eliminar la opción de especificar num_envs, ahora es automático
    parser.add_argument('--topo', type=str, default="SIMPLE", help="Topology type (e.g., SIMPLE, UNIDIR)")
    parser.add_argument('--link_rate', type=int, default=100)
    # Añadir argumentos para min/max payload
    parser.add_argument('--min-payload', type=int, default=DEFAULT_MIN_PAYLOAD, help=f"Tamaño mínimo de payload en bytes (default: {DEFAULT_MIN_PAYLOAD})")
    parser.add_argument('--max-payload', type=int, default=DEFAULT_MAX_PAYLOAD, help=f"Tamaño máximo de payload en bytes (default: {DEFAULT_MAX_PAYLOAD})")
    # Cambio: Hacer --model opcional con un valor por defecto de None
    parser.add_argument('--model', type=str, default=None, 
                       help="Ruta opcional a un modelo pre-entrenado. Por defecto no carga ninguno.")
    # ------------- argumentos para la separación probabilística -------------
    parser.add_argument('--gap-mode', type=str, default='fixed',
                        choices=['fixed', 'uniform', 'exponential', 'gaussian', 'pareto'],
                        help="Modo de cálculo del gap en µs entre creaciones de paquetes")
    parser.add_argument('--pkt-gap', type=int, default=0,
                        help="• fixed ⇒ valor constante\n"
                             "• exponential ⇒ media μ (λ = 1/μ)")
    parser.add_argument('--gap-uniform', type=int, nargs=2, metavar=('MIN', 'MAX'),
                        help="Sólo con --gap-mode uniform: intervalo [MIN,MAX]")
    parser.add_argument('--gap-gauss', type=int, nargs=2, metavar=('MEAN','STD'),
                        help="Sólo con --gap-mode gaussian: media μ y desvío σ (µs)")
    parser.add_argument('--gap-pareto', type=float, nargs=2, metavar=('ALPHA','XM'),
                        help="Sólo con --gap-mode pareto: shape α y scale xm")
    # ------------- interfaz unificada de distribución de gaps ---------------
    parser.add_argument('--dist', type=str, default='fixed',
                        choices=['fixed', 'uniform', 'exponential', 'gaussian', 'pareto'],
                        help="Tipo de distribución de separación de paquetes")
    parser.add_argument('--dist-params', type=float, nargs='+', default=[],
                        help="Parámetros numéricos de la distribución (ver doc)")
    parser.add_argument('--arrival-dist', type=str, default='set',
                        choices=['set', 'uniform', 'exponential'],
                        help="Distribución para el período de cada flujo")
    # Añadir opción para controlar el curriculum learning
    parser.add_argument('--curriculum', action='store_true', default=True,
                      help='Usar curriculum learning adaptativo (por defecto activado)')
    parser.add_argument('--no-curriculum', action='store_false', dest='curriculum',
                      help='Desactivar curriculum learning adaptativo')
   
    args = parser.parse_args()


    if args.link_rate is not None:
        support_link_rates = [100, 1000]
        assert args.link_rate in support_link_rates, \
            f"Unknown link rate {args.link_rate}, which is not in supported link rates {support_link_rates}"

    # Validar rango de payload
    if args.min_payload > args.max_payload:
        logging.error(f"Error: min-payload ({args.min_payload}) no puede ser mayor que max-payload ({args.max_payload})")
        sys.exit(1)
    if args.min_payload < 1 or args.max_payload < 1:
        logging.error("Error: min-payload y max-payload deben ser >= 1")
        sys.exit(1)

    # Eliminar procesamiento de jitters
    TOPO = args.topo
    # NUM_ENVS = args.num_envs  # Esta línea se elimina

    # 👉 Aplicar el valor elegido antes de crear cualquier entorno
    from core.network.net import Net
    if args.pkt_gap < 0:
        logging.error("Error: pkt-gap debe ser >= 0")
        sys.exit(1)
    # -------- aplicar configuración global del gap -------- #
    Net.PACKET_GAP_MODE = args.gap_mode
    if args.gap_mode == 'uniform':
        if args.gap_uniform is None or len(args.gap_uniform) != 2:
            logging.error("Debe proporcionar --gap-uniform MIN MAX con --gap-mode uniform")
            sys.exit(1)
        Net.PACKET_GAP_UNIFORM = tuple(args.gap_uniform)
    Net.PACKET_GAP_EXTRA = max(args.pkt_gap, 0)

    if args.gap_mode == 'gaussian':
        if args.gap_gauss is None or len(args.gap_gauss) != 2:
            logging.error("Debe proporcionar --gap-gauss MEAN STD con --gap-mode gaussian")
            sys.exit(1)
        Net.PACKET_GAP_GAUSS = tuple(args.gap_gauss)

    if args.gap_mode == 'pareto':
        if args.gap_pareto is None or len(args.gap_pareto) != 2:
            logging.error("Debe proporcionar --gap-pareto ALPHA XM con --gap-mode pareto")
            sys.exit(1)
        Net.PACKET_GAP_PARETO = tuple(args.gap_pareto)

    # -------- aplicar configuración global del gap -------- #
    try:
        Net.set_gap_distribution(args.dist, args.dist_params)
    except AssertionError as e:
        logging.error(e)
        sys.exit(1)

    log_config(os.path.join(OUT_DIR, f"train.log"), logging.DEBUG)

    logging.info(args)

    done = False
    i = 0
    MONITOR_DIR = None
    while not done:
        try:
            MONITOR_DIR = os.path.join(MONITOR_ROOT_DIR, str(i))
            os.makedirs(MONITOR_DIR, exist_ok=False)
            done = True
        except OSError:
            i += 1
            continue
    assert MONITOR_DIR is not None

    logging.info("start training...")
    # metrics variable is ignored since it's now None
    train(args.topo, args.time_steps,
          MONITOR_DIR,  # Pasar MONITOR_DIR como parámetro
          num_flows=args.num_flows,
          pre_trained_model=args.model,  # Usar args.model (que podría ser None)
          link_rate=args.link_rate,
          min_payload=args.min_payload, # Pasar min/max payload
          max_payload=args.max_payload,
          use_curriculum=args.curriculum)

    # Add try-except block around plotting
    try:
        plot_results(MONITOR_DIR)
    except Exception as e:
        logging.error(f"Error during plotting: {e}")
        logging.info("Continuing without plotting.")

    # Remove metrics summary code
    logging.info(f"Training completed successfully.")

    # Ya no necesitamos pasar la ruta del modelo, test la determinará automáticamente
    test(args.topo, args.num_flows, NUM_ENVS, alg=DRL_ALG, link_rate=args.link_rate, min_payload=args.min_payload, max_payload=args.max_payload)


if __name__ == "__main__":
    main()
