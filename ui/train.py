import argparse
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import os
import sys
import shutil  # Para eliminar directorios recursivamente
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed


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
             use_curriculum: bool = True, seed: int | None = None):
    def _init():
        # Set environment-specific seeds
        if seed is not None:
            random.seed(seed + rank)
            np.random.seed(seed + rank)
            set_random_seed(seed + rank)
            
        graph = generate_graph(topo, link_rate)

        # Simplificar - eliminar jitters
        # Cualquier variante "UNIDIR*" se trata como unidireccional
        is_unidir = topo.startswith("UNIDIR")
        # Pasar el rango de payload al generador
        if is_unidir:
            flow_generator = UniDirectionalFlowGenerator(
                graph,
                seed=seed + rank if seed is not None else None,
                min_payload=min_payload,
                max_payload=max_payload,
            )
        else:
            flow_generator = FlowGenerator(
                graph,
                seed=seed + rank if seed is not None else None,
                min_payload=min_payload,
                max_payload=max_payload,
            )

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
        if seed is not None:
            env.reset(seed=seed + rank)
        return env

    return _init


def train(topo: str, num_time_steps, monitor_dir, num_flows=NUM_FLOWS, pre_trained_model=None,
          link_rate: int = 100, min_payload: int = DEFAULT_MIN_PAYLOAD, max_payload: int = DEFAULT_MAX_PAYLOAD,
          use_curriculum: bool = True, show_log: bool = True, seed: int | None = None):
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
        make_env(num_flows, i, topo, monitor_dir, link_rate=link_rate,
                 min_payload=min_payload, max_payload=max_payload,
                 use_curriculum=use_curriculum, seed=seed)
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
        make_env(num_flows, i, topo, monitor_dir, training=False, link_rate=link_rate,
                 min_payload=min_payload, max_payload=max_payload, use_curriculum=False,
                 seed=seed)
        for i in range(n_envs)
        ])
    
    # Crear callback de evaluación y métricas
    callbacks = [
        EvalCallback(eval_env, 
                   best_model_save_path=get_best_model_path(topo=topo, alg=DRL_ALG),
                   log_path=OUT_DIR, 
                   eval_freq=max(10000 // n_envs, 1))
    ]

    # ══════════════════════════════════════════════════════════════════
    #  📊 NUEVO: Variables para medir tiempo de convergencia
    # ══════════════════════════════════════════════════════════════════
    convergence_data = {
        "rewards": [],           # Historial de rewards promedio
        "episode_numbers": [],   # Números de episodio
        "timestamps": [],        # Timestamps reales
        "convergence_episode": None,    # Episodio donde converge
        "convergence_time": None,       # Tiempo real de convergencia
        "is_converged": False,          # Si ya convergió
        "stability_window": 100,        # Ventana para medir estabilidad
        "stability_threshold": 0.05,    # Umbral de variación para convergencia (5%)
        "min_episodes_for_convergence": 200  # Mínimo de episodios antes de declarar convergencia
    }
    
    
    start_time = time.time()
    
    # Callback para capturar métricas de convergencia
    class ConvergenceCallback(BaseCallback):
        def __init__(self, convergence_data, max_timesteps, verbose=0):
            super().__init__(verbose)
            self.convergence_data = convergence_data
            self.episode_count = 0
            self.max_timesteps = max_timesteps
            self.timesteps_count = 0
            self.logged_stop = False
            
        def _on_step(self) -> bool:
            # CONTROL ESTRICTO: Usar el contador interno del modelo
            current_timesteps = self.model.num_timesteps
            
            # FORZAR PARADA EXACTA cuando alcance el límite
            if current_timesteps >= self.max_timesteps:
                if not self.logged_stop:
                    logging.info(f"🛑 PARADA FORZADA: Alcanzado límite exacto de {self.max_timesteps} timesteps")
                    logging.info(f"🔢 Timesteps del modelo: {current_timesteps}")
                    self.logged_stop = True
                return False  # Detener inmediatamente
            
            # Capturar datos cada vez que termina un episodio
            if len(self.locals.get('dones', [])) > 0 and any(self.locals['dones']):
                self.episode_count += 1
                
                # Obtener reward promedio de los entornos activos
                if 'infos' in self.locals:
                    episode_rewards = []
                    for info in self.locals['infos']:
                        if isinstance(info, dict) and 'episode' in info:
                            episode_rewards.append(info['episode']['r'])
                    
                    if episode_rewards:
                        avg_reward = sum(episode_rewards) / len(episode_rewards)
                        current_time = time.time()
                        
                        # Guardar datos
                        self.convergence_data["rewards"].append(avg_reward)
                        self.convergence_data["episode_numbers"].append(self.episode_count)
                        self.convergence_data["timestamps"].append(current_time)
                        
                        # Verificar convergencia si tenemos suficientes datos
                        self._check_convergence()
                        
                        # Log de progreso cada 100 episodios
                        if self.episode_count % 100 == 0:
                            logging.info(f"📊 Episodio {self.episode_count}, Timesteps: {current_timesteps}/{self.max_timesteps}, Reward promedio: {avg_reward:.3f}")
                        
            return True
            
        def _check_convergence(self):
            """Verifica si el algoritmo ha convergido basándose en la estabilidad de rewards"""
            data = self.convergence_data
            
            # No verificar hasta tener suficientes episodios
            if (len(data["rewards"]) < data["min_episodes_for_convergence"] or 
                data["is_converged"]):
                return
                
            window_size = data["stability_window"]
            threshold = data["stability_threshold"]
            
            # Verificar si tenemos suficientes datos para la ventana
            if len(data["rewards"]) < window_size:
                return
                
            # Obtener los últimos rewards en la ventana
            recent_rewards = data["rewards"][-window_size:]
            
            # Calcular estadísticas de estabilidad
            mean_reward = sum(recent_rewards) / len(recent_rewards)
            
            if mean_reward == 0:  # Evitar división por cero
                return
                
            # Calcular coeficiente de variación (desviación estándar / media)
            variance = sum((r - mean_reward) ** 2 for r in recent_rewards) / len(recent_rewards)
            std_dev = variance ** 0.5
            coefficient_of_variation = std_dev / abs(mean_reward)
            
            # Declarar convergencia si la variación es menor al umbral
            if coefficient_of_variation <= threshold:
                data["is_converged"] = True
                data["convergence_episode"] = data["episode_numbers"][-1]
                data["convergence_time"] = data["timestamps"][-1] - data["timestamps"][0]
                
                logging.info(
                    f"🎯 CONVERGENCIA DETECTADA en episodio {data['convergence_episode']} "
                    f"(Coef. Variación: {coefficient_of_variation:.4f} ≤ {threshold})"
                )
    
    # Crear callback de convergencia con límite de timesteps
    convergence_callback = ConvergenceCallback(convergence_data, num_time_steps)
    
    # Combinar callbacks existentes con el de convergencia
    callbacks = [
        EvalCallback(eval_env, 
                   best_model_save_path=get_best_model_path(topo=topo, alg=DRL_ALG),
                   log_path=OUT_DIR, 
                   eval_freq=max(10000 // n_envs, 1)),
        convergence_callback
    ]

    # ══════════════════════════════════════════════════════════════════
    #  🎯 ENTRENAMIENTO CON CONTROL EXACTO DE TIMESTEPS - VERSIÓN CORREGIDA
    # ══════════════════════════════════════════════════════════════════
    logging.info(f"Iniciando entrenamiento por EXACTAMENTE {num_time_steps} timesteps...")
    logging.info(f"Usando {n_envs} entornos en paralelo")
    
    # Crear un callback adicional más simple que solo controle timesteps
    class StrictTimestepCallback(BaseCallback):
        def __init__(self, max_timesteps, verbose=0):
            super().__init__(verbose)
            self.max_timesteps = max_timesteps
            self.logged = False
            
        def _on_step(self) -> bool:
            if self.model.num_timesteps >= self.max_timesteps:
                if not self.logged:
                    logging.info(f"🚨 PARADA ESTRICTA: {self.model.num_timesteps} timesteps alcanzados")
                    self.logged = True
                return False
            return True
    
    # Combinar callbacks con el de parada estricta como el último
    callbacks = [
        EvalCallback(eval_env, 
                   best_model_save_path=get_best_model_path(topo=topo, alg=DRL_ALG),
                   log_path=OUT_DIR, 
                   eval_freq=max(10000 // n_envs, 1)),
        convergence_callback,
        StrictTimestepCallback(num_time_steps)  # Este tiene la prioridad final
    ]

    # Entrenar con configuración más estricta
    model.learn(
        total_timesteps=num_time_steps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True
    )
    
    # Verificar timesteps finales con mayor detalle
    actual_timesteps = model.num_timesteps
    
    logging.info(f"✅ Entrenamiento completado")
    logging.info(f"🎯 Timesteps solicitados: {num_time_steps}")
    logging.info(f"📊 Timesteps ejecutados: {actual_timesteps}")
    
    if actual_timesteps == num_time_steps:
        logging.info(f"🎯 PERFECTO: Timesteps exactos ejecutados")
    elif actual_timesteps > num_time_steps:
        sobrepaso = actual_timesteps - num_time_steps
        sobrepaso_pct = (sobrepaso / num_time_steps) * 100
        logging.warning(f"⚠️ SOBREPASO: {sobrepaso} timesteps adicionales ({sobrepaso_pct:.2f}%)")
    else:
        deficit = num_time_steps - actual_timesteps
        logging.warning(f"⚠️ DÉFICIT: {deficit} timesteps menos de lo esperado")
    
    end_time = time.time()
    total_training_time = end_time - start_time
    
    logging.info("Training complete.")

    # ══════════════════════════════════════════════════════════════════
    #  📊 ANÁLISIS Y REPORTE DE CONVERGENCIA
    # ══════════════════════════════════════════════════════════════════
    if show_log:
        print("\n" + "="*80)
        print("🎯 ANÁLISIS DE CONVERGENCIA DEL ENTRENAMIENTO")
        print("="*80)
        
        total_episodes = len(convergence_data["rewards"])
        
        if convergence_data["is_converged"]:
            convergence_episode = convergence_data["convergence_episode"]
            convergence_time_seconds = convergence_data["convergence_time"]
            convergence_percentage = (convergence_episode / total_episodes) * 100
            
            print(f"✅ CONVERGENCIA ALCANZADA:")
            print(f"   📈 Episodio de convergencia: {convergence_episode}/{total_episodes} ({convergence_percentage:.1f}%)")
            print(f"   ⏱️  Tiempo hasta convergencia: {convergence_time_seconds:.1f} segundos")
            print(f"   📊 Ventana de estabilidad: {convergence_data['stability_window']} episodios")
            print(f"   🎚️  Umbral de variación: {convergence_data['stability_threshold']*100:.1f}%")
            
            # Calcular eficiencia del entrenamiento
            efficiency = (total_training_time - convergence_time_seconds) / total_training_time * 100
            if efficiency > 0:
                print(f"   ⚡ Tiempo 'desperdiciado' post-convergencia: {efficiency:.1f}% del entrenamiento")
        
        else:
            print(f"⚠️  NO SE DETECTÓ CONVERGENCIA:")
            print(f"   📈 Episodios totales: {total_episodes}")
            print(f"   ⏱️  Tiempo total: {total_training_time:.1f} segundos")
            print(f"   📊 Ventana requerida: {convergence_data['stability_window']} episodios estables")
            print(f"   🎚️  Umbral requerido: variación ≤ {convergence_data['stability_threshold']*100:.1f}%")
            
            # Analizar la tendencia final
            if len(convergence_data["rewards"]) >= 50:
                recent_50 = convergence_data["rewards"][-50:]
                mean_recent = sum(recent_50) / len(recent_50)
                variance_recent = sum((r - mean_recent) ** 2 for r in recent_50) / len(recent_50)
                std_recent = variance_recent ** 0.5
                cv_recent = std_recent / abs(mean_recent) if mean_recent != 0 else float('inf')
                
                print(f"   📉 Variación en últimos 50 episodios: {cv_recent*100:.2f}%")
                
                if cv_recent <= convergence_data['stability_threshold'] * 2:  # Doble del umbral
                    print(f"   💡 Sugerencia: El algoritmo está cerca de converger, considere más episodios")
        
        # Estadísticas generales del entrenamiento
        if convergence_data["rewards"]:
            max_reward = max(convergence_data["rewards"])
            min_reward = min(convergence_data["rewards"])
            final_reward = convergence_data["rewards"][-1]
            avg_reward = sum(convergence_data["rewards"]) / len(convergence_data["rewards"])
            
            print(f"\n📊 ESTADÍSTICAS DE RECOMPENSAS:")
            print(f"   🏆 Máxima: {max_reward:.3f}")
            print(f"   📉 Mínima: {min_reward:.3f}")
            print(f"   🎯 Final: {final_reward:.3f}")
            print(f"   📈 Promedio: {avg_reward:.3f}")
        
        print("="*80)
    else:
        # Siempre mostrar un resumen básico aunque show_log sea False
        total_episodes = len(convergence_data["rewards"])
        if convergence_data["is_converged"]:
            print(f"✅ Convergencia alcanzada en episodio {convergence_data['convergence_episode']}/{total_episodes}")
        else:
            print(f"⚠️ Sin convergencia detectada en {total_episodes} episodios")
    
    # Log para archivo (siempre se escribe en el log)
    if convergence_data["is_converged"]:
        logging.info(
            f"🎯 Convergencia: episodio {convergence_data['convergence_episode']} "
            f"({convergence_data['convergence_time']:.1f}s) de {total_episodes} episodios totales"
        )
    else:
        logging.info(f"⚠️ Sin convergencia detectada en {total_episodes} episodios ({total_training_time:.1f}s)")

    # NUEVO: Ejecutar un episodio final y guardar los detalles del scheduling
    if show_log:
        logging.info("Ejecutando episodio final para generar informe detallado...")
        print("\n🔍 Ejecutando episodio final para validar el entrenamiento...")
    
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
    
    # Mostrar resultado del episodio final si show_log está activado
    if final_info and final_info.get('ScheduleRes'):
        if show_log:
            print(f"✅ Episodio de validación exitoso: {len(final_info['ScheduleRes'])} enlaces programados")
            print(f"📊 Flujos programados: {final_info.get('scheduled_flows', 0)}/{final_info.get('total_flows', num_flows)}")
        logging.info(f"Scheduling final exitoso con {len(final_info['ScheduleRes'])} enlaces programados")
    elif show_log:
        print("⚠️ Episodio de validación no completó el scheduling exitosamente")

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
    # Eliminar la opción de especificar num_envs, ahora es automático
    parser.add_argument('--topo', type=str, default="SIMPLE", help="Topology type (e.g., SIMPLE, UNIDIR)")
    parser.add_argument('--link_rate', type=int, default=100)
    # Añadir argumentos para min/max payload
    parser.add_argument('--min-payload', type=int, default=DEFAULT_MIN_PAYLOAD, help=f"Tamaño mínimo de payload en bytes (default: {DEFAULT_MIN_PAYLOAD})")
    parser.add_argument('--max-payload', type=int, default=DEFAULT_MAX_PAYLOAD, help=f"Tamaño máximo de payload en bytes (default: {DEFAULT_MAX_PAYLOAD})")
    # Cambio: Hacer --model opcional con un valor por defecto de None
    parser.add_argument('--model', type=str, default=None, 
                       help="Ruta opcional a un modelo pre-entrenado. Por defecto no carga ninguno.")
    # ------------- distribución del gap ------------- 
    # Ahora **todo** se controla con --dist y --dist-params
    parser.add_argument('--dist', type=str, default='fixed',
                        choices=['fixed', 'uniform', 'exponential', 'gaussian', 'pareto'],
                        help="Tipo de distribución de separación de paquetes")
    parser.add_argument('--dist-params', type=float, nargs='+', default=[],
                        help="Parámetros numéricos de la distribución (ver doc)")

    # --- NUEVAS OPCIONES PARA COINCIDIR CON ui/test.py --------------------
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generar visualización TSN al terminar')
    parser.add_argument('--show-log', action='store_true', default=True,
                        help='Mostrar información detallada del entrenamiento y scheduling')
    parser.add_argument('--gcl-threshold', type=int, default=30,
                        help='Umbral (µs) para generar entradas GCL')
                        
    # Añadir opción para controlar el curriculum learning
    parser.add_argument('--curriculum', action='store_true', default=False,
                      help='Activar curriculum learning adaptativo')
    parser.add_argument('--no-curriculum', action='store_false', dest='curriculum',
                      help='Desactivar curriculum learning adaptativo')
   
    # NUEVO: Añadir argumento para la semilla
    parser.add_argument('--seed', type=int, default=None,
                        help='Semilla para reproducibilidad (por defecto aleatoria)')
    
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
    
    # -------- configuración única (vigente) -------- #
    try:
        Net.set_gap_distribution(args.dist, args.dist_params)
    except (AssertionError, ValueError) as e:
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
    # Pasar show_log al método train
    train(args.topo, args.time_steps,
          MONITOR_DIR,  # Pasar MONITOR_DIR como parámetro
          num_flows=args.num_flows,
          pre_trained_model=args.model,  # Usar args.model (que podría ser None)
          link_rate=args.link_rate,
          min_payload=args.min_payload, # Pasar min/max payload
          max_payload=args.max_payload,
          use_curriculum=args.curriculum,
          show_log=args.show_log,
          seed=args.seed)  # ← NUEVO: Pasar show_log y seed

    # Add try-except block around plotting
    try:
        plot_results(MONITOR_DIR)
    except Exception as e:
        logging.error(f"Error during plotting: {e}")
        logging.info("Continuing without plotting.")

    # Remove metrics summary code
    logging.info(f"Training completed successfully.")

    # Al terminar training, invocamos test con los mismos args
    test(
        args.topo,
        args.num_flows,
        NUM_ENVS,
        alg=DRL_ALG,
        link_rate=args.link_rate,
        min_payload=args.min_payload,
        max_payload=args.max_payload,
        visualize=args.visualize,
        show_log=args.show_log,  # ← NUEVO: Pasar show_log también a test
        gcl_threshold=args.gcl_threshold,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
