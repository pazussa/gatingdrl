import logging
import os
from typing import Dict, List, Tuple
import math
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO
# Usamos el mismo extractor que durante el entrenamiento
from core.learning.encoder import FeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from tools.definitions import OUT_DIR
from core.network.net import Flow, Link, Network, Net
from core.network.operation import Operation
from core.learning.environment import NetEnv

# Tipo para resultados
ScheduleRes = Dict[Link, List[Tuple[Flow, Operation]]]

class DrlScheduler:
    """Scheduler TSN usando Deep Reinforcement Learning con MaskablePPO"""
    
    def __init__(self, network: Network, num_envs=1, timeout_s=300, use_curriculum=False):
        """Inicializa el scheduler con una red y opcionalmente número de entornos"""
        self.network = network
        self.num_flows = len(network.flows)
        self.num_envs = num_envs
        self.timeout_s = timeout_s
        # Explicitly pass curriculum_enabled=False for testing
        self.env = DummyVecEnv([
            lambda: NetEnv(network, curriculum_enabled=use_curriculum, initial_complexity=1.0) 
            for _ in range(num_envs)
        ])
        # Mantener consistencia con el extractor usado al entrenar.
        policy_kwargs = dict(features_extractor_class=FeaturesExtractor)
        self.model = MaskablePPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            policy_kwargs=policy_kwargs,
        )
        self.res = None
        logging.info(f"Scheduler inicializado con {self.num_flows} flujos (curriculum: {use_curriculum})")
        
    def load_model(self, filepath: str, alg='MaskablePPO'):
        """Carga un modelo pre-entrenado"""
        if filepath.endswith('.zip'):
            filepath = filepath[:-4]
        if not os.path.isfile(f"{filepath}.zip"):
            logging.error(f"Modelo no encontrado: {filepath}.zip")
            return False
        try:
            self.model = MaskablePPO.load(filepath, self.env)
            logging.info(f"Modelo cargado: {filepath}")
            return True
        except Exception as e:
            logging.error(f"Error cargando modelo: {e}")
            return False
        
    def schedule(self) -> bool:
        """Ejecuta el scheduling usando el modelo DRL"""
        self.res = None
        
        # Intentar hasta 100 episodios como máximo
        for _ in range(100):   # +100 % intentos ⇒ mayor tasa de éxito
            obs = self.env.reset()
            done = False
            
            # Procesar un episodio completo
            while not done:
                # Predecir acción con máscara
                # `env_method` devuelve una lista de arrays (uno por entorno);
                # apilamos para obtener shape (n_envs, n_actions).
                action_masks = np.vstack(self.env.env_method('action_masks'))
                # Asegurar dtype int8 para plena compatibilidad
                action, _ = self.model.predict(
                    obs, deterministic=True, action_masks=action_masks.astype(np.int8)
                )
                
                # Ejecutar acción
                obs, _, dones, infos = self.env.step(action)

                # ⬇️  Propagar correctamente la terminación del episodio
                done = any(dones)

                if done:
                    for i, is_done in enumerate(dones):
                        if is_done and infos[i].get('success'):
                            self.res = infos[i].get('ScheduleRes')
                            return True      # éxito ⇒ abandonar bucle de episodios
                    # Reiniciar si terminó sin éxito
                    obs = self.env.reset()
                    done = False
            
        # Si llegamos aquí, no se encontró solución
        return False
    
    def get_res(self) -> ScheduleRes:
        """Retorna el resultado del scheduling"""
        return self.res

class ResAnalyzer:
    """Analiza y guarda resultados del scheduling"""
    
    # Threshold for GCL entry generation - easily modifiable class variable
    DEFAULT_GCL_GAP_THRESHOLD = 30
    
    def __init__(self, network: Network, results: ScheduleRes):
        """
        Inicializa el analizador y guarda resultados
        
        Args:
            network: Red TSN
            results: Resultado del scheduling
        """
        self.network = network
        self.results = results
        self.analyzer_id = id(self) # Generar un ID único para el analizador
        
        # Set instance variable from class default
        self.gap_threshold_us = self.DEFAULT_GCL_GAP_THRESHOLD
        
        # --- NUEVO: Calcular y almacenar tablas GCL estáticas ---
        self._gcl_tables = self._calculate_gcl_tables(self.gap_threshold_us)
        # --- FIN NUEVO ---

        # Guardar resultados a archivo
        if results:
            # Asegurarse de que el directorio OUT_DIR existe
            os.makedirs(OUT_DIR, exist_ok=True)
            
            # Usar el formato "by_link" consistentemente
            filename = os.path.join(OUT_DIR, f'schedule_res_by_link_{self.analyzer_id}.log')
            try:
                with open(filename, 'w') as f:
                    for link, operations in results.items():
                        f.write(f"Enlace: {link}\n")
                        for flow, op in operations:
                            f.write(f"  Flujo: {flow.flow_id}, Op: {op}\n")
                        f.write("\n")
                logging.info(f"Resultados guardados en {filename}")
            except Exception as e:
                logging.error(f"Error al guardar resultados: {e}")

    # --- NUEVO: Método para calcular las tablas GCL --------------------------
    def _calculate_gcl_tables(self, gap_thr_us: int = None) -> Dict[Link, List[Tuple[int, int]]]:
        """
        Genera la tabla GCL (lista de pares «tiempo, estado») para cada
        puerto-switch.

        ▸ Sólo se insertan 0/1 cuando el hueco entre la recepción de un paquete
          y el comienzo del siguiente supera `gap_thr_us` µs (default: valor de self.gap_threshold_us).
        ▸ Se enlaza el último paquete del hiperperíodo con el primero para que
          el cierre final también quede reflejado.
        ▸ IMPORTANTE: Para cada par de eventos, se añade un 0 (cierre) y un 1 (apertura)
        """
        from math import lcm
        
        # El umbral viene del atributo de instancia (por defecto 30 µs,
        # o el que se haya pasado vía --gcl-threshold).  **NO** lo
        # sobre-escribimos aquí; así generamos la "tabla corta".
        gcl_tables: Dict[Link, List[Tuple[int, int]]] = {}
        if not self.results:
            return gcl_tables

        for link, ops in self.results.items():
            # Sólo puertos cuyo ORIGEN es un switch ("S…", excluyendo "SRV…")
            src = link.link_id[0] if isinstance(link.link_id, tuple) else link.link_id.split("-")[0]
            if not (src.startswith("S") and not src.startswith("SRV")):
                continue

            # 1️⃣  Ordenar operaciones por inicio real (gating_time o start_time)
            ops_sorted = sorted(ops, key=lambda p: (p[1].gating_time or p[1].start_time))
            n = len(ops_sorted)
            if n == 0:
                continue

            # 2️⃣  Calcular hiperperíodo de ese puerto
            gcl_cycle = 1
            for f, _ in ops_sorted:
                gcl_cycle = lcm(gcl_cycle, f.period)

            # 3️⃣  Analizar cada operación – reset de listas por-link
            all_transmission_times: list[tuple[int, str]] = []
            all_reception_times:    list[tuple[int, str]] = []
            
            hyperperiod_link = gcl_cycle  # Hiperperiodo para este enlace

            #     Crear un par de eventos 0/1 por paquete
            for i in range(n):
                f_curr, op_curr = ops_sorted[i]
                
                # Índice del siguiente paquete (con wraparound)
                next_idx = (i + 1) % n
                f_next, op_next = ops_sorted[next_idx]

                # Tiempo cuando termina de llegar este paquete (necesitamos cerrar el gate)
                close_time = op_curr.reception_time
                
                # Tiempo cuando inicia la transmisión del siguiente paquete (reabrimos el gate)
                open_time = op_next.gating_time or op_next.start_time
                
                # Si es el último paquete, añadir un período para el wraparound
                if i == n - 1:
                    open_time += f_next.period

                # Calcular el gap entre recepción y siguiente transmisión
                gap = open_time - close_time
                
                # Para cada paquete, repetirlo durante todo el hiperperíodo
                repetitions = hyperperiod_link // f_curr.period
                for rep in range(repetitions):
                    offset = rep * f_curr.period
                    # Guardar tiempo de inicio y recepción (normalizado al hiperperiodo)
                    tx_t = (op_curr.start_time + offset) % hyperperiod_link
                    rx_t = (op_curr.reception_time + offset) % hyperperiod_link
                    all_transmission_times.append((tx_t, f_curr.flow_id))
                    all_reception_times.append((rx_t, f_curr.flow_id))

            # Ordenar los tiempos
            all_transmission_times.sort(key=lambda x: x[0])
            all_reception_times.sort(key=lambda x: x[0])
            
            # PASO 2: Generar eventos GCL con la tabla COMPLETA
            gcl_close_events: list[tuple[int,str,int,str]] = []
            
            # Buscar gaps significativos entre recepción y siguiente transmisión
            for rx_time, rx_flow in all_reception_times:
                
                # ➊ Iniciar variables cada iteración
                next_tx_time: int | None = None
                next_tx_flow: str | None = None

                # Buscar la siguiente transmisión > rx_time
                for tx_time, tx_flow in all_transmission_times:
                    if tx_time > rx_time:
                        next_tx_time = tx_time
                        next_tx_flow = tx_flow
                        break

                # Si no hay ninguna (wraparound) usa la primera del ciclo + hiperperíodo
                if next_tx_time is None and all_transmission_times:
                    first_tx_time, first_tx_flow = all_transmission_times[0]
                    next_tx_time = first_tx_time + hyperperiod_link
                    next_tx_flow = first_tx_flow
                
                # Protección extra – si, aun así, no existe TX, saltar este RX
                if next_tx_time is None:
                    continue
                 
                # Calcular el gap siempre (tabla completa – sin filtro)
                gap = (next_tx_time - rx_time) % hyperperiod_link

                #  Sólo añadimos el par 0/1 cuando el hueco supera
                #  el threshold definido por el usuario.
                if gap > gap_thr_us:
                    # Añadir eventos de cierre/apertura
                    gcl_close_events.append(
                        (rx_time, rx_flow, next_tx_time, next_tx_flow)
                    )

            # PASO 3: Generar los pares de eventos 0/1 para cada gap significativo
            events: List[Tuple[int, int]] = []
            for rx_time, rx_flow, next_tx_time, next_tx_flow in gcl_close_events:
                # Añadir evento de cierre (0) en el tiempo de recepción
                events.append((rx_time, 0))
                    
                # Añadir evento de apertura (1) cuando empieza el siguiente paquete
                events.append((next_tx_time % hyperperiod_link, 1))

            # 4️⃣  Ordenar todos los eventos por tiempo
            events.sort(key=lambda x: (x[0], x[1]))
            
            # 5️⃣  Eliminar estados duplicados o redundantes consecutivos
            final_table: List[Tuple[int, int]] = []
            last_state: int | None = None
            for t, s in events:
                if s != last_state:  # Solo añadir si cambia el estado
                    final_table.append((t, s))
                    last_state = s

            # 6️⃣  Garantizar que la tabla empiece "abierta" en t = 0 µs
            if not final_table or final_table[0][0] != 0:
                final_table.insert(0, (0, 1))
            elif final_table[0][0] == 0 and final_table[0][1] == 0:
                # Si el primer evento es cerrar en t=0, añadir apertura en t=0 antes
                final_table.insert(0, (0, 1))

            gcl_tables[link] = final_table

        return gcl_tables

    # --- NUEVO: Método para recalcular GCL con threshold diferente ---
    def recalculate_gcl_tables(self, new_threshold_us: int) -> Dict[Link, List[Tuple[int, int]]]:
        """
        Recalcula las tablas GCL con un nuevo threshold.
        
        Args:
            new_threshold_us: Nuevo valor de threshold en µs
            
        Returns:
            Diccionario con las nuevas tablas GCL
        """
        # Update instance threshold
        self.gap_threshold_us = new_threshold_us
        
        # Recalculate tables
        self._gcl_tables = self._calculate_gcl_tables(new_threshold_us)
        
        # Log the change
        print(f"GCL tables recalculated with threshold: {new_threshold_us}µs")
        
        return self._gcl_tables

    # --- NUEVO: Método para imprimir información de los flujos ---
    def print_flow_info(self):
        """Imprime una tabla con información detallada de cada flujo, incluyendo tamaños de paquete."""
        if not self.network or not self.network.flows:
            print("\nNo hay información de flujos disponible.")
            return
            
        print("\n" + "="*80)
        print("INFORMACIÓN DE FLUJOS")
        print("="*80)
        
        # Definir formato de tabla
        format_str = "{:<8} | {:<8} | {:<8} | {:<10} | {:<12} | {:<6}"
        
        # Imprimir cabecera
        print(format_str.format("Flujo", "Origen", "Destino", "Período (µs)", "Payload (B)", "Hops"))
        print("-"*8 + " | " + "-"*8 + " | " + "-"*8 + " | " + "-"*10 + " | " + "-"*12 + " | " + "-"*6)
        
        # Imprimir cada flujo
        scheduled_flows = set()
        if self.results:
            for link_ops in self.results.values():
                for flow, _ in link_ops:
                    scheduled_flows.add(flow.flow_id)
        
        for flow in self.network.flows:
            # Verificar si el flujo fue programado exitosamente
            status = "✓" if flow.flow_id in scheduled_flows else ""
            
            # Calcular número de hops
            num_hops = len(flow.path)
            
            # Imprimir información
            print(format_str.format(
                flow.flow_id, 
                flow.src_id, 
                flow.dst_id, 
                flow.period, 
                flow.payload,
                f"{num_hops} {status}"
            ))
            
        print("="*80 + "\n")

    # --- NUEVO: Método para imprimir las tablas GCL ---
    def print_gcl_tables(self):
        """
        Imprime las tablas GCL estáticas.
        Muestra solo la tabla GCL generada sin referencia a umbrales.
        """
        if not self._gcl_tables:
            print("\nNo se generaron tablas GCL (posiblemente no hubo gating o scheduling falló).")
            return

        print("\n" + "="*80)
        print("TABLA GCL GENERADA (t, estado)")
        print("="*80)

        for link, table in self._gcl_tables.items():
            # Re-calcular gcl_cycle aquí para mostrarlo
            gated_ops = [(f, op) for f, op in self.results.get(link, []) if op.gating_time is not None]
            if not gated_ops: continue
            gcl_cycle = 1
            for f, _ in gated_ops:
                gcl_cycle = math.lcm(gcl_cycle, f.period)

            print(f"\n--- Enlace: {link.link_id} (Ciclo GCL: {gcl_cycle} µs) ---")
            print(f"{'Tiempo (µs)':<12} | {'Estado':<6}")
            print(f"{'-'*12} | {'-'*6}")
            for time, state in table:
                print(f"{time:<12} | {state:<6}")

        print("="*80 + "\n")
