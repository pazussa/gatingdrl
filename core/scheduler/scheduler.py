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
from core.learning.environment import NetworkEnvironment

# Tipo para resultados
ScheduleRes = Dict[Link, List[Tuple[Flow, Operation]]]

class DrlScheduler:
    """Scheduler TSN usando Deep Reinforcement Learning con MaskablePPO"""
    
    def __init__(self, network: Network, num_envs=1, timeout_s=300, use_curriculum=False):
        """Inicializa el scheduler con una red y opcionalmente número de entornos"""
        self.network = network
        self.num_flows = len(network.flows)
        self.timeout_s = timeout_s
        # ──────────────────────────────────────────────────────────────
        #  INFERENCIA ⇒ un solo entorno
        #     Con varios envs el método schedule() se detiene en cuanto
        #     cualquiera termina (sea éxito o fallo), de modo que el
        #     primer fallo aborta el episodio y el recuento queda a 0.
        # ──────────────────────────────────────────────────────────────
        self.num_envs = 1
        self.env = DummyVecEnv([
            lambda: NetworkEnvironment(
                network,
                curriculum_enabled=use_curriculum,
                initial_complexity=1.0   # 100 % de los flujos en test
            )
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
        
    def schedule(self, max_episodes: int = 50, patience: int = 5) -> int:
        """
        Intenta como máximo ``max_episodes`` y se detiene si durante
        ``patience`` episodios seguidos no mejora el mejor resultado.

        Devuelve cuántos flujos se lograron programar en el mejor
        episodio encontrado (0 → ninguno).  La solución parcial queda en
        ``self.res`` y puede recuperarse con ``get_res()``.
        """

        self.res = None
        best_res: ScheduleRes | None = None
        best_count = 0
        no_improve = 0

        log = logging.getLogger(__name__)

        for ep in range(1, max_episodes + 1):
            obs = self.env.reset()
            done = False

            while not done:
                action_masks = np.vstack(self.env.env_method('action_masks'))
                action, _ = self.model.predict(
                    obs, deterministic=True,
                    action_masks=action_masks.astype(np.int8)
                )
                obs, _, dones, infos = self.env.step(action)
                done = any(dones)

                if not done:
                    continue

                # ---------- fin de episodio ----------
                scheduled_cnt = 0
                schedule_res = None
                for i, is_done in enumerate(dones):
                    if not is_done:
                        continue
                    schedule_res = infos[i].get('ScheduleRes')
                    if schedule_res:
                        scheduled_cnt = {
                            f.flow_id
                            for link_ops in schedule_res.values()
                            for f, _ in link_ops
                        }.__len__()
                    break

                if scheduled_cnt > best_count:
                    best_count = scheduled_cnt
                    best_res = schedule_res
                    no_improve = 0
                    log.info(f"[scheduler] episodio {ep}: "
                             f"nueva mejor marca {best_count}/{self.num_flows}")
                    if best_count == self.num_flows:          # ¡perfecto!
                        self.res = best_res
                        return best_count
                else:
                    no_improve += 1

                if no_improve >= patience:
                    log.warning(f"[scheduler] sin mejora en {patience} episodios; "
                                f"mejor = {best_count}/{self.num_flows}.")
                    self.res = best_res
                    return best_count

        # agotó max_episodes
        logging.getLogger(__name__).warning(
            f"[scheduler] alcanzado límite de {max_episodes} episodios; "
            f"mejor = {best_count}/{self.num_flows}")
        self.res = best_res
        return best_count

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

        # ───────────────  RESUMEN GLOBAL  ───────────────
        total_sched = len(scheduled_flows)
        total_flows = len(self.network.flows)
        print("-"*80)
        print(f"Programados con éxito: {total_sched}/{total_flows} flujos")
        print("="*80 + "\n")

    # --- NUEVO: Método para imprimir las tablas GCL ---
    def print_gcl_tables(self):
        """Print the generated GCL tables for visualization and debugging."""
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

    def calculate_latency_metrics(self):
        """
        Calcula métricas de latencia extremo-a-extremo para todos los flujos programados.
        
        Returns:
            dict: Diccionario con métricas de latencia (promedio, jitter, máxima, muestras)
        """
        import statistics as _stat
        
        print("🔍 INICIANDO CÁLCULO DE MÉTRICAS DE LATENCIA...")
        print(f"📊 Tenemos {len(self.network.flows)} flujos totales")
        print(f"📊 Tenemos {len(self.results)} enlaces con resultados")
        
        latencies = []
        flows_processed = []
        
        # Para cada flujo programado, calcular su latencia E2E
        for flow in self.network.flows:
            flow_id = flow.flow_id
            
            # Buscar todas las operaciones de este flujo
            flow_operations = []
            
            for link, operations in self.results.items():
                for f, op in operations:
                    if f.flow_id == flow_id:
                        flow_operations.append((link, op))
            
            print(f"🔎 Flujo {flow_id}: encontradas {len(flow_operations)} operaciones")
            
            # Si el flujo tiene operaciones programadas
            if flow_operations:
                flows_processed.append(flow_id)
                
                if len(flow_operations) == 1:
                    # Flujo de un solo hop
                    _, op = flow_operations[0]
                    latency = op.reception_time - op.start_time
                    latencies.append(latency)
                    print(f"  ✅ {flow_id} (1 hop): {op.start_time} → {op.reception_time} = {latency} µs")
                
                else:
                    # Flujo multi-hop: ordenar por start_time para asegurar orden correcto
                    sorted_ops = sorted(flow_operations, key=lambda x: x[1].start_time)
                    first_op = sorted_ops[0][1]   # Primera operación
                    last_op = sorted_ops[-1][1]   # Última operación
                    
                    latency = last_op.reception_time - first_op.start_time
                    latencies.append(latency)
                    print(f"  ✅ {flow_id} ({len(flow_operations)} hops): {first_op.start_time} → {last_op.reception_time} = {latency} µs")
            else:
                print(f"  ❌ {flow_id}: sin operaciones programadas")
        
        print(f"\n📈 RESUMEN: {len(flows_processed)} flujos procesados de {len(self.network.flows)} totales")
        print(f"📊 Flujos con latencias calculadas: {flows_processed}")
        
        if not latencies:
            print("⚠️  NO SE ENCONTRARON LATENCIAS PARA CALCULAR")
            print("🔥 FORZANDO RETORNO DE MÉTRICAS VACÍAS")
            return {
                "average": 0,
                "jitter": 0,
                "maximum": 0,
                "minimum": 0,
                "samples": []
            }
        
        # Calcular estadísticas
        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        min_lat = min(latencies)
        jitter = _stat.pstdev(latencies) if len(latencies) > 1 else 0
        
        # FORZAR SALIDA MÚLTIPLE
        print("\n" + "="*80)
        print("🎯 MÉTRICAS DE LATENCIA EXTREMO-A-EXTREMO")
        print("="*80)
        print(f"📊 Promedio: {avg_lat:.1f} µs")
        print(f"📊 Jitter:   {jitter:.1f} µs") 
        print(f"📊 Máxima:   {max_lat} µs")
        print(f"📊 Mínima:   {min_lat} µs")
        print(f"📊 Muestras: {len(latencies)} flujos")
        print(f"📊 Valores:  {latencies}")
        print("="*80)
        
        # También usar logging
        logging.info("🎯 MÉTRICAS DE LATENCIA EXTREMO-A-EXTREMO")
        logging.info(f"📊 Promedio: {avg_lat:.1f} µs | Jitter: {jitter:.1f} µs | Máxima: {max_lat} µs | Mínima: {min_lat} µs | Muestras: {len(latencies)}")
        
        return {
            "average": avg_lat,
            "jitter": jitter,
            "maximum": max_lat,
            "minimum": min_lat,
            "samples": latencies.copy()
        }
    
    def calculate_link_utilization(self):
        """
        Calcula la utilización de cada enlace como porcentaje del tiempo ocupado
        durante el hiperperíodo.
        
        
        Returns:
            dict: Diccionario con utilización por enlace y estadísticas globales
        """
        import math
        
        print("🔗 INICIANDO CÁLCULO DE UTILIZACIÓN DE ENLACES...")
        
        if not self.results:
            print("⚠️  No hay resultados de scheduling para analizar")
            return {"link_utilizations": {}, "global_stats": {}}
        
        link_utilizations = {}
        all_utilizations = []
        
        # Calcular hiperperíodo global
        all_periods = set()
        for link, operations in self.results.items():
            for flow, _ in operations:
                all_periods.add(flow.period)
        
        hyperperiod = 1
        for period in all_periods:
            hyperperiod = math.lcm(hyperperiod, period)
        
        print(f"📊 Hiperperíodo global: {hyperperiod} µs")
        
        # Calcular utilización para cada enlace
        for link, operations in self.results.items():
            if not operations:
                continue
                
            link_id = link.link_id if hasattr(link, 'link_id') else str(link)
            print(f"\n🔍 Analizando enlace: {link_id}")
            
            # Tiempo total ocupado en el hiperperíodo
            total_busy_time = 0
            transmission_events = []
            
            # Para cada operación, calcular todas sus repeticiones en el hiperperíodo
            for flow, operation in operations:
                # Tiempo de transmisión por paquete
                transmission_time = operation.end_time - (operation.gating_time or operation.start_time)
                
                # Número de repeticiones en el hiperperíodo
                repetitions = hyperperiod // flow.period
                
                # Tiempo total de todas las repeticiones
                flow_total_time = transmission_time * repetitions
                total_busy_time += flow_total_time
                
                print(f"  ➤ Flujo {flow.flow_id}: {transmission_time}µs × {repetitions} repeticiones = {flow_total_time}µs")
                
                # Guardar eventos para verificación (opcional)
                for rep in range(repetitions):
                    offset = rep * flow.period
                    start_tx = (operation.gating_time or operation.start_time) + offset
                    end_tx = operation.end_time + offset
                    transmission_events.append((start_tx, end_tx, flow.flow_id))
            
            # Calcular utilización como porcentaje
            utilization_percent = (total_busy_time / hyperperiod) * 100
            
            link_utilizations[str(link_id)] = {
                "utilization_percent": utilization_percent,
                "busy_time_us": total_busy_time,
                "hyperperiod_us": hyperperiod,
                "num_flows": len(operations),
                "transmission_events": len(transmission_events)
            }
            
            all_utilizations.append(utilization_percent)
            
            print(f"  📈 Utilización: {utilization_percent:.2f}% ({total_busy_time}/{hyperperiod} µs)")
        
        # Estadísticas globales
        if all_utilizations:
            global_stats = {
                "average_utilization": sum(all_utilizations) / len(all_utilizations),
                "max_utilization": max(all_utilizations),
                "min_utilization": min(all_utilizations),
                "total_links": len(all_utilizations),
                "hyperperiod_us": hyperperiod
            }
        else:
            global_stats = {
                "average_utilization": 0,
                "max_utilization": 0,
                "min_utilization": 0,
                "total_links": 0,
                "hyperperiod_us": hyperperiod
            }
        
        # Mostrar resumen
        print("\n" + "="*80)
        print("🔗 UTILIZACIÓN DE ENLACES")
        print("="*80)
        
        # Tabla detallada por enlace
        print(f"{'Enlace':<25} | {'Utilización':<12} | {'Tiempo Ocupado':<15} | {'Flujos':<6}")
        print("-"*25 + " | " + "-"*12 + " | " + "-"*15 + " | " + "-"*6)
        
        for link_str, stats in link_utilizations.items():
            print(f"{link_str:<25} | {stats['utilization_percent']:>10.2f}% | "
                  f"{stats['busy_time_us']:>13} µs | {stats['num_flows']:>4}")
        
        print("-"*80)
        print(f"📊 ESTADÍSTICAS GLOBALES:")
        print(f"   • Utilización promedio: {global_stats['average_utilization']:.2f}%")
        print(f"   • Utilización máxima:   {global_stats['max_utilization']:.2f}%")
        print(f"   • Utilización mínima:   {global_stats['min_utilization']:.2f}%")
        print(f"   • Enlaces analizados:   {global_stats['total_links']}")
        print(f"   • Hiperperíodo:         {global_stats['hyperperiod_us']} µs")
        print("="*80 + "\n")
        
        # Log también las métricas
        logging.info(
            f"🔗 Utilización de Enlaces → "
            f"Promedio: {global_stats['average_utilization']:.2f}% | "
            f"Máxima: {global_stats['max_utilization']:.2f}% | "
            f"Mínima: {global_stats['min_utilization']:.2f}% | "
            f"Enlaces: {global_stats['total_links']}"
        )
        
        return {
            "link_utilizations": link_utilizations,
            "global_stats": global_stats
        }
