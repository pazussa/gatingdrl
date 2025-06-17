import math
import numpy as np
from core.learning.env_utils import ErrorType, SchedulingError, find_next_event_time
from core.network.net import Net
from core.network.operation import Operation
# import directo (la función sigue existiendo)
from core.learning.env_actions import process_step_action
import math, statistics as _stat
# Add missing import for logging
import logging


def step(self, action):
    """
    Ejecuta un paso en el entorno con la acción dada.
    
    Este método implementa la lógica principal para:
    - Procesar la acción del agente
    - Calcular tiempos de transmisión
    - Manejar conflictos
    - Actualizar el estado del entorno
    - Calcular la recompensa
    
    Args:
        action: Acción multidimensional del agente RL
        
    Returns:
        Tuple: (observación, recompensa, terminado, truncado, info)
    """
    try:
        # ──────────────────────────────────────────────────────────────
        #  Inicializar métricas de latencia a 0 – se actualizarán al final
        #  del episodio si todos los flujos se completan con éxito.
        # ──────────────────────────────────────────────────────────────
        avg_lat = jitter = max_lat = 0

        # NUEVO: Extraer la selección de flujo de la acción y aplicarla ANTES de procesar
        flow_selection = int(action[-1])  # La última dimensión es la selección de flujo
        
        # IMPORTANTE: Aplicar la selección de flujo inmediatamente
        flow_reward_adj = 0.0
        agent_selected = False                     # Controla si se aplicó la selección RL
        original_flow_idx = self.current_flow_idx  # Guardar para comprobar si cambió
        
        if hasattr(self, 'current_candidate_flows') and self.current_candidate_flows:
            if 0 <= flow_selection < len(self.current_candidate_flows):
                selected_idx = self.current_candidate_flows[flow_selection]
                if not self.flow_completed[selected_idx]:
                    # Verificar que el flujo seleccionado tiene un hop válido para programar
                    if self.flow_progress[selected_idx] < len(self.flows[selected_idx].path):
                        self.current_flow_idx = selected_idx
                        agent_selected = True
                        
                        # Añadir información de debug para seguimiento
                        # ↓  Pasa a DEBUG para no saturar la consola
                        self.logger.debug(
                            "Agente seleccionó flujo %s (idx %d) de candidatos: %s",
                            self.flows[selected_idx].flow_id,
                            selected_idx,
                            [self.flows[idx].flow_id for idx in self.current_candidate_flows],
                        )
                        
                        # Evaluar si la selección fue buena basándose en características
                        selected_flow = self.flows[selected_idx]
                        
                        # Calcular recompensa por selección inteligente
                        period_factor   = 1.0 - (selected_flow.period / 10000)
                        payload_factor  = 1.0 - (selected_flow.payload / Net.MTU)
                        remaining       = len(selected_flow.path) - self.flow_progress[selected_idx]
                        progress_factor = remaining / len(selected_flow.path)

                        flow_reward_adj = (period_factor * 0.15 +
                                           payload_factor * 0.15 +
                                           progress_factor * 0.15)
                        
                        self.logger.debug(f"Flow selection: {flow_selection} → candidate #{selected_idx} (Flow {selected_flow.flow_id})")
                        self.logger.debug(f"Flow reward adjustment: +{flow_reward_adj:.4f}")
        
        # ──────────────────────────────────────────────────────────────
        #  ENFORCE PRIORITY SCHEDULING - IMMEDIATE FORWARDING FROM SWITCHES
        # ──────────────────────────────────────────────────────────────
        fifo_idx = self._next_fifo_idx()
        if fifo_idx is None:
            # No quedan flujos pendientes – debería terminar normalmente
            return self._get_observation(), 0, True, False, {"success": True}

        # Respetar la selección del agente salvo que no haya seleccionado
        if not agent_selected and self.current_flow_idx != fifo_idx:
            # Priorizar la transmisión desde switches
            prev_flow = self.flows[self.current_flow_idx]
            new_flow = self.flows[fifo_idx]
            prev_hop_idx = self.flow_progress[self.current_flow_idx]
            new_hop_idx = self.flow_progress[fifo_idx]
            
            # Verificar si es un cambio a un flujo que está en un switch esperando
            if new_hop_idx > 0:
                prev_link_id = new_flow.path[new_hop_idx - 1]
                dst_node = prev_link_id[1] if isinstance(prev_link_id, tuple) else prev_link_id.split('-')[1]
                is_at_switch = dst_node.startswith('S') and not dst_node.startswith('SRV')
                
                if is_at_switch:
                    self.logger.debug(f"Priorizando transmisión inmediata desde switch: flujo {new_flow.flow_id}")
            
            # Actualizar el flujo actual
            self.current_flow_idx = fifo_idx

        # A partir de aquí TODO el código sigue igual: ya trabajamos con el
        # flujo correcto; si posteriormente no cabe en el período, se lanzará
        # el SchedulingError habitual y el episodio terminará "con error".
        
        # Procesar la acción y obtener la información necesaria
        # ➋  Desestructuramos el nuevo elemento `guard_factor`
        (flow, hop_idx, link, gating, trans_time,
         guard_time, guard_factor,                # ← aquí
         offset_us, switch_gap, sw_src,
         is_egress_from_switch, gcl_strategy) = process_step_action(self, action)

        # ------------------------------------------------------------ #
        # 1. Calcular tiempos                                          #
        # ------------------------------------------------------------ #
        if hop_idx == 0:        # ---------- primer hop ----------
            # Registrar *exactamente* el instante en que se libera el primer bit
            # del paquete en el cliente → op.start_time (no global_time).
            if self.flow_first_tx[self.current_flow_idx] is None:
                # El objeto `op` se crea unas líneas más abajo; de momento
                # guardamos el valor provisional y lo sobrescribiremos enseguida.
                self.flow_first_tx[self.current_flow_idx] = -1

            # Si no se entrega una red, construir topología y flujos sencillos
            if self.flow_first_tx[self.current_flow_idx] is None:
                self.flow_first_tx[self.current_flow_idx] = self.global_time
            # ❶  Primer hop: basta con que el enlace esté libre
            # ➡️  Sólo el *primer* enlace respeta Net.PACKET_GAP_EXTRA
            earliest = max(
                self.link_busy_until[link],
                self.global_queue_busy_until,
                self.last_packet_start + self._next_packet_gap()
            )  # Removed offset_us
            # Obtener el switch de destino para este paquete
            dst_node = link.link_id[1] if isinstance(link.link_id, tuple) else link.link_id.split('-')[1]
            if dst_node.startswith('S'):
                # Asegurar separación mínima de 1μs entre llegadas al switch
                # Calcula el tiempo de llegada *potencial* al switch
                potential_arrival_time = earliest + trans_time + Net.DELAY_PROP
                min_arrival_time = self.switch_last_arrival[dst_node] + switch_gap
                if potential_arrival_time < min_arrival_time:
                    # Ajustar el tiempo de inicio para garantizar 1μs de separación en destino
                    delay = min_arrival_time - potential_arrival_time
                    earliest += delay
                    # Actualizar el tiempo de llegada real
                    arrival_time = min_arrival_time
                else:
                    arrival_time = potential_arrival_time
                # Actualizar el último tiempo de llegada registrado para este switch
                self.switch_last_arrival[dst_node] = arrival_time

            # Re-calcular la ventana tras cualquier retraso aplicado
            latest = earliest + Net.SYNC_ERROR

            # Para el primer hop desde ES, no hay gating. Dequeue es el inicio más temprano.
            offset   = 0            # sin margen de sincronía
            dequeue  = earliest     # comienza tan pronto el enlace queda libre
            end_time = dequeue + trans_time
            if end_time > flow.period:
                 # Primer hop (sale de una ES): no hay switch para crear espera
                 raise SchedulingError(ErrorType.PeriodExceed, "Excedió período")

            # --- Crear objeto Operation para hop_idx == 0 ---
            op_start_time = earliest
            op_gating_time = None # No hay gating desde ES
            op_latest_time = latest # Usamos latest calculado
            op_end_time = end_time
            # Crear la operación AHORA para que 'op' esté definida
            op = Operation(op_start_time, op_gating_time, op_latest_time, op_end_time)

            # ➕ GUARDAR datos que el visualizador necesita
            op.guard_factor      = guard_factor      # decisión RL
            op.min_gap_value     = switch_gap        # decisión RL
            op.guard_time        = guard_time        # longitud real del guard-band

            # ⏱️  ahora sí: fijamos el instante real de partida
            if self.flow_first_tx[self.current_flow_idx] == -1:
                self.flow_first_tx[self.current_flow_idx] = op_start_time

        else:                   # ---------- hops siguientes ----------
            # ❷  Resto de hops:
            prev_link_id = flow.path[hop_idx - 1]
            prev_link = self.link_dict[prev_link_id]
            prev_op = self.links_operations[prev_link][-1][1]

            # Tiempo base: cuando el paquete está listo en el nodo actual
            packet_ready_time = prev_op.reception_time

            # Earliest possible start considerando sólo llegada y disponibilidad del ENLACE
            # ⚠️  En hops posteriores **no** aplicamos la separación global:
            earliest_possible_start_on_link = max(packet_ready_time,
                                         self.link_busy_until[link],
                                         self.global_queue_busy_until)  # Removed offset_us

            # Earliest start considerando también la disponibilidad del PUERTO del SWITCH (si aplica)
            if is_egress_from_switch:
                # FCFS se mantiene, pero no registramos la espera (no la decide el agente)
                final_earliest_start = max(earliest_possible_start_on_link,
                                            self.switch_busy_until[sw_src])
            else:
                final_earliest_start = earliest_possible_start_on_link

            # Calcular la ventana 'latest' basada en el inicio más temprano real
            latest = final_earliest_start + Net.SYNC_ERROR

            # Determinar el tiempo real de DEQUEUE (inicio de transmisión)
            offset   = 0            # sin margen de sincronía
            dequeue  = final_earliest_start
            
            # Calcular tiempo de fin de transmisión
            end_time = dequeue + trans_time
            if end_time > flow.period:
                # Si no cabe en su periodo, abortar sin intentos de recolocación
                raise SchedulingError(ErrorType.PeriodExceed, "Excedió período")

            # --- Crear objeto Operation ---
            # start_time: Cuándo podría haber empezado (llegada + disponibilidad enlace)
            # gating_time: Cuándo empezó realmente (dequeue), si aplica gating
            # latest_time: Límite superior de la ventana para gating
            # end_time: Cuándo terminó la transmisión
            op_start_time = earliest_possible_start_on_link
            op_gating_time = dequeue if gating and is_egress_from_switch else None 
            # Importante: Si hay gating, latest_time debe ser igual a gating_time
            if gating and is_egress_from_switch:
                op_latest_time = op_gating_time  # Si hay gating, ambos deben ser iguales
            else:
                op_latest_time = latest  # Sin gating, latest_time mantiene su valor normal

            op_end_time = end_time

            op = Operation(op_start_time, op_gating_time, op_latest_time, op_end_time)

            # ➕ Actualizar también en la rama "hops > 0"
            op.guard_factor  = guard_factor
            op.min_gap_value = switch_gap
            op.guard_time    = guard_time
            # No guardamos esperas que no sean decisión del agente

            # ── Nuevo: garantizar separación mínima entre llegadas al switch destino ──
            dst_node = link.link_id[1] if isinstance(link.link_id, tuple) \
                       else link.link_id.split('-')[1]
            if dst_node.startswith('S'):                       # sólo switches reales
                arrival = op.reception_time - Net.DELAY_PROC_RX
                min_arrival = self.switch_last_arrival[dst_node] + switch_gap
                if arrival < min_arrival:
                    delay = min_arrival - arrival
                    op.add(delay)              # ajusta *todos* los tiempos de la operación
                    op.min_gap_wait += delay   # registrar espera por gap mínimo
                    dequeue   += delay
                    end_time  += delay
                    op_start_time += delay
                    op_end_time   += delay
                    arrival = min_arrival
                # Registrar llegada para el siguiente paquete
                self.switch_last_arrival[dst_node] = arrival

        # --- Regla *un‑solo‑paquete‑switch* ---
        # 2. Crear operación temporal                                  #
        # ------------------------------------------------------------ #
        # op ya está creado con los tiempos correctos
        self.temp_operations.append((link, op))

        # Resolver conflictos por desplazamiento
        offset = self._check_temp_operations()
        max_iter = 16  # salvaguarda contra bucles infinitos
        while offset is not None and max_iter:
            # Desplazar la operación según el offset de conflicto
            op_start_time += offset
            
            # Actualizar TODAS las propiedades temporales
            if op_gating_time is not None:
                # Con gating, todo se desplaza por igual
                op_gating_time += offset
                op_latest_time += offset  # latest siempre alineado con gating
            else:
                # Sin gating, latest avanza con start (son independientes)
                op_latest_time += offset
            
            # Tiempo final siempre se recalcula respecto al inicio real
            op_end_time = (op_gating_time if op_gating_time is not None else op_start_time) + trans_time

            # Recrear la operación con los nuevos tiempos
            op = Operation(op_start_time, op_gating_time, op_latest_time, op_end_time)
            
            # Si hay gating, validar que aún está dentro del período del flujo
            if op_gating_time is not None and op_end_time > flow.period:
                # El inicio real ocurriría después del final del período
                raise SchedulingError(
                    ErrorType.PeriodExceed, 
                    "Flow cycles into next period"
                )
            
            # Recrear el array de operaciones temporales con la actualizada
            self.temp_operations = [(link, op)]
            
            # Volver a verificar conflictos
            offset = self._check_temp_operations()
            max_iter -= 1
            
            # Use default fixed conflict resolution strategy
            # (previous conflict_strategy action dimension was removed)
            # Apply a small minimum offset to ensure progress
            if offset is not None and offset == 0:
                offset = max(1, int(switch_gap * Net.SWITCH_GAP_MIN))

        if max_iter == 0 and offset is not None:
            raise SchedulingError(
                ErrorType.PeriodExceed,
                "Failed to resolve conflict after 16 iterations"
            )

        #  ⛔  Ya no se generan ni reservan reglas GCL durante el scheduling.

        # ---------- REWARD SHAPING ----------
        GB_PEN      = 0.05   # guard-band (sí la decide RL)

        reward = 1.0
        reward -= GB_PEN * (guard_time / flow.period)

        # NUEVO: Añadir el ajuste de recompensa por selección inteligente de flujo
        reward += flow_reward_adj

    except SchedulingError as e:
        # Un flujo no cabe en su período ─ cerramos el episodio,
        #  pero entregando TODO lo que sí se ha programado hasta ahora.
        self.logger.debug(f"Fallo: {e.msg} (flujo {self.current_flow_idx})")

        partial_res = {lnk: ops.copy()          # copia superficial es suficiente
                       for lnk, ops in self.links_operations.items()}

        return self._get_observation(), -1, True, False, {
            "success": False,                  # episodio fallido
            "ScheduleRes": partial_res,        # ← planificación parcial
        }

    # ------------------------------------------------------------ #
    # 3. Avanzar progreso del flujo                                #
    # ------------------------------------------------------------ #
    self.links_operations[link].append((flow, op))
    self.temp_operations.clear()

    # 🌐 Registrar sólo si es el *primer* hop del flujo
    if hop_idx == 0:
        self.last_packet_start = op_start_time

    # ❷  Marcar el enlace como ocupado hasta que el paquete esté completamente recibido Y PROCESADO
    # Usar reception_time que ya incluye DELAY_PROP + DELAY_PROC_RX
    self.link_busy_until[link] = op.reception_time  # En lugar de op.end_time + Net.DELAY_PROP
    # 🔒 Mantener la sección crítica ocupada hasta que el frame se recibe
    self.global_queue_busy_until = op.reception_time

    if is_egress_from_switch:                 # ❷ liberar switch al terminar
        # Mantener el puerto bloqueado también durante la guard-band escogida
        # para reflejar exactamente la reserva temporal del modelo matemático
        self.switch_busy_until[sw_src] = op.end_time + guard_time

    self.flow_progress[self.current_flow_idx] += 1


    # ❸  El "reloj" global se redefine como el evento más temprano pendiente
    next_events = [*self.link_busy_until.values(),
                  *self.switch_busy_until.values()]
    # Si no quedan eventos pendientes, mantenemos el reloj en lugar de "rebobinar" a 0
    self.global_time = min(next_events, default=self.global_time)

    # ¿Terminó este flujo?
    if self.flow_progress[self.current_flow_idx] == len(flow.path):
        self.flow_completed[self.current_flow_idx] = True
        # ---------- verificación latencia extremo-a-extremo ----------
        fst = self.flow_first_tx[self.current_flow_idx]
        e2e_latency = op.reception_time - fst if fst is not None else 0
        
        # NUEVO: Guardar latencia e2e para estadísticas globales
        self._flow_latencies.append(e2e_latency)

        # ➕ Registrar la muestra en el acumulador global
        self._latency_samples.append(e2e_latency)
        
        # *Incluir* la propagación y el procesamiento de RX en el presupuesto
        # 💡 Tomar el peor‑caso acumulativo sobre la ruta completa
        hops = len(flow.path)
        e2e_budget = (flow.e2e_delay +                      # presupuesto nominal
                      Net.DELAY_PROP   * hops +            # propagación
                      Net.DELAY_PROC_RX * hops +           # procesado RX
                      guard_time        * (hops - 1))      # guard‑band por hop
        if e2e_latency > e2e_budget:
            raise SchedulingError(ErrorType.PeriodExceed,
                                  f"E2E delay {e2e_latency} > {e2e_budget}")
        reward += 2

    # ¿Terminó episodio?
    done = all(self.flow_completed)
    
    # Después de procesar un hop de un flujo, si el destino es un switch,
    # inmediatamente preparar el siguiente hop para transmisión
    if not done and hop_idx < len(flow.path) - 1:
        dst_node = link.link_id[1] if isinstance(link.link_id, tuple) else link.link_id.split('-')[1]
        if dst_node.startswith('S') and not dst_node.startswith('SRV'):
            # Este paquete llegó a un switch, marcar como alta prioridad
            # para ser procesado en el siguiente paso
            arrival_time = op.reception_time
            self.switch_last_arrival[dst_node] = min(arrival_time, self.switch_last_arrival[dst_node])
            
            # Actualizar el reloj global para favorecer el procesamiento inmediato
            # de este paquete que acaba de llegar al switch
            if arrival_time < self.global_time:
                next_events = [*self.link_busy_until.values(), *self.switch_busy_until.values(), arrival_time]
                self.global_time = min(next_events)

    # Gestionar el curriculum learning
    if done and all(self.flow_completed):
        # Episodio exitoso: incrementar contador de éxitos consecutivos
        self.consecutive_successes += 1
        # Añadir bonificación de recompensa proporcional al nivel de complejidad
        reward += 5.0 * self.current_complexity

        # ──────────────────────────────────────────────────────────────
        #  〽️  Calculamos las estadísticas de latencia del episodio
        # ──────────────────────────────────────────────────────────────
        if self._latency_samples:
            avg_lat = sum(self._latency_samples) / len(self._latency_samples)
            max_lat = max(self._latency_samples)
            jitter  = _stat.pstdev(self._latency_samples) if len(self._latency_samples) > 1 else 0
            self.logger.info(
                f"⏱️  Latencia promedio={avg_lat:.1f} µs · "
                f"jitter={jitter:.1f} µs · "
                f"máxima={max_lat} µs"
            )
        else:
            avg_lat = max_lat = jitter = 0
        
        # Mostrar información del progreso del curriculum
        if self.curriculum_enabled:
            self.logger.info(f"Éxito con {len(self.flows)}/{self.total_flows} flujos (complejidad: {self.current_complexity:.2f}, éxitos: {self.consecutive_successes}/3)")
    
    info = {
        "success": done,
        "ScheduleRes": self.links_operations.copy() if done else None,
        "curriculum_level": self.current_complexity,
        "num_flows": len(self.flows),

        # ─── métricas de latencia E2E ───
        "latency_us": {
            "average": avg_lat,
            "jitter" : jitter,
            "maximum": max_lat,
            "samples": self._latency_samples.copy(),
        },
        # NUEVO: Añadir información sobre selección de flujos
        "flow_selection": {
            "current_flow_idx": self.current_flow_idx,
            "available_candidates": getattr(self, 'current_candidate_flows', []),
            "selected_option": flow_selection,
            "reward_adj": flow_reward_adj
        }
    }

    # ────────────────────────────────────────────────────────────────
    #  Al terminar el episodio (todos los flujos entregados) → métricas
    # ────────────────────────────────────────────────────────────────
    if done and self._flow_latencies:
        avg_lat = sum(self._flow_latencies) / len(self._flow_latencies)
        max_lat = max(self._flow_latencies)
        import statistics as _st
        jitter = _st.pstdev(self._flow_latencies) if len(self._flow_latencies) > 1 else 0

        # Log amigable
        self.logger.info(
            f"⏱️  Latencia promedio={avg_lat:.0f} µs · "
            f"jitter={jitter:.0f} µs · máxima={max_lat:.0f} µs"
        )

        # Añadir al diccionario `info`
        info["latency_us"] = {
            "average": avg_lat,
            "jitter":  jitter,
            "max":     max_lat,
        }

    return self._get_observation(), reward, done, False, info
