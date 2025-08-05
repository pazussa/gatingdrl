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


def step(self, command):
    """
    Realiza un paso en el entorno según la acción proporcionada.
    
    Este método implementa la lógica principal para:
    - Procesar la acción del agente
    - Calcular tiempos de transmisión
    - Manejar conflictos
    - Actualizar el estado del entorno
    - Calcular la recompensa
    
    Args:
        command: Acción multidimensional del agente RL
        
    Returns:
        Tuple: (observación, recompensa, terminado, truncado, metadata)
    """
    try:
        # ──────────────────────────────────────────────────────────────
        #  Inicializar métricas de latencia a 0 – se actualizarán al final
        #  del episodio si todos los flujos se completan con éxito.
        # ──────────────────────────────────────────────────────────────
        mean_delay = delay_variance = peak_delay = 0

        # NUEVO: Extraer la selección de flujo de la acción y aplicarla ANTES de procesar
        stream_choice = int(command[-1])  # La última dimensión es la selección de flujo
        
        # IMPORTANTE: Aplicar la selección de flujo inmediatamente
        selection_bonus = 0.0
        policy_applied = False                     # Controla si se aplicó la selección RL
        backup_stream_id = self.active_stream_id  # Guardar para comprobar si cambió
        
        if hasattr(self, 'active_nominees') and self.active_nominees:
            if 0 <= stream_choice < len(self.active_nominees):
                chosen_identifier = self.active_nominees[stream_choice]
                if not self.stream_finished[chosen_identifier]:
                    # Verificar que el flujo seleccionado tiene un hop válido para programar
                    if self.stream_advancement[chosen_identifier] < len(self.traffic_streams[chosen_identifier].path):
                        self.active_stream_id = chosen_identifier
                        policy_applied = True
                        
                        # Añadir información de debug para seguimiento
                        # ↓  Pasa a DEBUG para no saturar la consola
                        self.event_recorder.debug(
                            "Agente seleccionó flujo %s (identifier %d) de nominees: %s",
                            self.traffic_streams[chosen_identifier].flow_id,
                            chosen_identifier,
                            [self.traffic_streams[identifier].flow_id for identifier in self.active_nominees],
                        )
                        
                        # Evaluar si la selección fue buena basándose en características
                        chosen_stream = self.traffic_streams[chosen_identifier]
                        
                        # Calcular recompensa por selección inteligente
                        frequency_coefficient   = 1.0 - (chosen_stream.period / 10000)
                        size_coefficient  = 1.0 - (chosen_stream.payload / Net.MTU)
                        pending_segments       = len(chosen_stream.path) - self.stream_advancement[chosen_identifier]
                        completion_ratio = pending_segments / len(chosen_stream.path)

                        selection_bonus = (frequency_coefficient * 0.15 +
                                           size_coefficient * 0.15 +
                                           completion_ratio * 0.15)
                        
                        self.event_recorder.debug(f"Flow selection: {stream_choice} → candidate #{chosen_identifier} (Flow {chosen_stream.flow_id})")
                        self.event_recorder.debug(f"Flow performance_score adjustment: +{selection_bonus:.4f}")
        
        # ──────────────────────────────────────────────────────────────
        #  ENFORCE PRIORITY SCHEDULING - IMMEDIATE FORWARDING FROM SWITCHES
        # ──────────────────────────────────────────────────────────────
        queue_position = self._next_fifo_idx()
        if queue_position is None:
            # No quedan flujos pendientes – debería terminar normalmente
            return self._get_observation(), 0, True, False, {"success": True}

        # Respetar la selección del agente salvo que no haya seleccionado
        if not policy_applied and self.active_stream_id != queue_position:
            # Priorizar la transmisión desde switches
            previous_stream = self.traffic_streams[self.active_stream_id]
            next_stream = self.traffic_streams[queue_position]
            prior_segment = self.stream_advancement[self.active_stream_id]
            next_segment = self.stream_advancement[queue_position]
            
            # Verificar si es un cambio a un flujo que está en un switch esperando
            if next_segment > 0:
                upstream_identifier = next_stream.path[next_segment - 1]
                target_endpoint = upstream_identifier[1] if isinstance(upstream_identifier, tuple) else upstream_identifier.split('-')[1]
                at_network_node = target_endpoint.startswith('S') and not target_endpoint.startswith('SRV')
                
                if at_network_node:
                    self.event_recorder.debug(f"Priorizando transmisión inmediata desde switch: flujo {next_stream.flow_id}")
            
            # Actualizar el flujo actual
            self.active_stream_id = queue_position

        # A partir de aquí TODO el código sigue igual: ya trabajamos con el
        # flujo correcto; si posteriormente no cabe en el período, se lanzará
        # el SchedulingError habitual y el episodio terminará "con error".
        
        # Procesar la acción y obtener la información necesaria
        # ➋  Desestructuramos el nuevo elemento `protection_multiplier`
        (data_stream, segment_index, network_connection, time_synchronization, transmission_duration,
         safety_interval, protection_multiplier,                # ← aquí
         timing_offset, inter_packet_spacing, sw_src,
         outbound_from_switch, scheduling_policy) = process_step_action(self, command)

        # ------------------------------------------------------------ #
        # 1. Calcular tiempos                                          #
        # ------------------------------------------------------------ #
        if segment_index == 0:        # ---------- primer hop ----------
            # Registrar *exactamente* el instante en que se libera el primer bit
            # del paquete en el cliente → operation_record.start_time (no simulation_clock).
            if self.initial_transmission[self.active_stream_id] is None:
                # El objeto `operation_record` se crea unas líneas más abajo; de momento
                # guardamos el valor provisional y lo sobrescribiremos enseguida.
                self.initial_transmission[self.active_stream_id] = -1

            # Si no se entrega una red, construir topología y flujos sencillos
            if self.initial_transmission[self.active_stream_id] is None:
                self.initial_transmission[self.active_stream_id] = self.simulation_clock
            # ❶  Primer hop: basta con que el enlace esté libre
            # ➡️  Sólo el *primer* enlace respeta Net.PACKET_GAP_EXTRA
            minimum_start = max(
                self.connection_free_time[network_connection],
                self.system_busy_time,
                self.last_packet_start + self._next_packet_gap()
            )  # Removed timing_offset
            # Obtener el switch de destino para este paquete
            target_endpoint = network_connection.link_id[1] if isinstance(network_connection.link_id, tuple) else network_connection.link_id.split('-')[1]
            if target_endpoint.startswith('S'):
                # Asegurar separación mínima de 1μs entre llegadas al switch
                # Calcula el tiempo de llegada *potencial* al switch
                estimated_arrival = minimum_start + transmission_duration + Net.DELAY_PROP
                minimum_ingress = self.node_last_reception[target_endpoint] + inter_packet_spacing
                if estimated_arrival < minimum_ingress:
                    # Ajustar el tiempo de inicio para garantizar 1μs de separación en destino
                    latency_offset = minimum_ingress - estimated_arrival
                    minimum_start += latency_offset
                    # Actualizar el tiempo de llegada real
                    reception_instant = minimum_ingress
                else:
                    reception_instant = estimated_arrival
                # Actualizar el último tiempo de llegada registrado para este switch
                self.node_last_reception[target_endpoint] = reception_instant

            # Re-calcular la ventana tras cualquier retraso aplicado
            maximum_time = minimum_start + Net.SYNC_ERROR

            # Para el primer hop desde ES, no hay time_synchronization. Dequeue es el inicio más temprano.
            time_adjustment   = 0            # sin margen de sincronía
            transmission_start  = minimum_start     # comienza tan pronto el enlace queda libre
            completion_instant = transmission_start + transmission_duration
            if completion_instant > data_stream.period:
                 # Primer hop (sale de una ES): no hay switch para crear espera
                 raise SchedulingError(ErrorType.DEADLINE_VIOLATION, "Excedió período")

            # --- Crear objeto Operation para segment_index == 0 ---
            operation_begin = minimum_start
            synchronization_time = None # No hay time_synchronization desde ES
            deadline_time = maximum_time # Usamos maximum_time calculado
            operation_finish = completion_instant
            # Crear la operación AHORA para que 'operation_record' esté definida
            operation_record = Operation(operation_begin, synchronization_time, deadline_time, operation_finish)

            # ➕ GUARDAR datos que el visualizador necesita
            operation_record.protection_multiplier      = protection_multiplier      # decisión RL
            operation_record.min_gap_value     = inter_packet_spacing        # decisión RL
            operation_record.safety_interval        = safety_interval        # longitud real del guard-band

            # ⏱️  ahora sí: fijamos el instante real de partida
            if self.initial_transmission[self.active_stream_id] == -1:
                self.initial_transmission[self.active_stream_id] = operation_begin

        else:                   # ---------- path_segments siguientes ----------
            # ❷  Resto de path_segments:
            upstream_identifier = data_stream.path[segment_index - 1]
            upstream_connection = self.connection_registry[upstream_identifier]
            previous_operation = self.connection_activities[upstream_connection][-1][1]

            # Tiempo base: cuando el paquete está listo en el nodo actual
            data_ready_instant = previous_operation.reception_time

            # Earliest possible start considerando sólo llegada y disponibilidad del ENLACE
            # ⚠️  En path_segments posteriores **no** aplicamos la separación global:
            link_availability = max(data_ready_instant,
                                         self.connection_free_time[network_connection],
                                         self.system_busy_time)  # Removed timing_offset

            # Earliest start considerando también la disponibilidad del PUERTO del SWITCH (si aplica)
            if outbound_from_switch:
                # FCFS se mantiene, pero no registramos la espera (no la decide el agente)
                actual_start_time = max(link_availability,
                                            self.node_free_time[sw_src])
            else:
                actual_start_time = link_availability

            # Calcular la ventana 'maximum_time' basada en el inicio más temprano real
            maximum_time = actual_start_time + Net.SYNC_ERROR

            # Determinar el tiempo real de DEQUEUE (inicio de transmisión)
            time_adjustment   = 0            # sin margen de sincronía
            transmission_start  = actual_start_time
            
            # Calcular tiempo de fin de transmisión
            completion_instant = transmission_start + transmission_duration
            if completion_instant > data_stream.period:
                # Si no cabe en su periodo, abortar sin intentos de recolocación
                raise SchedulingError(ErrorType.DEADLINE_VIOLATION, "Excedió período")

            # --- Crear objeto Operation ---
            # start_time: Cuándo podría haber empezado (llegada + disponibilidad enlace)
            # gating_time: Cuándo empezó realmente (transmission_start), si aplica time_synchronization
            # latest_time: Límite superior de la ventana para time_synchronization
            # completion_instant: Cuándo terminó la transmisión
            operation_begin = link_availability
            synchronization_time = transmission_start if time_synchronization and outbound_from_switch else None 
            # Importante: Si hay time_synchronization, latest_time debe ser igual a gating_time
            if time_synchronization and outbound_from_switch:
                deadline_time = synchronization_time  # Si hay time_synchronization, ambos deben ser iguales
            else:
                deadline_time = maximum_time  # Sin time_synchronization, latest_time mantiene su valor normal

            operation_finish = completion_instant

            operation_record = Operation(operation_begin, synchronization_time, deadline_time, operation_finish)

            # ➕ Actualizar también en la rama "path_segments > 0"
            operation_record.protection_multiplier  = protection_multiplier
            operation_record.min_gap_value = inter_packet_spacing
            operation_record.safety_interval    = safety_interval
            # No guardamos esperas que no sean decisión del agente

            # ── Nuevo: garantizar separación mínima entre llegadas al switch destino ──
            target_endpoint = network_connection.link_id[1] if isinstance(network_connection.link_id, tuple) \
                       else network_connection.link_id.split('-')[1]
            if target_endpoint.startswith('S'):                       # sólo switches reales
                ingress_time = operation_record.reception_time - Net.DELAY_PROC_RX
                earliest_reception = self.node_last_reception[target_endpoint] + inter_packet_spacing
                if ingress_time < earliest_reception:
                    latency_offset = int(earliest_reception - ingress_time)  # Convertir a entero
                    operation_record.add(latency_offset)              # ajusta *todos* los tiempos de la operación
                    operation_record.min_gap_wait += latency_offset   # registrar espera por gap mínimo
                    transmission_start   += latency_offset
                    completion_instant  += latency_offset
                    operation_begin += latency_offset
                    operation_finish   += latency_offset
                    # CRUCIAL: Actualizar también las variables locales para conflict resolution
                    if synchronization_time is not None:
                        synchronization_time += latency_offset
                        deadline_time += latency_offset  # Mantener igualdad deadline_time == synchronization_time
                    else:
                        deadline_time += latency_offset
                    ingress_time = int(earliest_reception)  # También convertir a entero
                # Registrar llegada para el siguiente paquete
                self.node_last_reception[target_endpoint] = ingress_time

        # --- Regla *un‑solo‑paquete‑switch* ---
        # 2. Crear operación temporal                                  #
        # ------------------------------------------------------------ #
        # operation_record ya está creado con los tiempos correctos
        self.provisional_activities.append((network_connection, operation_record))

        # Resolver conflictos por desplazamiento
        time_adjustment = self._check_temp_operations()
        iteration_limit = 16  # salvaguarda contra bucles infinitos
        iterations_used = 0  # Contador para métricas
        
        while time_adjustment is not None and iteration_limit:
            iterations_used += 1
            # Desplazar la operación según el time_adjustment de conflicto
            operation_begin += time_adjustment
            
            # Actualizar TODAS las propiedades temporales
            if synchronization_time is not None:
                # Con time_synchronization, todo se desplaza por igual
                synchronization_time += time_adjustment
                deadline_time = synchronization_time  # latest_time debe ser igual a gating_time cuando hay time_synchronization
            else:
                # Sin time_synchronization, maximum_time avanza con start (son independientes)
                deadline_time += time_adjustment
            
            # Tiempo final siempre se recalcula respecto al inicio real
            operation_finish = (synchronization_time if synchronization_time is not None else operation_begin) + transmission_duration

            # Recrear la operación con los nuevos tiempos
            operation_record = Operation(operation_begin, synchronization_time, deadline_time, operation_finish)
            
            # Validar que la operación está dentro del período del flujo (tanto con como sin time_synchronization)
            if operation_finish > data_stream.period:
                # La operación se extiende más allá del final del período
                raise SchedulingError(
                    ErrorType.DEADLINE_VIOLATION, 
                    f"Operation completion {operation_finish} exceeds flow period {data_stream.period}"
                )
            
            # Recrear el array de operaciones temporales con la actualizada
            self.provisional_activities = [(network_connection, operation_record)]
            
            # Volver a verificar conflictos
            time_adjustment = self._check_temp_operations()
            iteration_limit -= 1
            
            # Use default fixed conflict resolution strategy
            # (previous conflict_strategy command dimension was removed)
            # Apply a small minimum time_adjustment to ensure progress
            if time_adjustment is not None and time_adjustment == 0:
                time_adjustment = max(1, int(inter_packet_spacing * Net.SWITCH_GAP_MIN))

        # Registrar métricas de resolución de conflictos
        try:
            from tools.complexity_metrics import get_metrics
            metrics = get_metrics()
            metrics.record_conflict_resolution(iterations_used)
        except ImportError:
            pass  # Métricas opcionales

        if iteration_limit == 0 and time_adjustment is not None:
            raise SchedulingError(
                ErrorType.DEADLINE_VIOLATION,
                "Failed to resolve conflict after 16 iterations"
            )

        #  ⛔  Ya no se generan ni reservan reglas GCL durante el scheduling.

        # ---------- REWARD SHAPING ----------
        guard_penalty      = 0.01   # Reducir penalización por guard-band de 0.05 a 0.01

        performance_score = 0.5  # Recompensa base más moderada (era 1.0)
        performance_score -= guard_penalty * (safety_interval / data_stream.period)

        # NUEVO: Añadir el ajuste de recompensa por selección inteligente de flujo
        performance_score += selection_bonus
        
        # Añadir recompensa por completar un hop exitosamente
        performance_score += 0.1  # Pequeña recompensa por progreso
        
        # NUEVO: Normalizar recompensa para evitar explosiones
        performance_score = np.clip(performance_score, -2.0, 2.0)  # Limitar entre -2 y +2

    except SchedulingError as e:
        # Un flujo no cabe en su período ─ simplemente lo saltamos y continuamos
        self.event_recorder.debug(f"Fallo: {e.error_message} (flujo {self.active_stream_id}) - SALTANDO")
        
        # Marcar el flujo actual como terminado (fallido) para continuar
        programmed_flows = sum(1 for finished in self.stream_finished if finished)
        remaining_flows = sum(1 for finished in self.stream_finished if not finished)
        
        self.event_recorder.info(f"Progreso parcial: {programmed_flows} flujos completados, {remaining_flows} pendientes")
        self.event_recorder.info(f"❌ Saltando flujo {self.traffic_streams[self.active_stream_id].flow_id} por fallo de scheduling")
        
        # Marcar el flujo como terminado (fallido)
        self.stream_finished[self.active_stream_id] = True
        
        # Aplicar penalización por flujo fallido ESCALADA según progreso
        # En lugar de -100 fijo, penalizar proporcionalmente
        completed_flows = sum(1 for finished in self.stream_finished if finished)
        total_flows = len(self.stream_finished)
        progress_ratio = (completed_flows - 1) / total_flows  # -1 porque acabamos de marcar uno como fallido
        
        # Penalización más suave: de -1 a -10 según cuántos flujos llevamos completados
        performance_score = -1 - (9 * progress_ratio)
        
        # NUEVO: Aplicar normalización también a las penalizaciones
        performance_score = np.clip(performance_score, -2.0, 2.0)
        
        # Limpiar actividades provisionales
        self.provisional_activities.clear()
        
        # Verificar si el episodio está completo
        episode_complete = all(self.stream_finished)
        
        # Retornar inmediatamente sin procesar el flujo fallido
        metadata = {
            "success": episode_complete,
            "ScheduleRes": self.connection_activities.copy() if episode_complete else None,
            "curriculum_level": self.difficulty_level,
            "stream_count": len(self.traffic_streams),
            "latency_us": {
                "average": 0,
                "delay_variance": 0,
                "maximum": 0,
                "samples": [],
            },
            "stream_choice": {
                "active_stream_id": self.active_stream_id,
                "available_candidates": getattr(self, 'active_nominees', []),
                "selected_option": stream_choice,
                "reward_adj": selection_bonus
            }
        }
        
        return self._get_observation(), performance_score, episode_complete, False, metadata

    # ------------------------------------------------------------ #
    # 3. Avanzar progreso del flujo                                #
    # ------------------------------------------------------------ #
    self.connection_activities[network_connection].append((data_stream, operation_record))
    self.provisional_activities.clear()

    # 🌐 Registrar sólo si es el *primer* hop del flujo
    if segment_index == 0:
        self.last_packet_start = operation_begin

    # ❷  Marcar el enlace como ocupado hasta que el paquete esté completamente recibido Y PROCESADO
    # Usar reception_time que ya incluye DELAY_PROP + DELAY_PROC_RX
    self.connection_free_time[network_connection] = operation_record.reception_time  # En lugar de operation_record.completion_instant + Net.DELAY_PROP
    # 🔒 Mantener la sección crítica ocupada hasta que el frame se recibe
    self.system_busy_time = operation_record.reception_time

    if outbound_from_switch:                 # ❷ liberar switch al terminar
        # Mantener el puerto bloqueado también durante la guard-band escogida
        # para reflejar exactamente la reserva temporal del modelo matemático
        self.node_free_time[sw_src] = operation_record.completion_instant + safety_interval

    self.stream_advancement[self.active_stream_id] += 1


    # ❸  El "reloj" global se redefine como el evento más temprano pendiente
    pending_activities = [*self.connection_free_time.values(),
                  *self.node_free_time.values()]
    # Si no quedan eventos pendientes, mantenemos el reloj en lugar de "rebobinar" a 0
    self.simulation_clock = min(pending_activities, default=self.simulation_clock)

    # ¿Terminó este flujo?
    if self.stream_advancement[self.active_stream_id] == len(data_stream.path):
        self.stream_finished[self.active_stream_id] = True
        
        # Registrar métricas del flujo completado
        try:
            from tools.complexity_metrics import get_metrics
            metrics = get_metrics()
            metrics.record_flow_processing(data_stream.flow_id, len(data_stream.path))
        except ImportError:
            pass  # Métricas opcionales
            
        # ---------- verificación latencia extremo-a-extremo ----------
        first_timestamp = self.initial_transmission[self.active_stream_id]
        total_delay = operation_record.reception_time - first_timestamp if first_timestamp is not None else 0
        
        # NUEVO: Guardar latencia e2e para estadísticas globales
        self._flow_latencies.append(total_delay)

        # ➕ Registrar la muestra en el acumulador global
        self._latency_samples.append(total_delay)
        
        # *Incluir* la propagación y el procesamiento de RX en el presupuesto
        # 💡 Tomar el peor‑caso acumulativo sobre la ruta completa
        path_segments = len(data_stream.path)
        end_to_end_allowance = (data_stream.e2e_delay +                      # presupuesto nominal
                      Net.DELAY_PROP   * path_segments +            # propagación
                      Net.DELAY_PROC_RX * path_segments +           # procesado RX
                      safety_interval        * (path_segments - 1))      # guard‑band por hop
        if total_delay > end_to_end_allowance:
            raise SchedulingError(ErrorType.DEADLINE_VIOLATION,
                                  f"E2E latency_offset {total_delay} > {end_to_end_allowance}")
        performance_score += 2

    # ¿Terminó episodio?
    episode_complete = all(self.stream_finished)
    
    # Después de procesar un hop de un flujo, si el destino es un switch,
    # inmediatamente preparar el siguiente hop para transmisión
    if not episode_complete and segment_index < len(data_stream.path) - 1:
        target_endpoint = network_connection.link_id[1] if isinstance(network_connection.link_id, tuple) else network_connection.link_id.split('-')[1]
        if target_endpoint.startswith('S') and not target_endpoint.startswith('SRV'):
            # Este paquete llegó a un switch, marcar como alta prioridad
            # para ser procesado en el siguiente paso
            reception_instant = operation_record.reception_time
            self.node_last_reception[target_endpoint] = min(reception_instant, self.node_last_reception[target_endpoint])
            
            # Actualizar el reloj global para favorecer el procesamiento inmediato
            # de este paquete que acaba de llegar al switch
            if reception_instant < self.simulation_clock:
                pending_activities = [*self.connection_free_time.values(), *self.node_free_time.values(), reception_instant]
                self.simulation_clock = min(pending_activities)

    # Gestionar el curriculum learning
    if episode_complete and all(self.stream_finished):
        # Episodio exitoso: incrementar contador de éxitos consecutivos
        self.streak_count += 1
        # Añadir bonificación de recompensa proporcional al nivel de complejidad
        # MODERADA la bonificación para evitar recompensas extremas
        performance_score += min(2.0 * self.difficulty_level, 10.0)  # Máximo +10
        
        # NUEVO: Normalizar bonificación final también
        performance_score = np.clip(performance_score, -2.0, 15.0)  # Permitir hasta +15 para éxito completo

        # ──────────────────────────────────────────────────────────────
        #  〽️  Calculamos las estadísticas de latencia del episodio
        # ──────────────────────────────────────────────────────────────
        if self._latency_samples:
            mean_delay = sum(self._latency_samples) / len(self._latency_samples)
            peak_delay = max(self._latency_samples)
            delay_variance  = _stat.pstdev(self._latency_samples) if len(self._latency_samples) > 1 else 0
            self.event_recorder.info(
                f"⏱️  Latencia promedio={mean_delay:.1f} µs · "
                f"delay_variance={delay_variance:.1f} µs · "
                f"máxima={peak_delay} µs"
            )
        else:
            mean_delay = peak_delay = delay_variance = 0
        
        # Mostrar información del progreso del curriculum
        if self.adaptive_learning:
            self.event_recorder.info(f"Éxito con {len(self.traffic_streams)}/{self.complete_stream_count} flujos (complejidad: {self.difficulty_level:.2f}, éxitos: {self.streak_count}/3)")
    
    metadata = {
        "success": episode_complete,
        "ScheduleRes": self.connection_activities.copy() if episode_complete else None,
        "curriculum_level": self.difficulty_level,
        "stream_count": len(self.traffic_streams),

        # ─── métricas de latencia E2E ───
        "latency_us": {
            "average": mean_delay,
            "delay_variance" : delay_variance,
            "maximum": peak_delay,
            "samples": self._latency_samples.copy(),
        },
        # NUEVO: Añadir información sobre selección de flujos
        "stream_choice": {
            "active_stream_id": self.active_stream_id,
            "available_candidates": getattr(self, 'active_nominees', []),
            "selected_option": stream_choice,
            "reward_adj": selection_bonus
        }
    }

    # ────────────────────────────────────────────────────────────────
    #  Al terminar el episodio (todos los flujos entregados) → métricas
    # ────────────────────────────────────────────────────────────────
    if episode_complete and self._flow_latencies:
        mean_delay = sum(self._flow_latencies) / len(self._flow_latencies)
        peak_delay = max(self._flow_latencies)
        import statistics as _st
        delay_variance = _st.pstdev(self._flow_latencies) if len(self._flow_latencies) > 1 else 0

        # Log amigable
        self.event_recorder.info(
            f"⏱️  Latencia promedio={mean_delay:.0f} µs · "
            f"delay_variance={delay_variance:.0f} µs · máxima={peak_delay:.0f} µs"
        )

        # Añadir al diccionario `metadata`
        metadata["latency_us"] = {
            "average": mean_delay,
            "delay_variance":  delay_variance,
            "max":     peak_delay,
        }

    return self._get_observation(), performance_score, episode_complete, False, metadata
