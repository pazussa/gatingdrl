import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from core.network.net import Net
import os
import math
import re
import colorsys
from collections import defaultdict
import webbrowser
import logging


from tools.definitions import OUT_DIR
from core.scheduler.scheduler import ScheduleRes

def visualize_tsn_schedule_plotly(schedule_res: ScheduleRes, save_path=None):
    """
    Genera una visualización interactiva de la programación TSN usando Plotly.
    
    Args:
        schedule_res: Resultado de la programación
        save_path: Ruta para guardar el HTML, por defecto es 'out/tsn_schedule_plotly.html'
    """
    if not schedule_res:
        print("No hay resultados de programación para visualizar.")
        return
 
    
    # Organizar datos por enlace
    link_data = defaultdict(list)
    all_periods = set()
    all_flow_ids = set()
    max_end_time = 0
    
    print("\nProcesando datos para visualización Plotly...")
    
    # ----- NUEVO: Extraer información de ocupación de switches -----
    switch_busy_periods = defaultdict(list)
    
    # Track which legend entries have been shown globally
    legend_shown = set()
    
    # Extraer y procesar datos
    for network_connection, operations in schedule_res.items():
        link_str = str(network_connection)
        match = re.search(r"Link\('([^']+)', '([^']+)'\)", link_str)
        if not match:
            print(f"Error: No se pudo extraer origen/destino de {link_str}")
            continue
        
        source_node, destination_node = match.group(1), match.group(2)
        link_name = f"{source_node} → {destination_node}"
        
        flow_ids = [data_stream.flow_id for data_stream, _ in operations]
        print(f"Enlace: {link_name}, Flujos: {flow_ids}")
        
        for data_stream, operation in operations:
            all_flow_ids.add(data_stream.flow_id)
            all_periods.add(data_stream.period)
            
            # Calculate earliest_time on the fly based on the current Operation structure
            earliest_time = operation.start_time if operation.gating_time is None else operation.gating_time
            
            # Guardamos también el guard-band y el desglose de esperas
            link_data[link_name].append({
                'flow_id'        : data_stream.flow_id,
                'period'         : data_stream.period,
                'start_time'     : operation.start_time,
                'earliest_time'  : earliest_time,          # calculado
                'gating_time'    : operation.gating_time,
                'latest_time'    : operation.latest_time,
                'completion_instant'       : operation.completion_instant,
                'reception_time' : operation.reception_time,
                # ➊ NUEVOS campos → visualización de bloques temporales
                'safety_interval'     : getattr(operation, 'safety_interval', 0),
                'wait_breakdown' : operation.wait_breakdown,
                'min_gap_wait'   : getattr(operation, 'min_gap_wait', 0),
                # ---- acción RL (se mantiene) ----
                'offset_idx'     : getattr(operation, 'offset_idx', None),
                'timing_offset'      : getattr(operation, 'timing_offset',  None),
            })
            
            max_end_time = max(max_end_time, operation.completion_instant)
            
            # ACTUALIZADO: Extraer ocupación del switch si este enlace sale de un switch
            if source_node.startswith('S') and not source_node.startswith('SRV'):
                # El puerto del switch está ocupado durante la transmisión SOLAMENTE
                # El switch termina de estar ocupado cuando el paquete sale completamente
                switch_busy_start = earliest_time
                # Mostrar la ocupación real del puerto: transmisión + guard-band
                safety_interval = network_connection.interference_time() if hasattr(network_connection, "interference_time") else 1.22
                switch_busy_end = operation.completion_instant + safety_interval
                
                # Almacenar período de ocupación para el switch
                switch_busy_periods[source_node].append({
                    'flow_id': data_stream.flow_id,
                    'start': switch_busy_start,
                    'end': switch_busy_end,  # El switch termina su trabajo cuando completa la transmisión
                    'period': data_stream.period
                })


    # Calcular el hiperperíodo
    hyperperiod = 1
    for period in all_periods:
        hyperperiod = math.lcm(hyperperiod, period)
    
    print(f"Hiperperíodo calculado: {hyperperiod}µs")

    # ▸ valor máximo de Δsw  →  nos permite ampliar el eje-X hacia la izquierda
    global_max_gap = max(
        (operation_record['min_gap_wait'] for link_ops in link_data.values() for operation_record in link_ops),
        default=0
    )
    
    # Ordenar enlaces para visualización
    # Considera como "switch-network_connection" todo enlace cuyo **origen** sea S<n>
    switch_links = [lnk for lnk in link_data.keys() if re.match(r'^S\d+\s+→', lnk)]
    client_links = [lnk for lnk in link_data.keys() if lnk not in switch_links]
    sorted_links = sorted(switch_links) + sorted(client_links)
    
    # --- TODOS los path_segments de switch van con gate ⇒ un solo esquema de colores ---
    # IMPORTANTE: Definir flow_colors ANTES de cualquier referencia
    flow_colors = {}
    for iterator, flow_id in enumerate(sorted(all_flow_ids)):
        hue = (iterator * 0.618033988749895) % 1
        r, network_graph, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        flow_colors[flow_id] = f'rgb({int(r*255)},{int(network_graph*255)},{int(b*255)})'
    
    # ----- NUEVO: Añadir switches a la lista de elementos a visualizar -----
    # Ordenar switches para mostrarlos primero
    sorted_switches = sorted(switch_busy_periods.keys())
    
    # Crear figura de Plotly con subplots compartiendo el eje X
    fig = make_subplots(
        rows=3, 
        cols=1,
        row_heights=[0.15, 0.70, 0.15],  # Proporciones ajustadas para incluir switches
        vertical_spacing=0.02,     # Reducir espacio entre gráficas para mejor integración visual
        shared_xaxes=True,         # Compartir eje X para que el zoom se sincronice
        subplot_titles=["Ocupación de Switches", "Programación de Flujos TSN", "Gate Control List (GCL)"]
    )
    
    # -- NUEVO: SECCIÓN 1: OCUPACIÓN DE SWITCHES --
    for iterator, switch_name in enumerate(sorted_switches):
        periods = switch_busy_periods[switch_name]
        
        # Replicar períodos para todo el hiperperíodo
        for period_info in periods:
            flow_id = period_info['flow_id']
            flow_period = period_info['period']
            repetitions = hyperperiod // flow_period
            
            for rep in range(repetitions):
                time_offset = rep * flow_period
                start_time = period_info['start'] + time_offset
                completion_instant = period_info['end'] + time_offset
                duration = completion_instant - start_time
                
                # Añadir barra para el período ocupado
                fig.add_trace(
                    go.Bar(
                        x=[duration],
                        y=[switch_name],
                        orientation='h',
                        base=[start_time],
                        name=f"{switch_name} ocupado ({flow_id})",
                        marker=dict(
                            color='rgba(150,150,150,0.7)',
                            pattern=dict(
                                shape="/",
                                solidity=0.3,
                                fgcolor="black"
                            )
                        ),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f"Switch: {switch_name}<br>Ocupado por flujo: {flow_id}<br>Inicio: {start_time}µs<br>Fin: {completion_instant}µs",
                    ),
                    row=1, col=1
                )
    
    # -- SECCIÓN 2: GRÁFICO PRINCIPAL DE BARRAS (ahora en la segunda fila) --
    # Recopilar eventos de GCL
    gcl_events = []
    for network_connection, operations in schedule_res.items():
        link_str = str(network_connection)
        match = re.search(r"Link\('([^']+)', '([^']+)'\)", link_str)
        if not match:
            continue
        source_node, destination_node = match.group(1), match.group(2)
        link_name = f"{source_node} → {destination_node}"
        
        # Solo procesar enlaces que salen de un switch
        if not source_node.startswith('S'):
            continue
        
        # Tratar todos los flujos como críticos para un GCL periódico
        link_operations = operations
        if link_operations:
            print(f"Generando GCL periódico para enlace switch: {link_name}")
            
            # Calcular el hiperperíodo para todos los flujos
            periods = [data_stream.period for data_stream, _ in link_operations]
            hyperperiod_link = 1
            for period in periods:
                hyperperiod_link = math.lcm(hyperperiod_link, period)
                
            # Generar eventos GCL para todos los flujos con el mismo formato
            # ------------------------------------------------------------------
            # MODIFICACIÓN: Generar eventos "0" (cerrar) para TODOS los paquetes
            # en su tiempo de recepción, sin filtrar por tamaño de gap.
            # ------------------------------------------------------------------

            # 1) Ordenar operaciones por inicio real de transmisión
            ops_sorted = sorted(link_operations,
                                key=lambda probability: (probability[1].gating_time or probability[1].start_time))
            n = len(ops_sorted)

            if n < 2:
                continue

            # 2) Hiperperíodo individual del enlace
            periods = [data_stream.period for data_stream, _ in ops_sorted]
            hyperperiod_link = 1
            for probability in periods:
                hyperperiod_link = math.lcm(hyperperiod_link, probability)

            # Variable para el umbral de gap (usado para filtrar qué eventos mostrar)
            gap_thr_us = 50  # umbral de espacio mínimo para crear entradas GCL
            
            # PASO 1: Recopilar todos los tiempos de transmisión y recepción
            all_transmission_times = []
            all_reception_times = []
            
            # Recopilar todos los tiempos de transmisión y recepción
            for iterator in range(n):
                f_curr, op_curr = ops_sorted[iterator]
                tx_start = op_curr.gating_time if op_curr.gating_time is not None else op_curr.start_time
                # Para cada paquete, repetirlo durante todo el hiperperíodo
                repetitions = hyperperiod_link // f_curr.period
                for rep in range(repetitions):
                    time_adjustment = rep * f_curr.period
                    # Guardar tiempo de inicio y recepción (normalizado al hiperperíodo)
                    tx_t = (tx_start + time_adjustment) % hyperperiod_link
                    rx_t = (op_curr.reception_time + time_adjustment) % hyperperiod_link
                    all_transmission_times.append((tx_t, f_curr.flow_id))
                    all_reception_times.append((rx_t, f_curr.flow_id))
            
            # Ordenar los tiempos
            all_transmission_times.sort(key=lambda item: item[0])
            all_reception_times.sort(key=lambda item: item[0])
            
            # PASO 2: Generar eventos GCL analizando los gaps significativos
            gcl_close_events = []  # Lista temporal para eventos de cierre (0)
            
            # Buscar gaps significativos entre recepción y siguiente transmisión
            for iterator in range(len(all_reception_times)):
                rx_time, rx_flow = all_reception_times[iterator]
                
                # Encontrar el siguiente tiempo de transmisión después de esta recepción
                next_tx_time = None
                next_tx_flow = None
                
                for tx_time, tx_flow in all_transmission_times:
                    # Búsqueda circular (considerando el wraparound del hiperperíodo)
                    if tx_time > rx_time:
                        # Caso normal: siguiente TX está después de RX en este ciclo
                        next_tx_time = tx_time
                        next_tx_flow = tx_flow
                        break
                
                # Si no se encontró ninguno, buscar el primero (wraparound)
                if next_tx_time is None and all_transmission_times:
                    next_tx_time = all_transmission_times[0][0] + hyperperiod_link
                    next_tx_flow = all_transmission_times[0][1]
                
                # Calcular el gap (si hay transmisiones)
                if next_tx_time is not None:
                    gap = next_tx_time - rx_time
                    if gap < 0:
                        gap += hyperperiod_link  # Ajustar para gaps negativos (wraparound)
                    
                    # Solo considerar gaps que superen el umbral
                    if gap > gap_thr_us:
                        # Añadir eventos de cierre/apertura
                        gcl_close_events.append((rx_time, rx_flow, next_tx_time, next_tx_flow))
            
            # PASO 3: Generar los pares de eventos 0/1 para cada gap significativo
            for close_time, close_flow, next_tx_time, next_tx_flow in gcl_close_events:
                # Calcular repeticiones para todo el hiperperíodo
                repetitions = hyperperiod_link // hyperperiod_link  # Simplificado a 1
                
                for rep in range(repetitions):
                    time_adjustment = rep * hyperperiod_link
                    
                    # Añadir evento de cierre (0) en el tiempo de recepción
                    close_t = (close_time + time_adjustment) % hyperperiod_link
                    gcl_events.append((close_t, 0, close_flow, link_name, True))
                    
                    # Añadir evento de apertura (1) EXACTAMENTE cuando empieza el siguiente paquete
                    open_t = (next_tx_time + time_adjustment) % hyperperiod_link
                    gcl_events.append((open_t, 1, next_tx_flow, link_name, True))

    # Ordenar eventos por tiempo
    gcl_events.sort(key=lambda item: item[0])
    
    # Para barras muy estrechas o marcadores de ventana de transmisión, mejorar visibilidad
    for iterator, link_name in enumerate(sorted_links):
        if (link_name not in link_data):
            continue
            
        operations = link_data[link_name]
        
        for op_data in operations:
            flow_id = op_data['flow_id']
            flow_period = op_data['period']
            repetitions = hyperperiod // flow_period
            
            for rep in range(repetitions):
                # Calcular tiempos con el desplazamiento del período
                time_offset = rep * flow_period
                # Usar los tiempos recalculados si hubo time_adjustment
                start_time = op_data['start_time'] + time_offset
                completion_instant = op_data['completion_instant'] + time_offset
                earliest_time = op_data['earliest_time'] + time_offset  # Use the computed value
                latest_time = op_data['latest_time'] + time_offset
                gating_time = op_data['gating_time'] + time_offset if op_data['gating_time'] is not None else None
                reception_time = op_data['reception_time'] + time_offset if op_data['reception_time'] is not None else None

                # Tiempo de transmisión real (desde gating_time o start_time si no hay time_synchronization)
                actual_start_time = gating_time if gating_time is not None else start_time
                transmission_duration = completion_instant - actual_start_time

                # --- NUEVO: Barra de Tiempo de Espera con distinción de tipos ---
                if gating_time is not None and gating_time > start_time:
                    wait_duration = gating_time - start_time
                    
                    # Solo dibujamos la barra base gris (Total)
                    fig.add_trace(
                        go.Bar(
                            x=[wait_duration],
                            y=[link_name],
                            orientation='h',
                            base=[start_time],
                            name="Espera Total",
                            marker=dict(
                                color='rgba(200,200,200,0.3)',
                                line=dict(width=1, color='black'),
                            ),
                            showlegend=False,
                            hoverinfo='none',
                        ),
                        row=2, col=1
                    )

                # ───────── BARRAS DE ESPERA DESGLOSADAS ─────────
                if gating_time is not None and gating_time > start_time:
                    wb = op_data['wait_breakdown']          # dict: min_gap / other / total

                    # Dibujar la base gris con la espera total
                    fig.add_trace(
                        go.Bar(
                            x=[wb['total']],
                            y=[link_name],
                            orientation='h',
                            base=[start_time],
                            name="Espera Total",
                            marker=dict(color='rgba(200,200,200,0.25)'),
                            showlegend=False,
                            hoverinfo='none',
                        ),
                        row=2, col=1
                    )

                    waits = [
                        # Solo las esperas controladas por RL
                        ('min_gap', 'rgba(220, 20, 60,0.8)', "\\", "Separación mínima"),
                        ('other',   'rgba(120,120,120,0.5)', "",   "Otros"),
                    ]

                    offset_acc = 0
                    for key, color, pattern, label in waits:
                        w = wb.get(key, 0)
                        if w == 0:
                            continue
                        
                        # Check if this legend entry has been shown before
                        legend_key = f"wait_{key}"
                        show_legend = legend_key not in legend_shown
                        if show_legend:
                            legend_shown.add(legend_key)
                            
                        fig.add_trace(
                            go.Bar(
                                x=[w],
                                y=[link_name],
                                orientation='h',
                                base=[start_time + offset_acc],
                                name=label,
                                marker=dict(color=color,
                                            pattern=dict(shape=pattern, solidity=0.35)),
                                showlegend=show_legend,
                                legendgroup=legend_key,
                                hoverinfo='text',
                                hovertext=f"{label}: {w} µs<br>Flujo: {flow_id}",
                            ),
                            row=2, col=1
                        )
                        offset_acc += w

                # ---------------------------------------------------------------
                #  BLOQUE 2 · ESPERA EN EL SWITCH  (Δsw y resto de waits)
                # ---------------------------------------------------------------
                wb = op_data['wait_breakdown']
                # ▶️  dibujamos SIEMPRE que exista min_gap_wait, con o sin time_synchronization
                if op_data['min_gap_wait'] > 0 and not wb:
                    # reconstruir diccionario vacío
                    wb = {'min_gap': op_data['min_gap_wait'],
                          'other'  : 0,
                          'total'  : op_data['min_gap_wait']}

                if wb:
                    # Dibujar la base gris con la espera total
                    fig.add_trace(
                        go.Bar(
                            x=[wb['total']],
                            y=[link_name],
                            orientation='h',
                            base=[start_time],
                            name="Espera Total",
                            marker=dict(color='rgba(200,200,200,0.25)'),
                            showlegend=False,
                            hoverinfo='none',
                        ),
                        row=2, col=1
                    )

                    waits = [
                        ('min_gap', 'rgba(220, 20, 60,0.8)', "\\", "Δ sw   (gap mín.)"),
                        ('other',   'rgba(120,120,120,0.5)', "",   "Otros"),
                    ]

                    offset_acc = 0
                    for key, color, pattern, label in waits:
                        w = wb.get(key, 0)
                        if w == 0:
                            continue
                            
                        # Check if this legend entry has been shown before
                        legend_key = f"wait_{key}"
                        show_legend = legend_key not in legend_shown
                        if show_legend:
                            legend_shown.add(legend_key)
                            
                        fig.add_trace(
                            go.Bar(
                                x=[w],
                                y=[link_name],
                                orientation='h',
                                base=[start_time + offset_acc],
                                name=label,
                                marker=dict(color=color,
                                            pattern=dict(shape=pattern, solidity=0.35)),
                                showlegend=show_legend,
                                legendgroup=legend_key,
                                hoverinfo='text',
                                hovertext=f"{label}: {w} µs<br>Flujo: {flow_id}",
                            ),
                            row=2, col=1
                        )
                        offset_acc += w

                # ---------------------------------------------------------------
                #  BLOQUE 3 · GUARD-BAND  (γe·dmax)
                # ---------------------------------------------------------------
                safety_interval = op_data['safety_interval']
                if safety_interval > 0:
                    # Check if guard-band legend has been shown before
                    show_guard_legend = "guard_band" not in legend_shown
                    if show_guard_legend:
                        legend_shown.add("guard_band")
                        
                    fig.add_trace(
                        go.Bar(
                            x=[safety_interval],
                            y=[link_name],
                            orientation='h',
                            base=[completion_instant],
                            name="Guard-band",
                            legendgroup="guard_band",
                            marker=dict(
                                color='rgba(30,144,255,0.5)',
                                pattern=dict(shape="/", solidity=0.35)
                            ),
                            showlegend=show_guard_legend,
                            hoverinfo='text',
                            hovertext=f"Guard-band: {safety_interval} µs<br>Flujo: {flow_id}",
                        ),
                        row=2, col=1
                    )

                # --- Barra principal para transmisión ---
                fig.add_trace(
                    go.Bar(
                        x=[transmission_duration],
                        y=[link_name],
                        orientation='h',
                        base=[actual_start_time],
                        name=flow_id,                    # sigue como texto interno
                        marker=dict(
                            color=flow_colors[flow_id],
                            opacity=0.8,
                            line=dict(width=1, color='black')
                        ),
                        text=flow_id,
                        textposition='inside',
                        insidetextanchor='middle',
                        hoverinfo='text',
                        hovertext=(
                            f"Flujo: {flow_id}"
                            f"<br>Período: {flow_period}µs"
                            f"<br>Inicio Tx: {actual_start_time}µs"
                            f"<br>Fin Tx: {completion_instant}µs"
                            f"<br>Recibido: {reception_time}µs"
                        ),
                        showlegend=False,                # ← leyenda desactivada
                    ),
                    row=2, col=1
                )

                # Obtener detalles de las decisiones del agente si están disponibles
                protection_multiplier = getattr(operation, 'protection_multiplier', 1.0)
                min_gap = getattr(operation, 'min_gap_value', 1.0)
                
                # Tooltip completo con parámetros de RL
                hover_text = (
                    f"Flujo: {data_stream.flow_id}"
                    f"<br>Período: {data_stream.period}µs"
                    f"<br>Inicio Tx: {actual_start_time}µs"
                    f"<br>Fin Tx: {completion_instant}µs"
                    f"<br>Recibido: {reception_time}µs"
                    f"<br>Guard Factor: {protection_multiplier:.2f}"
                    f"<br>Separación Mín: {min_gap:.1f}µs"
                )

                # Marcadores ---
                # Marcador para start_time (cuándo podría haber empezado)
                fig.add_trace(
                    go.Scatter(
                        x=[start_time],
                        y=[link_name],
                        mode='markers',
                        marker=dict(symbol='line-ns', size=15, color='green', line=dict(width=2)),
                        name='Start Time (Available)',
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f"Disponible (Start Time): {start_time}µs<br>Flujo: {flow_id}",
                    ),
                    row=2, col=1
                )

                # Marcador para gating_time (cuándo empezó realmente si hubo time_synchronization)
                if gating_time is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[gating_time],
                            y=[link_name],
                            mode='markers',
                            marker=dict(symbol='line-ns', size=18, color='red', line=dict(width=3)),
                            name='Gate Time (Actual Start)',
                            showlegend=False,
                            hoverinfo='text',
                            hovertext=f"Inicio Real (Gate Time): {gating_time}µs<br>Flujo: {flow_id}",
                        ),
                        row=2, col=1
                    )

                # Marcador para latest_time (límite ventana time_synchronization)
                # Solo relevante si hay time_synchronization
                if gating_time is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[latest_time],
                            y=[link_name],
                            mode='markers',
                            marker=dict(symbol='line-ns', size=15, color='orange', line=dict(width=2)),
                            name='Latest Time',
                            showlegend=False,
                            hoverinfo='text',
                            hovertext=f"Latest Time: {latest_time}µs<br>Flujo: {flow_id}",
                        ),
                        row=2, col=1
                    )
    
    # -- SECCIÓN 3 (GCL): una fila por switch --------------------------------
    #  ① Agrupamos los eventos por switch   →  gcl_by_switch[source_node] = [...]
    #  ② Para cada switch añadimos *dos* trazas (línea punteada + markers)

    # gcl_events se genera más arriba; aquí lo re-estructuramos:
    gcl_by_switch = defaultdict(list)
    for t, state, flow_id, link_name, is_gating in gcl_events:
        # el nombre de la fila será, probability.ej.,  "GCL S1"
        src_sw = link_name.split(' ')[0]          # 'S1' de "S1 → SRV1"
        gcl_by_switch[src_sw].append((t, state, flow_id, is_gating))

    if gcl_by_switch:
        print(f"Dibujando GCL para {len(gcl_by_switch)} switches")
        # Aseguramos orden estable
        for sw_name in sorted(gcl_by_switch.keys()):
            events = sorted(gcl_by_switch[sw_name], key=lambda item: item[0])
            if not events:
                continue

            # Descomponer para construir arrays
            times   = [e[0] for e in events]
            states  = [e[1] for e in events]
            colors  = ['rgba(0,0,255,0.9)' if s == 0 else 'rgba(0,180,0,0.9)'
                       for s in states]
            htexts  = [("Cierre" if s == 0 else "Apertura") +
                       f"<br>{sw_name}<br>t={t}µs<br>Flujo={flow_generator}"
                       for (t, s, flow_generator, _) in events]

            # Línea guía (gris) para esa fila
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[f"GCL {sw_name}"]*len(times),
                    mode='lines',
                    line=dict(color='lightgray', width=1.5, dash='dot'),
                    showlegend=False,
                    hoverinfo='none',
                ),
                row=3, col=1
            )
            # Marcadores con 0/1
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[f"GCL {sw_name}"]*len(times),
                    mode='markers+text',
                    marker=dict(symbol='circle', size=15, color=colors,
                                line=dict(width=2, color='black')),
                    text=[str(s) for s in states],
                    textposition='middle center',
                    textfont=dict(color='white', size=10,
                                  family='Arial Black'),
                    name=f"GCL {sw_name}",
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=htexts,
                ),
                row=3, col=1
            )
    else:
        fig.add_annotation(
            x=hyperperiod/2, y=0,
            text="No hay eventos GCL para mostrar",
            showarrow=False, font=dict(size=12, color="gray"),
            row=3, col=1
        )
        
    # Añadir línea vertical para el hiperperíodo
    fig.add_vline(
        x=hyperperiod,
        line_width=2,
        line_dash="solid",
        line_color="blue",
        annotation_text=f"Hiperperíodo: {hyperperiod}µs",
        annotation_position="top",
        annotation_font_size=12,
        annotation_font_color="blue"
    )
    
    # Configuración de diseño mejorada para ocupar toda la página
    fig.update_layout(
        title=f"Programación TSN de Flujos (Hiperperíodo: {hyperperiod}µs)",
        barmode='overlay',
        height=max(800, len(sorted_links) * 45 + 250),  # Altura ajustada para mejor visualización
        width=1400,                                     # Ancho aumentado para mejor visualización
        margin=dict(l=50, r=50, t=80, b=80),           # Márgenes reducidos
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,                                    # Ajustar posición de leyenda
            xanchor="center",
            x=0.5,
            title="Flujos"
        ),
        plot_bgcolor='white',
        hovermode='closest',
    )
    
    # Agregar leyenda interactiva para filtrar por tipo de evento
    fig.update_layout(
        legend=dict(
            orientation="v",     # vertical
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,             # fuera del área de trazado
            font=dict(size=11),
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        )
    )
    
    # Vincular los ejes X para que el zoom se sincronice entre gráficas
    tick_interval = max(100, hyperperiod // 20)  # máx. 20 ticks
    left_pad = max(global_max_gap * 1.1, 10)      # algo de margen
    fig.update_xaxes(
        title="Tiempo (µs)",
        range=[-left_pad, hyperperiod*1.02],
        gridcolor='lightgray',
        griddash='dot',
        tickvals=list(range(0, hyperperiod + tick_interval, tick_interval)),
        row=1, col=1
    )
    
    fig.update_xaxes(
        title="Tiempo (µs)",
        range=[-left_pad, hyperperiod*1.02],
        showgrid=True,
        gridcolor='lightgray',
        griddash='dot',
        tickvals=list(range(0, hyperperiod + tick_interval, tick_interval)),
        row=2, col=1,
        matches='x'  # Esto sincroniza este eje con el eje X de la primera gráfica
    )
    
    fig.update_xaxes(
        title="Tiempo (µs)",
        range=[-left_pad, hyperperiod*1.02],
        showgrid=True,
        gridcolor='lightgray',
        griddash='dot',
        tickvals=list(range(0, hyperperiod + tick_interval, tick_interval)),
        row=3, col=1,
        matches='x'  # Esto sincroniza este eje con el eje X de la primera gráfica
    )
    
    # Mejorar el estilo de las etiquetas del eje Y
    fig.update_yaxes(
        title="Switches",
        row=1, col=1,
        linecolor='black',
        gridcolor='rgba(200,200,200,0.3)'
    )
    
    fig.update_yaxes(
        title="Enlaces",
        row=2, col=1,
        linecolor='black',
        gridcolor='rgba(200,200,200,0.3)'
    )
    
    # Manejo especial para el eje Y del GCL
    gcl_rows = [f"GCL {sw}" for sw in sorted(gcl_by_switch.keys())] or ["GCL"]
    fig.update_yaxes(
        showticklabels=True,
        tickvals=gcl_rows,
        ticktext=gcl_rows,
        row=3, col=1,
        linecolor='black',
        gridcolor='rgba(200,200,200,0.3)'
    )
    
    # Agregar leyenda adicional para símbolos
    symbols_legend = [
        dict(name="Disponible (Start)", marker=dict(color="green", symbol="line-ns", size=10)),
        dict(name="Inicio Real (Gate)", marker=dict(color="red", symbol="line-ns", size=10)),
        dict(name="Último Inicio (Latest)", marker=dict(color="orange", symbol="line-ns", size=10)),
        dict(name="Espera por FCFS/Switch Ocupado", marker=dict(color="rgba(100,100,255,0.7)", symbol="square", size=10)),
        dict(name="Espera por Separación Mínima", marker=dict(color="rgba(220,20,60,0.7)", symbol="square", size=10)),
    ]
    
    for item in symbols_legend:
        # Asegurarse de que el marcador no contenga 'pattern'
        marker_config = item.get('marker', {})
        if 'pattern' in marker_config:
             del marker_config['pattern'] # Eliminar si existe por error
                
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=marker_config, # Usar la configuración corregida
                name=item.get('name', ''),
                showlegend=True
            )
        )
    
    # Agregar leyenda para la ocupación de switches
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name="Switch Ocupado",
            marker=dict(
                color='rgba(150,150,150,0.7)',
                pattern=dict(shape="/", solidity=0.3, fgcolor="black")
            ),
            showlegend=True
        )
    )
    
    # ── Leyendas para bloques temporales ──────────────────────────────────
    fig.add_trace(
        go.Bar(
            x=[None], y=[None], name="Guard-band",
            marker=dict(color='rgba(30,144,255,0.5)',
                        pattern=dict(shape="/", solidity=0.35)),
            showlegend=True)
    )
    # ya no es necesario añadir trazas «fantasma»;
    # la barra Δ sw real se encarga de la entrada en la leyenda.
    
    # Añadimos marcador de «Transmisión» genérico (color neutro)
    fig.add_trace(
        go.Bar(
            x=[None], y=[None], name="Transmisión",
            marker=dict(color='rgba(160,160,160,0.7)'),
            showlegend=True)
    )
    
    # Guardar como HTML interactivo con opciones para mejor visualización
    if save_path is None:
        save_path = os.path.join(OUT_DIR, 'tsn_schedule_plotly.html')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(
        save_path,
        include_plotlyjs='cdn',
        full_html=True,
        include_mathjax='cdn',
        config={
            'scrollZoom': True,            # Permitir zoom con rueda del ratón
            'displayModeBar': True,        # Mostrar barra de herramientas
            'displaylogo': False,          # No mostrar logo de Plotly
            'toImageButtonOptions': {      # Configuración para guardar imagen
                'format': 'png',
                'filename': 'tsn_schedule',
                'height': 1200,
                'width': 1800,
                'scale': 2                 # Alta resolución
            }
        }
    )
    print(f"Visualización interactiva TSN guardada en: {save_path}")
    
    # Try to open in browser with error handling
    try:
        webbrowser.open('file://' + os.path.abspath(save_path))
    except Exception as e:
        logging.warning(f"Could not open browser automatically: {e}")
        print(f"Please open the visualization manually at: file://{os.path.abspath(save_path)}")
    
    return fig

def visualize_lcm_cycle_plotly(schedule_res: ScheduleRes, save_path=None):
    """
    Función de compatibilidad que redirige a visualize_tsn_schedule_plotly
    """
    print("La funcionalidad de visualización del hiperperíodo está integrada en visualize_tsn_schedule_plotly.")
    print("Llamando a visualize_tsn_schedule_plotly...")
    return visualize_tsn_schedule_plotly(schedule_res, save_path)

if __name__ == "__main__":
    print("Este módulo debe ser importado y utilizado desde test.py")
    print("Ejemplo: visualize_tsn_schedule_plotly(scheduler.get_res())")
