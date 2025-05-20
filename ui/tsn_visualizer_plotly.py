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
    
    # Extraer y procesar datos
    for link, operations in schedule_res.items():
        link_str = str(link)
        match = re.search(r"Link\('([^']+)', '([^']+)'\)", link_str)
        if not match:
            print(f"Error: No se pudo extraer origen/destino de {link_str}")
            continue
        
        src, dst = match.group(1), match.group(2)
        link_name = f"{src} → {dst}"
        
        flow_ids = [flow.flow_id for flow, _ in operations]
        print(f"Enlace: {link_name}, Flujos: {flow_ids}")
        
        for flow, operation in operations:
            all_flow_ids.add(flow.flow_id)
            all_periods.add(flow.period)
            
            # Calculate earliest_time on the fly based on the current Operation structure
            earliest_time = operation.start_time if operation.gating_time is None else operation.gating_time
            
            link_data[link_name].append({
                'flow_id': flow.flow_id,
                'period': flow.period,
                'start_time': operation.start_time,
                'earliest_time': earliest_time,  # Computed value
                'gating_time': operation.gating_time,
                'latest_time': operation.latest_time,
                'end_time': operation.end_time,
                'reception_time': operation.reception_time,  # visualización
                # ---- NEW: acción RL ----
                'offset_idx': getattr(operation, 'offset_idx', None),
                'offset_us' : getattr(operation, 'offset_us',  None),
            })
            
            max_end_time = max(max_end_time, operation.end_time)
            
            # ACTUALIZADO: Extraer ocupación del switch si este enlace sale de un switch
            if src.startswith('S') and not src.startswith('SRV'):
                # El puerto del switch está ocupado durante la transmisión SOLAMENTE
                # El switch termina de estar ocupado cuando el paquete sale completamente
                guard_time = link.interference_time() if hasattr(link, "interference_time") else 1.22
                switch_busy_start = earliest_time
                switch_busy_end = operation.end_time  # CORREGIDO: Eliminar guard_time adicional
                
                # Almacenar período de ocupación para el switch
                switch_busy_periods[src].append({
                    'flow_id': flow.flow_id,
                    'start': switch_busy_start,
                    'end': switch_busy_end,  # El switch termina su trabajo cuando completa la transmisión
                    'period': flow.period
                })
    
    # Calcular el hiperperíodo
    hyperperiod = 1
    for period in all_periods:
        hyperperiod = math.lcm(hyperperiod, period)
    
    print(f"Hiperperíodo calculado: {hyperperiod}µs")
    
    # Ordenar enlaces para visualización
    # Considera como "switch-link" todo enlace cuyo **origen** sea S<n>
    switch_links = [lnk for lnk in link_data.keys() if re.match(r'^S\d+\s+→', lnk)]
    client_links = [lnk for lnk in link_data.keys() if lnk not in switch_links]
    sorted_links = sorted(switch_links) + sorted(client_links)
    
    # --- TODOS los hops de switch van con gate ⇒ un solo esquema de colores ---
    # IMPORTANTE: Definir flow_colors ANTES de cualquier referencia
    flow_colors = {}
    for i, flow_id in enumerate(sorted(all_flow_ids)):
        hue = (i * 0.618033988749895) % 1
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        flow_colors[flow_id] = f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
    
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
    for i, switch_name in enumerate(sorted_switches):
        periods = switch_busy_periods[switch_name]
        
        # Replicar períodos para todo el hiperperíodo
        for period_info in periods:
            flow_id = period_info['flow_id']
            flow_period = period_info['period']
            repetitions = hyperperiod // flow_period
            
            for rep in range(repetitions):
                time_offset = rep * flow_period
                start_time = period_info['start'] + time_offset
                end_time = period_info['end'] + time_offset
                duration = end_time - start_time
                
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
                                shape="x",
                                solidity=0.3,
                                fgcolor="black"
                            )
                        ),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f"Switch: {switch_name}<br>Ocupado por flujo: {flow_id}<br>Inicio: {start_time}µs<br>Fin: {end_time}µs",
                    ),
                    row=1, col=1
                )
    
    # -- SECCIÓN 2: GRÁFICO PRINCIPAL DE BARRAS (ahora en la segunda fila) --
    # Recopilar eventos de GCL
    gcl_events = []
    for link, operations in schedule_res.items():
        link_str = str(link)
        match = re.search(r"Link\('([^']+)', '([^']+)'\)", link_str)
        if not match:
            continue
        src, dst = match.group(1), match.group(2)
        link_name = f"{src} → {dst}"
        
        # Solo procesar enlaces que salen de un switch
        if not src.startswith('S'):
            continue
        
        # Tratar todos los flujos como críticos para un GCL periódico
        link_operations = operations
        if link_operations:
            print(f"Generando GCL periódico para enlace switch: {link_name}")
            
            # Calcular el hiperperíodo para todos los flujos
            periods = [flow.period for flow, _ in link_operations]
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
                                key=lambda p: (p[1].gating_time or p[1].start_time))
            n = len(ops_sorted)

            if n < 2:
                continue

            # 2) Hiperperíodo individual del enlace
            periods = [flow.period for flow, _ in ops_sorted]
            hyperperiod_link = 1
            for p in periods:
                hyperperiod_link = math.lcm(hyperperiod_link, p)

            # Variable para el umbral de gap (usado para filtrar qué eventos mostrar)
            gap_thr_us = 50  # umbral de espacio mínimo para crear entradas GCL
            
            # PASO 1: Recopilar todos los tiempos de transmisión y recepción
            all_transmission_times = []
            all_reception_times = []
            
            # Recopilar todos los tiempos de transmisión y recepción
            for i in range(n):
                f_curr, op_curr = ops_sorted[i]
                tx_start = op_curr.gating_time if op_curr.gating_time is not None else op_curr.start_time
                # Para cada paquete, repetirlo durante todo el hiperperíodo
                repetitions = hyperperiod_link // f_curr.period
                for rep in range(repetitions):
                    offset = rep * f_curr.period
                    # Guardar tiempo de inicio y recepción (normalizado al hiperperíodo)
                    tx_t = (tx_start + offset) % hyperperiod_link
                    rx_t = (op_curr.reception_time + offset) % hyperperiod_link
                    all_transmission_times.append((tx_t, f_curr.flow_id))
                    all_reception_times.append((rx_t, f_curr.flow_id))
            
            # Ordenar los tiempos
            all_transmission_times.sort(key=lambda x: x[0])
            all_reception_times.sort(key=lambda x: x[0])
            
            # PASO 2: Generar eventos GCL analizando los gaps significativos
            gcl_close_events = []  # Lista temporal para eventos de cierre (0)
            
            # Buscar gaps significativos entre recepción y siguiente transmisión
            for i in range(len(all_reception_times)):
                rx_time, rx_flow = all_reception_times[i]
                
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
                    offset = rep * hyperperiod_link
                    
                    # Añadir evento de cierre (0) en el tiempo de recepción
                    close_t = (close_time + offset) % hyperperiod_link
                    gcl_events.append((close_t, 0, close_flow, link_name, True))
                    
                    # Añadir evento de apertura (1) EXACTAMENTE cuando empieza el siguiente paquete
                    open_t = (next_tx_time + offset) % hyperperiod_link
                    gcl_events.append((open_t, 1, next_tx_flow, link_name, True))

    # Ordenar eventos por tiempo
    gcl_events.sort(key=lambda x: x[0])
    
    # Para barras muy estrechas o marcadores de ventana de transmisión, mejorar visibilidad
    for i, link_name in enumerate(sorted_links):
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
                # Usar los tiempos recalculados si hubo offset
                start_time = op_data['start_time'] + time_offset
                end_time = op_data['end_time'] + time_offset
                earliest_time = op_data['earliest_time'] + time_offset  # Use the computed value
                latest_time = op_data['latest_time'] + time_offset
                gating_time = op_data['gating_time'] + time_offset if op_data['gating_time'] is not None else None
                reception_time = op_data['reception_time'] + time_offset if op_data['reception_time'] is not None else None

                # Tiempo de transmisión real (desde gating_time o start_time si no hay gating)
                actual_start_time = gating_time if gating_time is not None else start_time
                transmission_duration = end_time - actual_start_time

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
                    wb = op.wait_breakdown   # dict con 'min_gap', 'other', 'total'

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
                        fig.add_trace(
                            go.Bar(
                                x=[w],
                                y=[link_name],
                                orientation='h',
                                base=[start_time + offset_acc],
                                name=label,
                                marker=dict(color=color,
                                            pattern=dict(shape=pattern, solidity=0.35)),
                                showlegend=(rep == 0),
                                legendgroup=f"wait_{key}",
                                hoverinfo='text',
                                hovertext=f"{label}: {w}µs<br>Flujo: {flow_id}",
                            ),
                            row=2, col=1
                        )
                        offset_acc += w

                # Guard-band visualization removed

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
                            f"<br>Fin Tx: {end_time}µs"
                            f"<br>Recibido: {reception_time}µs"
                        ),
                        showlegend=False,                # ← leyenda desactivada
                    ),
                    row=2, col=1
                )

                # Obtener detalles de las decisiones del agente si están disponibles
                guard_factor = getattr(operation, 'guard_factor', 1.0)
                min_gap = getattr(operation, 'min_gap_value', 1.0)
                
                # Tooltip completo con parámetros de RL
                hover_text = (
                    f"Flujo: {flow.flow_id}"
                    f"<br>Período: {flow.period}µs"
                    f"<br>Inicio Tx: {actual_start_time}µs"
                    f"<br>Fin Tx: {end_time}µs"
                    f"<br>Recibido: {reception_time}µs"
                    f"<br>Guard Factor: {guard_factor:.2f}"
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

                # Marcador para gating_time (cuándo empezó realmente si hubo gating)
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

                # Marcador para latest_time (límite ventana gating)
                # Solo relevante si hay gating
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
    
    # -- SECCIÓN 3: REPRESENTACIÓN DEL GCL (ahora en la tercera fila) --
    if gcl_events:
        print(f"Dibujando {len(gcl_events)} eventos GCL")
        
        # Crear representación del GCL como un gráfico mejorado
        gcl_times = []
        gcl_values = []
        gcl_texts = []
        gcl_colors = []
        
        # Debug adicional para identificar el problema
        print("\nDETALLE DE EVENTOS GCL ORDENADOS:")
        print(f"{'Tiempo':8} | {'Estado':6} | {'Flujo':5} | {'Tipo':10}")
        print(f"{'-'*8} | {'-'*6} | {'-'*5} | {'-'*10}")
        
        # Mostrar eventos de cierre (0) Y apertura (1)
        for time, state, flow_id, link_name, is_gating in gcl_events:
            tipo = "Gated" if is_gating else "Non-Gated"
            print(f"{time:8} | {state:6} | {flow_id:5} | {tipo:10}")
            
            gcl_times.append(time)
            gcl_values.append(state)  # Incluir ambos estados: 0 y 1
            
            event_type = "Cierre" if state == 0 else "Apertura"
            gcl_texts.append(f"{event_type} de gate<br>Tiempo: {time}µs<br>Flujo: {flow_id}<br>Tipo: {tipo}")
            # Azul para estado 0 (cierre), Verde para estado 1 (apertura)
            gcl_colors.append('rgba(0,0,255,0.9)' if state == 0 else 'rgba(0,180,0,0.9)')

        # ------------------------------------------------------------------
        #  MOSTRAR TABLA GCL EN CONSOLA PARA VERIFICACIÓN RÁPIDA
        # ------------------------------------------------------------------
        print("\n" + "="*60)
        print("TABLA GCL GENERADA (t, estado)")
        print("-"*60)
        for t, v in zip(gcl_times, gcl_values):
            print(f"{t:>8} µs | {v}")
        print("="*60 + "\n")
        
        # Línea para conectar los eventos GCL y hacerlos más visibles
        fig.add_trace(
            go.Scatter(
                x=gcl_times,
                y=['GCL'] * len(gcl_times),
                mode='lines',
                line=dict(color='lightgray', width=1.5, dash='dot'),
                showlegend=False,
                hoverinfo='none'
            ),
            row=3, col=1
        )
        
        # Dibujar los eventos GCL con texto forzado para distinguir entre 0 y 1
        fig.add_trace(
            go.Scatter(
                x=gcl_times,
                y=['GCL'] * len(gcl_times),
                mode='markers+text',
                marker=dict(
                    size=15,  # Tamaño fijo para todos los eventos
                    color=gcl_colors,
                    symbol='circle',
                    line=dict(width=2, color='black')
                ),
                text=[str(v) for v in gcl_values],  # Mostrar "0" o "1" según el valor
                textposition='middle center',
                textfont=dict(color='white', size=10, family='Arial Black'),
                name='GCL Events',
                showlegend=False,
                hoverinfo='text',
                hovertext=gcl_texts,
            ),
            row=3, col=1
        )
    else:
        # Si no hay eventos GCL, mostrar un mensaje
        fig.add_annotation(
            x=hyperperiod/2,
            y=0,
            text="No hay eventos GCL para mostrar",
            showarrow=False,
            font=dict(size=12, color="gray"),
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
            title="Tipo de evento",
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        )
    )
    
    # Vincular los ejes X para que el zoom se sincronice entre gráficas
    tick_interval = max(100, hyperperiod // 20)  # máx. 20 ticks
    fig.update_xaxes(
        title="Tiempo (µs)",
        range=[-hyperperiod*0.02, hyperperiod*1.02],
        gridcolor='lightgray',
        griddash='dot',
        tickvals=list(range(0, hyperperiod + tick_interval, tick_interval)),
        row=1, col=1
    )
    
    fig.update_xaxes(
        title="Tiempo (µs)",
        range=[-hyperperiod*0.02, hyperperiod*1.02],
        showgrid=True,
        gridcolor='lightgray',
        griddash='dot',
        tickvals=list(range(0, hyperperiod + tick_interval, tick_interval)),
        row=2, col=1,
        matches='x'  # Esto sincroniza este eje con el eje X de la primera gráfica
    )
    
    fig.update_xaxes(
        title="Tiempo (µs)",
        range=[-hyperperiod*0.02, hyperperiod*1.02],
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
    fig.update_yaxes(
        showticklabels=True,                # Mostrar etiquetas
        tickvals=['GCL'],                   # Establecer valores específicos
        ticktext=['GCL'],                   # Establecer texto para esos valores
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
                pattern=dict(shape="x", solidity=0.3, fgcolor="black")
            ),
            showlegend=True
        )
    )
    
    # ── leyendas limpiadas: sólo min-gap (decisión RL) ──
    fig.add_trace(
        go.Bar(
            x=[None], y=[None], name="Espera ▸ min-gap",
            marker=dict(color='rgba(220, 20, 60,0.8)',
                        pattern=dict(shape="\\", solidity=0.35)),
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
