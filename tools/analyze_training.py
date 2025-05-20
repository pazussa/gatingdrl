#!/usr/bin/env python3

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.definitions import OUT_DIR

def load_metrics(json_path=None):
    """Carga métricas de un archivo .json de entrenamiento"""
    if json_path is None:
        # Buscar el último archivo metrics.json
        metrics_files = [f for f in os.listdir(OUT_DIR) if f.startswith('training_metrics_') and f.endswith('.json')]
        if not metrics_files:
            print("No se encontraron archivos de métricas de entrenamiento")
            return None
        
        # Ordenar por fecha de modificación y usar el más reciente
        metrics_files.sort(key=lambda x: os.path.getmtime(os.path.join(OUT_DIR, x)), reverse=True)
        json_path = os.path.join(OUT_DIR, metrics_files[0])
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error cargando métricas: {e}")
        return None

def analyze_rewards(metrics, save_dir=None):
    """Analiza y visualiza la evolución de recompensas"""
    if not metrics or 'rewards' not in metrics:
        print("No hay datos de recompensas disponibles")
        return
    
    rewards = metrics['rewards']
    if not rewards:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Gráfico de recompensa por episodio
    plt.plot(rewards, 'b-', alpha=0.6)
    
    # Añadir suavizado (media móvil)
    window = min(10, len(rewards) // 5 + 1)
    if window > 1:
        smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        valid_idx = np.arange(len(smooth_rewards)) + window - 1
        plt.plot(valid_idx, smooth_rewards, 'r-', linewidth=2, label=f'Media móvil ({window} eps)')
    
    # Estadísticas de recompensas
    plt.axhline(y=np.mean(rewards), color='g', linestyle='--', label=f'Media: {np.mean(rewards):.2f}')
    
    # Tramos de análisis
    n = len(rewards)
    if n >= 30:  # Solo si hay suficientes episodios
        first_third = np.mean(rewards[:n//3])
        mid_third = np.mean(rewards[n//3:2*n//3])
        last_third = np.mean(rewards[2*n//3:])
        
        plt.axhline(y=first_third, color='c', linestyle=':', alpha=0.7, 
                   label=f'Primer tercio: {first_third:.2f}')
        plt.axhline(y=last_third, color='m', linestyle=':', alpha=0.7,
                   label=f'Último tercio: {last_third:.2f}')
        
        # Indicar tendencia
        if last_third > first_third * 1.1:
            trend = "↗️ Mejora"
        elif last_third < first_third * 0.9:
            trend = "↘️ Deterioro"
        else:
            trend = "→ Estable"
            
        plt.text(0.02, 0.02, f"Tendencia: {trend}", transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Evolución de Recompensas por Episodio', fontsize=14)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'rewards_analysis.png'))
    
    plt.show()

def analyze_action_distribution(metrics, save_dir=None):
    """Analiza la distribución de decisiones del agente"""
    if not metrics or 'action_distributions' not in metrics:
        print("No hay datos de distribuciones de acciones disponibles")
        return
    
    action_dist = metrics.get('action_distributions', {})
    if not action_dist:
        return
    
    # Crear un DataFrame para análisis
    action_data = {}
    for action_name, dist in action_dist.items():
        if isinstance(dist, dict):  # Asegurar que es un diccionario
            action_data[action_name] = dist
    
    if not action_data:
        return
        
    # Analizar cada dimensión de acción
    plt.figure(figsize=(15, 10))
    
    n_actions = len(action_data)
    n_cols = 2
    n_rows = (n_actions + n_cols - 1) // n_cols
    
    action_names = {
        "offset": "Offset temporal (µs)",
        "gcl_strategy": "Estrategia GCL",
        "guard_factor": "Guard factor",
        "priority": "Prioridad",
        "switch_gap": "Gap mínimo",
        "jitter": "Control jitter"
    }
    
    for i, (action, dist) in enumerate(action_data.items()):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Extraer valores y frecuencias
        values = sorted(map(int, dist.keys()))
        counts = [dist.get(str(v), 0) for v in values]
        
        # Calcular entropía normalizada para medir aleatoriedad
        total = sum(counts)
        if total > 0:
            probs = [count/total for count in counts]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(len(values)) if len(values) > 0 else 0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
            entropy_str = f"Entropía: {norm_entropy:.2f}"
            
            # Análisis de distribución
            if norm_entropy > 0.95:
                analysis = "Muy uniforme - posible indecisión"
            elif norm_entropy > 0.85:
                analysis = "Bastante uniforme - poca preferencia"
            elif norm_entropy < 0.3:
                analysis = "Muy concentrada - fuerte preferencia"
            else:
                analysis = "Distribución normal"
        else:
            entropy_str = ""
            analysis = "Datos insuficientes"
        
        # Crear gráfico
        plt.bar(values, counts)
        plt.title(f"{action_names.get(action, action)}\n{entropy_str}\n{analysis}")
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'action_distribution.png'))
    
    plt.show()

def analyze_correlation(metrics, save_dir=None):
    """Analiza correlaciones entre decisiones y resultados"""
    # Aquí se implementaría el análisis de correlación entre dimensiones
    # y entre decisiones y rendimiento (recompensas)
    pass

def main():
    parser = argparse.ArgumentParser(description='Analiza métricas de entrenamiento DRL')
    parser.add_argument('--file', type=str, help='Ruta al archivo JSON de métricas (opcional)')
    parser.add_argument('--save-dir', type=str, help='Directorio donde guardar los gráficos (opcional)')
    parser.add_argument('--no-plots', action='store_true', help='No mostrar gráficos, solo guardarlos')
    
    args = parser.parse_args()
    
    # Cargar métricas
    metrics = load_metrics(args.file)
    if not metrics:
        print("No se pudieron cargar las métricas")
        return
    
    # Configurar backend de matplotlib
    if args.no_plots:
        plt.switch_backend('Agg')  # No mostrar gráficos
    
    # Realizar análisis
    print("Analizando recompensas...")
    analyze_rewards(metrics, args.save_dir)
    
    print("Analizando distribuciones de decisiones...")
    analyze_action_distribution(metrics, args.save_dir)
    
    print("Análisis completado")

if __name__ == "__main__":
    main()
