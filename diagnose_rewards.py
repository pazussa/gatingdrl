#!/usr/bin/env python3
"""
Script para diagnosticar problemas de recompensas en el entrenamiento.
Analiza el log de entrenamiento para identificar patrones problemáticos.
"""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def extract_rewards_from_log(log_file: str) -> List[float]:
    """Extrae las recompensas promedio del archivo de log"""
    rewards = []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Buscar patrones de recompensa promedio
    patterns = [
        r'ep_rew_mean\s+\|\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
        r'mean_reward\s+\|\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
        r'episode_reward=([+-]?\d*\.?\d+)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            try:
                reward = float(match)
                rewards.append(reward)
            except ValueError:
                continue
    
    return rewards

def analyze_reward_collapse(rewards: List[float]) -> dict:
    """Analiza el colapso de recompensas"""
    if len(rewards) < 10:
        return {"error": "Insuficientes datos de recompensas"}
    
    analysis = {}
    
    # Encontrar el punto de colapso
    max_reward = max(rewards)
    max_idx = rewards.index(max_reward)
    
    # Buscar cuando las recompensas se vuelven consistentemente negativas
    collapse_point = None
    for i in range(max_idx, len(rewards)):
        if rewards[i] < 0:
            # Verificar si las siguientes también son negativas
            negative_streak = 0
            for j in range(i, min(i + 10, len(rewards))):
                if rewards[j] < 0:
                    negative_streak += 1
            
            if negative_streak >= 5:  # 5 recompensas negativas consecutivas
                collapse_point = i
                break
    
    analysis["max_reward"] = max_reward
    analysis["max_reward_episode"] = max_idx
    analysis["collapse_point"] = collapse_point
    
    if collapse_point:
        analysis["collapse_reward"] = rewards[collapse_point]
        analysis["episodes_to_collapse"] = collapse_point - max_idx
        
        # Analizar la magnitud del colapso
        final_rewards = rewards[collapse_point:collapse_point+20] if collapse_point+20 < len(rewards) else rewards[collapse_point:]
        analysis["avg_after_collapse"] = sum(final_rewards) / len(final_rewards)
        analysis["collapse_magnitude"] = max_reward - analysis["avg_after_collapse"]
    
    # Estadísticas generales
    analysis["total_episodes"] = len(rewards)
    analysis["final_reward"] = rewards[-1]
    analysis["overall_trend"] = "decline" if rewards[-1] < rewards[0] else "improve"
    
    return analysis

def plot_reward_evolution(rewards: List[float], analysis: dict):
    """Grafica la evolución de recompensas con anotaciones"""
    plt.figure(figsize=(12, 8))
    
    episodes = list(range(len(rewards)))
    plt.plot(episodes, rewards, 'b-', alpha=0.7, linewidth=1)
    
    # Añadir línea de media móvil
    if len(rewards) > 10:
        window = min(10, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'Media móvil ({window})')
    
    # Marcar puntos importantes
    if "max_reward_episode" in analysis:
        plt.axvline(x=analysis["max_reward_episode"], color='g', linestyle='--', 
                   label=f'Máximo: {analysis["max_reward"]:.2f}')
    
    if analysis.get("collapse_point"):
        plt.axvline(x=analysis["collapse_point"], color='r', linestyle='--', 
                   label=f'Colapso: episodio {analysis["collapse_point"]}')
    
    # Línea de referencia en 0
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Promedio')
    plt.title('Evolución de Recompensas - Diagnóstico de Colapso')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Añadir texto con análisis
    textstr = f"""
    Episodios totales: {analysis["total_episodes"]}
    Recompensa máxima: {analysis.get("max_reward", "N/A"):.2f}
    Recompensa final: {analysis["final_reward"]:.2f}
    """
    
    if analysis.get("collapse_point"):
        textstr += f"""
    Punto de colapso: {analysis["collapse_point"]}
    Magnitud del colapso: {analysis.get("collapse_magnitude", 0):.2f}
    """
    
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('reward_diagnosis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_recommendations(analysis: dict) -> List[str]:
    """Genera recomendaciones basadas en el análisis"""
    recommendations = []
    
    if analysis.get("collapse_point"):
        recommendations.append("🚨 PROBLEMA DETECTADO: Colapso de recompensas")
        recommendations.append("✅ Recomendación 1: Reducir penalizaciones por fallo de -100 a -1~-10")
        recommendations.append("✅ Recomendación 2: Hacer curriculum learning más conservador")
        recommendations.append("✅ Recomendación 3: Normalizar recompensas entre -2 y +15")
        recommendations.append("✅ Recomendación 4: Reducir learning rate de 3e-4 a 1e-4")
        recommendations.append("✅ Recomendación 5: Añadir clipping del value function")
        
        if analysis.get("collapse_magnitude", 0) > 100:
            recommendations.append("⚠️ Colapso severo: Considerar reiniciar entrenamiento con ajustes")
    
    if analysis["total_episodes"] > 100 and analysis["final_reward"] < -10:
        recommendations.append("📊 Recompensas consistentemente negativas: Revisar función de recompensa")
    
    if analysis.get("episodes_to_collapse", 0) < 50:
        recommendations.append("⏰ Colapso rápido: El modelo no está explorando adecuadamente")
    
    return recommendations

def main():
    if len(sys.argv) < 2:
        print("Uso: python diagnose_rewards.py <archivo_log>")
        print("Ejemplo: python diagnose_rewards.py train1.log")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    print("🔍 DIAGNÓSTICO DE RECOMPENSAS")
    print("=" * 50)
    
    try:
        rewards = extract_rewards_from_log(log_file)
        print(f"📊 Extraídas {len(rewards)} muestras de recompensa")
        
        if len(rewards) == 0:
            print("❌ No se encontraron recompensas en el log")
            return
        
        # Análisis
        analysis = analyze_reward_collapse(rewards)
        
        if "error" in analysis:
            print(f"❌ Error: {analysis['error']}")
            return
        
        # Mostrar resultados
        print(f"\n📈 ANÁLISIS DE RESULTADOS:")
        print(f"   Total episodios: {analysis['total_episodes']}")
        print(f"   Recompensa máxima: {analysis.get('max_reward', 'N/A'):.2f} (episodio {analysis.get('max_reward_episode', 'N/A')})")
        print(f"   Recompensa final: {analysis['final_reward']:.2f}")
        
        if analysis.get("collapse_point"):
            print(f"\n💥 COLAPSO DETECTADO:")
            print(f"   Punto de colapso: Episodio {analysis['collapse_point']}")
            print(f"   Recompensa en colapso: {analysis['collapse_reward']:.2f}")
            print(f"   Episodios hasta colapso: {analysis['episodes_to_collapse']}")
            print(f"   Magnitud del colapso: {analysis.get('collapse_magnitude', 0):.2f}")
        else:
            print("\n✅ No se detectó colapso dramático")
        
        # Generar recomendaciones
        recommendations = generate_recommendations(analysis)
        if recommendations:
            print(f"\n🛠️ RECOMENDACIONES:")
            for rec in recommendations:
                print(f"   {rec}")
        
        # Generar gráfico
        print(f"\n📊 Generando gráfico de diagnóstico...")
        plot_reward_evolution(rewards, analysis)
        print(f"   Gráfico guardado como 'reward_diagnosis.png'")
        
    except FileNotFoundError:
        print(f"❌ No se pudo encontrar el archivo: {log_file}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    main()
