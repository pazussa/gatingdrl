#!/usr/bin/env python3
"""
Script para medir métricas de complejidad computacional del scheduler DRL-TSN.
Ejecuta un entrenamiento corto y genera un reporte de las métricas observadas.
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Medir métricas de complejidad computacional")
    parser.add_argument('--time_steps', type=int, default=10000, 
                       help="Timesteps de entrenamiento (default: 10000)")
    parser.add_argument('--stream_count', type=int, default=20,
                       help="Número de flujos (default: 20)")
    parser.add_argument('--topo', type=str, default="SIMPLE",
                       help="Topología (default: SIMPLE)")
    parser.add_argument('--link_rate', type=int, default=100,
                       help="Velocidad de enlace (default: 100)")
    
    args = parser.parse_args()
    
    print("🔍 INICIANDO MEDICIÓN DE MÉTRICAS DE COMPLEJIDAD COMPUTACIONAL")
    print("=" * 70)
    print(f"Parámetros:")
    print(f"  - Timesteps: {args.time_steps}")
    print(f"  - Flujos: {args.stream_count}")
    print(f"  - Topología: {args.topo}")
    print(f"  - Velocidad enlace: {args.link_rate} Mbps")
    print("=" * 70)
    
    # Construir comando de entrenamiento
    cmd = [
        sys.executable, "ui/train.py",
        "--time_steps", str(args.time_steps),
        "--stream_count", str(args.stream_count),
        "--topo", args.topo,
        "--link_rate", str(args.link_rate),
        "--show-log"  # Mostrar logs detallados
    ]
    
    print(f"🚀 Ejecutando: {' '.join(cmd)}")
    print()
    
    try:
        # Ejecutar entrenamiento
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Medición completada exitosamente!")
            
            # Buscar el archivo de métricas
            out_dir = "out"
            metrics_file = os.path.join(out_dir, "complexity_metrics.txt")
            
            if os.path.exists(metrics_file):
                print(f"\n📊 Reporte de métricas guardado en: {metrics_file}")
                print("\n" + "="*70)
                print("RESUMEN DE MÉTRICAS:")
                print("="*70)
                
                # Mostrar contenido del archivo
                with open(metrics_file, 'r') as f:
                    content = f.read()
                    print(content)
            else:
                print(f"⚠️ No se encontró el archivo de métricas en: {metrics_file}")
                
        else:
            print(f"❌ Error en la ejecución (código: {result.returncode})")
            
    except KeyboardInterrupt:
        print("\n🛑 Medición interrumpida por el usuario")
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")

if __name__ == "__main__":
    main()
