"""
Herramientas para medir métricas de complejidad computacional del scheduler DRL-TSN.
Estas métricas permiten validar las estimaciones teóricas de complejidad contra la implementación real.
"""

import math
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import statistics

class ComplexityMetrics:
    """Captura y analiza métricas de complejidad computacional del scheduler."""
    
    def __init__(self):
        # Métricas de iteraciones de resolución de conflictos
        self.conflict_iterations: List[int] = []
        self.total_conflict_checks = 0
        
        # Métricas de hiperperíodos
        self.hyperperiods: List[int] = []
        self.period_pairs: List[Tuple[int, int]] = []
        
        # Métricas de flujos y saltos
        self.flows_processed = 0
        self.total_hops = 0
        self.hops_per_flow: List[int] = []
        
        # Métricas de tiempo
        self.operation_times: List[float] = []
        self.check_isolation_times: List[float] = []
        
        # Configuración de períodos observados
        self.observed_periods: set = set()
        
        self.logger = logging.getLogger(__name__)
    
    def record_conflict_resolution(self, iterations_used: int):
        """Registra el número de iteraciones usadas en resolución de conflictos."""
        self.conflict_iterations.append(iterations_used)
    
    def record_hyperperiod_calculation(self, period1: int, period2: int, hyperperiod: int):
        """Registra un cálculo de hiperperíodo."""
        self.period_pairs.append((period1, period2))
        self.hyperperiods.append(hyperperiod)
        self.observed_periods.add(period1)
        self.observed_periods.add(period2)
    
    def record_flow_processing(self, flow_id: str, num_hops: int):
        """Registra el procesamiento de un flujo."""
        self.flows_processed += 1
        self.total_hops += num_hops
        self.hops_per_flow.append(num_hops)
    
    def record_operation_time(self, duration: float):
        """Registra el tiempo de una operación de scheduling."""
        self.operation_times.append(duration)
    
    def record_isolation_check_time(self, duration: float):
        """Registra el tiempo de verificación de aislamiento."""
        self.check_isolation_times.append(duration)
    
    def get_average_iterations(self) -> float:
        """Calcula I_avg - número promedio de iteraciones de resolución de conflictos."""
        if not self.conflict_iterations:
            return 0.0
        return statistics.mean(self.conflict_iterations)
    
    def get_max_hyperperiod_ratio(self) -> float:
        """Calcula H_max/T_min - relación hiperperíodo máximo / período mínimo."""
        if not self.observed_periods:
            return 0.0
        
        max_hyperperiod = max(self.hyperperiods) if self.hyperperiods else 0
        min_period = min(self.observed_periods)
        
        return max_hyperperiod / min_period if min_period > 0 else 0.0
    
    def get_average_hops_per_flow(self) -> float:
        """Calcula H - número promedio de saltos por flujo."""
        if not self.hops_per_flow:
            return 0.0
        return statistics.mean(self.hops_per_flow)
    
    def get_theoretical_complexity_estimate(self) -> float:
        """Calcula la estimación teórica de complejidad O((F·H)² × I_avg × H_max/T_min)."""
        F = self.flows_processed
        H = self.get_average_hops_per_flow()
        I_avg = self.get_average_iterations()
        H_max_T_min = self.get_max_hyperperiod_ratio()
        
        if F == 0 or H == 0:
            return 0.0
        
        K = F * H  # Número total de operaciones
        complexity = (K ** 2) * I_avg * H_max_T_min
        
        return complexity
    
    def generate_report(self) -> str:
        """Genera un reporte completo de las métricas de complejidad."""
        report = []
        report.append("=" * 80)
        report.append("📊 REPORTE DE MÉTRICAS DE COMPLEJIDAD COMPUTACIONAL")
        report.append("=" * 80)
        
        # Métricas básicas
        F = self.flows_processed
        H = self.get_average_hops_per_flow()
        I_avg = self.get_average_iterations()
        H_max_T_min = self.get_max_hyperperiod_ratio()
        K = F * H if H > 0 else 0
        
        report.append(f"🔢 PARÁMETROS FUNDAMENTALES:")
        report.append(f"   F (flujos procesados): {F}")
        report.append(f"   H (saltos promedio por flujo): {H:.2f}")
        report.append(f"   K (operaciones totales): {K:.0f}")
        report.append(f"   I_avg (iteraciones promedio de conflictos): {I_avg:.2f}")
        report.append(f"   H_max/T_min (ratio hiperperíodo/período): {H_max_T_min:.2f}")
        
        # Distribuciones
        if self.conflict_iterations:
            report.append(f"\n📈 DISTRIBUCIÓN DE ITERACIONES DE CONFLICTOS:")
            report.append(f"   Mínimo: {min(self.conflict_iterations)}")
            report.append(f"   Máximo: {max(self.conflict_iterations)}")
            report.append(f"   Mediana: {statistics.median(self.conflict_iterations):.2f}")
            report.append(f"   Desviación estándar: {statistics.stdev(self.conflict_iterations) if len(self.conflict_iterations) > 1 else 0:.2f}")
        
        if self.hops_per_flow:
            report.append(f"\n🛤️  DISTRIBUCIÓN DE SALTOS POR FLUJO:")
            report.append(f"   Mínimo: {min(self.hops_per_flow)}")
            report.append(f"   Máximo: {max(self.hops_per_flow)}")
            report.append(f"   Mediana: {statistics.median(self.hops_per_flow):.2f}")
        
        if self.observed_periods:
            periods = sorted(self.observed_periods)
            report.append(f"\n⏱️  PERÍODOS OBSERVADOS:")
            report.append(f"   Rango: {min(periods)} - {max(periods)} µs")
            report.append(f"   Cantidad de períodos únicos: {len(periods)}")
            report.append(f"   Períodos: {periods[:10]}{'...' if len(periods) > 10 else ''}")
        
        # Estimación de complejidad
        theoretical_complexity = self.get_theoretical_complexity_estimate()
        report.append(f"\n🧮 ESTIMACIÓN DE COMPLEJIDAD:")
        report.append(f"   Fórmula: O((F·H)² × I_avg × H_max/T_min)")
        report.append(f"   Valor calculado: {theoretical_complexity:.2e}")
        
        # Comparación con valores LaTeX
        report.append(f"\n📋 COMPARACIÓN CON VALORES TEÓRICOS LATEX:")
        report.append(f"   I_avg actual: {I_avg:.2f} vs teórico: ~2.0")
        report.append(f"   H_max/T_min actual: {H_max_T_min:.2f} vs teórico: ~10.0")
        report.append(f"   F actual: {F} vs teórico: ≤200")
        report.append(f"   H actual: {H:.2f} vs teórico: ≤5")
        
        # Tiempos de ejecución
        if self.operation_times:
            avg_op_time = statistics.mean(self.operation_times) * 1000  # convertir a ms
            report.append(f"\n⏱️  TIEMPOS DE EJECUCIÓN:")
            report.append(f"   Tiempo promedio por operación: {avg_op_time:.3f} ms")
            report.append(f"   Operaciones registradas: {len(self.operation_times)}")
        
        if self.check_isolation_times:
            avg_check_time = statistics.mean(self.check_isolation_times) * 1000  # ms
            report.append(f"   Tiempo promedio check_isolation: {avg_check_time:.3f} ms")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def log_summary(self):
        """Registra un resumen en el log."""
        self.logger.info(f"Complexity metrics: F={self.flows_processed}, "
                        f"H_avg={self.get_average_hops_per_flow():.2f}, "
                        f"I_avg={self.get_average_iterations():.2f}, "
                        f"H_max/T_min={self.get_max_hyperperiod_ratio():.2f}")


# Instancia global para capturar métricas
global_metrics = ComplexityMetrics()

def get_metrics() -> ComplexityMetrics:
    """Obtiene la instancia global de métricas."""
    return global_metrics

def reset_metrics():
    """Reinicia las métricas globales."""
    global global_metrics
    global_metrics = ComplexityMetrics()
